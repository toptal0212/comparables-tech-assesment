"""Greedy longest-match phrase scanner over a controlled vocabulary.

One scanner serves both sides of the system, which is the point: the same code
that lifts topics out of a company description at ingest time resolves topics in
a user's query at search time. If the two used different matching logic they
would drift, and a topic that indexes one way but parses another simply never
matches.

Algorithm: walk the token stream left to right; at each position try the longest
n-gram first and take the first hit. Longest-first is what makes
"supply chain visibility" win over the shorter "supply chain", and non-
overlapping consumption stops one phrase being counted twice.

Cost is O(n * max_ngram) dictionary lookups — a few microseconds for a query,
and about 25 seconds for the whole 50k corpus, which happens once at ingest.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.taxonomy import normalize_tokens


@dataclass(frozen=True)
class Match:
    canonical: str
    #: Token span consumed, as a half-open [start, end) range into the input.
    start: int
    end: int
    #: The surface tokens that produced the match, useful for explaining results.
    text: str


class PhraseMatcher:
    """Maps alias phrases to canonical values.

    The supplied index is re-keyed through `normalize_tokens` so that the stored
    keys and the incoming tokens are singularised identically. Without that step
    "smart grids" in a query would not reach the "smart grid" topic.
    """

    def __init__(self, alias_index: dict[str, str]) -> None:
        self._index: dict[str, str] = {}
        max_n = 1
        for alias, canonical in alias_index.items():
            key = " ".join(normalize_tokens(alias))
            if not key:
                continue
            # setdefault: an earlier (more canonical) alias keeps the slot when
            # two aliases normalise to the same key.
            self._index.setdefault(key, canonical)
            max_n = max(max_n, key.count(" ") + 1)
        self._max_ngram = max_n

    @property
    def max_ngram(self) -> int:
        return self._max_ngram

    def find_all(self, tokens: list[str]) -> list[Match]:
        """Every non-overlapping canonical value present, in order of appearance."""
        matches: list[Match] = []
        i = 0
        n = len(tokens)
        while i < n:
            # Longest window first so the most specific phrase wins.
            upper = min(self._max_ngram, n - i)
            for size in range(upper, 0, -1):
                key = " ".join(tokens[i : i + size])
                canonical = self._index.get(key)
                if canonical is not None:
                    matches.append(Match(canonical, i, i + size, key))
                    i += size
                    break
            else:
                i += 1
        return matches

    def find_unique(self, tokens: list[str]) -> list[str]:
        """Distinct canonical values, order preserved."""
        seen: dict[str, None] = {}
        for m in self.find_all(tokens):
            seen.setdefault(m.canonical, None)
        return list(seen)

    def scan_text(self, text: str) -> list[str]:
        return self.find_unique(normalize_tokens(text))
