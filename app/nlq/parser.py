"""Natural-language query parser.

Turns "Find fintech companies in Finland founded after 2015 with revenue between
10M and 100M" into structured filters plus a residual text query.

Why rules rather than an LLM
----------------------------
An LLM call would parse a wider space of phrasings, and it would also add
300-800ms to a request with a 200ms budget, introduce a network dependency on
the hot path, cost money per query, and produce different output for the same
input from one day to the next. The queries this system actually receives are
short, templated and drawn from a closed vocabulary of countries, industries,
topics and numeric predicates. Rules cover that in ~0.2ms, deterministically,
and every decision is inspectable.

The honest limit: phrasings outside the alias tables fall through to lexical and
semantic retrieval rather than becoming filters. That degrades gracefully — the
query still returns sensible results, just without the hard constraint. If usage
showed a long tail of unparsed queries, the right move is an LLM parser behind a
cache keyed on the normalised query, with these rules as the synchronous
fallback. The `ParsedQuery` contract would not change.

Pipeline
--------
1. Extract numeric constraints (year / employees / revenue) and consume their
   tokens, so "500M" cannot later be mistaken for anything else.
2. Match topics on what remains — most specific first.
3. Match industries, then locations, on what still remains.
4. Whatever is left, minus stopwords and corpus boilerplate, becomes the lexical
   query.

Order matters. Numbers are consumed before phrases because "5G analytics"
contains a digit; topics before industries because "energy forecasting" must not
be shredded into the Energy industry plus a stray "forecasting".
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.models.search import NumericRange, ParsedQuery
from app.nlq.matcher import PhraseMatcher
from app.taxonomy import (
    INDUSTRY_INDEX,
    LOCATION_INDEX,
    REGION_ALIASES,
    TOPIC_INDEX,
    buckets_overlapping,
    industry_for_topic,
    normalize_tokens,
)

# Sentinel written over consumed positions. Contains no word characters, so it
# can never take part in a later phrase match, and keeps indices aligned.
_CONSUMED = "\x00"

_topic_matcher = PhraseMatcher(TOPIC_INDEX)
_industry_matcher = PhraseMatcher(INDUSTRY_INDEX)
_location_matcher = PhraseMatcher(LOCATION_INDEX)
_region_matcher = PhraseMatcher({k: k for k in REGION_ALIASES})

# ---------------------------------------------------------------------------
# Numeric vocabulary
# ---------------------------------------------------------------------------

_NUMBER = re.compile(r"^(\d+(?:[.,]\d+)?)(k|m|mm|b|bn)?$")

_MAGNITUDE: dict[str, float] = {
    "k": 1e3, "thousand": 1e3,
    "m": 1e6, "mm": 1e6, "million": 1e6, "millions": 1e6,
    "b": 1e9, "bn": 1e9, "billion": 1e9, "billions": 1e9,
}

# Comparator phrases, longest first so "at least" beats a bare "least" and
# "no more than" is not read as "more than".
_GREATER: tuple[tuple[str, ...], ...] = (
    ("greater", "than"), ("more", "than"), ("at", "least"), ("no", "fewer", "than"),
    ("no", "less", "than"), ("north", "of"), ("in", "excess", "of"),
    ("over",), ("above",), ("exceeding",), ("beyond",), ("min",), ("minimum",),
)
_LESS: tuple[tuple[str, ...], ...] = (
    ("no", "more", "than"), ("not", "exceeding"), ("fewer", "than"), ("less", "than"),
    ("at", "most"), ("up", "to"), ("under",), ("below",), ("max",), ("maximum",),
)
_AFTER: tuple[tuple[str, ...], ...] = (
    ("later", "than"), ("after",), ("since",), ("post",), ("from",),
)
_BEFORE: tuple[tuple[str, ...], ...] = (
    ("earlier", "than"), ("prior", "to"), ("before",), ("pre",), ("until",),
)
_BETWEEN: tuple[tuple[str, ...], ...] = (("between",), ("from",))

# Bounds are exclusive for "more/less than" and inclusive for "at least/at
# most", because that is what the words mean and the difference is observable.
_EXCLUSIVE_GREATER = {("greater", "than"), ("more", "than"), ("over",), ("above",),
                      ("exceeding",), ("beyond",), ("north", "of"), ("in", "excess", "of")}
_EXCLUSIVE_LESS = {("fewer", "than"), ("less", "than"), ("under",), ("below",),
                   ("not", "exceeding")}

_FIELD_WORDS: dict[str, str] = {}
for _w in ("employee", "staff", "headcount", "people", "person", "worker", "fte", "hire"):
    _FIELD_WORDS[_w] = "employees"
for _w in ("revenue", "turnover", "sale", "arr", "income", "earnings", "topline"):
    _FIELD_WORDS[_w] = "revenue"
for _w in ("founded", "established", "incorporated", "started", "launched", "founding",
           "inception", "began", "created"):
    _FIELD_WORDS[_w] = "founded"

_YEAR_MIN, _YEAR_MAX = 1800, 2100

# ---------------------------------------------------------------------------
# Terms removed from the lexical residue
# ---------------------------------------------------------------------------

_STOPWORDS = frozenset(
    """a an the and or but of in on at to for with from by as is are be been being
    that which who whom whose this these those it its their our your my me we us
    you they them he she his her i do does did doing have has had having will would
    shall should can could may might must not no nor so than then there here when
    where why how all any both each few more most other some such only own same too
    very just about into over under again further once""".split()
)

# Query framing verbs and nouns that carry no retrieval signal.
_QUERY_VERBS = frozenset(
    """find show get list search retrieve display give tell fetch identify locate
    looking look want need seeking seek return surface company companies business
    businesses firm firms organisation organisations organization organizations
    startup startups scaleup vendor vendors provider providers player players
    working focused focus building build make making use using used based
    specialising specialistng specializing specialise specialize operating please
    target targets result results top best relevant similar like""".split()
)

# Corpus boilerplate. These words appear ~1,000 times each, spread evenly across
# every industry, so leaving them in the lexical query pulls in thousands of
# unrelated documents. "US technology companies building infrastructure for data
# pipelines" is the motivating case: bare "infrastructure" matches ~8k rows.
_CORPUS_NOISE = frozenset(
    """ai powered data driven machine learning real time cloud native automated
    distributed predictive platform solution engine software system infrastructure""".split()
)

# Field keywords ('founded', 'employees', 'revenue') are scaffolding for the
# numeric constraints and carry no lexical signal of their own.
_LEXICAL_DROP = _STOPWORDS | _QUERY_VERBS | _CORPUS_NOISE | frozenset(_FIELD_WORDS)

# "us" is both a country and a pronoun. After these words it is the pronoun.
_US_PRONOUN_CONTEXT = frozenset({"show", "tell", "give", "send", "let", "get", "find"})

_SOFT_SIGNAL_WORDS: dict[str, str] = {
    "startup": "early-stage",
    "scaleup": "growth-stage",
    "sme": "small",
    "small": "small",
    "large": "large",
    "enterprise": "large",
    "young": "early-stage",
    "emerging": "early-stage",
    "established": "mature",
    "mature": "mature",
}


@dataclass
class _Amount:
    value: float
    end: int
    had_magnitude: bool


def _parse_amount(tokens: list[str], i: int) -> _Amount | None:
    """Read a number at position `i`, absorbing a trailing magnitude word."""
    if i >= len(tokens):
        return None
    m = _NUMBER.match(tokens[i])
    if not m:
        return None
    value = float(m.group(1).replace(",", "."))
    suffix = m.group(2)
    if suffix:
        return _Amount(value * _MAGNITUDE[suffix], i + 1, True)
    # "500 million"
    if i + 1 < len(tokens) and tokens[i + 1] in _MAGNITUDE:
        return _Amount(value * _MAGNITUDE[tokens[i + 1]], i + 2, True)
    return _Amount(value, i + 1, False)


def _match_phrase(
    tokens: list[str], i: int, phrases: tuple[tuple[str, ...], ...]
) -> tuple[str, ...] | None:
    for phrase in phrases:
        if tuple(tokens[i : i + len(phrase)]) == phrase:
            return phrase
    return None


def _infer_field(
    tokens: list[str], comparator_start: int, amount_end: int, amount: _Amount
) -> str:
    """Decide whether a constraint is about employees, revenue or founding year.

    Explicit context word wins; otherwise the shape of the number decides.

    `tokens` must be the *partially consumed* list. Already-consumed positions
    act as barriers so a scan cannot borrow context from a neighbouring
    constraint. Without that barrier, "founded after 2018 with fewer than 100
    employees" reads the `founded` belonging to the first constraint and files
    the headcount limit under founding year.
    """
    # Look back from the comparator: "revenue over 500M", "founded after 2015".
    for j in range(comparator_start - 1, max(-1, comparator_start - 5), -1):
        if j < 0:
            break
        if tokens[j] == _CONSUMED:
            break
        if (field := _FIELD_WORDS.get(tokens[j])) is not None:
            return field
    # Look forward past the number: "fewer than 20 employees".
    for j in range(amount_end, min(len(tokens), amount_end + 4)):
        if tokens[j] == _CONSUMED:
            break
        if (field := _FIELD_WORDS.get(tokens[j])) is not None:
            return field
    # No context. A magnitude suffix means money; a bare four-digit number in a
    # plausible range means a year; anything else is a headcount.
    if amount.had_magnitude:
        return "revenue"
    if amount.value == int(amount.value) and _YEAR_MIN <= amount.value <= _YEAR_MAX:
        return "founded"
    return "employees"


def _apply(ranges: dict[str, NumericRange], field: str, **kw: object) -> None:
    r = ranges.setdefault(field, NumericRange())
    for key, value in kw.items():
        setattr(r, key, value)


def _extract_numeric(tokens: list[str]) -> tuple[dict[str, NumericRange], list[str]]:
    """Pull numeric constraints out, returning them and the token list with the
    consumed positions blanked."""
    ranges: dict[str, NumericRange] = {}
    out = list(tokens)
    n = len(tokens)
    i = 0

    while i < n:
        # -- "between X and Y" / "from X to Y" ----------------------------
        if (phrase := _match_phrase(tokens, i, _BETWEEN)) is not None:
            lo = _parse_amount(tokens, i + len(phrase))
            if lo is not None:
                joiner = lo.end
                if joiner < n and tokens[joiner] in ("and", "to", "-"):
                    hi = _parse_amount(tokens, joiner + 1)
                    if hi is not None:
                        field = _infer_field(out, i, hi.end, hi)
                        _apply(ranges, field, min=lo.value, max=hi.value,
                               min_inclusive=True, max_inclusive=True)
                        for k in range(i, hi.end):
                            out[k] = _CONSUMED
                        i = hi.end
                        continue

        # -- comparator + amount ------------------------------------------
        handled = False
        for phrases, kind in (
            (_GREATER, "gt"), (_LESS, "lt"), (_AFTER, "after"), (_BEFORE, "before")
        ):
            phrase = _match_phrase(tokens, i, phrases)
            if phrase is None:
                continue
            amount = _parse_amount(tokens, i + len(phrase))
            if amount is None:
                continue
            field = _infer_field(out, i, amount.end, amount)

            if kind == "gt":
                _apply(ranges, field, min=amount.value,
                       min_inclusive=phrase not in _EXCLUSIVE_GREATER)
            elif kind == "lt":
                _apply(ranges, field, max=amount.value,
                       max_inclusive=phrase not in _EXCLUSIVE_LESS)
            elif kind == "after":
                # "after 2015" excludes 2015; "since 2015" and "from 2015" include it.
                _apply(ranges, field, min=amount.value,
                       min_inclusive=phrase in (("since",), ("from",)))
            else:
                _apply(ranges, field, max=amount.value, max_inclusive=False)

            for k in range(i, amount.end):
                out[k] = _CONSUMED
            i = amount.end
            handled = True
            break
        if handled:
            continue

        # -- bare "X to Y" ("with 50 to 250 employees") --------------------
        lo = _parse_amount(tokens, i)
        if lo is not None and lo.end < n and tokens[lo.end] in ("to", "-"):
            hi = _parse_amount(tokens, lo.end + 1)
            if hi is not None:
                field = _infer_field(out, i, hi.end, hi)
                _apply(ranges, field, min=lo.value, max=hi.value,
                       min_inclusive=True, max_inclusive=True)
                for k in range(i, hi.end):
                    out[k] = _CONSUMED
                i = hi.end
                continue

        # -- bare amount governed by a nearby field word -------------------
        # "founded in 2015", "revenue of 2 billion", "with 500 employees".
        # Requires an explicit field word so that incidental numbers ("top 10
        # companies") are left alone.
        if lo is not None:
            field = None
            for j in range(i - 1, max(-1, i - 4), -1):
                if j < 0 or out[j] == _CONSUMED:
                    break
                if (f := _FIELD_WORDS.get(out[j])) is not None:
                    field = f
                    break
            if field is None:
                for j in range(lo.end, min(n, lo.end + 3)):
                    if out[j] == _CONSUMED:
                        break
                    if (f := _FIELD_WORDS.get(out[j])) is not None:
                        field = f
                        break

            plausible_year = (
                not lo.had_magnitude
                and lo.value == int(lo.value)
                and _YEAR_MIN <= lo.value <= _YEAR_MAX
            )
            if field == "founded" and not plausible_year:
                field = None  # "founded" next to a non-year number: not a year.

            if field is not None:
                # An exact value. For revenue this resolves, via bucket overlap,
                # to the single bucket containing the amount.
                _apply(ranges, field, min=lo.value, max=lo.value,
                       min_inclusive=True, max_inclusive=True)
                for k in range(i, lo.end):
                    out[k] = _CONSUMED
                i = lo.end
                continue

        i += 1

    return ranges, out


def _mask(tokens: list[str], start: int, end: int) -> None:
    for k in range(start, end):
        tokens[k] = _CONSUMED


def parse(query: str) -> ParsedQuery:
    """Parse a natural-language company query into filters plus residual text."""
    tokens = normalize_tokens(query)
    notes: list[str] = []

    # 1 -- numeric constraints, consumed first ----------------------------
    ranges, working = _extract_numeric(tokens)

    founded = ranges.get("founded")
    employees = ranges.get("employees")
    revenue = ranges.get("revenue")

    if founded:
        notes.append(f"founded {founded.describe()}")
    if employees:
        notes.append(f"employees {employees.describe()}")
    if revenue:
        notes.append(f"revenue {revenue.describe()}")

    # 2 -- topics (most specific) -----------------------------------------
    topics: list[str] = []
    for m in _topic_matcher.find_all(working):
        if m.canonical not in topics:
            topics.append(m.canonical)
        _mask(working, m.start, m.end)
    if topics:
        notes.append(f"topics: {', '.join(topics)}")

    # 3 -- industries ------------------------------------------------------
    industries: list[str] = []
    for m in _industry_matcher.find_all(working):
        if m.canonical not in industries:
            industries.append(m.canonical)
        _mask(working, m.start, m.end)

    # A topic implies exactly one industry in this corpus. When the query also
    # named industries, keep only those the topics agree with. This is what
    # stops "Automotive software companies … autonomous driving" from carrying
    # both Automotive and Technology: the topic settles it.
    implied = {ind for t in topics if (ind := industry_for_topic(t))}
    if implied and industries:
        agreed = [i for i in industries if i in implied]
        if agreed and len(agreed) < len(industries):
            dropped = [i for i in industries if i not in agreed]
            notes.append(f"narrowed industry to {', '.join(agreed)} (dropped {', '.join(dropped)})")
            industries = agreed
    if industries:
        notes.append(f"industries: {', '.join(industries)}")

    # 4 -- locations, then regional shorthands -----------------------------
    locations: list[str] = []
    for m in _location_matcher.find_all(working):
        if m.text == "us" and m.start > 0 and tokens[m.start - 1] in _US_PRONOUN_CONTEXT:
            continue  # "show us companies" — pronoun, not country
        if m.canonical not in locations:
            locations.append(m.canonical)
        _mask(working, m.start, m.end)

    for m in _region_matcher.find_all(working):
        expanded = REGION_ALIASES.get(m.canonical, set())
        added = [c for c in sorted(expanded) if c not in locations]
        if added:
            locations.extend(added)
            notes.append(f"expanded '{m.canonical}' to {', '.join(added)}")
        _mask(working, m.start, m.end)

    if locations:
        notes.append(f"locations: {', '.join(locations)}")

    # 5 -- soft signals ----------------------------------------------------
    soft: list[str] = []
    for tok in working:
        if tok in _SOFT_SIGNAL_WORDS:
            label = _SOFT_SIGNAL_WORDS[tok]
            if label not in soft:
                soft.append(label)
    # Deliberately not enforced. 90% of this corpus has 500+ employees, so
    # turning "startups" into a hard size filter would empty most result sets
    # for a word the user used loosely.
    if soft:
        notes.append(f"soft signals (not enforced as filters): {', '.join(soft)}")

    # 6 -- lexical residue -------------------------------------------------
    text_terms = [
        tok
        for tok in working
        if tok != _CONSUMED and tok not in _LEXICAL_DROP and not _NUMBER.match(tok) and len(tok) > 1
    ]

    revenue_buckets = buckets_overlapping(
        revenue.min if revenue else None,
        revenue.max if revenue else None,
    ) if revenue else []

    return ParsedQuery(
        raw=query,
        locations=locations,
        industries=industries,
        topics=topics,
        founded=founded,
        employees=employees,
        revenue=revenue,
        revenue_buckets=revenue_buckets,
        text_terms=text_terms,
        semantic_text=query.strip(),
        soft_signals=soft,
        notes=notes,
    )
