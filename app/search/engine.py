"""Hybrid search: structured filtering, then lexical + semantic retrieval fused.

Request flow
------------
    parse -> resolve filters -> mask -> [relax if needed]
          -> retrieve (BM25 || vector) -> fuse -> hydrate

Filtering runs first, always. It costs 0.4ms, produces an exact match count, and
makes the vector search cheaper the more selective it is. Everything downstream
operates on the surviving candidate set.

Fusion
------
Reciprocal Rank Fusion combines the two ranked retrievers:

    score(d) = sum over retrievers of  w / (k + rank(d))

RRF is used rather than a weighted sum of raw scores because BM25 scores and
cosine similarities are not comparable — BM25 is unbounded and corpus-relative,
cosine is bounded [-1, 1] — so any fixed blend of the two is arbitrary and
drifts as the corpus changes. Ranks are directly comparable. k=60 is the
constant from the original RRF paper and is not sensitive enough to tune here.

Topic overlap is added as a third, bounded term rather than a third ranked list.
As a ranked list it would be almost all ties, since thousands of companies share
any single topic. Scaling it by 1/k puts it on the same footing as a first-place
finish in one retriever, so a company matching both requested topics outranks
one matching either alone without swamping the text signals.

Boolean semantics for topics
----------------------------
Multiple topics are OR-ed, then companies matching more of them are ranked
higher. This is deliberate and it contradicts a literal reading of the queries:
"focused on diagnostics **and** patient monitoring" is a conjunction in English.
But every company in this corpus except eleven carries exactly one topic, so
strict conjunction returns zero results for that query and three others like it.
The user means "in the diagnostics / patient-monitoring space". OR-with-boost
delivers that, and the ranking still puts genuine both-topic matches first.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

import numpy as np

from app.config import settings
from app.core.logging import get_logger
from app.core.metrics import metrics
from app.models.company import CompanyOut, CompanyRecord, ScoreBreakdown
from app.models.search import (
    ParsedQuery,
    RelaxationInfo,
    SearchFilters,
    SearchRequest,
    SearchResponse,
    SimilarRequest,
    SimilarResponse,
)
from app.nlq.parser import parse
from app.search.columns import ColumnStore, FilterSpec
from app.search.state import SearchRuntime
from app.taxonomy import buckets_overlapping

log = get_logger(__name__)

# Order in which constraints are surrendered when a query is unsatisfiable.
# Least central first.
#
# Location and topic are never dropped. They are the substance of the request —
# a user asking for Finnish fintech does not want German biotech, and returning
# it under a "relaxed" banner is worse than an honest empty page. Numeric bounds
# are usually approximate in the user's head ("fewer than 20 employees" means
# "small"), and industry is frequently redundant because the topic implies it.
RELAXATION_ORDER: tuple[str, ...] = ("employees", "revenue", "founded", "industry")


# Awarded to candidates that satisfy every original constraint when the search
# had to be relaxed. Deliberately far larger than the maximum achievable fused
# score (w_keyword/k + w_vector/k + w_topic/k = 0.067 at defaults), so exact
# matches always precede widened ones regardless of relevance ordering within
# each group.
EXACT_MATCH_BONUS = 1.0

# Applied to near-misses when a numeric constraint is dropped.
#
# Sized to dominate text-relevance ordering within the relaxed set, which is a
# deliberate choice for this corpus: every candidate that survives relaxation
# already satisfies the retained filters, and their descriptions are drawn from
# the same handful of templates, so the BM25 and cosine differences between them
# are close to noise. Proximity to what the user actually asked for is the more
# meaningful signal. It stays well below EXACT_MATCH_BONUS, so genuine full
# matches are never displaced.
RELAX_PROXIMITY_WEIGHT = 0.05


@dataclass
class _Candidate:
    row: int
    keyword_rrf: float = 0.0
    vector_rrf: float = 0.0
    topic_score: float = 0.0
    exact_bonus: float = 0.0
    proximity: float = 0.0
    keyword_rank: int | None = None
    vector_rank: int | None = None
    matched_topics: list[str] = field(default_factory=list)

    @property
    def total(self) -> float:
        return (
            self.keyword_rrf
            + self.vector_rrf
            + self.topic_score
            + self.exact_bonus
            + self.proximity
        )


class SearchEngine:
    def __init__(self, runtime: SearchRuntime, columns: ColumnStore) -> None:
        self.rt = runtime
        self.columns = columns

    # -- public API -------------------------------------------------------

    async def search(self, request: SearchRequest) -> SearchResponse:
        started = time.perf_counter()
        timings: dict[str, float] = {}

        with _stage(timings, "parse"):
            parsed = parse(request.query)

        spec = self._build_spec(parsed, request.filters)

        with _stage(timings, "filter"):
            strict_mask = self.columns.mask(spec)
            strict_total = int(strict_mask.sum())

        # -- relaxation ---------------------------------------------------
        mask = strict_mask
        relaxation = RelaxationInfo(applied=False, strict_result_count=strict_total)
        if (
            request.allow_relaxation
            and settings.relaxation_enabled
            and strict_total < settings.relaxation_min_results
        ):
            mask, relaxation = self._relax(spec, strict_total)

        total = int(mask.sum())

        # Fuse enough candidates to serve the requested page. Without this a
        # deep offset silently returns nothing.
        pool = max(settings.candidate_pool, request.offset + request.limit)

        with _stage(timings, "retrieve"):
            candidates = await self._retrieve(parsed, request, mask, pool)

        # A relaxed search still contains the companies that satisfied every
        # original constraint. Those are what the user actually asked for, so
        # they lead — otherwise relaxing a query can bury the one exact match
        # underneath fifty approximate ones, which is worse than not relaxing.
        if relaxation.applied:
            if strict_total:
                for cand in candidates.values():
                    if strict_mask[cand.row]:
                        cand.exact_bonus = EXACT_MATCH_BONUS
            self._apply_proximity(spec, relaxation.dropped, candidates)

        with _stage(timings, "rank"):
            # Ties broken by company id so paging is stable across requests.
            ordered = sorted(
                candidates.values(),
                key=lambda c: (-c.total, int(self.columns.ids[c.row])),
            )
            page = ordered[request.offset : request.offset + request.limit]

        with _stage(timings, "hydrate"):
            results = await self._hydrate(page, parsed, explain=request.explain)

        took = (time.perf_counter() - started) * 1000
        metrics.observe_ms("search_total_ms", took, mode=request.mode)
        metrics.incr("search_requests_total", mode=request.mode)
        if relaxation.applied:
            metrics.incr("search_relaxed_total")

        return SearchResponse(
            query=request.query,
            mode=request.mode,
            parsed=parsed,
            results=results,
            total=total,
            limit=request.limit,
            offset=request.offset,
            took_ms=round(took, 2),
            relaxation=relaxation,
            timings_ms=timings if request.explain else {},
        )

    async def similar(self, company_id: int, request: SimilarRequest) -> SimilarResponse | None:
        """Companies most like `company_id`.

        Uses the seed's stored document vector directly — no embedding call, so
        this path avoids the 25ms that dominates a text query.
        """
        started = time.perf_counter()
        seed = await self.rt.db.get(company_id)
        if seed is None:
            return None

        row = self.columns.row_of(company_id)
        if row is None or self.rt.vector is None:
            # Without vectors, fall back to topic overlap, which is still a
            # reasonable notion of "similar" in this corpus.
            return await self._similar_by_topic(seed, request, started)

        spec = FilterSpec(
            exclude_ids=[company_id],
            industries=[seed.industry] if request.same_industry and seed.industry else (),
            locations=[seed.location] if request.same_location and seed.location else (),
        )
        mask = self.columns.mask(spec)
        query_vec = self.rt.vector.vector_for_row(row)
        rows, scores = self.rt.vector.search(query_vec, request.limit, mask=mask)

        ids = [int(self.columns.ids[r]) for r in rows]
        records = await self.rt.db.get_many(ids)
        results = []
        for r, score in zip(rows, scores, strict=False):
            cid = int(self.columns.ids[r])
            rec = records.get(cid)
            if rec is None:
                continue
            shared = [t for t in rec.topics if t in seed.topics]
            results.append(
                CompanyOut.from_record(
                    rec,
                    score=round(float(score), 6),
                    matched_on=[f"shared topic: {t}" for t in shared] or ["semantic similarity"],
                    score_breakdown=ScoreBreakdown(vector=round(float(score), 6))
                    if request.explain
                    else None,
                )
            )

        took = (time.perf_counter() - started) * 1000
        metrics.observe_ms("similar_total_ms", took)
        return SimilarResponse(
            seed=CompanyOut.from_record(seed),
            results=results,
            total=int(mask.sum()),
            took_ms=round(took, 2),
        )

    # -- filter construction ----------------------------------------------

    def _build_spec(self, parsed: ParsedQuery, explicit: SearchFilters | None) -> FilterSpec:
        """Merge parsed filters with any the caller supplied.

        Explicit always wins. A UI shows the parse as editable chips; when a user
        corrects one, their correction must not be re-overridden by the parser.
        """
        spec = FilterSpec(
            locations=list(parsed.locations),
            industries=list(parsed.industries),
            topics=list(parsed.topics),
            founded_min=parsed.founded.min if parsed.founded else None,
            founded_max=parsed.founded.max if parsed.founded else None,
            founded_min_inclusive=parsed.founded.min_inclusive if parsed.founded else True,
            founded_max_inclusive=parsed.founded.max_inclusive if parsed.founded else True,
            employees_min=parsed.employees.min if parsed.employees else None,
            employees_max=parsed.employees.max if parsed.employees else None,
            employees_min_inclusive=parsed.employees.min_inclusive if parsed.employees else True,
            employees_max_inclusive=parsed.employees.max_inclusive if parsed.employees else True,
            revenue_buckets=list(parsed.revenue_buckets),
        )
        if explicit is None:
            return spec

        if explicit.locations is not None:
            spec.locations = explicit.locations
        if explicit.industries is not None:
            spec.industries = explicit.industries
        if explicit.topics is not None:
            spec.topics = explicit.topics
        if explicit.founded_after is not None:
            spec.founded_min, spec.founded_min_inclusive = explicit.founded_after, True
        if explicit.founded_before is not None:
            spec.founded_max, spec.founded_max_inclusive = explicit.founded_before, True
        if explicit.min_employees is not None:
            spec.employees_min, spec.employees_min_inclusive = explicit.min_employees, True
        if explicit.max_employees is not None:
            spec.employees_max, spec.employees_max_inclusive = explicit.max_employees, True
        if explicit.min_revenue is not None or explicit.max_revenue is not None:
            spec.revenue_buckets = buckets_overlapping(explicit.min_revenue, explicit.max_revenue)
        return spec

    def _relax(self, spec: FilterSpec, strict_total: int) -> tuple[np.ndarray, RelaxationInfo]:
        """Drop constraints one at a time until enough results survive.

        Returns the relaxed mask and a record of what was given up. The response
        always states this — silently widening a search is how a system loses a
        user's trust.
        """
        dropped: list[str] = []
        active = set(spec.active_constraints())
        current = spec
        mask = self.columns.mask(current)

        for constraint in RELAXATION_ORDER:
            if constraint not in active:
                continue
            current = current.without(constraint)
            dropped.append(constraint)
            mask = self.columns.mask(current)
            if int(mask.sum()) >= settings.relaxation_min_results:
                break

        count = int(mask.sum())
        if not dropped or count == strict_total:
            return self.columns.mask(spec), RelaxationInfo(
                applied=False, strict_result_count=strict_total
            )

        readable = ", ".join(dropped)
        log.info(
            "search_relaxed",
            extra={"dropped": dropped, "strict": strict_total, "relaxed": count},
        )
        matched = (
            "No companies matched"
            if strict_total == 0
            else f"Only {strict_total} company matched"
            if strict_total == 1
            else f"Only {strict_total} companies matched"
        )
        constraint_word = "constraint was" if len(dropped) == 1 else "constraints were"
        return mask, RelaxationInfo(
            applied=True,
            dropped=dropped,
            strict_result_count=strict_total,
            message=(
                f"{matched} every constraint, so the {readable} {constraint_word} "
                f"relaxed. Showing {count} wider matches"
                + (
                    f", with the {strict_total} exact "
                    f"{'match' if strict_total == 1 else 'matches'} listed first."
                    if strict_total
                    else "."
                )
            ),
        )

    def _apply_proximity(
        self, spec: FilterSpec, dropped: list[str], candidates: dict[int, _Candidate]
    ) -> None:
        """Favour near-misses when a numeric constraint has been dropped.

        Asking for "fewer than 20 employees" and being shown a 3,000-person
        company ahead of a 114-person one is technically a relaxed match and
        practically useless. The bonus decays with relative distance past the
        original bound, so the companies closest to what was asked for surface
        first within the widened set.

        Only applies to employees and founding year. Revenue is stored as a
        bucket, so distance past a bound is not meaningfully measurable.

        The two fields need different distance scales. Headcount is perceived
        relatively — 40 people is "twice 20", so the bound itself is the scale.
        Years are perceived absolutely: a company founded in 2010 against a 2018
        bound is eight years early, not 0.4% early. Normalising years by 2018
        would make every distance ~0 and hand every candidate the full bonus,
        which is exactly the bug this scale table avoids.
        """
        # (column, lo, hi, scale) — scale None means "use the bound itself".
        YEAR_SCALE = 10.0
        columns = {
            "employees": (self.columns.employees, spec.employees_min, spec.employees_max, None),
            "founded": (self.columns.founded, spec.founded_min, spec.founded_max, YEAR_SCALE),
        }
        for constraint in dropped:
            entry = columns.get(constraint)
            if entry is None:
                continue
            column, lo, hi, scale = entry
            if lo is None and hi is None:
                continue
            for cand in candidates.values():
                value = float(column[cand.row])
                if value < 0:
                    continue
                if hi is not None and value > hi:
                    distance = (value - hi) / (scale or max(abs(hi), 1.0))
                elif lo is not None and value < lo:
                    distance = (lo - value) / (scale or max(abs(lo), 1.0))
                else:
                    distance = 0.0
                cand.proximity += RELAX_PROXIMITY_WEIGHT / (1.0 + distance)

    # -- retrieval --------------------------------------------------------

    async def _retrieve(
        self,
        parsed: ParsedQuery,
        request: SearchRequest,
        mask: np.ndarray,
        pool: int,
    ) -> dict[int, _Candidate]:
        candidates: dict[int, _Candidate] = {}
        if not mask.any():
            return candidates

        want_keyword = request.mode in ("hybrid", "keyword")
        want_vector = request.mode in ("hybrid", "semantic") and self.rt.semantic_enabled

        tasks = []
        if want_keyword:
            tasks.append(self._keyword_pass(parsed, mask, pool))
        if want_vector:
            tasks.append(self._vector_pass(parsed, mask, pool))

        # Both retrievers are independent; run them concurrently so the request
        # costs max(embed+vector, bm25) rather than the sum.
        results = await asyncio.gather(*tasks) if tasks else []

        for contribution in results:
            for row, (rrf, rank, kind) in contribution.items():
                cand = candidates.setdefault(row, _Candidate(row=row))
                if kind == "keyword":
                    cand.keyword_rrf, cand.keyword_rank = rrf, rank
                else:
                    cand.vector_rrf, cand.vector_rank = rrf, rank

        # A filter-only query ("companies in Sweden with 50 to 250 employees")
        # produces no lexical terms and, in keyword mode, no candidates at all.
        # Fall back to the filtered set so the request still returns its matches.
        if not candidates:
            rows = np.flatnonzero(mask)[:pool]
            for row in rows:
                candidates[int(row)] = _Candidate(row=int(row))

        self._apply_topic_scores(parsed, candidates)
        return candidates

    async def _keyword_pass(
        self, parsed: ParsedQuery, mask: np.ndarray, pool: int
    ) -> dict[int, tuple[float, int, str]]:
        terms = list(dict.fromkeys(parsed.text_terms + parsed.topics))
        if not terms:
            return {}

        selected = np.flatnonzero(mask)
        # Push the candidate set into SQL when it is small enough to be a useful
        # restriction; above that the planner does better scanning, and the mask
        # is re-applied below regardless.
        allowed = (
            [int(self.columns.ids[r]) for r in selected] if selected.size <= 4000 else None
        )

        hits = await self.rt.keyword.search(terms, limit=pool, allowed_ids=allowed)

        out: dict[int, tuple[float, int, str]] = {}
        rank = 0
        for hit in hits:
            row = self.columns.row_of(hit.company_id)
            if row is None or not mask[row]:
                continue  # outside the filter; only possible on the unrestricted path
            out[row] = (settings.weight_keyword / (settings.rrf_k + rank), rank, "keyword")
            rank += 1
        return out

    async def _vector_pass(
        self, parsed: ParsedQuery, mask: np.ndarray, pool: int
    ) -> dict[int, tuple[float, int, str]]:
        if self.rt.embedder is None or self.rt.vector is None:
            return {}
        query_vec = await self.rt.embedder.embed_query(parsed.semantic_text or parsed.raw)
        rows, _scores = self.rt.vector.search(query_vec, pool, mask=mask)
        return {
            int(row): (settings.weight_vector / (settings.rrf_k + rank), rank, "vector")
            for rank, row in enumerate(rows)
        }

    def _apply_topic_scores(
        self, parsed: ParsedQuery, candidates: dict[int, _Candidate]
    ) -> None:
        """Boost candidates by how many of the requested topics they carry."""
        if not parsed.topics:
            return
        wanted = list(parsed.topics)
        n = len(wanted)
        membership = {
            topic: self.columns.topic_postings.get(topic) for topic in wanted
        }
        row_sets = {
            topic: set(rows.tolist()) for topic, rows in membership.items() if rows is not None
        }
        for cand in candidates.values():
            matched = [t for t, rows in row_sets.items() if cand.row in rows]
            if matched:
                cand.matched_topics = matched
                # Scaled by 1/rrf_k so a full topic match is worth roughly a
                # first-place finish in one retriever — influential, not decisive.
                cand.topic_score = settings.weight_topic * (len(matched) / n) / settings.rrf_k

    # -- output -----------------------------------------------------------

    async def _hydrate(
        self, page: list[_Candidate], parsed: ParsedQuery, *, explain: bool
    ) -> list[CompanyOut]:
        if not page:
            return []
        ids = [int(self.columns.ids[c.row]) for c in page]
        records = await self.rt.db.get_many(ids)

        out: list[CompanyOut] = []
        for cand in page:
            cid = int(self.columns.ids[cand.row])
            rec = records.get(cid)
            if rec is None:
                continue
            out.append(
                CompanyOut.from_record(
                    rec,
                    score=round(cand.total, 6),
                    matched_on=self._explain_match(rec, cand, parsed),
                    score_breakdown=ScoreBreakdown(
                        keyword=round(cand.keyword_rrf, 6),
                        vector=round(cand.vector_rrf, 6),
                        topic=round(cand.topic_score, 6),
                        keyword_rank=cand.keyword_rank,
                        vector_rank=cand.vector_rank,
                    )
                    if explain
                    else None,
                )
            )
        return out

    @staticmethod
    def _explain_match(
        rec: CompanyRecord, cand: _Candidate, parsed: ParsedQuery
    ) -> list[str]:
        reasons: list[str] = []
        if cand.exact_bonus:
            reasons.append("matches every requested constraint")
        for topic in cand.matched_topics:
            reasons.append(f"topic: {topic}")
        if parsed.locations and rec.location in parsed.locations:
            reasons.append(f"location: {rec.location}")
        if parsed.industries and rec.industry in parsed.industries:
            reasons.append(f"industry: {rec.industry}")
        if cand.keyword_rank is not None:
            reasons.append(f"keyword rank {cand.keyword_rank + 1}")
        if cand.vector_rank is not None:
            reasons.append(f"semantic rank {cand.vector_rank + 1}")
        return reasons

    async def _similar_by_topic(
        self, seed: CompanyRecord, request: SimilarRequest, started: float
    ) -> SimilarResponse:
        """Topic-overlap fallback when no vector index is loaded."""
        spec = FilterSpec(
            topics=seed.topics,
            exclude_ids=[seed.id],
            industries=[seed.industry] if request.same_industry and seed.industry else (),
            locations=[seed.location] if request.same_location and seed.location else (),
        )
        mask = self.columns.mask(spec)
        rows = np.flatnonzero(mask)[: request.limit]
        ids = [int(self.columns.ids[r]) for r in rows]
        records = await self.rt.db.get_many(ids)
        results = [
            CompanyOut.from_record(
                records[cid],
                score=1.0,
                matched_on=[f"shared topic: {t}" for t in records[cid].topics if t in seed.topics],
            )
            for cid in ids
            if cid in records
        ]
        return SimilarResponse(
            seed=CompanyOut.from_record(seed),
            results=results,
            total=int(mask.sum()),
            took_ms=round((time.perf_counter() - started) * 1000, 2),
        )


class _stage:
    """Context manager recording a stage duration into `timings`."""

    def __init__(self, timings: dict[str, float], name: str) -> None:
        self.timings = timings
        self.name = name

    def __enter__(self) -> _stage:
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self.timings[self.name] = round((time.perf_counter() - self._t0) * 1000, 3)


# `_stage` is used with `with` inside async functions; both stages it wraps are
# synchronous, so a plain context manager is correct and cheaper than an async one.
