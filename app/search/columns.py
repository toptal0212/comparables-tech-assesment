"""In-memory columnar store for structured filtering.

Why this exists as a separate structure from SQLite
---------------------------------------------------
The vector retriever scores candidates with a dense matrix multiply over numpy
arrays. To restrict that to companies matching "Finland, Fintech, founded after
2015" it needs a boolean mask over row positions — not a set of ids from a SQL
query. Round-tripping ids out of SQLite and back into numpy on every request
would cost more than the search itself.

So the filterable fields are held as parallel numpy arrays, laid out in the same
row order as the vector matrix. Producing a mask is then a handful of vectorised
comparisons: about 120µs across 50k rows, against 2.5ms for the vector search it
gates. Cheap enough to run first, always, which is what makes it possible to
answer "how many companies match these filters" exactly rather than estimating.

Memory is ~1MB per 50k companies. At 5M it is roughly 100MB, still comfortable.

Topics use an inverted index rather than a column
-------------------------------------------------
Topic membership is many-to-many, so it cannot be a scalar column. A dense
boolean matrix (rows x topics) would work at 27 topics but scales badly — a
500-topic taxonomy over 5M companies would be 2.5GB. Postings lists cost only
what is actually populated, and a topic filter touches only the rows that carry
those topics (a few thousand) instead of scanning all of them.

Sentinels
---------
Missing numeric values are stored as -1, and every numeric predicate also
requires `>= 0`. A company with an unknown founding year must not satisfy
"founded before 2005" — absence of data is not evidence of a small number.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np

from app.core.logging import get_logger
from app.models.company import CompanyRecord
from app.taxonomy import REVENUE_BUCKET_ORDER

log = get_logger(__name__)

MISSING = -1


@dataclass
class FilterSpec:
    """Resolved structured constraints. All fields AND together."""

    locations: Sequence[str] = ()
    industries: Sequence[str] = ()
    #: Topics are OR-ed against each other, then AND-ed with everything else.
    #: See docs/DESIGN.md — strict topic conjunction returns nothing on a corpus
    #: where all but 11 companies carry exactly one topic.
    topics: Sequence[str] = ()

    founded_min: float | None = None
    founded_max: float | None = None
    founded_min_inclusive: bool = True
    founded_max_inclusive: bool = True

    employees_min: float | None = None
    employees_max: float | None = None
    employees_min_inclusive: bool = True
    employees_max_inclusive: bool = True

    revenue_buckets: Sequence[str] = ()

    #: Row positions to exclude, used by /similar to drop the seed company.
    exclude_ids: Sequence[int] = ()

    def active_constraints(self) -> list[str]:
        """Names of the constraints in play, used to drive relaxation."""
        out = []
        if self.locations:
            out.append("location")
        if self.industries:
            out.append("industry")
        if self.topics:
            out.append("topic")
        if self.founded_min is not None or self.founded_max is not None:
            out.append("founded")
        if self.employees_min is not None or self.employees_max is not None:
            out.append("employees")
        if self.revenue_buckets:
            out.append("revenue")
        return out

    def without(self, constraint: str) -> FilterSpec:
        """A copy with one constraint dropped."""
        clone = FilterSpec(**self.__dict__)
        if constraint == "location":
            clone.locations = ()
        elif constraint == "industry":
            clone.industries = ()
        elif constraint == "topic":
            clone.topics = ()
        elif constraint == "founded":
            clone.founded_min = clone.founded_max = None
        elif constraint == "employees":
            clone.employees_min = clone.employees_max = None
        elif constraint == "revenue":
            clone.revenue_buckets = ()
        return clone

    @property
    def is_empty(self) -> bool:
        return not self.active_constraints()


@dataclass
class ColumnStore:
    ids: np.ndarray                       # int64, ascending
    industry: np.ndarray                  # int16 codes
    location: np.ndarray                  # int16 codes
    founded: np.ndarray                   # int32, MISSING for unknown
    employees: np.ndarray                 # int64, MISSING for unknown
    revenue_bucket: np.ndarray            # int8 index into REVENUE_BUCKET_ORDER
    topic_postings: dict[str, np.ndarray] # topic -> row positions
    industry_codes: dict[str, int]
    location_codes: dict[str, int]
    _all_true: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0, bool))

    @property
    def size(self) -> int:
        return int(self.ids.shape[0])

    # -- construction -----------------------------------------------------

    @classmethod
    def build(cls, records: Iterable[CompanyRecord]) -> ColumnStore:
        recs = sorted(records, key=lambda r: r.id)
        n = len(recs)

        industry_codes: dict[str, int] = {}
        location_codes: dict[str, int] = {}
        postings: dict[str, list[int]] = {}

        ids = np.empty(n, dtype=np.int64)
        industry = np.empty(n, dtype=np.int16)
        location = np.empty(n, dtype=np.int16)
        founded = np.full(n, MISSING, dtype=np.int32)
        employees = np.full(n, MISSING, dtype=np.int64)
        revenue = np.full(n, MISSING, dtype=np.int8)

        bucket_index = {label: i for i, label in enumerate(REVENUE_BUCKET_ORDER)}

        for row, rec in enumerate(recs):
            ids[row] = rec.id
            industry[row] = industry_codes.setdefault(rec.industry, len(industry_codes))
            location[row] = location_codes.setdefault(rec.location, len(location_codes))
            if rec.founded_year is not None:
                founded[row] = rec.founded_year
            if rec.employee_count is not None:
                employees[row] = rec.employee_count
            if rec.revenue_range in bucket_index:
                revenue[row] = bucket_index[rec.revenue_range]
            for topic in rec.topics:
                postings.setdefault(topic, []).append(row)

        store = cls(
            ids=ids,
            industry=industry,
            location=location,
            founded=founded,
            employees=employees,
            revenue_bucket=revenue,
            topic_postings={t: np.asarray(rows, dtype=np.int64) for t, rows in postings.items()},
            industry_codes=industry_codes,
            location_codes=location_codes,
            _all_true=np.ones(n, dtype=bool),
        )
        log.info(
            "column_store_built",
            extra={
                "rows": n,
                "topics": len(postings),
                "industries": len(industry_codes),
                "locations": len(location_codes),
                "bytes": int(
                    ids.nbytes + industry.nbytes + location.nbytes
                    + founded.nbytes + employees.nbytes + revenue.nbytes
                ),
            },
        )
        return store

    # -- lookups ----------------------------------------------------------

    def row_of(self, company_id: int) -> int | None:
        """Row position for a company id, or None. Binary search over sorted ids."""
        pos = int(np.searchsorted(self.ids, company_id))
        if pos < self.size and int(self.ids[pos]) == company_id:
            return pos
        return None

    def rows_of(self, company_ids: Sequence[int]) -> np.ndarray:
        """Row positions for ids, silently dropping ids that are not present."""
        if len(company_ids) == 0:
            return np.empty(0, dtype=np.int64)
        wanted = np.asarray(company_ids, dtype=np.int64)
        pos = np.searchsorted(self.ids, wanted)
        pos = np.clip(pos, 0, self.size - 1)
        return pos[self.ids[pos] == wanted]

    def ids_for(self, rows: np.ndarray) -> np.ndarray:
        return self.ids[rows]

    # -- filtering --------------------------------------------------------

    @staticmethod
    def _code_lut(codes: Sequence[int], size: int) -> np.ndarray:
        """Boolean lookup table for a set of category codes.

        Indexing a small LUT by the code column (`lut[column]`) is a single
        gather, where `np.isin` sorts its argument and binary-searches per row.
        Measured over 50k rows, moving the category predicates to LUTs took a
        full six-constraint filter from 1.10ms to 0.40ms p50. Worth having: the
        mask runs on every request and gates a 2.5ms vector search.
        """
        lut = np.zeros(size, dtype=bool)
        lut[list(codes)] = True
        return lut

    def mask(self, spec: FilterSpec) -> np.ndarray:
        """Boolean array over rows: True where the company satisfies every constraint."""
        if spec.is_empty and not spec.exclude_ids:
            # A fresh array, not the cached one: callers combine masks in place
            # and must never be handed a shared buffer.
            return np.ones(self.size, dtype=bool)

        mask = np.ones(self.size, dtype=bool)

        if spec.locations:
            codes = [
                self.location_codes[loc] for loc in spec.locations if loc in self.location_codes
            ]
            if not codes:
                # A location we have never seen cannot match anything. Returning
                # all-False here is correct and avoids a pointless scan.
                return np.zeros(self.size, dtype=bool)
            mask &= self._code_lut(codes, len(self.location_codes))[self.location]

        if spec.industries:
            codes = [
                self.industry_codes[ind] for ind in spec.industries if ind in self.industry_codes
            ]
            if not codes:
                return np.zeros(self.size, dtype=bool)
            mask &= self._code_lut(codes, len(self.industry_codes))[self.industry]

        if spec.topics:
            # Union the postings lists, which touches only rows carrying these
            # topics rather than scanning the whole column.
            topic_mask = np.zeros(self.size, dtype=bool)
            matched_any = False
            for topic in spec.topics:
                rows = self.topic_postings.get(topic)
                if rows is not None and rows.size:
                    topic_mask[rows] = True
                    matched_any = True
            if not matched_any:
                return np.zeros(self.size, dtype=bool)
            mask &= topic_mask

        mask = self._apply_range(
            mask, self.founded, spec.founded_min, spec.founded_max,
            spec.founded_min_inclusive, spec.founded_max_inclusive,
        )
        mask = self._apply_range(
            mask, self.employees, spec.employees_min, spec.employees_max,
            spec.employees_min_inclusive, spec.employees_max_inclusive,
        )

        if spec.revenue_buckets:
            codes = [
                REVENUE_BUCKET_ORDER.index(b)
                for b in spec.revenue_buckets
                if b in REVENUE_BUCKET_ORDER
            ]
            if not codes:
                return np.zeros(self.size, dtype=bool)
            # Shift by one so the MISSING sentinel (-1) lands on index 0 instead
            # of wrapping to the end of the table under negative indexing.
            lut = np.zeros(len(REVENUE_BUCKET_ORDER) + 1, dtype=bool)
            lut[[c + 1 for c in codes]] = True
            mask &= lut[self.revenue_bucket + 1]

        if spec.exclude_ids:
            rows = self.rows_of(list(spec.exclude_ids))
            if rows.size:
                mask[rows] = False

        return mask

    @staticmethod
    def _apply_range(
        mask: np.ndarray,
        column: np.ndarray,
        lo: float | None,
        hi: float | None,
        lo_inclusive: bool,
        hi_inclusive: bool,
    ) -> np.ndarray:
        if lo is None and hi is None:
            return mask
        # Unknown values never satisfy a numeric predicate: absence of data is
        # not evidence of a small number. In-place `&=` throughout, so a
        # multi-constraint filter allocates one temporary per predicate rather
        # than a fresh mask each time.
        mask &= column != MISSING
        if lo is not None:
            mask &= (column >= lo) if lo_inclusive else (column > lo)
        if hi is not None:
            mask &= (column <= hi) if hi_inclusive else (column < hi)
        return mask
