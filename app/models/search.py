"""Search request/response contracts and the parsed-query representation.

The parsed query is part of the public response, not an internal detail. A user
who types a natural-language sentence and gets ten companies back has no way to
tell whether the system understood "founded after 2015" or quietly ignored it.
Returning the interpretation makes the system debuggable by the person using it,
and it is what lets the UI show editable filter chips.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.models.company import CompanyOut

SearchMode = Literal["hybrid", "keyword", "semantic"]


class NumericRange(BaseModel):
    """A half-open-or-closed numeric interval, with bound inclusivity recorded.

    Inclusivity is explicit because natural language distinguishes it and the
    difference is visible in results: "founded after 2015" excludes 2015, while
    "founded from 2015" includes it.
    """

    min: float | None = None
    max: float | None = None
    min_inclusive: bool = True
    max_inclusive: bool = True

    def describe(self, unit: str = "") -> str:
        suffix = f" {unit}" if unit else ""
        lo_op = ">=" if self.min_inclusive else ">"
        hi_op = "<=" if self.max_inclusive else "<"
        if self.min is not None and self.max is not None:
            return f"{lo_op} {_fmt(self.min)} and {hi_op} {_fmt(self.max)}{suffix}"
        if self.min is not None:
            return f"{lo_op} {_fmt(self.min)}{suffix}"
        if self.max is not None:
            return f"{hi_op} {_fmt(self.max)}{suffix}"
        return "any"


def _fmt(v: float) -> str:
    if v >= 1e9:
        return f"{v / 1e9:g}B"
    if v >= 1e6:
        return f"{v / 1e6:g}M"
    if v >= 1e3 and v == int(v) and v >= 1e4:
        return f"{v / 1e3:g}K"
    return f"{v:g}"


class ParsedQuery(BaseModel):
    """Structured interpretation of a natural-language query."""

    raw: str
    locations: list[str] = Field(default_factory=list)
    industries: list[str] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)

    founded: NumericRange | None = None
    employees: NumericRange | None = None
    revenue: NumericRange | None = None
    #: Bucket labels whose numeric span overlaps `revenue`.
    revenue_buckets: list[str] = Field(default_factory=list)

    #: Residual terms after entities and noise are removed; what BM25 receives.
    text_terms: list[str] = Field(default_factory=list)
    #: Text handed to the embedding model. Usually the raw query.
    semantic_text: str = ""

    #: Recognised but intentionally not enforced, e.g. "startups".
    soft_signals: list[str] = Field(default_factory=list)
    #: Ordered human-readable account of what the parser did.
    notes: list[str] = Field(default_factory=list)

    @property
    def has_structured_filters(self) -> bool:
        return bool(
            self.locations
            or self.industries
            or self.founded
            or self.employees
            or self.revenue_buckets
        )


class SearchFilters(BaseModel):
    """Explicit filters supplied by the caller.

    These are merged over whatever the parser inferred, so a UI can let a user
    correct a misparse without rewriting their sentence. An explicit value always
    wins over an inferred one.
    """

    model_config = ConfigDict(extra="forbid")

    locations: list[str] | None = None
    industries: list[str] | None = None
    topics: list[str] | None = None
    founded_after: int | None = None
    founded_before: int | None = None
    min_employees: int | None = None
    max_employees: int | None = None
    min_revenue: float | None = None
    max_revenue: float | None = None


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: Annotated[str, Field(min_length=1, max_length=1000)]
    limit: Annotated[int, Field(ge=1, le=100)] = 10
    offset: Annotated[int, Field(ge=0, le=10_000)] = 0

    #: "hybrid" is the product default. The single-retriever modes exist so the
    #: contribution of each can be measured independently — the ablation in
    #: docs/SELF_ASSESSMENT.md is produced with them.
    mode: SearchMode = "hybrid"

    filters: SearchFilters | None = None

    #: Include per-result score breakdowns.
    explain: bool = False

    #: Allow dropping the least central constraint when strict filtering yields
    #: too little. Disable to get exact-only semantics.
    allow_relaxation: bool = True


class RelaxationInfo(BaseModel):
    """What was given up to produce a non-empty result set."""

    applied: bool = False
    #: Constraints dropped, in the order they were dropped.
    dropped: list[str] = Field(default_factory=list)
    strict_result_count: int = 0
    message: str | None = None


class SearchResponse(BaseModel):
    query: str
    mode: SearchMode
    parsed: ParsedQuery
    results: list[CompanyOut]
    #: Total matching the filters, before limit/offset. Exact, not an estimate.
    total: int
    limit: int
    offset: int
    took_ms: float
    relaxation: RelaxationInfo = Field(default_factory=RelaxationInfo)
    #: Per-stage timings when `explain` is set; used by the benchmark harness.
    timings_ms: dict[str, float] = Field(default_factory=dict)


class SimilarRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    limit: Annotated[int, Field(ge=1, le=100)] = 10
    #: Restrict to the seed company's own industry.
    same_industry: bool = False
    #: Restrict to the seed company's own country.
    same_location: bool = False
    explain: bool = False


class SimilarResponse(BaseModel):
    seed: CompanyOut
    results: list[CompanyOut]
    total: int
    took_ms: float
