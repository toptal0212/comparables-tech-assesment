"""Company schemas.

Three shapes, deliberately kept apart:

* `CompanyIn`  — what a client may POST. Tolerant: accepts common field-name
  variants and either a bucket label or a raw number for revenue.
* `CompanyRecord` — the enriched internal form that gets persisted, with derived
  fields (topics, numeric revenue bounds) filled in.
* `CompanyOut` — what the API returns, plus per-result scoring metadata.

The tolerance in `CompanyIn` is intentional. The brief says to use the provided
dataset as-is, and we do; but the bonus ingestion endpoint is the path by which
*other* data arrives, and in practice that data never matches your schema
exactly. Accepting `country` for `location` costs one alias and removes a whole
class of integration friction.
"""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator

from app.taxonomy import (
    INDUSTRY_INDEX,
    LOCATION_INDEX,
    REVENUE_BUCKETS,
    normalize,
)

# The corpus spans 1995-2024. The bounds below are wide enough to accept real
# companies while still rejecting typos and epoch-seconds pasted into the field.
MIN_YEAR = 1600
MAX_YEAR = 2100


def bucket_for_amount(amount: float) -> str:
    """Map a raw revenue figure onto the dataset's bucket labels."""
    for label, (lo, hi) in REVENUE_BUCKETS.items():
        if lo <= amount < hi:
            return label
    return "500M+"


class CompanyIn(BaseModel):
    """Inbound company payload."""

    model_config = ConfigDict(populate_by_name=True, str_strip_whitespace=True)

    id: int | None = Field(
        default=None,
        description="Optional. Omit to let the service assign one.",
    )
    name: Annotated[str, Field(min_length=1, max_length=512)] = Field(
        validation_alias=AliasChoices("name", "company_name", "company")
    )
    description: Annotated[str, Field(max_length=8000)] = Field(
        default="",
        validation_alias=AliasChoices("description", "summary", "about", "overview"),
    )
    industry: str = Field(
        default="",
        validation_alias=AliasChoices("industry", "sector", "vertical"),
    )
    location: str = Field(
        default="",
        validation_alias=AliasChoices("location", "country", "hq", "headquarters"),
    )
    founded_year: int | None = Field(
        default=None,
        validation_alias=AliasChoices("founded_year", "founded", "year_founded", "founding_year"),
    )
    employee_count: int | None = Field(
        default=None,
        ge=0,
        le=10_000_000,
        validation_alias=AliasChoices("employee_count", "employees", "headcount", "staff"),
    )
    revenue_range: str | None = Field(
        default=None,
        validation_alias=AliasChoices("revenue_range", "revenue", "revenue_band", "turnover"),
    )

    @field_validator("founded_year")
    @classmethod
    def _year_in_range(cls, v: int | None) -> int | None:
        if v is None:
            return None
        if not (MIN_YEAR <= v <= MAX_YEAR):
            raise ValueError(f"founded_year must be between {MIN_YEAR} and {MAX_YEAR}")
        return v

    @model_validator(mode="before")
    @classmethod
    def _coerce_numeric_revenue(cls, data: Any) -> Any:
        """Accept `revenue: 25000000` as well as `revenue_range: "10M-50M"`.

        Callers integrating a numeric source should not have to learn our bucket
        labels to talk to us.
        """
        if not isinstance(data, dict):
            return data
        for key in ("revenue_range", "revenue", "turnover"):
            value = data.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                data[key] = bucket_for_amount(float(value))
                break
        return data

    @field_validator("industry")
    @classmethod
    def _canon_industry(cls, v: str) -> str:
        # Snap to the canonical spelling when we recognise it; otherwise keep the
        # caller's value rather than dropping data we simply have no alias for.
        return INDUSTRY_INDEX.get(normalize(v), v)

    @field_validator("location")
    @classmethod
    def _canon_location(cls, v: str) -> str:
        return LOCATION_INDEX.get(normalize(v), v)

    @field_validator("revenue_range")
    @classmethod
    def _validate_bucket(cls, v: str | None) -> str | None:
        if v is None or v == "":
            return None
        if v in REVENUE_BUCKETS:
            return v
        # Tolerate spacing/case variants of a known label before giving up.
        squashed = v.replace(" ", "").upper()
        for label in REVENUE_BUCKETS:
            if label.replace(" ", "").upper() == squashed:
                return label
        raise ValueError(
            f"revenue_range must be one of {list(REVENUE_BUCKETS)} or a numeric amount"
        )


class CompanyRecord(BaseModel):
    """Enriched, persisted form."""

    id: int
    name: str
    description: str
    industry: str
    location: str
    founded_year: int | None
    employee_count: int | None
    revenue_range: str | None
    #: Canonical topics lifted from the description at ingest time.
    topics: list[str] = Field(default_factory=list)
    #: Numeric span of `revenue_range`, denormalised so range queries avoid a join.
    revenue_min: float | None = None
    revenue_max: float | None = None


class ScoreBreakdown(BaseModel):
    """Per-retriever contribution, so a result can explain itself.

    Returned on every hit. Being able to see *why* something ranked where it did
    is the difference between a search system you can tune and one you can only
    guess at.
    """

    keyword: float = 0.0
    vector: float = 0.0
    topic: float = 0.0
    keyword_rank: int | None = None
    vector_rank: int | None = None


class CompanyOut(BaseModel):
    id: int
    name: str
    description: str
    industry: str
    location: str
    founded_year: int | None = None
    employee_count: int | None = None
    revenue_range: str | None = None
    topics: list[str] = Field(default_factory=list)

    score: float = 0.0
    score_breakdown: ScoreBreakdown | None = None
    #: Human-readable reasons, e.g. ["topic: fraud detection", "location: Finland"].
    matched_on: list[str] = Field(default_factory=list)

    @classmethod
    def from_record(cls, rec: CompanyRecord, **extra: Any) -> CompanyOut:
        return cls(**rec.model_dump(exclude={"revenue_min", "revenue_max"}), **extra)
