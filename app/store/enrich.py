"""Turn an inbound company into the enriched form we index.

Everything expensive or ambiguous is resolved once here, at write time, so the
read path stays cheap. Concretely this does three things:

1. **Lifts topics out of the description.** This is the important one. Topic is
   the only discriminating text in the corpus, so promoting it from prose to a
   structured list converts the dominant part of relevance from fuzzy scoring
   into exact set intersection.

2. **Denormalises revenue bounds.** The stored value is a bucket label; queries
   are numeric. Precomputing the numeric span lets range filters run as plain
   column comparisons.

3. **Backfills industry from topic when it is missing.** Only when absent — a
   caller who states an industry is always believed, even if it disagrees with
   what the topic implies.

Ingest-time cost is roughly 25s for the full 50k corpus, single threaded, and it
happens once per build.
"""

from __future__ import annotations

from app.models.company import CompanyIn, CompanyRecord
from app.nlq.matcher import PhraseMatcher
from app.taxonomy import (
    REVENUE_BUCKETS,
    TOPIC_INDEX,
    industry_for_topic,
)

# Built once at import; the matcher is stateless and safe to share.
_topic_matcher = PhraseMatcher(TOPIC_INDEX)


def extract_topics(text: str) -> list[str]:
    """Canonical topics mentioned anywhere in `text`.

    A scan rather than a template match on purpose. Most descriptions in the
    corpus are single-topic and templated, but the handful of hand-written ones
    carry several ("fraud detection, banking analytics, and risk assessment"),
    and a template parser would drop all but the first.
    """
    return _topic_matcher.scan_text(text)


def revenue_bounds(bucket: str | None) -> tuple[float | None, float | None]:
    if not bucket:
        return None, None
    span = REVENUE_BUCKETS.get(bucket)
    return span if span else (None, None)


def enrich(company: CompanyIn, *, company_id: int) -> CompanyRecord:
    """Produce the persisted record for a validated inbound company."""
    # Scan the name too: "Nordic Fintech Solutions" and "GridPulse Analytics"
    # carry topic signal that the description sometimes repeats and sometimes
    # does not.
    topics = extract_topics(f"{company.name}. {company.description}")

    industry = company.industry
    if not industry and topics:
        # Every topic in this corpus implies exactly one industry, so the first
        # match is unambiguous. Only used to fill a gap, never to override.
        industry = industry_for_topic(topics[0]) or ""

    rev_min, rev_max = revenue_bounds(company.revenue_range)

    return CompanyRecord(
        id=company_id,
        name=company.name,
        description=company.description,
        industry=industry,
        location=company.location,
        founded_year=company.founded_year,
        employee_count=company.employee_count,
        revenue_range=company.revenue_range,
        topics=topics,
        revenue_min=rev_min,
        revenue_max=rev_max,
    )
