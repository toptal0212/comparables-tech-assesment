"""Enrichment, persistence and the FTS keyword index."""

from __future__ import annotations

import pytest

from app.models.company import CompanyIn, bucket_for_amount
from app.search.keyword import build_match_expression
from app.store.enrich import enrich, extract_topics, revenue_bounds

# --- enrichment -------------------------------------------------------------


def test_topics_are_extracted_from_compound_descriptions():
    """A template parser would keep only the first topic here."""
    topics = extract_topics(
        "AI-powered drug discovery and molecular analysis platform for biotech research teams."
    )
    assert set(topics) == {"drug discovery", "molecular analysis"}


def test_boilerplate_head_yields_no_topic():
    """The generated head alone must not imply anything."""
    assert extract_topics("cloud-native platform for") == []
    assert extract_topics("AI-powered infrastructure") == []


def test_industry_is_inferred_only_when_absent():
    inferred = enrich(
        CompanyIn(name="X", description="Engine for drug discovery.", industry=""),
        company_id=1,
    )
    assert inferred.industry == "Biotech"

    # A stated industry is believed even when the topic disagrees.
    stated = enrich(
        CompanyIn(name="X", description="Engine for drug discovery.", industry="Retail"),
        company_id=2,
    )
    assert stated.industry == "Retail"


@pytest.mark.parametrize(
    ("amount", "bucket"),
    [(0, "0-1M"), (999_999, "0-1M"), (1e6, "1M-10M"), (25e6, "10M-50M"),
     (75e6, "50M-100M"), (250e6, "100M-500M"), (2e9, "500M+")],
)
def test_numeric_revenue_maps_to_bucket(amount, bucket):
    assert bucket_for_amount(amount) == bucket


def test_revenue_bounds_round_trip():
    assert revenue_bounds("10M-50M") == (1e7, 5e7)
    assert revenue_bounds(None) == (None, None)
    assert revenue_bounds("nonsense") == (None, None)


def test_field_aliases_are_accepted():
    """The ingestion endpoint has to tolerate data not shaped like ours."""
    company = CompanyIn.model_validate(
        {
            "company_name": "Helsinki Fraud AI",
            "summary": "Real-time platform for fraud detection.",
            "sector": "fintech",
            "country": "finland",
            "founded": 2021,
            "employees": 30,
            "revenue": 25_000_000,
        }
    )
    assert company.name == "Helsinki Fraud AI"
    assert company.industry == "Fintech"   # canonicalised from "fintech"
    assert company.location == "Finland"   # canonicalised from "finland"
    assert company.revenue_range == "10M-50M"  # bucketed from a raw number
    assert company.founded_year == 2021
    assert company.employee_count == 30


def test_implausible_year_is_rejected():
    with pytest.raises(ValueError):
        CompanyIn(name="X", founded_year=1_700_000_000)  # epoch seconds


def test_unknown_revenue_label_is_rejected():
    with pytest.raises(ValueError):
        CompanyIn(name="X", revenue_range="a lot")


# --- FTS query construction -------------------------------------------------


@pytest.mark.parametrize(
    "hostile",
    ["e-commerce", 'say "hi"', "AND OR NEAR", "a*b", "col:val", "^caret",
     "NOT x", "(unbalanced", 'quote"inject', "-minus"],
)
def test_match_expression_neutralises_fts_syntax(hostile):
    """FTS5 treats -, *, ", :, ^ and the boolean words as operators.

    Unescaped, "e-commerce" parses as a NOT and silently returns wrong results.
    """
    expr = build_match_expression([hostile])
    if expr is None:
        return
    for operator in ('"', "*", ":", "^", "-", "("):
        inner = expr.strip('"')
        assert operator not in inner, f"{operator!r} survived in {expr!r}"


def test_match_expression_drops_useless_terms():
    assert build_match_expression([]) is None
    assert build_match_expression(["", " ", "a"]) is None
    assert build_match_expression(["fraud detection"]) == '"fraud detection"'
    assert build_match_expression(["a", "fraud"]) == '"fraud"'


# --- persistence ------------------------------------------------------------


async def test_upsert_and_read_back(db, corpus):
    assert await db.count() == len(corpus)
    first = await db.get(1)
    assert first is not None and first.topics


async def test_upsert_is_idempotent(db, corpus):
    before = await db.count()
    record = enrich(corpus[0], company_id=corpus[0].id or 1)
    await db.upsert_many([record])
    assert await db.count() == before


async def test_update_changes_fields_and_fts(db):
    from app.search.keyword import KeywordIndex

    record = await db.get(1)
    record.name = "Zzyzx Unmistakable Corp"
    await db.upsert_many([record])

    hits = await KeywordIndex(db).search(["zzyzx"], limit=5)
    assert [h.company_id for h in hits] == [1]


async def test_delete_removes_from_fts(db):
    from app.search.keyword import KeywordIndex

    record = await db.get(2)
    record.name = "Deletable Unique Marker"
    await db.upsert_many([record])
    assert await KeywordIndex(db).search(["deletable"], limit=5)

    assert await db.delete(2) is True
    assert await db.get(2) is None
    assert await KeywordIndex(db).search(["deletable"], limit=5) == []
    assert await db.delete(2) is False


async def test_get_many_returns_only_existing(db):
    found = await db.get_many([1, 2, 999_999])
    assert set(found) == {1, 2}
    assert await db.get_many([]) == {}


async def test_keyword_scores_are_higher_is_better(db):
    """bm25() returns negatives; the boundary flips the sign once."""
    from app.search.keyword import KeywordIndex

    hits = await KeywordIndex(db).search(["fraud detection"], limit=10)
    assert hits
    assert all(h.score > 0 for h in hits)
    assert hits == sorted(hits, key=lambda h: -h.score)


async def test_keyword_respects_allowed_ids(db):
    from app.search.keyword import KeywordIndex

    index = KeywordIndex(db)
    everything = await index.search(["fraud detection"], limit=50)
    assert len(everything) > 1
    target = everything[1].company_id
    restricted = await index.search(["fraud detection"], limit=50, allowed_ids=[target])
    assert [h.company_id for h in restricted] == [target]
    assert await index.search(["fraud detection"], limit=50, allowed_ids=[]) == []
