"""Query parser behaviour, driven by the brief's own example queries."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.nlq.parser import parse

QUERIES = json.loads(
    (Path(__file__).parent.parent / "scripts" / "example_queries.json").read_text(
        encoding="utf-8"
    )
)["queries"]


@pytest.mark.parametrize("case", QUERIES, ids=lambda c: f"Q{c['id']}")
def test_example_queries_parse_as_expected(case):
    """Every query in the brief, against its recorded expected parse."""
    parsed = parse(case["text"])
    expect = case["expect"]

    if "locations" in expect:
        assert parsed.locations == expect["locations"]
    if "industries" in expect:
        assert parsed.industries == expect["industries"]
    if "topics" in expect:
        assert sorted(parsed.topics) == sorted(expect["topics"])
    if "soft_signals" in expect:
        assert parsed.soft_signals == expect["soft_signals"]
    if "revenue_buckets" in expect:
        assert parsed.revenue_buckets == expect["revenue_buckets"]

    for field in ("founded", "employees", "revenue"):
        if field not in expect:
            continue
        actual = getattr(parsed, field)
        assert actual is not None, f"{field} not parsed from {case['text']!r}"
        for key, value in expect[field].items():
            assert getattr(actual, key) == value, f"{field}.{key}"


@pytest.mark.parametrize("case", QUERIES, ids=lambda c: f"Q{c['id']}")
def test_residue_contains_no_scaffolding(case):
    """Framing verbs, field keywords and boilerplate must not reach BM25.

    "US technology companies building infrastructure for data pipelines" would
    otherwise send bare "infrastructure" to the lexical retriever, which matches
    ~8k unrelated rows in the real corpus.
    """
    residue = parse(case["text"]).text_terms
    for banned in ("companies", "find", "show", "working", "founded", "revenue",
                   "employees", "employee", "infrastructure", "ai", "platform",
                   "software", "using", "focused", "building"):
        assert banned not in residue, f"{banned!r} leaked into residue: {residue}"


def test_constraint_does_not_borrow_context_from_its_neighbour():
    """Regression: Q14 filed a headcount limit under founding year.

    "founded after 2018 with fewer than 100 employees" — the backward scan for
    "fewer than" reached past its own clause and found the "founded" belonging
    to the previous constraint. Consumed tokens now act as barriers.
    """
    parsed = parse(
        "Show German companies founded after 2018 with fewer than 100 employees "
        "working on drug discovery."
    )
    assert parsed.founded is not None
    assert parsed.founded.min == 2018 and parsed.founded.max is None
    assert parsed.employees is not None
    assert parsed.employees.max == 100 and parsed.employees.min is None


def test_bound_inclusivity_follows_the_words():
    assert parse("founded after 2015").founded.min_inclusive is False
    assert parse("founded since 2015").founded.min_inclusive is True
    assert parse("at least 500 employees").employees.min_inclusive is True
    assert parse("more than 500 employees").employees.min_inclusive is False
    assert parse("fewer than 20 employees").employees.max_inclusive is False
    assert parse("up to 20 employees").employees.max_inclusive is True


def test_topics_disambiguate_industry():
    """"Automotive software" matches two industries; the topic settles it."""
    parsed = parse("Automotive software companies in Germany working on autonomous driving.")
    assert parsed.industries == ["Automotive"]
    assert "Technology" not in parsed.industries


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("Show us fintech companies", []),          # pronoun, not the USA
        ("US technology companies", ["USA"]),
        ("companies in the United States", ["USA"]),
    ],
)
def test_us_pronoun_guard(query, expected):
    assert parse(query).locations == expected


def test_no_is_not_norway():
    assert parse("companies with no revenue in Norway").locations == ["Norway"]
    assert parse("companies with no revenue").locations == []


def test_regions_expand():
    assert set(parse("Nordic fintech").locations) == {"Finland", "Sweden", "Norway"}


@pytest.mark.parametrize(
    ("query", "field", "attr", "value"),
    [
        ("firms with revenue of 2 billion", "revenue", "min", 2e9),
        ("companies with 500 employees", "employees", "min", 500),
        ("biotech founded in 2015", "founded", "min", 2015),
        ("turnover up to 5 million", "revenue", "max", 5e6),
        ("between 10M and 100M revenue", "revenue", "min", 1e7),
        ("with 50 to 250 employees", "employees", "max", 250),
    ],
)
def test_numeric_forms(query, field, attr, value):
    parsed = parse(query)
    actual = getattr(parsed, field)
    assert actual is not None, f"{field} not parsed from {query!r}"
    assert getattr(actual, attr) == value


def test_incidental_numbers_are_ignored():
    """"top 10 companies" has no field keyword, so 10 is not a constraint."""
    parsed = parse("top 10 companies in Sweden")
    assert parsed.employees is None
    assert parsed.founded is None
    assert parsed.locations == ["Sweden"]


def test_startup_is_soft_not_enforced():
    """90% of the corpus has 500+ staff; a hard size filter would empty it."""
    parsed = parse("Biotech startups in Germany working on drug discovery.")
    assert parsed.soft_signals == ["early-stage"]
    assert parsed.employees is None
    assert parsed.founded is None


@pytest.mark.parametrize(
    "query",
    ["", "   ", "!!!", "a", "the and or of", "🙂", "x" * 999],
)
def test_degenerate_input_does_not_raise(query):
    parsed = parse(query)
    assert parsed.raw == query


def test_parse_is_deterministic():
    q = "Find fintech companies in Finland founded after 2015 with revenue between 10M and 100M."
    first = parse(q).model_dump()
    for _ in range(5):
        assert parse(q).model_dump() == first
