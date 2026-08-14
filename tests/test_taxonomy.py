"""Taxonomy invariants.

These are regression guards for a class of bug that is silent when it happens:
a bad alias does not raise, it just mislabels thousands of companies and makes
search quietly worse.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.store.enrich import extract_topics
from app.taxonomy import (
    INDUSTRY_INDEX,
    LOCATION_INDEX,
    REVENUE_BUCKETS,
    TOPIC_BY_NAME,
    TOPICS,
    buckets_overlapping,
    find_noise_aliases,
    industry_for_topic,
    normalize,
    normalize_tokens,
    singularize,
)

DATASET = Path(__file__).parent.parent / "sample_dataset" / "companies.json"


def test_no_alias_collides_with_description_boilerplate():
    """Every topic alias must discriminate on its own.

    Bare "cloud" and "infrastructure" once did not: both appear in the generated
    description head ("cloud-native software for …", "AI-powered infrastructure
    for …"), so they attached a Technology topic to ~13k companies across every
    other industry.
    """
    assert find_noise_aliases() == []


def test_every_topic_maps_to_exactly_one_industry():
    for topic in TOPICS:
        assert topic.industry, f"{topic.name} has no industry"
        assert industry_for_topic(topic.name) == topic.industry


def test_no_alias_shadows_another_topics_canonical_name():
    canonical = {normalize(t.name) for t in TOPICS}
    for topic in TOPICS:
        for alias in topic.aliases:
            key = normalize(alias)
            if key in canonical and key != normalize(topic.name):
                pytest.fail(f"alias {alias!r} of {topic.name!r} shadows another topic")


@pytest.mark.parametrize(
    ("token", "expected"),
    [
        ("grids", "grid"),
        ("companies", "company"),
        ("analytics", "analytics"),  # kept: canonical form is plural
        ("payments", "payments"),
        ("diagnostics", "diagnostics"),
        ("business", "business"),  # -ss must not lose its s
    ],
)
def test_singularize(token, expected):
    assert singularize(token) == expected


def test_normalize_strips_accents_and_punctuation():
    assert normalize("Zürich, Ltd.") == "zurich ltd"
    assert normalize("E-Commerce") == "e commerce"
    # "+" survives because "500M+" is a bucket label.
    assert "+" in normalize("500M+")


@pytest.mark.parametrize(
    ("lo", "hi", "expected"),
    [
        (500e6, None, ["500M+"]),
        (100e6, None, ["100M-500M", "500M+"]),
        (None, 10e6, ["0-1M", "1M-10M"]),
        (10e6, 100e6, ["10M-50M", "50M-100M"]),
        (200e6, None, ["100M-500M", "500M+"]),  # partial overlap counts
        (None, None, list(REVENUE_BUCKETS)),
    ],
)
def test_revenue_bucket_overlap(lo, hi, expected):
    assert buckets_overlapping(lo, hi) == expected


def test_ambiguous_aliases_are_absent():
    """Words too common in English to be safe as entity aliases."""
    for dangerous in ("it", "data"):
        assert INDUSTRY_INDEX.get(dangerous) is None, f"{dangerous!r} is too generic"
    # "companies with no revenue" must not resolve to Norway.
    assert LOCATION_INDEX.get("no") is None


def test_multiword_aliases_still_resolve():
    assert INDUSTRY_INDEX[normalize("information technology")] == "Technology"
    assert LOCATION_INDEX[normalize("united kingdom")] == "UK"
    assert LOCATION_INDEX[normalize("the netherlands")] == "Netherlands"


def test_topic_lookup_is_singularisation_stable():
    """A plural in a query must reach the singular canonical topic."""
    assert extract_topics("we build smart grids") == ["smart grid"]
    assert " ".join(normalize_tokens("smart grids")) == "smart grid"


@pytest.mark.skipif(not DATASET.exists(), reason="sample dataset not present")
def test_extraction_never_contradicts_the_corpus():
    """Empirical guard over the real 50k corpus.

    Stronger than the lexical check above: every topic the extractor pulls out
    of a description must imply that company's own industry. Any contradiction
    means an alias is firing on text it should not.
    """
    data = json.loads(DATASET.read_text(encoding="utf-8"))
    contradictions = []
    without_topic = 0
    for record in data:
        topics = extract_topics(f"{record['name']}. {record['description']}")
        if not topics:
            without_topic += 1
        for topic in topics:
            if industry_for_topic(topic) != record["industry"]:
                contradictions.append((record["id"], topic, record["industry"]))

    assert contradictions[:5] == [], f"{len(contradictions)} topic/industry contradictions"
    assert without_topic == 0, f"{without_topic} records yielded no topic"


@pytest.mark.skipif(not DATASET.exists(), reason="sample dataset not present")
def test_every_corpus_topic_is_in_the_taxonomy():
    """The taxonomy must cover what the corpus actually contains."""
    data = json.loads(DATASET.read_text(encoding="utf-8"))
    covered = {t for r in data for t in extract_topics(r["description"])}
    assert covered <= set(TOPIC_BY_NAME)
    # All 27 known topics should appear; a missing one means dead vocabulary.
    assert len(covered) == len(TOPICS)
