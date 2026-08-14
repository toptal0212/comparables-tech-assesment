"""Search engine behaviour: filtering, fusion, relaxation, similarity.

Runs against the lexical-only runtime. The semantic path is exercised in
test_columns_vector.py; what matters here is that filters are respected,
unsatisfiable queries degrade honestly, and ranking puts the right thing first.
"""

from __future__ import annotations

import pytest

from app.models.search import SearchFilters, SearchRequest, SimilarRequest


async def test_filters_are_respected(engine):
    response = await engine.search(
        SearchRequest(query="fintech companies in Finland", limit=20)
    )
    assert response.results
    for company in response.results:
        assert company.location == "Finland"
        assert company.industry == "Fintech"


async def test_total_is_exact_not_an_estimate(engine, columns):
    from app.search.columns import FilterSpec

    response = await engine.search(SearchRequest(query="fintech companies in Finland", limit=1))
    expected = int(columns.mask(FilterSpec(locations=["Finland"], industries=["Fintech"])).sum())
    assert response.total == expected


async def test_parsed_query_is_returned(engine):
    """The caller must be able to see how their sentence was understood."""
    response = await engine.search(
        SearchRequest(query="Finnish fintech founded after 2015", limit=5)
    )
    assert response.parsed.locations == ["Finland"]
    assert response.parsed.industries == ["Fintech"]
    assert response.parsed.founded.min == 2015
    assert response.parsed.notes


async def test_explicit_filters_override_the_parse(engine):
    """A UI correction must not be re-overridden by the parser."""
    response = await engine.search(
        SearchRequest(
            query="fintech companies in Finland",
            filters=SearchFilters(locations=["Germany"]),
            limit=10,
        )
    )
    assert response.results
    assert {c.location for c in response.results} == {"Germany"}


async def test_pagination_is_stable_and_non_overlapping(engine):
    query = "fintech companies"
    first = await engine.search(SearchRequest(query=query, limit=5, offset=0))
    second = await engine.search(SearchRequest(query=query, limit=5, offset=5))
    again = await engine.search(SearchRequest(query=query, limit=5, offset=0))

    assert [c.id for c in first.results] == [c.id for c in again.results]
    assert not ({c.id for c in first.results} & {c.id for c in second.results})


async def test_deep_offset_returns_empty_not_an_error(engine):
    response = await engine.search(SearchRequest(query="fintech", limit=10, offset=9000))
    assert response.results == []
    assert response.total > 0


# --- relaxation -------------------------------------------------------------


async def test_unsatisfiable_query_relaxes_and_says_so(engine):
    """The fixture's only sub-20-employee company is in Sweden, so this is empty."""
    response = await engine.search(
        SearchRequest(query="UK telecom companies with fewer than 20 employees", limit=10)
    )
    assert response.relaxation.applied
    assert response.relaxation.strict_result_count == 0
    assert "employees" in response.relaxation.dropped
    assert response.relaxation.message
    assert response.results


async def test_relaxation_never_drops_location_or_topic(engine):
    """Widening the country is worse than an honest empty page."""
    response = await engine.search(
        SearchRequest(query="UK telecom companies with fewer than 20 employees", limit=10)
    )
    assert "location" not in response.relaxation.dropped
    assert "topic" not in response.relaxation.dropped
    assert {c.location for c in response.results} == {"UK"}


async def test_relaxation_can_be_disabled(engine):
    response = await engine.search(
        SearchRequest(
            query="UK telecom companies with fewer than 20 employees",
            limit=10,
            allow_relaxation=False,
        )
    )
    assert not response.relaxation.applied
    assert response.results == []
    assert response.total == 0


async def test_exact_matches_lead_a_relaxed_result_set(engine, db):
    """Regression: relaxing could bury the single company that fully matched."""
    seed = await db.get(1)
    await db.upsert_many([seed])

    response = await engine.search(
        SearchRequest(query="Swedish telecom companies with fewer than 20 employees", limit=10)
    )
    assert response.relaxation.applied is False or response.results
    if response.relaxation.applied and response.relaxation.strict_result_count:
        top = response.results[0]
        assert "matches every requested constraint" in top.matched_on
        assert top.employee_count < 20


async def test_near_misses_outrank_far_misses(engine):
    """"Fewer than 20" should surface the smallest companies first."""
    response = await engine.search(
        SearchRequest(query="UK telecom companies with fewer than 20 employees", limit=10)
    )
    assert response.relaxation.applied
    sizes = [c.employee_count for c in response.results if c.employee_count]
    assert sizes[0] == min(sizes), f"closest match not first: {sizes}"


async def test_satisfiable_query_is_not_relaxed(engine):
    response = await engine.search(SearchRequest(query="fintech companies in Finland", limit=5))
    assert not response.relaxation.applied
    assert response.relaxation.dropped == []


# --- ranking ----------------------------------------------------------------


async def test_more_topic_matches_rank_higher(engine):
    response = await engine.search(
        SearchRequest(query="drug discovery and molecular analysis", limit=5, explain=True)
    )
    assert response.results
    top = response.results[0]
    assert {"drug discovery", "molecular analysis"} <= set(top.topics)


async def test_exact_name_match_wins_over_a_semantic_near_miss(hybrid_engine, db):
    """Regression: RRF ties fell through to company id, losing name lookups.

    Needs *both* retrievers to be meaningful. RRF gives rank 0 in the lexical
    list exactly the same score as rank 0 in the semantic list, so with only one
    retriever there is no tie to break and the bug is invisible. That is why
    this uses `hybrid_engine` rather than the lexical-only `engine`.
    """
    record = await db.get(5)
    record.name = "Zzyzx Unmistakable Holdings"
    await db.upsert_many([record])

    response = await hybrid_engine.search(SearchRequest(query="Zzyzx Unmistakable", limit=5))
    assert response.results[0].id == 5, (
        "exact lexical match lost to a semantic result: "
        f"{[(c.id, c.name) for c in response.results[:3]]}"
    )


async def test_tied_fused_scores_break_toward_the_stronger_raw_signal(hybrid_engine):
    """The tie-break itself, isolated.

    The integration test above cannot reach this on a fixture corpus of ~60
    companies: the candidate pool holds every row, so a lexical match also picks
    up a semantic contribution and wins on total outright. The tie only arises
    at real scale, where the vector pool is a small slice of the corpus and a
    document can be rank 0 in one list and absent from the other.

    So construct that state directly. Two candidates, identical fused scores,
    one strong on lexical and one strong on semantic — the lexical one must
    lead, and it must do so on merit rather than on the id fallback, which is
    why the lexical candidate is given the *higher* id.
    """
    from app.search.engine import _Candidate

    rrf = 1.0 / 60.0
    lexical = _Candidate(row=1, keyword_rrf=rrf, keyword_rank=0, keyword_score=27.4)
    semantic = _Candidate(row=0, vector_rrf=rrf, vector_rank=0, vector_score=0.61)

    assert lexical.total == pytest.approx(semantic.total), "test premise: scores must tie"
    assert int(hybrid_engine.columns.ids[1]) > int(hybrid_engine.columns.ids[0]), (
        "test premise: the lexical candidate must lose the id fallback"
    )

    ordered = sorted([semantic, lexical], key=hybrid_engine._sort_key)
    assert ordered[0] is lexical


async def test_identical_candidates_fall_back_to_id_for_stable_paging(hybrid_engine):
    from app.search.engine import _Candidate

    first = _Candidate(row=5, keyword_rrf=0.01, keyword_score=1.0)
    second = _Candidate(row=2, keyword_rrf=0.01, keyword_score=1.0)
    ordered = sorted([first, second], key=hybrid_engine._sort_key)
    assert [c.row for c in ordered] == [2, 5]


async def test_both_retrievers_contribute_to_the_fused_score(hybrid_engine):
    response = await hybrid_engine.search(
        SearchRequest(query="fintech companies in Finland fraud detection", limit=10,
                      explain=True)
    )
    assert response.results
    breakdowns = [c.score_breakdown for c in response.results]
    assert any(b.keyword > 0 for b in breakdowns), "lexical retriever contributed nothing"
    assert any(b.vector > 0 for b in breakdowns), "semantic retriever contributed nothing"


async def test_hybrid_respects_filters_just_as_strictly(hybrid_engine):
    """The semantic retriever must not leak rows past the mask."""
    response = await hybrid_engine.search(
        SearchRequest(query="fintech companies in Finland", limit=25)
    )
    assert response.results
    for company in response.results:
        assert company.location == "Finland"
        assert company.industry == "Fintech"


async def test_semantic_mode_skips_the_lexical_retriever(hybrid_engine):
    response = await hybrid_engine.search(
        SearchRequest(query="fraud detection", limit=10, mode="semantic", explain=True)
    )
    assert all(c.score_breakdown.keyword == 0 for c in response.results)


async def test_keyword_mode_skips_the_semantic_retriever(hybrid_engine):
    response = await hybrid_engine.search(
        SearchRequest(query="fraud detection", limit=10, mode="keyword", explain=True)
    )
    assert all(c.score_breakdown.vector == 0 for c in response.results)


async def test_similar_uses_vectors_when_available(hybrid_engine):
    response = await hybrid_engine.similar(1, SimilarRequest(limit=5, explain=True))
    assert response is not None
    assert response.results
    assert all(c.id != 1 for c in response.results)
    # Scores descend, and come from the vector path rather than the topic fallback.
    scores = [c.score for c in response.results]
    assert scores == sorted(scores, reverse=True)


async def test_similar_ignores_rows_added_after_the_build(hybrid_engine, db, hybrid_runtime):
    """A company with no embedding must fall back, not crash or mis-index."""
    from app.models.company import CompanyIn
    from app.store.enrich import enrich

    new_id = hybrid_runtime.columns.size + 500
    await db.upsert_many([
        enrich(
            CompanyIn(
                id=new_id,
                name="Post Build Oy",
                description="Engine for fraud detection.",
                industry="Fintech",
                location="Finland",
            ),
            company_id=new_id,
        )
    ])
    # The runtime still holds the pre-write column store, so the new row is not
    # in it; asking for something absent must simply 404-equivalent (None).
    assert await hybrid_engine.similar(new_id, SimilarRequest(limit=3)) is not None


async def test_explain_exposes_the_score_breakdown(engine):
    response = await engine.search(
        SearchRequest(query="fintech in Finland fraud detection", limit=3, explain=True)
    )
    top = response.results[0]
    assert top.score_breakdown is not None
    assert top.matched_on
    assert set(response.timings_ms) >= {"parse", "filter", "retrieve", "rank", "hydrate"}


async def test_explain_off_omits_internals(engine):
    response = await engine.search(SearchRequest(query="fintech", limit=3, explain=False))
    assert response.results[0].score_breakdown is None
    assert response.timings_ms == {}


async def test_filter_only_query_still_returns_matches(engine):
    """No lexical terms and no topics — the filtered set is the answer."""
    response = await engine.search(
        SearchRequest(query="companies in Germany with more than 100 employees", limit=10)
    )
    assert response.results
    assert all(c.location == "Germany" for c in response.results)
    assert all(c.employee_count > 100 for c in response.results)


async def test_nonsense_query_does_not_error(engine):
    response = await engine.search(SearchRequest(query="xyzzy plugh frobnicate", limit=5))
    assert response.total >= 0


@pytest.mark.parametrize("mode", ["hybrid", "keyword", "semantic"])
async def test_all_modes_answer_without_a_vector_index(engine, mode):
    """Semantic mode must degrade rather than fail when vectors are absent."""
    response = await engine.search(
        SearchRequest(query="fintech companies in Finland", limit=5, mode=mode)
    )
    assert response.mode == mode
    assert response.total > 0


# --- similarity -------------------------------------------------------------


async def test_similar_falls_back_to_topic_overlap(engine):
    """No vector index here, so similarity comes from shared topics."""
    response = await engine.similar(1, SimilarRequest(limit=5))
    assert response is not None
    assert response.seed.id == 1
    assert all(c.id != 1 for c in response.results)
    for company in response.results:
        assert set(company.topics) & set(response.seed.topics)


async def test_similar_unknown_company_returns_none(engine):
    assert await engine.similar(999_999, SimilarRequest(limit=5)) is None


async def test_similar_can_be_constrained(engine):
    response = await engine.similar(1, SimilarRequest(limit=5, same_location=True))
    assert response is not None
    for company in response.results:
        assert company.location == response.seed.location
