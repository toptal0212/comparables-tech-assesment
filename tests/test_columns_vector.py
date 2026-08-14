"""Columnar filtering and the vector index.

The vector tests use synthetic embeddings rather than the real model: the
concern here is the index mechanics — masking, top-k, gather-vs-full, row
translation — none of which depend on what produced the vectors.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.search.columns import FilterSpec
from app.search.vector import RowMap, VectorIndex

# --- filter masks -----------------------------------------------------------


async def test_empty_spec_matches_everything(columns):
    assert int(columns.mask(FilterSpec()).sum()) == columns.size


async def test_mask_returns_a_fresh_array_each_time(columns):
    """Callers combine masks in place, so a shared buffer would corrupt state."""
    first = columns.mask(FilterSpec())
    first[0] = False
    assert columns.mask(FilterSpec())[0] is np.True_


async def test_unknown_category_matches_nothing(columns):
    assert int(columns.mask(FilterSpec(locations=["Atlantis"])).sum()) == 0
    assert int(columns.mask(FilterSpec(industries=["Sorcery"])).sum()) == 0
    assert int(columns.mask(FilterSpec(topics=["time travel"])).sum()) == 0


async def test_constraints_and_together(columns):
    finland = int(columns.mask(FilterSpec(locations=["Finland"])).sum())
    fintech = int(columns.mask(FilterSpec(industries=["Fintech"])).sum())
    both = int(columns.mask(FilterSpec(locations=["Finland"], industries=["Fintech"])).sum())
    assert both <= min(finland, fintech)


async def test_topics_or_together(columns):
    a = int(columns.mask(FilterSpec(topics=["fraud detection"])).sum())
    b = int(columns.mask(FilterSpec(topics=["banking analytics"])).sum())
    both = int(columns.mask(FilterSpec(topics=["fraud detection", "banking analytics"])).sum())
    assert both >= max(a, b)
    assert both <= a + b


async def test_missing_values_never_satisfy_a_numeric_predicate(columns, db):
    """Absence of data is not evidence of a small number."""
    unknown = next(
        r for r in [await db.get(i) for i in range(1, columns.size + 1)]
        if r and r.founded_year is None
    )
    row = columns.row_of(unknown.id)

    for spec in (
        FilterSpec(founded_max=2100),
        FilterSpec(founded_min=1000),
        FilterSpec(employees_max=10_000),
        FilterSpec(employees_min=0),
    ):
        assert not columns.mask(spec)[row]


async def test_bound_inclusivity_is_honoured(columns):
    inclusive = int(columns.mask(FilterSpec(founded_min=2010, founded_min_inclusive=True)).sum())
    exclusive = int(columns.mask(FilterSpec(founded_min=2010, founded_min_inclusive=False)).sum())
    assert inclusive >= exclusive


async def test_revenue_sentinel_does_not_wrap(columns, db):
    """revenue_bucket uses -1 for missing; naive LUT indexing would wrap to the
    last bucket and report unknown-revenue companies as 500M+."""
    unknown = next(
        r for r in [await db.get(i) for i in range(1, columns.size + 1)]
        if r and r.revenue_range is None
    )
    row = columns.row_of(unknown.id)
    assert not columns.mask(FilterSpec(revenue_buckets=["500M+"]))[row]
    assert not columns.mask(FilterSpec(revenue_buckets=["0-1M"]))[row]


async def test_exclude_ids(columns):
    full = int(columns.mask(FilterSpec()).sum())
    assert int(columns.mask(FilterSpec(exclude_ids=[1, 2])).sum()) == full - 2


async def test_row_lookup_round_trips(columns):
    row = columns.row_of(3)
    assert row is not None and int(columns.ids[row]) == 3
    assert columns.row_of(999_999) is None
    assert columns.rows_of([1, 999_999, 2]).tolist() == sorted(
        [columns.row_of(1), columns.row_of(2)]
    )


async def test_relaxation_helpers(columns):
    spec = FilterSpec(locations=["Finland"], employees_max=5, topics=["fraud detection"])
    assert set(spec.active_constraints()) == {"location", "employees", "topic"}
    relaxed = spec.without("employees")
    assert relaxed.employees_max is None
    assert relaxed.locations == ["Finland"]        # untouched
    assert spec.employees_max == 5                  # original unmodified
    assert FilterSpec().is_empty


# --- vector index -----------------------------------------------------------


def _index(n: int = 200, dim: int = 16, seed: int = 0) -> VectorIndex:
    rng = np.random.default_rng(seed)
    vectors = rng.normal(size=(n, dim)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return VectorIndex(vectors, np.arange(1, n + 1, dtype=np.int64), "synthetic")


def test_top_k_is_sorted_and_exact():
    index = _index()
    query = index.vectors[7]
    rows, scores = index.search(query, 5)
    assert rows[0] == 7                                   # a vector's best match is itself
    assert scores.tolist() == sorted(scores.tolist(), reverse=True)
    brute = np.argsort(-(index.vectors @ query))[:5]
    assert rows.tolist() == brute.tolist()


def test_search_honours_the_mask():
    index = _index()
    mask = np.zeros(index.size, dtype=bool)
    mask[[10, 20, 30]] = True
    rows, _ = index.search(index.vectors[7], 10, mask=mask)
    assert set(rows.tolist()) <= {10, 20, 30}


def test_gather_and_full_paths_agree():
    """The selectivity threshold is an optimisation and must not change results."""
    index = _index(n=1000, dim=16)
    query = index.vectors[3]
    rng = np.random.default_rng(1)

    for fraction in (0.02, 0.5):  # below and above the gather threshold
        mask = np.zeros(index.size, dtype=bool)
        mask[rng.choice(index.size, int(index.size * fraction), replace=False)] = True
        rows, scores = index.search(query, 10, mask=mask)

        candidates = np.flatnonzero(mask)
        expected = candidates[np.argsort(-(index.vectors[candidates] @ query))[:10]]
        assert rows.tolist() == expected.tolist()
        assert np.allclose(scores, index.vectors[rows] @ query, atol=1e-6)


def test_empty_mask_returns_nothing():
    index = _index()
    rows, scores = index.search(index.vectors[0], 5, mask=np.zeros(index.size, dtype=bool))
    assert rows.size == 0 and scores.size == 0


def test_excluded_rows_never_surface_via_negative_infinity():
    """Cosine scores are legitimately negative, so masked rows must be -inf,
    not 0 — otherwise they outrank genuine poor matches."""
    index = _index(n=50)
    mask = np.zeros(index.size, dtype=bool)
    mask[[0]] = True
    rows, _ = index.search(-index.vectors[0], 10, mask=mask)  # worst possible query
    assert rows.tolist() == [0]


def test_k_larger_than_corpus_is_safe():
    index = _index(n=5)
    rows, _ = index.search(index.vectors[0], 100)
    assert rows.size == 5


def test_length_mismatch_is_rejected():
    with pytest.raises(ValueError):
        VectorIndex(np.zeros((3, 4), dtype=np.float32), np.arange(2, dtype=np.int64))


# --- row translation --------------------------------------------------------


def test_row_map_is_identity_when_aligned():
    ids = np.arange(1, 11, dtype=np.int64)
    row_map = RowMap(ids, ids)
    assert row_map.aligned and row_map.coverage == 1.0
    mask = np.zeros(10, dtype=bool)
    mask[3] = True
    assert row_map.mask_to_vector_space(mask) is mask


def test_row_map_handles_companies_added_after_the_build():
    """Regression: one ingested company used to disable semantic search entirely.

    The column store gains a row; the vector matrix does not. Everything already
    embedded must keep working, with the new arrival simply absent.
    """
    vector_ids = np.array([1, 2, 3, 4], dtype=np.int64)
    column_ids = np.array([1, 2, 3, 4, 5], dtype=np.int64)  # 5 was just ingested
    row_map = RowMap(column_ids, vector_ids)

    assert not row_map.aligned
    assert row_map.coverage == pytest.approx(0.8)
    assert row_map.col_to_vec[4] == -1                    # no embedding for id 5

    mask = np.ones(5, dtype=bool)
    projected = row_map.mask_to_vector_space(mask)
    assert projected.tolist() == [True] * 4               # only the embedded four
    assert row_map.to_column_rows(np.array([0, 3])).tolist() == [0, 3]


def test_row_map_handles_deleted_companies():
    """A company embedded at build time but deleted since maps back to -1."""
    vector_ids = np.array([1, 2, 3], dtype=np.int64)
    column_ids = np.array([1, 3], dtype=np.int64)          # id 2 deleted
    row_map = RowMap(column_ids, vector_ids)
    assert row_map.vec_to_col[1] == -1
    assert row_map.to_column_rows(np.array([0, 1, 2])).tolist() == [0, -1, 1]


def test_row_map_with_no_overlap_reports_zero_coverage():
    row_map = RowMap(np.array([10, 11], dtype=np.int64), np.array([1, 2], dtype=np.int64))
    assert row_map.coverage == 0.0
    assert not row_map.mask_to_vector_space(np.ones(2, dtype=bool)).any()
