"""Dense vector retrieval — exact, in-process, no ANN.

Why exact brute force
---------------------
Measured on the target hardware, an exact top-k over 50,000 x 384 float32 takes
2.5ms p50. The query embedding that produces the search vector takes 18ms. An
approximate index (HNSW, IVF) would cut the 2.5ms to perhaps 0.3ms and leave the
18ms untouched — a 1% end-to-end gain in exchange for a C++ dependency, a build
step, tuning parameters, and approximate recall.

It also composes badly with filtering, which matters more here. Most queries in
this domain carry hard constraints, and pre-filtering an HNSW graph either
degrades recall (the reachable neighbourhood gets disconnected) or forces
over-fetching. A brute-force scan restricted to a boolean mask is exact by
construction and gets *faster* as filters get more selective, which is the
opposite of the usual ANN behaviour.

The crossover is around 1-2M documents. `VectorIndex` is deliberately a narrow
interface — `search(query, k, mask)` — so swapping in hnswlib or an external
vector database at that point touches this file and nothing else. The scaling
path is set out in docs/DESIGN.md.

Storage
-------
Vectors are L2-normalised at build time, so cosine similarity is a plain dot
product and the whole search is one BLAS matmul. The matrix is saved as .npy and
loaded on startup, because embedding 50k documents takes about two minutes and
no restart should pay that.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from app.core.logging import get_logger
from app.core.metrics import metrics

log = get_logger(__name__)

# Below this fraction of surviving rows, gathering the candidate submatrix beats
# scoring everything and masking. Gathering costs a copy of (n_selected x dim)
# plus that many dot products; the full path is a fixed ~3ms matmul at 50k rows.
#
# Measured, not guessed (50k x 384, k=400, median of 30):
#
#   selectivity   rows    full    gather
#         0.1%      50   3.06ms    0.05ms
#         1.0%     500   2.94ms    0.36ms
#         5.0%    2500   3.10ms    1.50ms
#         8.0%    4000   2.93ms    2.20ms
#        12.0%    6000   3.46ms    3.36ms   <- crossover
#        20.0%   10000   3.09ms    5.62ms
#        80.0%   40000   2.96ms   20.64ms
#
# Set at 10% rather than the measured 12% so the choice is made on the side
# where the gap is clearly in gather's favour, instead of on the noisy boundary.
_GATHER_THRESHOLD = 0.10


class RowMap:
    """Translation between column-store rows and vector-matrix rows.

    The two indexes are built from the same corpus in the same id order, so
    immediately after a build they are positionally identical. Ingestion breaks
    that: adding one company shifts every subsequent column row while the vector
    matrix stays as it was.

    The first version of this treated any length mismatch as a stale index and
    discarded the vectors entirely — one new company turned semantic search off
    for all 50,000. Mapping through ids instead means a newly ingested company
    simply does not participate in semantic ranking until the next full build,
    while everything already embedded keeps working. It is found by keyword and
    filters in the meantime.

    Both id arrays are sorted, so this is two binary searches at load time and a
    gather per query.
    """

    __slots__ = ("col_to_vec", "vec_to_col", "aligned", "coverage")

    def __init__(self, column_ids: np.ndarray, vector_ids: np.ndarray) -> None:
        n_cols = column_ids.shape[0]
        n_vecs = vector_ids.shape[0]

        self.aligned = n_cols == n_vecs and bool((column_ids == vector_ids).all())
        if self.aligned:
            identity = np.arange(n_cols, dtype=np.int64)
            self.col_to_vec = identity
            self.vec_to_col = identity
            self.coverage = 1.0
            return

        # column row -> vector row, or -1 when the company has no embedding yet.
        pos = np.searchsorted(vector_ids, column_ids)
        pos_clipped = np.clip(pos, 0, max(n_vecs - 1, 0))
        found = (n_vecs > 0) & (vector_ids[pos_clipped] == column_ids)
        self.col_to_vec = np.where(found, pos_clipped, -1).astype(np.int64)

        # vector row -> column row, or -1 when the company has since been deleted.
        rpos = np.searchsorted(column_ids, vector_ids)
        rpos_clipped = np.clip(rpos, 0, max(n_cols - 1, 0))
        rfound = (n_cols > 0) & (column_ids[rpos_clipped] == vector_ids)
        self.vec_to_col = np.where(rfound, rpos_clipped, -1).astype(np.int64)

        self.coverage = float(found.mean()) if n_cols else 0.0

    def mask_to_vector_space(self, column_mask: np.ndarray) -> np.ndarray:
        """Project a column-space boolean mask onto vector rows."""
        if self.aligned:
            return column_mask
        vector_mask = np.zeros(self.vec_to_col.shape[0], dtype=bool)
        targets = self.col_to_vec[np.flatnonzero(column_mask)]
        targets = targets[targets >= 0]
        if targets.size:
            vector_mask[targets] = True
        return vector_mask

    def to_column_rows(self, vector_rows: np.ndarray) -> np.ndarray:
        if self.aligned:
            return vector_rows
        return self.vec_to_col[vector_rows]


class VectorIndex:
    """Exact cosine top-k over a dense matrix, with optional row masking."""

    def __init__(self, vectors: np.ndarray, ids: np.ndarray, model_name: str = "") -> None:
        if vectors.shape[0] != ids.shape[0]:
            raise ValueError(
                f"vector/id length mismatch: {vectors.shape[0]} vs {ids.shape[0]}"
            )
        self.vectors = vectors
        self.ids = ids
        self.model_name = model_name

    @property
    def size(self) -> int:
        return int(self.vectors.shape[0])

    @property
    def dim(self) -> int:
        return int(self.vectors.shape[1])

    # -- search -----------------------------------------------------------

    def search(
        self,
        query: np.ndarray,
        k: int,
        mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Top `k` rows by cosine similarity.

        Returns `(rows, scores)` sorted best-first. `rows` are positions into
        the matrix, not company ids — the caller maps them through the column
        store, which shares this row order.
        """
        if self.size == 0 or k <= 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)

        with metrics.timer("vector_search_ms"):
            if mask is None:
                scores = self.vectors @ query
                rows = self._top_k(scores, k)
                return rows, scores[rows]

            candidates = np.flatnonzero(mask)
            if candidates.size == 0:
                return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)

            if candidates.size <= self.size * _GATHER_THRESHOLD:
                # Selective filter: score only what survives. This is why a
                # heavily-constrained query is *faster* here than an open one,
                # where an ANN index would be the other way round.
                sub_scores = self.vectors[candidates] @ query
                local = self._top_k(sub_scores, k)
                return candidates[local], sub_scores[local]

            # Broad filter: one full matmul beats gathering most of the matrix.
            # -inf rather than 0, because cosine scores are legitimately
            # negative and 0 would rank excluded rows above genuine poor matches.
            scores = self.vectors @ query
            scores = np.where(mask, scores, -np.inf)
            rows = self._top_k(scores, k)
            rows = rows[np.isfinite(scores[rows])]
            return rows, scores[rows]

    @staticmethod
    def _top_k(scores: np.ndarray, k: int) -> np.ndarray:
        """Indices of the k largest scores, best first.

        argpartition is O(n) to find the k-th largest and only the surviving k
        get sorted, versus O(n log n) to sort everything. At 50k rows with k=400
        that is roughly a 5x saving.
        """
        k = min(k, scores.shape[0])
        if k <= 0:
            return np.empty(0, dtype=np.int64)
        if k == scores.shape[0]:
            return np.argsort(-scores, kind="stable")
        part = np.argpartition(-scores, k - 1)[:k]
        return part[np.argsort(-scores[part], kind="stable")]

    def vector_for_row(self, row: int) -> np.ndarray:
        return self.vectors[row]

    # -- persistence ------------------------------------------------------

    def save(self, vectors_path: Path, ids_path: Path, meta_path: Path, **extra: object) -> None:
        vectors_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(vectors_path, self.vectors)
        np.save(ids_path, self.ids)
        meta = {
            "model": self.model_name,
            "count": self.size,
            "dim": self.dim,
            "built_at": datetime.now(UTC).isoformat(),
            **extra,
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        log.info("vector_index_saved", extra={"count": self.size, "dim": self.dim})

    @classmethod
    def load(
        cls,
        vectors_path: Path,
        ids_path: Path,
        meta_path: Path,
        *,
        expected_count: int | None = None,
        expected_model: str | None = None,
    ) -> VectorIndex | None:
        """Load a persisted index, or return None if it is absent or stale.

        The consistency checks matter on a restart. A container mounts a volume
        it did not write; the corpus may have been re-ingested since, or the
        configured model may have changed. Serving a stale index would return
        confidently wrong results, which is worse than returning lexical-only
        ones. So a mismatch is treated as "no vector index" and logged loudly.
        """
        if not (vectors_path.exists() and ids_path.exists()):
            log.info("vector_index_absent", extra={"path": str(vectors_path)})
            return None

        try:
            vectors = np.load(vectors_path)
            ids = np.load(ids_path)
        except Exception:
            log.exception("vector_index_unreadable", extra={"path": str(vectors_path)})
            return None

        model_name = ""
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                model_name = str(meta.get("model", ""))
            except Exception:
                log.warning("vector_index_meta_unreadable", exc_info=True)

        if expected_count is not None and vectors.shape[0] != expected_count:
            log.warning(
                "vector_index_stale_row_count",
                extra={"index_rows": int(vectors.shape[0]), "corpus_rows": expected_count},
            )
            return None

        if expected_model and model_name and model_name != expected_model:
            log.warning(
                "vector_index_model_mismatch",
                extra={"index_model": model_name, "configured_model": expected_model},
            )
            return None

        # np.load returns whatever dtype was written; downstream assumes float32
        # contiguous for BLAS to take the fast path.
        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)
        vectors = np.ascontiguousarray(vectors)

        log.info(
            "vector_index_loaded",
            extra={
                "count": int(vectors.shape[0]),
                "dim": int(vectors.shape[1]),
                "model": model_name,
                "mb": round(vectors.nbytes / 1e6, 1),
            },
        )
        return cls(vectors, ids, model_name)
