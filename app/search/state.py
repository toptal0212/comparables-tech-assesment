"""Shared runtime state for the search subsystem.

Holds the loaded index components and a readiness flag. Kept in one place for
two reasons:

1. Readiness is a single source of truth that `/health/ready` and every search
   handler consult, so a warming container cannot half-serve traffic.
2. Reindexing swaps a whole new `SearchRuntime` in atomically. Readers hold a
   reference to the object they started with, so an in-flight query never sees a
   half-rebuilt index and no reader-side locking is needed. The old runtime is
   garbage collected once its last reader finishes.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - import cycle guard, types only
    from app.search.embeddings import Embedder
    from app.search.keyword import KeywordIndex
    from app.search.vector import VectorIndex
    from app.store.db import Database


@dataclass
class SearchRuntime:
    """An immutable-by-convention bundle of everything a query needs."""

    db: Database
    keyword: KeywordIndex
    vector: VectorIndex | None
    embedder: Embedder | None
    doc_count: int = 0
    built_at: str | None = None
    model_name: str | None = None

    @property
    def semantic_enabled(self) -> bool:
        return self.vector is not None and self.embedder is not None


@dataclass
class IndexState:
    _runtime: SearchRuntime | None = None
    _reason: str = "index not loaded"
    _lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def is_ready(self) -> bool:
        return self._runtime is not None

    @property
    def not_ready_reason(self) -> str | None:
        return None if self.is_ready else self._reason

    @property
    def runtime(self) -> SearchRuntime | None:
        # Plain attribute read. Assignment of a reference is atomic under the
        # GIL, which is exactly the guarantee the swap-on-reindex design relies
        # on.
        return self._runtime

    def set_runtime(self, runtime: SearchRuntime) -> None:
        with self._lock:
            self._runtime = runtime

    def mark_unavailable(self, reason: str) -> None:
        with self._lock:
            self._runtime = None
            self._reason = reason

    def summary(self) -> dict[str, Any]:
        rt = self._runtime
        if rt is None:
            return {"ready": False, "reason": self._reason}
        return {
            "ready": True,
            "documents": rt.doc_count,
            "semantic_search": rt.semantic_enabled,
            "embedding_model": rt.model_name,
            "built_at": rt.built_at,
        }


_state = IndexState()


def get_index_state() -> IndexState:
    return _state
