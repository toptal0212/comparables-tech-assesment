"""Shared FastAPI dependencies.

The runtime is resolved once per request and the resulting `SearchEngine` binds
to that exact snapshot. If a reindex swaps a new runtime in mid-request, the
in-flight query keeps reading the one it started with and finishes against a
consistent view. That is the entire reason `SearchRuntime` is an immutable
bundle swapped by reference rather than a mutable object.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import Depends

from app.core.errors import ServiceUnavailableError
from app.core.security import require_read_access
from app.search.engine import SearchEngine
from app.search.state import SearchRuntime, get_index_state


def get_runtime() -> SearchRuntime:
    """The current search runtime, or 503 if the service is still warming."""
    state = get_index_state()
    runtime = state.runtime
    if runtime is None:
        raise ServiceUnavailableError(
            "Search index is not loaded yet.",
            {"reason": state.not_ready_reason},
        )
    return runtime


def get_engine(runtime: Annotated[SearchRuntime, Depends(get_runtime)]) -> SearchEngine:
    # Constructing the engine per request is two attribute assignments; it holds
    # no state of its own beyond the runtime it was handed.
    return SearchEngine(runtime, runtime.columns)


RuntimeDep = Annotated[SearchRuntime, Depends(get_runtime)]
EngineDep = Annotated[SearchEngine, Depends(get_engine)]


def require_read_dependency() -> Any:
    """Auth dependency for read routes.

    Wrapped in a function rather than applied directly so the read/write split
    is decided in one place. `require_read_access` is a no-op unless
    `REQUIRE_AUTH_FOR_READS` is set.
    """
    return Depends(require_read_access)
