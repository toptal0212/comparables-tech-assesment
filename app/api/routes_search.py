"""Search endpoints."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Query

from app.api.deps import EngineDep
from app.config import settings
from app.core.errors import AppError
from app.core.logging import get_logger
from app.core.metrics import metrics
from app.models.search import SearchRequest, SearchResponse

log = get_logger(__name__)

router = APIRouter(tags=["search"])


class _Timeout(AppError):
    status_code = 504
    code = "search_timeout"


@router.post(
    "/search",
    response_model=SearchResponse,
    summary="Natural-language company search",
    description=(
        "Accepts a natural-language query, extracts structured filters from it, "
        "and returns ranked companies.\n\n"
        "The response includes `parsed`, showing exactly how the query was "
        "interpreted, and `relaxation`, which is populated when the query was "
        "unsatisfiable as written and constraints had to be dropped.\n\n"
        "`mode` selects the retrieval strategy: `hybrid` (default) fuses BM25 "
        "and vector search, while `keyword` and `semantic` isolate one retriever "
        "each — useful for measuring their individual contribution."
    ),
)
async def search(request: SearchRequest, engine: EngineDep) -> SearchResponse:
    # A hard ceiling on a single search. Without it a pathological request can
    # occupy a worker indefinitely; with it the client gets a 504 and the worker
    # is returned to the pool.
    try:
        return await asyncio.wait_for(
            engine.search(request), timeout=settings.request_timeout_s
        )
    except TimeoutError:
        metrics.incr("search_timeouts_total")
        log.warning("search_timeout", extra={"query": request.query[:200]})
        raise _Timeout(
            f"Search exceeded {settings.request_timeout_s}s.",
            {"query": request.query[:200]},
        ) from None


@router.get(
    "/search",
    response_model=SearchResponse,
    summary="Natural-language company search (GET convenience form)",
    description=(
        "Equivalent to `POST /search` with a body of `{\"query\": q}`. Provided "
        "so a search can be shared as a URL or tried from a browser address bar; "
        "the POST form is the primary interface and supports explicit filters."
    ),
)
async def search_get(
    engine: EngineDep,
    q: str = Query(..., min_length=1, max_length=1000, description="Natural-language query"),
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0, le=10_000),
    mode: str = Query("hybrid", pattern="^(hybrid|keyword|semantic)$"),
    explain: bool = Query(False),
) -> SearchResponse:
    return await search(
        SearchRequest(
            query=q,
            limit=limit,
            offset=offset,
            mode=mode,  # type: ignore[arg-type]
            explain=explain,
        ),
        engine,
    )
