"""Ingestion endpoints — add, update and remove companies on a live service.

Consistency model
-----------------
A write goes to SQLite, which is the source of truth. The FTS index updates in
the same transaction via triggers, so a new company is findable by keyword and
by structured filter as soon as the derived column arrays are refreshed.

Those column arrays are a derived cache and have to be rebuilt — about 0.5s at
50k rows. Rebuilding is much simpler than incremental array maintenance
(resizing, gap management, and a correctness argument for each) and, for an
endpoint called occasionally rather than continuously, 0.5s is a fair trade. The
rebuilt runtime is swapped in atomically, so concurrent searches never observe a
half-updated index.

Semantic search is the one thing that lags. Embedding a new company requires the
model and a rewrite of the vector matrix, so newly ingested companies are
retrievable by keyword and filter immediately but do not enter the vector index
until the next full build. Rather than hide that, every write response states it
in `semantic_indexed`.

Batching
--------
`POST /companies` accepts a single object or a list. A caller inserting a
thousand companies should send one list, or send many requests with
`refresh=false` and finish with `POST /admin/reindex`, instead of paying the
rebuild a thousand times.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Body, Depends, Path, Query, Response, status
from pydantic import BaseModel, Field

from app.api.deps import RuntimeDep
from app.core.errors import BadRequestError, NotFoundError
from app.core.logging import get_logger
from app.core.security import require_write_access
from app.models.company import CompanyIn, CompanyOut
from app.search import bootstrap

log = get_logger(__name__)

router = APIRouter(tags=["ingestion"], dependencies=[Depends(require_write_access)])

MAX_BATCH = 5000


class IngestResult(BaseModel):
    written: int
    ids: list[int]
    #: Total corpus size after the write.
    corpus_size: int
    #: Whether the written companies are in the vector index. False until the
    #: next full build — they are still keyword- and filter-searchable.
    semantic_indexed: bool = False
    index_refreshed: bool = True
    note: str | None = None


@router.post(
    "/companies",
    response_model=IngestResult,
    status_code=status.HTTP_201_CREATED,
    summary="Add or update companies",
    description=(
        "Accepts a single company object or a list of them. Records with an "
        "existing `id` are updated; records without one are assigned the next "
        "available id.\n\n"
        "Set `refresh=false` when loading many batches, then call "
        "`POST /admin/reindex` once at the end."
    ),
)
async def ingest_companies(
    runtime: RuntimeDep,
    payload: Annotated[CompanyIn | list[CompanyIn], Body(...)],
    refresh: bool = Query(True, description="Rebuild the in-memory index after writing"),
) -> IngestResult:
    companies = payload if isinstance(payload, list) else [payload]
    if not companies:
        raise BadRequestError("No companies supplied.")
    if len(companies) > MAX_BATCH:
        raise BadRequestError(
            f"Batch too large: {len(companies)} exceeds the {MAX_BATCH} limit.",
            {"limit": MAX_BATCH},
        )

    # Import here rather than at module scope: enrich pulls in the taxonomy
    # matcher, which the read path already owns, and this keeps the API module's
    # import graph shallow.
    from app.store.enrich import enrich

    next_id = await runtime.db.next_id()
    records = []
    for company in companies:
        if company.id is None:
            company.id = next_id
            next_id += 1
        else:
            next_id = max(next_id, company.id + 1)
        records.append(enrich(company, company_id=company.id))

    written = await runtime.db.upsert_many(records)

    if refresh:
        await bootstrap.reload_columns()

    state_runtime = runtime if not refresh else None
    corpus_size = (
        state_runtime.doc_count if state_runtime else await runtime.db.count()
    )

    log.info(
        "companies_ingested",
        extra={"written": written, "refreshed": refresh, "corpus_size": corpus_size},
    )
    return IngestResult(
        written=written,
        ids=[r.id for r in records],
        corpus_size=corpus_size,
        semantic_indexed=False,
        index_refreshed=refresh,
        note=(
            "Searchable by keyword and filters immediately. Semantic search "
            "covers these records after the next full index build "
            "(python -m scripts.build_index)."
            if refresh
            else "Index not refreshed; call POST /admin/reindex to make these "
            "records searchable."
        ),
    )


@router.delete(
    "/companies/{company_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a company",
)
async def delete_company(
    runtime: RuntimeDep,
    company_id: int = Path(..., ge=1),
    refresh: bool = Query(True),
) -> Response:
    deleted = await runtime.db.delete(company_id)
    if not deleted:
        raise NotFoundError(f"No company with id {company_id}.", {"id": company_id})
    if refresh:
        await bootstrap.reload_columns()
    log.info("company_deleted", extra={"id": company_id})
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post(
    "/admin/reindex",
    summary="Rebuild the in-memory index from SQLite",
    description=(
        "Rebuilds the columnar filter arrays from the database. Use after a "
        "batch of `refresh=false` writes. Does not re-embed: the vector index "
        "is rebuilt only by `scripts/build_index.py`."
    ),
)
async def reindex() -> dict[str, Any]:
    await bootstrap.reload_columns()
    from app.search.state import get_index_state

    runtime = get_index_state().runtime
    return {
        "status": "ok",
        "corpus_size": runtime.doc_count if runtime else 0,
        "semantic_search": runtime.semantic_enabled if runtime else False,
    }


class EchoCompany(BaseModel):
    """Validation preview. Shows how a payload is normalised without writing it."""

    input: dict[str, Any]
    parsed: CompanyOut
    derived_topics: list[str] = Field(default_factory=list)


@router.post(
    "/companies/validate",
    response_model=EchoCompany,
    summary="Dry-run a company payload",
    description=(
        "Runs a payload through validation and enrichment and returns what would "
        "be stored, without writing anything. Useful when wiring up an "
        "integration: it shows which field aliases were accepted, which topics "
        "were extracted from the description, and how revenue was bucketed."
    ),
)
async def validate_company(payload: CompanyIn) -> EchoCompany:
    from app.store.enrich import enrich

    record = enrich(payload, company_id=payload.id or 0)
    return EchoCompany(
        input=payload.model_dump(),
        parsed=CompanyOut.from_record(record),
        derived_topics=record.topics,
    )
