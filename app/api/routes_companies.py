"""Company retrieval and similarity endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Path, Query

from app.api.deps import EngineDep, RuntimeDep
from app.core.errors import NotFoundError
from app.models.company import CompanyOut
from app.models.search import SimilarRequest, SimilarResponse

router = APIRouter(prefix="/companies", tags=["companies"])


@router.get(
    "/{company_id}",
    response_model=CompanyOut,
    summary="Fetch a single company",
)
async def get_company(
    runtime: RuntimeDep,
    company_id: int = Path(..., ge=1),
) -> CompanyOut:
    record = await runtime.db.get(company_id)
    if record is None:
        raise NotFoundError(f"No company with id {company_id}.", {"id": company_id})
    return CompanyOut.from_record(record)


@router.get(
    "/{company_id}/similar",
    response_model=SimilarResponse,
    summary="Companies similar to this one",
    description=(
        "Ranks the corpus by cosine similarity to the seed company's stored "
        "document vector. No embedding call is made, so this is the fastest "
        "path in the API — it skips the ~25ms a text query spends encoding.\n\n"
        "Similarity is topical, not geographic: location is deliberately "
        "excluded from the embedded text, so a Finnish fintech's nearest "
        "neighbours are fintechs anywhere rather than Finnish companies in "
        "unrelated sectors. Use `same_location` to constrain it explicitly.\n\n"
        "If no vector index is loaded, this falls back to ranking by shared "
        "topics."
    ),
)
async def similar_companies(
    engine: EngineDep,
    company_id: int = Path(..., ge=1),
    limit: int = Query(10, ge=1, le=100),
    same_industry: bool = Query(False, description="Restrict to the seed's industry"),
    same_location: bool = Query(False, description="Restrict to the seed's country"),
    explain: bool = Query(False),
) -> SimilarResponse:
    response = await engine.similar(
        company_id,
        SimilarRequest(
            limit=limit,
            same_industry=same_industry,
            same_location=same_location,
            explain=explain,
        ),
    )
    if response is None:
        raise NotFoundError(f"No company with id {company_id}.", {"id": company_id})
    return response
