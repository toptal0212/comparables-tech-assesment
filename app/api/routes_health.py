"""Health, readiness and metrics endpoints.

The liveness/readiness split is what makes rolling restarts safe, and the brief
explicitly calls out frequent restarts:

  * `/health/live`  — the process is running. Never touches the index. If this
    fails the orchestrator should kill and replace the container.
  * `/health/ready` — the index is loaded and queries will succeed. Returns 503
    while the service is still warming, which keeps the load balancer from
    routing traffic to a container that would only return errors.

Conflating the two causes a classic outage: the platform restarts a container
that was merely still warming up, and never converges.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Response, status

from app.config import settings
from app.core.metrics import metrics
from app.search.state import get_index_state

router = APIRouter(tags=["health"])


@router.get("/health", summary="Liveness + a short readiness summary")
async def health() -> dict[str, Any]:
    state = get_index_state()
    return {
        "status": "ok",
        "service": settings.app_name,
        "version": settings.version,
        "env": settings.env,
        "index": state.summary(),
    }


@router.get("/health/live", summary="Process is alive")
async def live() -> dict[str, str]:
    return {"status": "alive"}


@router.get("/health/ready", summary="Index loaded and able to serve queries")
async def ready(response: Response) -> dict[str, Any]:
    state = get_index_state()
    if not state.is_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {"status": "warming", "reason": state.not_ready_reason, "index": state.summary()}
    return {"status": "ready", "index": state.summary()}


@router.get("/metrics", summary="In-process counters and latency percentiles")
async def get_metrics() -> dict[str, Any]:
    return metrics.snapshot()
