"""Application entrypoint and composition root.

Startup policy is the important decision here. The service loads its index
inside the lifespan handler, but a failure to load does **not** abort startup:
the process comes up, `/health/live` succeeds, `/health/ready` reports 503 with
a reason, and `/metrics` still works. That is what lets an operator diagnose a
bad deploy through the same interface as a healthy one, instead of staring at a
container that crash-loops before it can serve a single request.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from app.api import routes_companies, routes_health, routes_ingest, routes_search
from app.api.deps import require_read_dependency
from app.config import settings
from app.core.errors import register_exception_handlers
from app.core.logging import configure_logging, get_logger
from app.core.middleware import RequestContextMiddleware
from app.core.security import RateLimitMiddleware
from app.search import bootstrap

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    configure_logging()
    log.info(
        "starting",
        extra={
            "version": settings.version,
            "env": settings.env,
            "data_dir": str(settings.data_dir),
        },
    )

    settings.data_dir.mkdir(parents=True, exist_ok=True)

    # Never raises: a failure here is recorded as a readiness failure so the
    # process still serves /health, /health/ready and /metrics. See the module
    # docstring for why that beats crash-looping.
    await bootstrap.startup()

    yield

    log.info("shutting down")
    await bootstrap.shutdown()


def create_app() -> FastAPI:
    configure_logging()

    app = FastAPI(
        title="Company Search API",
        version=settings.version,
        description=(
            "Natural-language search over a corpus of companies. "
            "Combines lexical (BM25), semantic (vector) and structured filtering."
        ),
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url=None,
        openapi_url="/openapi.json",
    )

    # Order matters: middleware added last runs outermost. RequestContext must
    # be outermost so the correlation id exists before anything else logs.
    app.add_middleware(GZipMiddleware, minimum_size=1024)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=False,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["x-request-id", "x-response-time-ms"],
    )
    app.add_middleware(RateLimitMiddleware)
    app.add_middleware(RequestContextMiddleware)

    register_exception_handlers(app)

    app.include_router(routes_health.router)
    # Read endpoints are public unless REQUIRE_AUTH_FOR_READS is set, so the
    # deployed demo is explorable from a browser. Write endpoints carry their
    # own unconditional auth dependency.
    app.include_router(routes_search.router, dependencies=[require_read_dependency()])
    app.include_router(routes_companies.router, dependencies=[require_read_dependency()])
    app.include_router(routes_ingest.router)

    return app


app = create_app()
