"""Request-scoped middleware: correlation id, timing, access logging.

Kept as a plain ASGI-level `BaseHTTPMiddleware` subclass rather than a decorator
so it wraps *everything*, including responses produced by exception handlers.
"""

from __future__ import annotations

import time

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from app.core.logging import get_logger, new_request_id, set_request_id
from app.core.metrics import metrics

log = get_logger("access")

# Health checks fire every few seconds from the platform's prober. Logging them
# drowns real traffic, and their latency is not interesting.
_QUIET_PATHS = frozenset({"/health", "/health/live", "/health/ready", "/metrics", "/favicon.ico"})


class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # Honour an inbound correlation id so traces survive a proxy hop.
        rid = request.headers.get("x-request-id") or new_request_id()
        set_request_id(rid)

        t0 = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            # The exception handler will format the body; record the failure
            # here so the metric counts it even though we re-raise.
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            metrics.incr("http_requests_total", status="500", path=_route_of(request))
            metrics.observe_ms("http_request_ms", elapsed_ms, path=_route_of(request))
            raise

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        path = _route_of(request)

        response.headers["x-request-id"] = rid
        response.headers["x-response-time-ms"] = f"{elapsed_ms:.1f}"

        if request.url.path not in _QUIET_PATHS:
            metrics.incr("http_requests_total", status=str(response.status_code), path=path)
            metrics.observe_ms("http_request_ms", elapsed_ms, path=path)
            log.info(
                "request",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "status": response.status_code,
                    "duration_ms": round(elapsed_ms, 2),
                },
            )
        return response


def _route_of(request: Request) -> str:
    """Templated route path (`/companies/{company_id}`) rather than the concrete
    URL, so metric cardinality stays bounded no matter how many ids are hit."""
    route = request.scope.get("route")
    return getattr(route, "path", request.url.path)
