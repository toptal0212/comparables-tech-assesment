"""API-key authentication and per-client rate limiting.

Authentication
--------------
Static API keys compared in constant time, supplied either as `X-API-Key` or as
`Authorization: Bearer <key>`. This is the right weight for the problem: the
service holds public company data, has no per-user state, and needs to keep
casual abuse off a free-tier container. OAuth or JWT would add an identity
provider, key rotation and token validation to a service with no users to
identify. If it ever needed per-tenant quotas or audit trails, the dependency
below is the seam to replace.

Reads are public by default and writes always require a key when any key is
configured, which is what lets the deployed demo be explored from a browser
while still refusing anonymous mutations. `REQUIRE_AUTH_FOR_READS=true` locks
reads down too.

Rate limiting
-------------
A token bucket per client, in process. Bucket state is a float and a timestamp,
refilled lazily on access rather than by a background sweeper, so there is no
timer and no work for idle clients.

The honest limitation: it is per-process. Two workers means twice the effective
limit, and it resets on restart. That is acceptable for the intended deployment
(a single small container) and the wrong answer for a multi-replica one, where
this should move to Redis. It is here to stop one client saturating a shared
vCPU, not to enforce billing.
"""

from __future__ import annotations

import hmac
import threading
import time
from collections import OrderedDict

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import JSONResponse, Response

from app.config import settings
from app.core.errors import UnauthorizedError
from app.core.logging import get_logger, get_request_id
from app.core.metrics import metrics

log = get_logger(__name__)

# Paths that must answer even when a caller is rate limited or unauthenticated.
# Health probes come from the platform, not the client, and locking them out
# turns a burst of traffic into a container restart.
_EXEMPT_PATHS = frozenset(
    {"/health", "/health/live", "/health/ready", "/metrics", "/docs", "/openapi.json", "/"}
)

# Ceiling on tracked clients. Without it, a spray of distinct source addresses
# would grow the bucket table without bound — the rate limiter becoming the
# memory-exhaustion vector is a bad trade.
_MAX_TRACKED_CLIENTS = 10_000


def extract_api_key(request: Request) -> str | None:
    if key := request.headers.get("x-api-key"):
        return key.strip()
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return None


def _key_is_valid(candidate: str) -> bool:
    """Constant-time comparison against every configured key.

    `hmac.compare_digest` rather than `in`, so response timing does not leak how
    much of a guessed key was correct. Every configured key is checked even
    after a match, so the work done is independent of which key was supplied.
    """
    valid = False
    for known in settings.api_key_set:
        if hmac.compare_digest(candidate, known):
            valid = True
    return valid


def require_write_access(request: Request) -> None:
    """Reject unauthenticated writes when API keys are configured."""
    if not settings.auth_enabled:
        return
    key = extract_api_key(request)
    if key is None:
        metrics.incr("auth_failures_total", reason="missing")
        raise UnauthorizedError(
            "This endpoint requires an API key.",
            {"how": "Send X-API-Key: <key> or Authorization: Bearer <key>"},
        )
    if not _key_is_valid(key):
        metrics.incr("auth_failures_total", reason="invalid")
        log.warning("auth_rejected", extra={"client": _client_id(request)})
        raise UnauthorizedError("Invalid API key.")


def require_read_access(request: Request) -> None:
    if not settings.require_auth_for_reads:
        return
    require_write_access(request)


def _client_id(request: Request) -> str:
    """Identify the caller for rate-limiting purposes.

    API key first, so a legitimate integration gets its own budget regardless of
    where it connects from. Otherwise the peer address. X-Forwarded-For is only
    honoured because every target platform terminates TLS at a proxy — the
    direct peer would be that proxy, giving all clients one shared bucket. It is
    client-supplied and therefore spoofable, which is precisely why this is a
    fair-use guard and not a security control.
    """
    if key := extract_api_key(request):
        return f"key:{key[:12]}"
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return f"ip:{forwarded.split(',')[0].strip()}"
    return f"ip:{request.client.host if request.client else 'unknown'}"


class TokenBucket:
    """Fixed-capacity bucket refilled at a steady rate."""

    __slots__ = ("tokens", "updated")

    def __init__(self, capacity: float) -> None:
        self.tokens = capacity
        self.updated = time.monotonic()

    def take(self, capacity: float, refill_per_s: float) -> tuple[bool, float]:
        """Consume one token. Returns (allowed, seconds until next token)."""
        now = time.monotonic()
        self.tokens = min(capacity, self.tokens + (now - self.updated) * refill_per_s)
        self.updated = now
        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return True, 0.0
        return False, (1.0 - self.tokens) / refill_per_s if refill_per_s else 60.0


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: object) -> None:
        super().__init__(app)  # type: ignore[arg-type]
        self._buckets: OrderedDict[str, TokenBucket] = OrderedDict()
        self._lock = threading.Lock()

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        if not settings.rate_limit_enabled or request.url.path in _EXEMPT_PATHS:
            return await call_next(request)

        client = _client_id(request)
        capacity = float(settings.rate_limit_burst)
        refill = settings.rate_limit_rpm / 60.0

        with self._lock:
            bucket = self._buckets.get(client)
            if bucket is None:
                bucket = TokenBucket(capacity)
                self._buckets[client] = bucket
                # Evict the least recently seen client, not the least active —
                # an idle bucket is full anyway, so dropping it costs nothing.
                while len(self._buckets) > _MAX_TRACKED_CLIENTS:
                    self._buckets.popitem(last=False)
            else:
                self._buckets.move_to_end(client)
            allowed, retry_after = bucket.take(capacity, refill)

        if not allowed:
            metrics.incr("rate_limited_total")
            log.info("rate_limited", extra={"client": client, "path": request.url.path})
            body: dict[str, object] = {
                "error": {
                    "code": "rate_limited",
                    "message": (
                        f"Rate limit exceeded: {settings.rate_limit_rpm} requests/minute "
                        f"with a burst of {settings.rate_limit_burst}."
                    ),
                }
            }
            if (rid := get_request_id()) is not None:
                body["request_id"] = rid
            return JSONResponse(
                status_code=429,
                content=body,
                headers={
                    "Retry-After": str(max(1, int(retry_after + 0.999))),
                    "X-RateLimit-Limit": str(settings.rate_limit_rpm),
                    "X-RateLimit-Remaining": "0",
                },
            )

        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(settings.rate_limit_rpm)
        response.headers["X-RateLimit-Remaining"] = str(int(bucket.tokens))
        return response
