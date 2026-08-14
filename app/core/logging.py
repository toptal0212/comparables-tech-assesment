"""Structured logging and request correlation.

Two things matter operationally: every line emitted while handling a request
carries the same `request_id`, and in production those lines are machine
parseable. A `ContextVar` gives us the first without threading a logger through
every call site, and it works correctly under asyncio because each task gets its
own copy.
"""

from __future__ import annotations

import json
import logging
import sys
import uuid
from contextvars import ContextVar
from typing import Any

from app.config import settings

_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)

# Attributes present on every LogRecord. Anything outside this set was attached
# by the caller via `extra=` and belongs in the structured output.
_STD_ATTRS = frozenset(
    """args asctime created exc_info exc_text filename funcName levelname levelno
    lineno module msecs message msg name pathname process processName
    relativeCreated stack_info thread threadName taskName""".split()
)


def new_request_id() -> str:
    return uuid.uuid4().hex[:16]


def set_request_id(value: str) -> None:
    _request_id.set(value)


def get_request_id() -> str | None:
    return _request_id.get()


class JsonFormatter(logging.Formatter):
    """One JSON object per line, with caller-supplied `extra` fields merged in."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if (rid := get_request_id()) is not None:
            payload["request_id"] = rid
        for key, value in record.__dict__.items():
            if key not in _STD_ATTRS and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


class TextFormatter(logging.Formatter):
    """Human-readable local format; keeps `extra` fields visible but compact."""

    def format(self, record: logging.LogRecord) -> str:
        rid = get_request_id()
        prefix = f"[{rid}] " if rid else ""
        extras = " ".join(
            f"{k}={v}"
            for k, v in record.__dict__.items()
            if k not in _STD_ATTRS and not k.startswith("_")
        )
        stamp = self.formatTime(record, "%H:%M:%S")
        line = f"{stamp} {record.levelname:<7} {prefix}{record.name}: {record.getMessage()}"
        if extras:
            line = f"{line}  {extras}"
        if record.exc_info:
            line = f"{line}\n{self.formatException(record.exc_info)}"
        return line


def configure_logging() -> None:
    """Install our formatter on the root logger.

    uvicorn installs its own handlers at startup. We clear them and let its
    loggers propagate to root instead, so access logs and application logs share
    one format and one destination.
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter() if settings.log_json else TextFormatter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(settings.log_level)

    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        lg = logging.getLogger(name)
        lg.handlers.clear()
        lg.propagate = True

    # These are chatty at INFO during index builds and tell us nothing we do not
    # already log ourselves.
    logging.getLogger("httpx").setLevel("WARNING")
    logging.getLogger("filelock").setLevel("WARNING")


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
