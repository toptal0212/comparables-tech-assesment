"""Serves the single-page UI at `/`.

Read from disk on each request in development so edits show up on refresh, and
cached in memory otherwise. The file is ~20KB, so the cache is free and the
read is cheap either way.

Mounting a StaticFiles app would be the conventional choice, but this is one
file, and a plain route keeps the root path under the same middleware — request
id, timing, rate limiting — as everything else.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import HTMLResponse, PlainTextResponse

from app.config import settings

router = APIRouter(tags=["ui"], include_in_schema=False)

UI_PATH = Path(__file__).resolve().parent.parent.parent / "ui" / "index.html"

_cached: str | None = None


def _load() -> str | None:
    global _cached
    if settings.env != "local" and _cached is not None:
        return _cached
    try:
        content = UI_PATH.read_text(encoding="utf-8")
    except OSError:
        return None
    _cached = content
    return content


@router.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    content = _load()
    if content is None:
        # The API is the product; a missing UI file is not a reason to 500.
        return HTMLResponse(
            "<h1>Company Search API</h1>"
            '<p>UI asset not found. The API is available at <a href="/docs">/docs</a>.</p>',
            status_code=200,
        )
    return HTMLResponse(content)


@router.get("/favicon.ico")
async def favicon() -> PlainTextResponse:
    # Browsers request this unprompted; answering 204 keeps it out of the error
    # logs without shipping an icon.
    return PlainTextResponse("", status_code=204)
