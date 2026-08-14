"""Async SQLite access layer.

Concurrency model
-----------------
SQLite in WAL mode allows any number of concurrent readers alongside a single
writer. This class mirrors that exactly:

* a small pool of read connections, handed out through an `asyncio.Queue`;
* one dedicated write connection behind an `asyncio.Lock`.

Every connection is an `aiosqlite` connection, which owns a thread and executes
statements there. That matters more than it looks: a synchronous `sqlite3` call
in a request handler blocks the whole event loop for the duration of the query,
so a 3ms FTS scan under 50 concurrent requests turns into 150ms of head-of-line
blocking. Pushing the work onto connection threads keeps the loop responsive and
lets reads genuinely overlap.

Pool size defaults to 4. SQLite reads are served from the page cache and are
short; more connections mostly buy thread-switching overhead, and each one costs
memory for its own page cache on a container with ~512MB.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Iterable, Sequence
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiosqlite

from app.core.logging import get_logger
from app.models.company import CompanyRecord

log = get_logger(__name__)

SCHEMA_PATH = Path(__file__).parent / "schema.sql"

# Applied to every connection. WAL and the busy timeout are per-database and
# persist, but the pragmas below are per-connection and must be re-issued.
_CONNECTION_PRAGMAS = (
    # Wait rather than immediately returning SQLITE_BUSY when a writer holds the
    # lock. Without this, a concurrent ingest can fail an unrelated read.
    "PRAGMA busy_timeout = 5000",
    # 64MB page cache per connection (negative value = kibibytes).
    "PRAGMA cache_size = -64000",
    # Keep temp b-trees (ORDER BY, GROUP BY) off disk.
    "PRAGMA temp_store = MEMORY",
    # Let SQLite memory-map the database file; removes a copy on every read.
    "PRAGMA mmap_size = 268435456",
)

_SELECT_COLUMNS = """
    id, name, description, industry, location, founded_year,
    employee_count, revenue_range, revenue_min, revenue_max, topics
"""


def _row_to_record(row: aiosqlite.Row | Sequence[Any]) -> CompanyRecord:
    return CompanyRecord(
        id=row[0],
        name=row[1],
        description=row[2],
        industry=row[3],
        location=row[4],
        founded_year=row[5],
        employee_count=row[6],
        revenue_range=row[7],
        revenue_min=row[8],
        revenue_max=row[9],
        topics=json.loads(row[10]) if row[10] else [],
    )


class Database:
    def __init__(self, path: Path, *, read_pool_size: int = 4) -> None:
        self.path = path
        self._read_pool_size = read_pool_size
        self._readers: asyncio.Queue[aiosqlite.Connection] = asyncio.Queue()
        self._all_readers: list[aiosqlite.Connection] = []
        self._writer: aiosqlite.Connection | None = None
        self._write_lock = asyncio.Lock()
        self._closed = False

    # -- lifecycle --------------------------------------------------------

    async def connect(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)

        # The writer runs the schema first, so readers never observe a
        # half-created database.
        self._writer = await self._open()
        await self._apply_schema(self._writer)

        for _ in range(self._read_pool_size):
            conn = await self._open()
            self._all_readers.append(conn)
            self._readers.put_nowait(conn)

        log.info(
            "database_connected",
            extra={"path": str(self.path), "read_pool": self._read_pool_size},
        )

    async def _open(self) -> aiosqlite.Connection:
        conn = await aiosqlite.connect(self.path)
        for pragma in _CONNECTION_PRAGMAS:
            await conn.execute(pragma)
        return conn

    async def _apply_schema(self, conn: aiosqlite.Connection) -> None:
        await conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
        await conn.commit()

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for conn in self._all_readers:
            await conn.close()
        if self._writer is not None:
            # Fold the WAL back into the main database so the next start does
            # not have to replay it. Meaningful on a cold container mounting a
            # volume that was left mid-write.
            try:
                await self._writer.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            except Exception:  # pragma: no cover - best effort on shutdown
                log.warning("wal_checkpoint_failed", exc_info=True)
            await self._writer.close()
        log.info("database_closed")

    @asynccontextmanager
    async def reader(self) -> AsyncIterator[aiosqlite.Connection]:
        conn = await self._readers.get()
        try:
            yield conn
        finally:
            self._readers.put_nowait(conn)

    @asynccontextmanager
    async def writer(self) -> AsyncIterator[aiosqlite.Connection]:
        if self._writer is None:
            raise RuntimeError("Database.connect() has not been called")
        async with self._write_lock:
            yield self._writer

    # -- reads ------------------------------------------------------------

    async def count(self) -> int:
        async with self.reader() as conn:
            async with conn.execute("SELECT count(*) FROM companies") as cur:
                row = await cur.fetchone()
        return int(row[0]) if row else 0

    async def get(self, company_id: int) -> CompanyRecord | None:
        async with self.reader() as conn:
            async with conn.execute(
                f"SELECT {_SELECT_COLUMNS} FROM companies WHERE id = ?", (company_id,)
            ) as cur:
                row = await cur.fetchone()
        return _row_to_record(row) if row else None

    async def get_many(self, ids: Sequence[int]) -> dict[int, CompanyRecord]:
        """Hydrate a set of ids in one round-trip, preserving nothing about order.

        The caller already knows the ranking; re-sorting in SQL would only add
        work. Returning a dict makes the caller's ordering explicit.
        """
        if not ids:
            return {}
        placeholders = ",".join("?" * len(ids))
        async with self.reader() as conn:
            async with conn.execute(
                f"SELECT {_SELECT_COLUMNS} FROM companies WHERE id IN ({placeholders})",
                tuple(ids),
            ) as cur:
                rows = await cur.fetchall()
        return {row[0]: _row_to_record(row) for row in rows}

    async def iter_all(self, batch_size: int = 2000) -> AsyncIterator[list[CompanyRecord]]:
        """Stream the whole corpus in batches.

        Used when building the vector index. Batched rather than one big
        fetchall so peak memory stays bounded — at 50k rows the difference is
        immaterial, at 5M it is the difference between working and not.
        """
        last_id = 0
        async with self.reader() as conn:
            while True:
                async with conn.execute(
                    f"SELECT {_SELECT_COLUMNS} FROM companies "
                    "WHERE id > ? ORDER BY id LIMIT ?",
                    (last_id, batch_size),
                ) as cur:
                    rows = await cur.fetchall()
                if not rows:
                    return
                last_id = rows[-1][0]
                yield [_row_to_record(r) for r in rows]

    # -- writes -----------------------------------------------------------

    async def upsert_many(self, records: Iterable[CompanyRecord]) -> int:
        """Insert or replace records. Returns how many were written.

        One transaction for the whole batch: 50k individual commits would be
        thousands of fsyncs, whereas one commit is a single flush.
        """
        now = datetime.now(timezone.utc).isoformat()
        rows = [
            (
                r.id, r.name, r.description, r.industry, r.location,
                r.founded_year, r.employee_count, r.revenue_range,
                r.revenue_min, r.revenue_max,
                json.dumps(r.topics), " ".join(r.topics), now,
            )
            for r in records
        ]
        if not rows:
            return 0
        async with self.writer() as conn:
            await conn.executemany(
                """
                INSERT INTO companies (
                    id, name, description, industry, location, founded_year,
                    employee_count, revenue_range, revenue_min, revenue_max,
                    topics, topics_text, updated_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(id) DO UPDATE SET
                    name = excluded.name,
                    description = excluded.description,
                    industry = excluded.industry,
                    location = excluded.location,
                    founded_year = excluded.founded_year,
                    employee_count = excluded.employee_count,
                    revenue_range = excluded.revenue_range,
                    revenue_min = excluded.revenue_min,
                    revenue_max = excluded.revenue_max,
                    topics = excluded.topics,
                    topics_text = excluded.topics_text,
                    updated_at = excluded.updated_at
                """,
                rows,
            )
            await conn.commit()
        return len(rows)

    async def delete(self, company_id: int) -> bool:
        async with self.writer() as conn:
            cur = await conn.execute("DELETE FROM companies WHERE id = ?", (company_id,))
            await conn.commit()
            return cur.rowcount > 0

    async def next_id(self) -> int:
        async with self.reader() as conn:
            async with conn.execute("SELECT coalesce(max(id), 0) FROM companies") as cur:
                row = await cur.fetchone()
        return int(row[0]) + 1 if row else 1

    # -- metadata ---------------------------------------------------------

    async def set_meta(self, key: str, value: str) -> None:
        async with self.writer() as conn:
            await conn.execute(
                "INSERT INTO index_meta (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )
            await conn.commit()

    async def get_meta(self, key: str) -> str | None:
        async with self.reader() as conn:
            async with conn.execute(
                "SELECT value FROM index_meta WHERE key = ?", (key,)
            ) as cur:
                row = await cur.fetchone()
        return row[0] if row else None
