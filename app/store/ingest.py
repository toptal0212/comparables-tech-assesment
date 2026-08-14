"""Bulk corpus loading.

Separate from `Database` because a full load is a batch job with different
constraints from a request: it is single-threaded, it owns the database, and it
is allowed to disable safety rails that only matter under concurrency. Using the
async layer here would cost a thread hop per statement and buy nothing.

Two decisions worth stating:

**Streaming reader.** The provided file is 13MB and would load fine with
`json.load`, but the brief asks how the system behaves as the corpus grows.
`_iter_json_array` decodes one object at a time out of a sliding buffer, so peak
memory is the buffer plus one record whether the file is 13MB or 13GB. JSON
Lines is also accepted and is the better format at that size.

**Triggers off during load.** The FTS triggers in schema.sql fire per row. For a
full corpus load that is 50,000 individual index updates; dropping them and
issuing a single `rebuild` afterwards does the same work in one pass. They are
restored before the function returns, so the ingestion endpoint still works.
"""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

from app.core.logging import get_logger
from app.models.company import CompanyIn
from app.store.db import SCHEMA_PATH
from app.store.enrich import enrich

log = get_logger(__name__)

_TRIGGERS = ("companies_ai", "companies_ad", "companies_au")


@dataclass
class IngestStats:
    read: int = 0
    written: int = 0
    invalid: int = 0
    elapsed_s: float = 0.0
    errors: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            "read": self.read,
            "written": self.written,
            "invalid": self.invalid,
            "elapsed_s": round(self.elapsed_s, 2),
            "rate_per_s": round(self.written / self.elapsed_s) if self.elapsed_s else 0,
        }


def _iter_json_array(fh: TextIO, chunk_size: int = 1 << 20) -> Iterator[dict[str, Any]]:
    """Yield objects from a top-level JSON array without holding it all in memory."""
    decoder = json.JSONDecoder()
    buf = fh.read(chunk_size)
    start = buf.find("[")
    if start < 0:
        raise ValueError("expected a top-level JSON array")
    buf = buf[start + 1 :]

    while True:
        buf = buf.lstrip()
        if buf[:1] == ",":
            buf = buf[1:].lstrip()
        if buf[:1] == "]":
            return
        if not buf:
            more = fh.read(chunk_size)
            if not more:
                return
            buf = more
            continue
        try:
            obj, end = decoder.raw_decode(buf)
        except json.JSONDecodeError:
            # Almost certainly a record straddling the buffer boundary. Pull
            # more input and retry; only give up when the file is exhausted.
            more = fh.read(chunk_size)
            if not more:
                raise
            buf += more
            continue
        yield obj
        buf = buf[end:]


def iter_dataset(path: Path) -> Iterator[dict[str, Any]]:
    """Stream records from a `.json` array or a `.jsonl` file."""
    with path.open("r", encoding="utf-8") as fh:
        if path.suffix == ".jsonl":
            for line in fh:
                if line := line.strip():
                    yield json.loads(line)
        else:
            yield from _iter_json_array(fh)


def bulk_ingest(
    db_path: Path,
    dataset_path: Path,
    *,
    batch_size: int = 5000,
    max_reported_errors: int = 20,
) -> IngestStats:
    """Load a dataset into a fresh or existing database.

    Invalid records are counted and skipped rather than aborting the load. A
    corpus of this size will contain a few malformed rows, and failing the whole
    import over them is the wrong default for a bulk tool.
    """
    started = time.perf_counter()
    stats = IngestStats()

    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
        # Safe for a batch load: if the process dies we re-run it from the
        # source file, so trading crash-durability for speed costs nothing.
        conn.execute("PRAGMA synchronous = OFF")
        conn.execute("PRAGMA journal_mode = WAL")

        for trigger in _TRIGGERS:
            conn.execute(f"DROP TRIGGER IF EXISTS {trigger}")
        conn.commit()

        now = datetime.now(timezone.utc).isoformat()
        batch: list[tuple[Any, ...]] = []
        next_id = _max_id(conn) + 1

        for raw in iter_dataset(dataset_path):
            stats.read += 1
            try:
                company = CompanyIn.model_validate(raw)
            except Exception as exc:
                stats.invalid += 1
                if len(stats.errors) < max_reported_errors:
                    stats.errors.append(f"record {stats.read}: {exc}")
                continue

            if company.id is None:
                company.id = next_id
                next_id += 1
            else:
                next_id = max(next_id, company.id + 1)

            record = enrich(company, company_id=company.id)
            batch.append(
                (
                    record.id, record.name, record.description, record.industry,
                    record.location, record.founded_year, record.employee_count,
                    record.revenue_range, record.revenue_min, record.revenue_max,
                    json.dumps(record.topics), " ".join(record.topics), now,
                )
            )

            if len(batch) >= batch_size:
                stats.written += _flush(conn, batch)
                batch.clear()

        if batch:
            stats.written += _flush(conn, batch)

        # Restore triggers, then build the whole FTS index in one pass.
        conn.executescript(_trigger_ddl())
        conn.execute("INSERT INTO companies_fts (companies_fts) VALUES ('rebuild')")
        # Refresh planner statistics so the composite indexes actually get used.
        conn.execute("ANALYZE")
        conn.execute(
            "INSERT INTO index_meta (key, value) VALUES ('corpus_loaded_at', ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (now,),
        )
        conn.commit()
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        conn.commit()
    finally:
        conn.close()

    stats.elapsed_s = time.perf_counter() - started
    log.info("bulk_ingest_complete", extra=stats.summary())
    return stats


def _flush(conn: sqlite3.Connection, batch: list[tuple[Any, ...]]) -> int:
    conn.executemany(
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
        batch,
    )
    conn.commit()
    return len(batch)


def _max_id(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT coalesce(max(id), 0) FROM companies").fetchone()
    return int(row[0]) if row else 0


def _trigger_ddl() -> str:
    """The trigger definitions, lifted back out of schema.sql.

    Read from the file rather than duplicated here so there is exactly one
    definition of each trigger and it cannot drift.
    """
    sql = SCHEMA_PATH.read_text(encoding="utf-8")
    out: list[str] = []
    for chunk in sql.split("CREATE TRIGGER")[1:]:
        body, _, _ = chunk.partition("END;")
        out.append(f"CREATE TRIGGER{body}END;")
    return "\n".join(out)
