"""Build the document embedding matrix.

Run as part of `scripts/build_index.py`, never at request time.

What gets embedded
------------------
Not the raw description. In this corpus a description is
"{modifier} {noun} for {topic}." where the head is drawn from 48 combinations
that appear uniformly across all ten industries. Embedding that verbatim means
most of every vector encodes boilerplate: "AI-powered platform for fraud
detection" and "AI-powered platform for drug discovery" share four of their six
words, so they land far closer together than two fintech companies phrased
differently. The noise dominates the signal.

So the embedded text is composed to foreground what discriminates:

    "{name}. {description} Focus: {topics}. Sector: {industry}."

Topics are appended explicitly, which amplifies them under mean pooling — they
are the only part of the text that separates one company from another.

**Location is deliberately excluded.** It is a hard filter, handled exactly by
the column store, and putting it in the vector would make a Finnish fintech more
similar to a Finnish biotech than to a German fintech. That is wrong for
topical similarity and it would corrupt `/companies/{id}/similar`, whose whole
job is "what else does what this company does".
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Any

import numpy as np

from app.config import settings
from app.core.logging import get_logger
from app.models.company import CompanyRecord
from app.search.embeddings import Embedder
from app.search.vector import VectorIndex

log = get_logger(__name__)


def embedding_text(record: CompanyRecord) -> str:
    """The text representation of a company that gets embedded."""
    parts = [f"{record.name}."]
    if record.description:
        parts.append(record.description)
    if record.topics:
        parts.append(f"Focus: {', '.join(record.topics)}.")
    if record.industry:
        parts.append(f"Sector: {record.industry}.")
    return " ".join(parts)


def build_vector_index(
    db_path: Path,
    data_dir: Path,
    *,
    batch_size: int | None = None,
    progress_every: int = 10_000,
) -> dict[str, Any]:
    """Embed the whole corpus and persist the matrix.

    Reads through plain sqlite3 rather than the async layer: this is a batch job
    with no event loop, and a thread hop per statement would buy nothing.
    """
    batch_size = batch_size or settings.embed_batch_size
    started = time.perf_counter()

    embedder = Embedder.try_create(
        settings.embedding_model,
        dim=settings.embedding_dim,
        # The build is throughput-bound with no concurrent requests to protect,
        # which is the opposite of the serving case, so let onnxruntime use
        # every core here. Serving uses settings.embed_threads (1).
        threads=0,
        cache_dir=str(settings.model_cache_dir) if settings.model_cache_dir else None,
    )
    if embedder is None:
        raise RuntimeError(f"embedding model unavailable: {settings.embedding_model}")

    conn = sqlite3.connect(db_path)
    try:
        total = conn.execute("SELECT count(*) FROM companies").fetchone()[0]
        log.info("vector_build_start", extra={"documents": total, "model": embedder.model_name})

        ids: list[int] = []
        chunks: list[np.ndarray] = []
        texts: list[str] = []
        pending_ids: list[int] = []
        done = 0

        # Ordered by id so the matrix rows line up with ColumnStore, which sorts
        # the same way. The two are indexed by the same row positions and would
        # silently mis-associate results if either ordering changed.
        cursor = conn.execute(
            """
            SELECT id, name, description, industry, location, founded_year,
                   employee_count, revenue_range, revenue_min, revenue_max, topics
            FROM companies ORDER BY id
            """
        )

        import json as _json

        for row in cursor:
            record = CompanyRecord(
                id=row[0], name=row[1], description=row[2], industry=row[3],
                location=row[4], founded_year=row[5], employee_count=row[6],
                revenue_range=row[7], revenue_min=row[8], revenue_max=row[9],
                topics=_json.loads(row[10]) if row[10] else [],
            )
            texts.append(embedding_text(record))
            pending_ids.append(record.id)

            if len(texts) >= batch_size:
                chunks.append(embedder.embed_documents(texts, batch_size=batch_size))
                ids.extend(pending_ids)
                done += len(texts)
                texts, pending_ids = [], []
                if done % progress_every < batch_size:
                    rate = done / (time.perf_counter() - started)
                    log.info(
                        "vector_build_progress",
                        extra={
                            "done": done,
                            "total": total,
                            "docs_per_s": round(rate),
                            "eta_s": round((total - done) / rate) if rate else None,
                        },
                    )

        if texts:
            chunks.append(embedder.embed_documents(texts, batch_size=batch_size))
            ids.extend(pending_ids)
            done += len(texts)
    finally:
        conn.close()

    if not chunks:
        raise RuntimeError("no documents to embed")

    vectors = np.vstack(chunks).astype(np.float32)
    id_array = np.asarray(ids, dtype=np.int64)

    index = VectorIndex(vectors, id_array, model_name=embedder.model_name)
    index.save(
        data_dir / "vectors.npy",
        data_dir / "vector_ids.npy",
        data_dir / "index_meta.json",
        embed_seconds=round(time.perf_counter() - started, 1),
    )

    elapsed = time.perf_counter() - started
    return {
        "documents": int(vectors.shape[0]),
        "dim": int(vectors.shape[1]),
        "seconds": round(elapsed, 1),
        "docs_per_s": round(vectors.shape[0] / elapsed),
        "megabytes": round(vectors.nbytes / 1e6, 1),
        "model": embedder.model_name,
    }
