"""Assemble the search runtime at startup, and rebuild it after ingestion.

Startup is where the brief's "frequent restarts" constraint is actually paid.
The sequence is: open the database, load the corpus into columnar arrays, mmap
the prebuilt vector matrix, load the embedding model. On the reference corpus
that is about 1.5 seconds — because nothing here computes anything. All the
expensive work happened in `scripts/build_index.py`.

Every stage degrades rather than aborting:

  * no database          -> not ready, /health/ready explains why, process lives
  * no vector matrix     -> lexical + filters, semantic_search reported false
  * no embedding model   -> same
  * stale vector matrix  -> treated as absent, and logged loudly

The only unrecoverable state is "no corpus at all", which is reported through
readiness rather than by crashing, so an operator can reach /health and /metrics
on a broken deploy instead of watching a container restart loop.
"""

from __future__ import annotations

import time

from app.config import settings
from app.core.logging import get_logger
from app.core.metrics import metrics
from app.search.columns import ColumnStore
from app.search.embeddings import Embedder
from app.search.keyword import KeywordIndex
from app.search.state import SearchRuntime, get_index_state
from app.search.vector import RowMap, VectorIndex
from app.store.db import Database

log = get_logger(__name__)

# Below this fraction of companies having an embedding, the matrix is treated
# as belonging to a different corpus rather than merely lagging behind one.
MIN_VECTOR_COVERAGE = 0.5


async def load_runtime(db: Database | None = None) -> SearchRuntime | None:
    """Build a `SearchRuntime` from whatever artifacts are on disk.

    Returns None when there is no usable corpus. The caller decides what that
    means; `startup` below records it as a readiness failure.
    """
    started = time.perf_counter()

    if db is None:
        db = Database(settings.db_path, read_pool_size=4)
        await db.connect()

    doc_count = await db.count()
    if doc_count == 0:
        log.error(
            "corpus_empty",
            extra={
                "db": str(settings.db_path),
                "hint": "run: python -m scripts.build_index",
            },
        )
        return None

    # -- columnar filter arrays ------------------------------------------
    t0 = time.perf_counter()
    records = []
    async for batch in db.iter_all():
        records.extend(batch)
    columns = ColumnStore.build(records)
    columns_ms = (time.perf_counter() - t0) * 1000

    # Free the record objects; the column store holds everything the filter path
    # needs, and hydration re-reads from SQLite. Keeping 50k pydantic models
    # resident would cost far more than the arrays themselves.
    del records

    # -- vector index -----------------------------------------------------
    t0 = time.perf_counter()
    vector: VectorIndex | None = None
    embedder: Embedder | None = None

    if settings.embeddings_enabled:
        # No expected_count here: a container restarting after ingestion will
        # legitimately find more rows in SQLite than in the matrix, and RowMap
        # handles that. A genuinely mismatched index is caught by the coverage
        # check below instead.
        vector = VectorIndex.load(
            settings.vectors_path,
            settings.vector_ids_path,
            settings.index_meta_path,
            expected_model=settings.embedding_model,
        )

        if vector is not None:
            embedder = Embedder.try_create(
                settings.embedding_model,
                dim=settings.embedding_dim,
                cache_size=settings.query_cache_size,
                threads=settings.embed_threads,
                cache_dir=str(settings.model_cache_dir)
                if settings.model_cache_dir
                else None,
            )
            if embedder is None:
                # Documents are embedded but queries cannot be. The matrix is
                # useless without a matching query encoder.
                log.warning("embedder_unavailable_discarding_vector_index")
                vector = None
    else:
        log.info("embeddings_disabled_by_config")

    vector_ms = (time.perf_counter() - t0) * 1000

    # Rows are looked up positionally in both structures, so the translation
    # between them has to be explicit. Identity after a clean build.
    row_map = RowMap(columns.ids, vector.ids) if vector is not None else None
    if row_map is not None and not row_map.aligned:
        if row_map.coverage < MIN_VECTOR_COVERAGE:
            # A handful of un-embedded new arrivals is expected. Coverage this
            # low means the matrix belongs to a different corpus, and using it
            # would silently degrade every result.
            log.error(
                "vector_index_coverage_too_low_discarding",
                extra={
                    "coverage": round(row_map.coverage, 4),
                    "vector_rows": vector.size if vector else 0,
                    "column_rows": columns.size,
                },
            )
            vector, embedder, row_map = None, None, None
        else:
            log.warning(
                "vector_index_partial_coverage",
                extra={
                    "vector_rows": vector.size if vector else 0,
                    "column_rows": columns.size,
                    "coverage": round(row_map.coverage, 4),
                },
            )

    runtime = SearchRuntime(
        db=db,
        keyword=KeywordIndex(db),
        columns=columns,
        vector=vector,
        embedder=embedder,
        row_map=row_map,
        doc_count=doc_count,
        built_at=await db.get_meta("corpus_loaded_at"),
        model_name=embedder.model_name if embedder else None,
    )

    total_ms = (time.perf_counter() - started) * 1000
    metrics.gauge("index_documents", doc_count)
    metrics.gauge("index_semantic_enabled", 1.0 if runtime.semantic_enabled else 0.0)
    log.info(
        "runtime_loaded",
        extra={
            "documents": doc_count,
            "semantic": runtime.semantic_enabled,
            "columns_ms": round(columns_ms),
            "vector_ms": round(vector_ms),
            "total_ms": round(total_ms),
        },
    )
    return runtime


async def startup() -> None:
    """Load the runtime and publish it, or record why readiness failed."""
    state = get_index_state()
    try:
        runtime = await load_runtime()
    except Exception as exc:
        log.exception("runtime_load_failed")
        state.mark_unavailable(f"index load failed: {type(exc).__name__}")
        return

    if runtime is None:
        state.mark_unavailable(
            "no corpus loaded — run `python -m scripts.build_index`"
        )
        return

    state.set_runtime(runtime)


async def shutdown() -> None:
    state = get_index_state()
    runtime = state.runtime
    state.mark_unavailable("shutting down")
    if runtime is not None:
        await runtime.db.close()


async def reload_columns() -> None:
    """Rebuild the in-memory column store from SQLite after a write.

    Ingestion writes straight to SQLite, which keeps the durable record and the
    FTS index correct immediately. The columnar arrays are a derived cache and
    have to be regenerated for new companies to become filterable.

    Rebuilding the whole store takes ~0.5s at 50k rows. That is acceptable for
    an occasional ingest and much simpler than incremental array maintenance,
    which would need resizing, gap management and its own correctness argument.
    The rebuilt runtime is swapped in atomically, so in-flight queries keep
    reading the old one and never see a partially updated index.

    Newly ingested companies are searchable by keyword and filters immediately;
    they gain semantic search at the next full index build, because embedding
    them requires the model and a matrix rewrite. This is stated in the API
    response so the behaviour is not a surprise.
    """
    state = get_index_state()
    current = state.runtime
    if current is None:
        await startup()
        return

    t0 = time.perf_counter()
    records = []
    async for batch in current.db.iter_all():
        records.extend(batch)
    columns = ColumnStore.build(records)
    doc_count = len(records)
    del records

    vector = current.vector
    row_map = RowMap(columns.ids, vector.ids) if vector is not None else None
    if row_map is not None and not row_map.aligned:
        # The corpus changed shape. The vector index is still valid for every
        # company it covers, so it is remapped rather than dropped: new arrivals
        # simply sit out semantic ranking until the next full build.
        log.info(
            "vector_index_remapped_after_write",
            extra={
                "vector_rows": vector.size if vector else 0,
                "corpus_rows": columns.size,
                "coverage": round(row_map.coverage, 4),
            },
        )

    state.set_runtime(
        SearchRuntime(
            db=current.db,
            keyword=current.keyword,
            columns=columns,
            vector=vector,
            embedder=current.embedder,
            row_map=row_map,
            doc_count=doc_count,
            built_at=await current.db.get_meta("corpus_loaded_at"),
            model_name=current.model_name,
        )
    )
    metrics.gauge("index_documents", doc_count)
    log.info(
        "columns_reloaded",
        extra={"documents": doc_count, "ms": round((time.perf_counter() - t0) * 1000)},
    )
