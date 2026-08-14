"""Sentence embeddings via ONNX Runtime.

Model: BAAI/bge-small-en-v1.5, 384 dimensions, ~130MB. Run through fastembed on
onnxruntime rather than sentence-transformers on PyTorch. torch would add ~2GB
to the image and several seconds to every cold start, for a model this size buys
nothing — and the brief calls out frequent restarts as a design constraint.

Three things this module is responsible for.

**Not blocking the event loop.** A query embedding takes ~25ms of pure CPU. Run
inline in a coroutine that stalls every other request on the worker; at 50
concurrent requests the tail becomes seconds. `embed_query` is therefore async
and hands the work to a thread. onnxruntime releases the GIL during inference,
so this is real parallelism, not just bookkeeping.

**Caching.** The embedding is a pure function of the query string and is the
single largest term in the latency budget — 25ms against a 200ms target, nearly
ten times the cost of the vector search it feeds. An LRU over normalised query
text turns a repeat query into a ~2us lookup. Search traffic is heavily
repetitive, so this is the highest-leverage cache in the system.

**Failing softly.** `try_create` returns None instead of raising. A missing model
file, a read-only cache directory or an incompatible CPU must degrade search to
lexical-plus-filters, not prevent the service from starting. This is not
hypothetical: the Hugging Face Xet download backend aborts with SIGILL on CPUs
lacking the instruction set it was built for, which is how it behaved on the
machine this was developed on.
"""

from __future__ import annotations

import asyncio
import threading
from collections import OrderedDict

import numpy as np

from app.core.logging import get_logger
from app.core.metrics import metrics

log = get_logger(__name__)


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """L2-normalise each row so a dot product is cosine similarity.

    Normalising once at build time removes a division from every query. The
    epsilon guards against a zero vector, which would otherwise produce NaNs
    that silently poison the ranking.
    """
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    np.maximum(norms, 1e-12, out=norms)
    return (matrix / norms).astype(np.float32, copy=False)


class Embedder:
    """Wraps a fastembed model with an async, cached query path."""

    def __init__(self, model: object, model_name: str, dim: int, cache_size: int = 2048) -> None:
        self._model = model
        self.model_name = model_name
        self.dim = dim
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._cache_size = cache_size
        self._cache_lock = threading.Lock()

    # -- construction -----------------------------------------------------

    @classmethod
    def try_create(
        cls,
        model_name: str,
        *,
        dim: int,
        cache_size: int = 2048,
        threads: int = 1,
        cache_dir: str | None = None,
    ) -> Embedder | None:
        """Load the model, or return None and log why.

        `threads=1` is deliberate, and was measured rather than assumed. Over 64
        queries on 6 cores it beat letting onnxruntime use all of them at every
        concurrency level tested — 26.6ms vs 33.5ms serially, and 73 vs 52 qps
        at 8 concurrent with a p95 of 168ms against 301ms. Fanning one small
        inference across cores mostly buys thread-coordination overhead, and
        under load the requests fight each other for the same cores.
        """
        try:
            from fastembed import TextEmbedding

            model = TextEmbedding(
                model_name=model_name, threads=threads, cache_dir=cache_dir
            )
            probe = next(iter(model.embed(["warmup"])))
            actual_dim = int(np.asarray(probe).shape[-1])
            if actual_dim != dim:
                log.warning(
                    "embedding_dim_mismatch",
                    extra={"configured": dim, "actual": actual_dim, "model": model_name},
                )
                dim = actual_dim
            log.info("embedder_ready", extra={"model": model_name, "dim": dim, "threads": threads})
            return cls(model, model_name, dim, cache_size)
        except Exception:
            log.exception(
                "embedder_unavailable_degrading_to_lexical", extra={"model": model_name}
            )
            return None

    # -- documents --------------------------------------------------------

    def embed_documents(self, texts: list[str], batch_size: int = 256) -> np.ndarray:
        """Embed a batch of documents. Synchronous: only used by the build job."""
        vectors = list(self._model.embed(texts, batch_size=batch_size))  # type: ignore[attr-defined]
        return normalize_rows(np.asarray(vectors, dtype=np.float32))

    # -- queries ----------------------------------------------------------

    def _embed_query_sync(self, text: str) -> np.ndarray:
        # bge models are trained asymmetrically: queries get an instruction
        # prefix that passages do not. fastembed exposes that as query_embed.
        # Measured neutral on this corpus (72/80 correct topics in top-10 either
        # way, same latency) because the documents are single sentences, but it
        # is the model's documented usage and would matter on longer passages.
        # Falls back to plain embed for models without the distinction.
        try:
            vector = next(iter(self._model.query_embed(text)))  # type: ignore[attr-defined]
        except (AttributeError, NotImplementedError):
            vector = next(iter(self._model.embed([text])))  # type: ignore[attr-defined]
        return normalize_rows(np.asarray(vector, dtype=np.float32).reshape(1, -1))[0]

    async def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query, cached, off the event loop."""
        key = text.strip().casefold()

        with self._cache_lock:
            hit = self._cache.get(key)
            if hit is not None:
                self._cache.move_to_end(key)
                metrics.incr("embedding_cache", result="hit")
                return hit

        metrics.incr("embedding_cache", result="miss")
        with metrics.timer("embed_query_ms"):
            vector = await asyncio.to_thread(self._embed_query_sync, text)

        with self._cache_lock:
            self._cache[key] = vector
            self._cache.move_to_end(key)
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)

        return vector

    @property
    def cache_size(self) -> int:
        with self._cache_lock:
            return len(self._cache)
