"""Application configuration.

Everything tunable lives here and is overridable by environment variable, so the
same image runs unchanged in local dev, CI and production. Defaults are chosen
to be safe for a small free-tier container (roughly 512MB RAM, 1 shared vCPU),
because that is the target deployment described in the brief.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # -- Service identity -------------------------------------------------
    app_name: str = "company-search"
    version: str = "1.0.0"
    env: Literal["local", "staging", "production"] = "local"

    # -- Logging ----------------------------------------------------------
    log_level: str = "INFO"
    # JSON lines in production (log collectors parse them); human-readable text
    # locally. Defaults flip automatically in `finalise()` below.
    log_json: bool | None = None

    # -- Paths ------------------------------------------------------------
    # `data_dir` is the only directory the service writes to at runtime. In
    # production it must be a persistent volume: it holds the SQLite database
    # (source of truth) and the derived vector index.
    data_dir: Path = REPO_ROOT / "data"
    dataset_path: Path = REPO_ROOT / "sample_dataset" / "companies.json"

    # -- Embeddings -------------------------------------------------------
    # bge-small-en-v1.5: 384 dimensions, ~130MB on disk. Measured ~25ms per
    # query embedding at embed_threads=1, which is the single largest term in
    # the request budget. Small enough to bake into the image, strong enough
    # for short business descriptions.
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384
    embed_batch_size: int = 256

    # Where the ONNX model files live. Pinned explicitly so the container
    # image can bake them in at build time; fastembed otherwise defaults to a
    # temp directory, which on a read-only or ephemeral filesystem means a
    # fresh ~130MB download on every cold start.
    model_cache_dir: Path | None = None

    # The service must still serve traffic when the embedding runtime is
    # unavailable (missing model files, an incompatible CPU, a cold volume).
    # With this off, search degrades to lexical + structured filtering rather
    # than failing to boot. See docs/DESIGN.md, "Graceful degradation".
    embeddings_enabled: bool = True

    # Query embeddings are pure functions of the query string and cost ~25ms —
    # by far the largest term in the latency budget. Caching them turns a repeat
    # query into a ~2us dictionary lookup.
    query_cache_size: int = 2048

    # onnxruntime threads per embedding call. 1 is not a typo: measured over 64
    # queries on 6 cores, threads=1 beat all-cores at every concurrency level
    # (73 vs 52 qps at 8 concurrent, p95 168ms vs 301ms). Letting one inference
    # fan out across cores makes concurrent requests fight each other.
    embed_threads: int = 1

    # -- Search behaviour -------------------------------------------------
    search_default_limit: int = 10
    search_max_limit: int = 100

    # How many candidates each retriever returns before fusion. Larger values
    # improve recall on heavily-filtered queries at a small CPU cost.
    candidate_pool: int = 400

    # Reciprocal Rank Fusion constant. 60 is the value from the original RRF
    # paper and is not sensitive enough to be worth tuning per-deployment.
    rrf_k: int = 60

    # Relative pull of each retriever during fusion.
    weight_keyword: float = 1.0
    weight_vector: float = 1.0
    # Topic matches are the strongest signal in this corpus (every description
    # reduces to one of 27 canonical topics), so they carry the most weight.
    weight_topic: float = 2.0

    # -- Constraint relaxation --------------------------------------------
    # Some realistic queries are unsatisfiable under strict AND semantics — two
    # of the brief's own example queries return zero rows. Rather than serve an
    # empty page, progressively drop the least central constraint and label the
    # response as relaxed.
    relaxation_enabled: bool = True
    relaxation_min_results: int = 3

    # -- Security ---------------------------------------------------------
    # Comma-separated list of accepted API keys. Empty means auth is disabled,
    # which is the default so the service is trivially explorable; production
    # deployments set API_KEYS.
    api_keys: str = ""
    # Read endpoints can be left public while writes stay authenticated. This is
    # what the public demo deployment uses.
    require_auth_for_reads: bool = False

    # -- Rate limiting ----------------------------------------------------
    # In-process token bucket per client. Protects a single small container from
    # a burst; a multi-replica deployment would move this to a shared store.
    rate_limit_enabled: bool = True
    rate_limit_rpm: int = 120
    rate_limit_burst: int = 30

    # -- HTTP -------------------------------------------------------------
    cors_origins: str = "*"
    # Upper bound on a single search. Prevents a pathological query from pinning
    # a worker; the client gets a 504 instead of hanging.
    request_timeout_s: float = 10.0

    # -- Derived paths ----------------------------------------------------
    @property
    def db_path(self) -> Path:
        return self.data_dir / "companies.db"

    @property
    def vectors_path(self) -> Path:
        """Document embedding matrix, float32, shape (n_docs, embedding_dim)."""
        return self.data_dir / "vectors.npy"

    @property
    def vector_ids_path(self) -> Path:
        """Row-index -> company id mapping for `vectors_path`."""
        return self.data_dir / "vector_ids.npy"

    @property
    def index_meta_path(self) -> Path:
        """Index provenance: row count, model name, build timestamp, checksum."""
        return self.data_dir / "index_meta.json"

    @property
    def api_key_set(self) -> frozenset[str]:
        return frozenset(k.strip() for k in self.api_keys.split(",") if k.strip())

    @property
    def auth_enabled(self) -> bool:
        return bool(self.api_key_set)

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @field_validator("log_level")
    @classmethod
    def _upper(cls, v: str) -> str:
        return v.upper()

    def finalise(self) -> Settings:
        """Apply defaults that depend on other fields."""
        if self.log_json is None:
            self.log_json = self.env != "local"
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton.

    Cached because it is read on hot paths (per request, for limits) and because
    a single consistent view of configuration is easier to reason about than
    re-reading the environment.
    """
    return Settings().finalise()


settings = get_settings()
