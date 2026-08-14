"""Shared test fixtures.

Environment is configured *before* anything from `app` is imported, because
`app.config` builds its settings singleton at import time. Everything points at
a temporary directory and embeddings are off by default, so the suite runs in
seconds with no model download and no network.

The fixture corpus is small (about 60 companies) and hand-shaped to exercise the
awkward cases rather than being a sample of the real data: missing values,
every revenue bucket, multi-topic descriptions, and a deliberately
unsatisfiable filter combination for the relaxation tests.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

# --- must run before any `app` import ---------------------------------------
_TMP = Path(tempfile.mkdtemp(prefix="company-search-tests-"))
os.environ.update(
    {
        "ENV": "local",
        "DATA_DIR": str(_TMP),
        "LOG_LEVEL": "WARNING",
        "EMBEDDINGS_ENABLED": "false",
        "RATE_LIMIT_ENABLED": "false",
        "API_KEYS": "",
        "RELAXATION_MIN_RESULTS": "3",
        # Keep the HF download path from ever being touched by an accidental
        # model load during the suite.
        "HF_HUB_OFFLINE": "1",
        "HF_HUB_DISABLE_XET": "1",
    }
)

from app.config import get_settings  # noqa: E402
from app.models.company import CompanyIn  # noqa: E402
from app.search.columns import ColumnStore  # noqa: E402
from app.search.keyword import KeywordIndex  # noqa: E402
from app.search.state import SearchRuntime  # noqa: E402
from app.store.db import Database  # noqa: E402
from app.store.enrich import enrich  # noqa: E402

TOPIC_BY_INDUSTRY = {
    "Fintech": ["fraud detection", "banking analytics", "payments"],
    "Healthcare": ["diagnostics", "patient monitoring"],
    "Biotech": ["drug discovery", "gene editing"],
    "Technology": ["data pipelines", "observability"],
    "Energy": ["energy forecasting", "smart grid"],
    "Retail": ["pricing optimization", "demand forecasting"],
    "Education": ["personalized learning"],
    "Telecom": ["5g analytics", "network optimization"],
}
LOCATIONS = ["Finland", "Germany", "UK", "Sweden", "USA", "Norway"]
BUCKETS = ["0-1M", "1M-10M", "10M-50M", "50M-100M", "100M-500M", "500M+"]


def build_corpus() -> list[CompanyIn]:
    """Deterministic fixture corpus."""
    companies: list[CompanyIn] = []
    next_id = 1

    for industry, topics in TOPIC_BY_INDUSTRY.items():
        for t_i, topic in enumerate(topics):
            for l_i, location in enumerate(LOCATIONS[:3]):
                companies.append(
                    CompanyIn(
                        id=next_id,
                        name=f"{industry}{next_id} Labs",
                        description=f"AI-powered platform for {topic}.",
                        industry=industry,
                        location=location,
                        founded_year=2000 + ((next_id * 3) % 24),
                        employee_count=10 + (next_id * 37) % 4000,
                        revenue_range=BUCKETS[(t_i + l_i + next_id) % len(BUCKETS)],
                    )
                )
                next_id += 1

    # Multi-topic, hand-written phrasing — the shape the template parser missed.
    companies.append(
        CompanyIn(
            id=next_id,
            name="Nordic Compound Systems",
            description=(
                "AI-powered drug discovery and molecular analysis platform for "
                "biotech research teams."
            ),
            industry="Biotech",
            location="Germany",
            founded_year=2019,
            employee_count=95,
            revenue_range="10M-50M",
        )
    )
    next_id += 1

    # Missing founded_year and employee_count: numeric predicates must exclude
    # these rather than treating absence as zero.
    companies.append(
        CompanyIn(
            id=next_id,
            name="Unknown Vintage Oy",
            description="Real-time engine for fraud detection.",
            industry="Fintech",
            location="Finland",
            founded_year=None,
            employee_count=None,
            revenue_range=None,
        )
    )
    next_id += 1

    # The only sub-20-employee company, and it is in Sweden — so a UK query with
    # "fewer than 20 employees" is unsatisfiable and must trigger relaxation.
    companies.append(
        CompanyIn(
            id=next_id,
            name="Tiny Telecom AB",
            description="Distributed system for 5G analytics.",
            industry="Telecom",
            location="Sweden",
            founded_year=2021,
            employee_count=8,
            revenue_range="0-1M",
        )
    )
    return companies


@pytest.fixture(scope="session")
def settings():
    get_settings.cache_clear()
    return get_settings()


@pytest.fixture(scope="session")
def corpus() -> list[CompanyIn]:
    return build_corpus()


@pytest.fixture
async def db(tmp_path: Path, corpus: list[CompanyIn]) -> Iterator[Database]:
    database = Database(tmp_path / "test.db", read_pool_size=2)
    await database.connect()
    await database.upsert_many([enrich(c, company_id=c.id or 0) for c in corpus])
    yield database
    await database.close()


@pytest.fixture
async def columns(db: Database) -> ColumnStore:
    records = []
    async for batch in db.iter_all():
        records.extend(batch)
    return ColumnStore.build(records)


@pytest.fixture
async def runtime(db: Database, columns: ColumnStore) -> SearchRuntime:
    """Lexical-only runtime: no vector index, no embedding model.

    Most engine behaviour — filtering, relaxation, keyword retrieval, topic
    scoring — is independent of the semantic path, and testing it without the
    model keeps the suite fast and offline. `test_vector.py` covers the vector
    layer directly with synthetic embeddings.
    """
    return SearchRuntime(
        db=db,
        keyword=KeywordIndex(db),
        columns=columns,
        vector=None,
        embedder=None,
        row_map=None,
        doc_count=columns.size,
    )


@pytest.fixture
async def engine(runtime: SearchRuntime):
    from app.search.engine import SearchEngine

    return SearchEngine(runtime, runtime.columns)


@pytest.fixture(scope="session")
def api_corpus_db(settings, corpus: list[CompanyIn]) -> Path:
    """Populate the database the application itself will open at startup.

    Session-scoped and built synchronously: the app's lifespan reads
    `settings.db_path` before any test runs, so this has to exist first.
    """
    import sqlite3

    from app.store.db import SCHEMA_PATH

    path = settings.db_path
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
    now = "2026-01-01T00:00:00+00:00"
    rows = []
    for company in corpus:
        record = enrich(company, company_id=company.id or 0)
        rows.append(
            (
                record.id, record.name, record.description, record.industry,
                record.location, record.founded_year, record.employee_count,
                record.revenue_range, record.revenue_min, record.revenue_max,
                json.dumps(record.topics), " ".join(record.topics), now,
            )
        )
    conn.executemany(
        "INSERT OR REPLACE INTO companies (id, name, description, industry, location,"
        " founded_year, employee_count, revenue_range, revenue_min, revenue_max,"
        " topics, topics_text, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        rows,
    )
    conn.execute("INSERT OR REPLACE INTO index_meta (key, value) VALUES ('corpus_loaded_at', ?)", (now,))
    conn.commit()
    conn.close()
    return path


@pytest.fixture
def client(api_corpus_db):
    """A TestClient with the real application lifespan.

    Function-scoped so each test gets a freshly started app: ingestion tests
    mutate the corpus, and leaking that between tests would make failures
    order-dependent.
    """
    import warnings

    from fastapi.testclient import TestClient

    from app.core.metrics import metrics
    from app.main import app

    metrics.reset()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with TestClient(app) as test_client:
            yield test_client


class StubEmbedder:
    """Deterministic stand-in for the ONNX embedder.

    Query vectors are derived by hashing the query string, so they are stable
    across runs but bear no relation to meaning. That is exactly what is wanted
    for testing fusion mechanics: it guarantees the semantic retriever returns a
    plausible-looking ranking that disagrees with the lexical one, which is the
    situation the tie-breaking logic exists to resolve. Semantic *quality* is
    not what these tests are for.
    """

    model_name = "stub"
    dim = 16

    def __init__(self, dim: int = 16) -> None:
        self.dim = dim

    def _vector(self, text: str) -> np.ndarray:
        import hashlib

        digest = hashlib.sha256(text.encode()).digest()
        seed = int.from_bytes(digest[:8], "big")
        vector = np.random.default_rng(seed).normal(size=self.dim).astype("float32")
        return vector / np.linalg.norm(vector)

    async def embed_query(self, text: str) -> np.ndarray:
        return self._vector(text)


@pytest.fixture
async def hybrid_runtime(db: Database, columns: ColumnStore) -> SearchRuntime:
    """Runtime with both retrievers active, using synthetic embeddings.

    The lexical-only `runtime` fixture cannot exercise fusion at all: with one
    retriever there are no cross-retriever ties, so it silently fails to cover
    the ranking logic. This fixture supplies the second ranked list.
    """
    from app.search.vector import RowMap, VectorIndex

    stub = StubEmbedder()
    vectors = np.vstack(
        [stub._vector(f"doc-{int(company_id)}") for company_id in columns.ids]
    ).astype("float32")

    vector_index = VectorIndex(vectors, columns.ids.copy(), "stub")
    return SearchRuntime(
        db=db,
        keyword=KeywordIndex(db),
        columns=columns,
        vector=vector_index,
        embedder=stub,
        row_map=RowMap(columns.ids, vector_index.ids),
        doc_count=columns.size,
    )


@pytest.fixture
async def hybrid_engine(hybrid_runtime: SearchRuntime):
    from app.search.engine import SearchEngine

    return SearchEngine(hybrid_runtime, hybrid_runtime.columns)
