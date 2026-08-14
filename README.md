# Company Search

Natural-language search over ~50,000 companies. Ask in plain English; get ranked
companies back, with the filters the system extracted from your sentence shown
alongside so you can see how it was understood.

```
POST /search  {"query": "Find fintech companies in Finland founded after 2015 with revenue between 10M and 100M."}
```

```
parsed   → location: Finland · industry: Fintech · founded > 2015 · revenue 10M-50M, 50M-100M
total    → 58 companies
took     → 46 ms cold · 10 ms warm
```

Built for the Comparables.ai backend assessment. Python 3.12, FastAPI, SQLite
FTS5 for lexical retrieval, ONNX sentence embeddings for semantic retrieval,
fused with reciprocal rank fusion.

**The reasoning behind every decision is in [`docs/DESIGN.md`](docs/DESIGN.md)
([PDF](docs/DESIGN.pdf)).** Measured results are in
[`reports/self_assessment.md`](reports/self_assessment.md).

---

## Quick start

### Docker (closest to how it deploys)

```bash
docker compose up --build          # first build ~8 min: 6 of those embed 50k docs
open http://localhost:8000         # UI
open http://localhost:8000/docs    # OpenAPI
```

The image ships the model **and** a prebuilt index, so containers start in ~1.5s
and need no network at runtime.

### Local

```bash
python -m venv .venv && . .venv/Scripts/activate   # Windows
# python -m venv .venv && source .venv/bin/activate  # macOS/Linux
pip install -r requirements-dev.txt

python -m scripts.build_index      # ~7 min: 11s ingest + ~6 min embedding
uvicorn app.main:app --reload
```

To skip the embedding step and run lexical + filters only (about 15 seconds):

```bash
python -m scripts.build_index --skip-vectors
EMBEDDINGS_ENABLED=false uvicorn app.main:app
```

> **If the model download aborts with `Illegal instruction` (SIGILL)**, set
> `HF_HUB_DISABLE_XET=1`. Hugging Face's Xet download backend is compiled for
> instruction sets that older and shared cloud CPUs lack. The Docker image sets
> this already.

---

## API

Interactive docs at `/docs`. Full schema at `/openapi.json`.

### `POST /search`

```bash
curl -s localhost:8000/search -H 'content-type: application/json' -d '{
  "query": "Show German healthcare companies focused on diagnostics and patient monitoring.",
  "limit": 10,
  "explain": true
}'
```

| Field | Default | Notes |
|---|---|---|
| `query` | required | Natural language, 1–1000 chars |
| `limit` / `offset` | 10 / 0 | |
| `mode` | `hybrid` | `keyword` and `semantic` isolate one retriever, for measuring each |
| `explain` | `false` | Adds per-result score breakdowns and stage timings |
| `filters` | — | Explicit filters; override anything parsed from the text |
| `allow_relaxation` | `true` | `false` gives exact-only semantics |

The response carries three things worth knowing about:

- **`parsed`** — how the query was interpreted. Without it, a user cannot tell
  whether "founded after 2015" was applied or quietly ignored.
- **`relaxation`** — populated when the query was unsatisfiable as written and
  constraints had to be dropped, naming which ones.
- **`matched_on`** / **`score_breakdown`** — why each result ranked where it did.

`GET /search?q=…` is available as a convenience form so a search can be shared
as a URL.

### `GET /companies/{id}/similar`

Reuses the seed company's stored vector, so it makes no embedding call — the
fastest path in the API. Similarity is **topical, not geographic**: location is
deliberately excluded from the embedding, so a Finnish fintech's neighbours are
fintechs anywhere. `?same_location=true` constrains it when that is wanted.

### Ingestion

```bash
curl -X POST localhost:8000/companies -H 'content-type: application/json' -d '{
  "company_name": "Helsinki Fraud AI",
  "summary": "Real-time platform for fraud detection and payments.",
  "sector": "fintech", "country": "finland",
  "founded": 2021, "employees": 30, "revenue": 25000000
}'
```

Accepts a single object or a list. Field-name variants (`company_name`,
`summary`, `sector`, `country`, `founded`, `employees`) and numeric revenue are
accepted and normalised — `25000000` becomes the `10M-50M` bucket, `finland`
becomes `Finland`, and topics are extracted from the description.

The company is **searchable by keyword and filters immediately**. Semantic
search covers it after the next full index build; the response says so in
`semantic_indexed` rather than leaving it to be discovered.

`POST /companies/validate` dry-runs a payload and shows exactly what would be
stored, without writing.

Other endpoints: `GET /companies/{id}`, `DELETE /companies/{id}`,
`POST /admin/reindex`, `GET /health`, `/health/live`, `/health/ready`,
`/metrics`.

---

## How it works

```
parse (rules, 0.2ms) → filter (numpy, 0.4ms) → relax if needed
                     → retrieve: BM25 ‖ vector  → fuse (RRF) → hydrate
```

Four decisions shape the whole system. Each is argued in full in
[`docs/DESIGN.md`](docs/DESIGN.md):

**Topics are a structured facet, not free text.** Profiling the corpus showed
49,999 of 50,000 descriptions are generated from `"{modifier} {noun} for
{topic}."` over just **27 distinct topics** and an 80-word vocabulary. The
modifier/noun head appears uniformly across every industry and carries no
signal — so "AI" and "infrastructure", both present in the brief's own example
queries, match thousands of unrelated companies. Topics are extracted once at
ingest, turning the dominant part of relevance into exact set intersection.

**Rules parse the query, not an LLM.** An LLM would cover more phrasings and
cost 300–800ms against a 200ms budget, plus a network dependency and
non-determinism. The queries draw on a closed vocabulary; rules cover it in
0.15ms and every decision is inspectable. Unparsed phrasings fall through to
retrieval rather than failing.

**Vector search is exact, not approximate.** The query embedding costs ~25ms;
an exact scan over 50,000×384 costs 2.9ms. ANN would optimise the 2.9ms and
leave the 25ms alone — ~1% end-to-end for a C++ dependency, a build step and
approximate recall. It also composes badly with filtering, which most queries
here use.

**Unsatisfiable queries relax, visibly.** Two of the brief's example queries
return zero rows under strict filtering. Rather than serve an empty page, the
least central constraints are dropped in order (never location or topic), the
response states what was surrendered, exact matches lead, and near-misses
outrank far ones.

---

## Performance

Measured in-process on a 6-core i5-9400F; see
[`reports/self_assessment.md`](reports/self_assessment.md) for the full run and
[`docs/DESIGN.md`](docs/DESIGN.md) for the stage-by-stage budget.

| | p50 | p95 | p99 |
|---|---:|---:|---:|
| Search, warm (embedding cached) | **10.5 ms** | 14.2 ms | 15.6 ms |
| Search, cold (unseen query) | **46.3 ms** | 49.8 ms | 50.7 ms |
| `/similar` (no embedding call) | **8.1 ms** | 9.7 ms | 10.1 ms |

| | |
|---|---:|
| Throughput, 16 concurrent | **127 rps**, p99 192 ms, 0 errors |
| Index load at startup | **1.3 s** |
| Full index build | ~6.5 min (build time, not runtime) |
| Memory | ~77 MB vectors + ~1 MB columns + model |

Latency is wall-clock at the client through the full ASGI stack, so it includes
validation and serialisation, not just search. All 17 example queries return
results with **zero filter violations**; two require relaxation.

Reproduce:

```bash
python -m scripts.benchmark                              # in-process
python -m scripts.benchmark --base-url https://your-app  # a deployment
```

The harness verifies correctness alongside latency: every non-relaxed result is
checked against the filters the parser extracted, and the run fails on any
violation. A fast search that ignores its filters is not a faster search.

---

## Testing

```bash
pytest -q                      # 202 tests, ~10s, no network, no model download
ruff check app/ tests/ scripts/
```

The suite runs offline with embeddings disabled; the vector layer is tested with
synthetic vectors, since its mechanics do not depend on what produced the
numbers. Every bug found during development has a regression test — the
constraint that borrowed its neighbour's field keyword, the aliases that
mislabelled ~13k companies, the sentinel that wrapped under negative indexing,
the ingest that disabled semantic search, and the ranking tie that lost exact
name matches.

The suite is mutation-tested rather than assumed correct: re-introducing the
alias bug fails four tests, including one that reports 12,140 topic/industry
contradictions across the real corpus.

---

## Deployment

See [`deploy/README.md`](deploy/README.md) for Fly.io, Render and AWS App
Runner, including which of them can actually persist ingested data (only Fly, on
a free plan).

```bash
fly launch --no-deploy --copy-config --config deploy/fly.toml
fly volumes create index_data --size 1 --region ams
fly secrets set API_KEYS="$(openssl rand -hex 24)"
fly deploy --config deploy/fly.toml
```

Configuration is entirely environment-driven; every value has a working default.
See [`.env.example`](.env.example).

---

## Layout

```
app/
  taxonomy.py        27 topics, aliases, revenue buckets — the domain vocabulary
  nlq/               query parsing: phrase matcher, numeric constraints
  search/            engine (fusion, relaxation), columns, vector, embeddings
  store/             SQLite schema, async access, bulk ingest, enrichment
  api/               routes: search, companies, ingestion, health, UI
  core/              logging, errors, metrics, security, middleware
scripts/             build_index · benchmark · make_pdf · example_queries.json
tests/               202 tests
docs/DESIGN.md       architecture and reasoning  (+ .pdf)
reports/             generated benchmark output
deploy/              fly.toml · render.yaml · aws-apprunner.yaml
ui/index.html        single-page UI, no build step
```

---

## Known limitations

Stated up front rather than buried:

- **No labelled relevance set.** Ranking quality is argued from spot checks and
  a constraint-satisfaction check, not measured against judgements. This is the
  largest gap and the first thing I would close.
- **Revenue is bucketed in the source data**, so "revenue over 200M" matches the
  whole 100M–500M bucket. Interval overlap is used rather than containment — a
  recall choice; requiring the whole bucket to satisfy the predicate would
  silently drop correct matches.
- **Rate limiting is per-process**, so N replicas means N× the limit. It exists
  to stop one client saturating a shared vCPU, not to enforce a quota.
- **Ingestion rebuilds the whole column store** (~0.5s at 50k). Fine for
  occasional writes; needs to become incremental at millions of rows.
- **The hand-written taxonomy does not generalise as-is.** 27 topics work
  because the corpus has 27. On a real corpus it would be derived by clustering
  and LLM labelling offline — the search path is unchanged, only the source of
  the vocabulary.
