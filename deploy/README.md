# Deployment

The image is self-contained: it ships the embedding model **and** a fully built
search index, so a container starts in about 1.5 seconds and needs no network
access to Hugging Face at runtime. Nothing is computed at boot.

```
docker build -t company-search .      # ~8 min: ~6 of those embed 50k documents
docker run -p 8000:8000 company-search
curl localhost:8000/health/ready
```

## Choosing a platform

|                        | Fly.io                | Render               | AWS App Runner        |
| ---------------------- | --------------------- | -------------------- | --------------------- |
| Free tier              | ongoing               | ongoing              | trial credits only    |
| Persistent volume      | yes, on free          | paid plans only      | no (ephemeral)        |
| Idle behaviour         | suspend, ~300ms wake  | spin down, ~30s wake | always on (billed)    |
| Ingested data persists | yes                   | paid plans only      | no                    |
| Config                 | `fly.toml`            | `render.yaml`        | `aws-apprunner.yaml`  |

**Fly.io is the recommended target.** It is the only one of the three offering a
persistent volume on a free plan, which is what makes the ingestion endpoint
meaningful — companies added through the API survive restarts instead of
vanishing with the container. Its suspend/resume is also fast enough not to
distort the latency story: a resumed machine answers in a few hundred
milliseconds, where a spun-down Render free service takes tens of seconds and a
suspended free-tier Postgres would take seconds on every cold query.

Search itself works identically on all three. The difference is only whether
*written* data survives.

## Fly.io

```bash
fly auth login
fly launch --no-deploy --copy-config --config deploy/fly.toml
fly volumes create index_data --size 1 --region ams
fly secrets set API_KEYS="$(openssl rand -hex 24)"   # required for writes
fly deploy --config deploy/fly.toml
fly open /docs
```

`fly logs` streams the structured JSON logs. `fly scale count 2` adds a machine;
see the note on workers below before raising `WEB_CONCURRENCY` instead.

## Render

Push to GitHub, then **New → Blueprint** pointed at the repo. `render.yaml`
generates an `API_KEYS` value automatically.

The free plan has no disk, so `/data` is ephemeral and re-seeded from the image
on each start. Search is unaffected; ingested companies are not retained. The
disk block in `render.yaml` is commented out and needs a paid plan.

## AWS App Runner

Push the image to ECR and create the service (commands are in
`aws-apprunner.yaml`). Use at least 1 vCPU — the embedding step is CPU-bound and
App Runner's 0.25 vCPU option pushes a ~25ms embed past 100ms, spending half the
latency budget before any search happens.

App Runner has no persistent storage. Retaining ingested data on AWS means ECS
with EFS, or moving the source of truth to RDS — the migration path in
`docs/DESIGN.md`.

## Configuration

Every value has a working default; see `.env.example` for the full list.

| Variable            | Default    | Notes                                              |
| ------------------- | ---------- | -------------------------------------------------- |
| `DATA_DIR`          | `/data`    | Must be the volume mount. Holds SQLite + vectors.   |
| `API_KEYS`          | *(empty)*  | Comma-separated. Empty disables auth entirely.      |
| `REQUIRE_AUTH_FOR_READS` | `false` | Reads are public by default so the demo is usable. |
| `WEB_CONCURRENCY`   | `1`        | See below.                                          |
| `RATE_LIMIT_RPM`    | `120`      | Per API key, or per forwarded IP.                   |
| `EMBEDDINGS_ENABLED`| `true`     | `false` runs lexical + filters only.                |

### Why one worker

Each worker process holds its own copy of the 77MB vector matrix, the column
arrays and the ONNX session — they are not shared. Two workers on a 512MB
instance is a straightforward OOM. The event loop already overlaps requests
(embedding runs on a thread pool, SQLite reads on a connection pool), so the
throughput gain would be small even where the memory exists.

Scale with replicas. Raise `WEB_CONCURRENCY` only on instances with RAM to back
it — roughly 400MB per additional worker.

### Rate limiting across replicas

The limiter is per process. Two replicas means twice the effective limit, and
buckets reset on restart. That is deliberate for a single small container: it
exists to stop one client saturating a shared vCPU, not to enforce a quota. A
multi-replica deployment should move it to Redis.

## Verifying a deployment

```bash
BASE=https://your-app.fly.dev

curl -s $BASE/health/ready | jq        # documents, semantic_search, coverage
curl -s -X POST $BASE/search -H 'content-type: application/json' \
  -d '{"query":"Find fintech companies in Finland working on fraud detection","limit":3,"explain":true}' | jq
curl -s "$BASE/companies/1/similar?limit=5" | jq '.results[].name'
curl -s $BASE/metrics | jq '.latency_ms'

# End-to-end latency against the deployed instance
python -m scripts.benchmark --base-url $BASE
```

A healthy `/health/ready` reports `semantic_search: true` and
`semantic_coverage: 1.0`. Coverage below 1.0 means companies have been ingested
since the last full index build — they are searchable by keyword and filters but
not yet by vector. Rebuild with `fly ssh console -C "docker-entrypoint.sh
build-index"`, or redeploy.
