# Self-Assessment Report

Generated 2026-08-14 08:08:58 UTC · transport: **in-process ASGI (no network)**

## Environment

| | |
|---|---|
| transport | in-process ASGI (no network) |
| documents | 50000 |
| semantic search | True |
| embedding model | BAAI/bge-small-en-v1.5 |
| index load time | 1.35s |
| python | 3.12.10 |

## Per-query results

All queries are the examples given in the assessment brief. Latency is warm (embedding cache primed), which is the steady state for repeated traffic; cold figures are below.

| # | Matches | p50 ms | p95 ms | Relaxed | Filter violations | Top result |
|---:|---:|---:|---:|:---:|---:|---|
| 1 | 305 | 12.15 | 13.34 | — | 0 | Nordic Fintech Solutions |
| 2 | 449 | 13.48 | 35.9 | — | 0 | Rhine Health Systems |
| 3 | 219 | 10.42 | 10.73 | — | 0 | Lambert Group |
| 4 | 414 | 14.17 | 15.05 | — | 0 | Roberts Ltd |
| 5 | 372 | 13.36 | 13.82 | — | 0 | CloudForge Systems |
| 6 | 73 | 8.6 | 9.41 | — | 0 | May Ltd |
| 7 | 312 | 13.47 | 13.85 | yes (employees) | 0 | Gallegos, Adams and Baker |
| 8 | 433 | 14.74 | 15.63 | — | 0 | RetailMind Europe |
| 9 | 313 | 12.03 | 13.21 | — | 0 | Warren-Ingram |
| 10 | 337 | 12.76 | 13.85 | — | 0 | French Ltd |
| 11 | 50 | 8.64 | 9.0 | — | 0 | Michael-Flores |
| 12 | 10 | 8.91 | 9.67 | — | 0 | Foster-Duncan |
| 13 | 58 | 5.7 | 7.31 | — | 0 | Nordic Fintech Solutions |
| 14 | 53 | 9.29 | 10.46 | yes (employees) | 0 | Helix BioCompute |
| 15 | 3 | 5.2 | 7.02 | — | 0 | Cook Inc |
| 16 | 39 | 5.72 | 7.86 | — | 0 | Baker and Sons |
| 17 | 103 | 9.94 | 10.81 | — | 0 | Taylor Group |

## Latency

| Case | p50 | p95 | p99 |
|---|---:|---:|---:|
| Warm (cache hit) | 10.46 | 14.2 | 15.63 |
| Cold (unseen query) | 46.27 | 49.82 | 50.72 |
| `/similar` | 8.05 | 9.67 | 10.09 |

### Where the time goes (warm, median across queries)

| Stage | ms |
|---|---:|
| filter | 0.379 |
| hydrate | 1.023 |
| parse | 0.206 |
| rank | 0.16 |
| retrieve | 5.711 |

## Retriever ablation

Same queries, one retriever at a time. Isolates what each contributes to cost.

| Mode | p50 | p95 | p99 |
|---|---:|---:|---:|
| keyword | 9.96 | 13.85 | 14.41 |
| semantic | 6.37 | 8.94 | 9.28 |
| hybrid | 10.61 | 14.54 | 14.7 |

## Throughput

| Concurrency | Requests | RPS | p50 | p95 | p99 | Errors |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 200 | 92.8 | 10.96 | 14.45 | 15.5 | 0 |
| 4 | 200 | 96.9 | 35.96 | 69.11 | 89.76 | 0 |
| 8 | 200 | 120.0 | 63.85 | 97.38 | 112.63 | 0 |
| 16 | 200 | 126.4 | 122.3 | 181.26 | 196.87 | 0 |

## Correctness

- Queries run: **17**
- Queries returning results: **17/17**
- Filter violations across all non-relaxed results: **0**
- Queries requiring relaxation: **2** (Q7, Q14)
