"""Latency, throughput and correctness harness.

    python -m scripts.benchmark                          # in-process
    python -m scripts.benchmark --base-url https://…     # a deployed instance
    python -m scripts.benchmark --output reports/self_assessment.md

Two modes, measuring different things:

* **in-process** drives the ASGI app directly, so the numbers are the service's
  own cost with no network in the way. This is what the design decisions were
  tuned against.
* **--base-url** goes over HTTP to a real deployment, so the numbers include TLS,
  the platform's proxy and the round trip. Always slower; the honest figure to
  quote for "what a user experiences".

The distinction matters when reading the report. A 30ms in-process p50 and a
90ms deployed p50 are not in conflict — the difference is the internet.

Correctness is checked alongside latency, because a fast search that ignores its
filters is not a faster search. Every non-relaxed result is verified against the
constraints the parser extracted.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Set before importing anything under `app`, which builds its settings singleton
# at import time. A benchmark exists to measure the search path; leaving the
# limiter on means it measures the limiter instead — the load stage trips 429
# within seconds and the run dies. Only affects the in-process transport; a
# --base-url run is governed by whatever the deployment is configured with.
os.environ.setdefault("RATE_LIMIT_ENABLED", "false")

import httpx  # noqa: E402

QUERIES_PATH = Path(__file__).parent / "example_queries.json"


@dataclass
class QueryResult:
    id: int
    text: str
    total: int
    returned: int
    latency_ms: list[float] = field(default_factory=list)
    relaxed: bool = False
    dropped: list[str] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)
    top_name: str = ""
    stage_ms: dict[str, float] = field(default_factory=dict)

    @property
    def p50(self) -> float:
        return _pct(self.latency_ms, 50)

    @property
    def p95(self) -> float:
        return _pct(self.latency_ms, 95)


def _pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round(p / 100 * (len(ordered) - 1))))
    return round(ordered[idx], 2)


def _check(parsed: dict[str, Any], company: dict[str, Any]) -> list[str]:
    """Verify one result against the filters the parser extracted."""
    problems = []
    if parsed["locations"] and company["location"] not in parsed["locations"]:
        problems.append(f"location {company['location']} not in {parsed['locations']}")
    if parsed["industries"] and company["industry"] not in parsed["industries"]:
        problems.append(f"industry {company['industry']} not in {parsed['industries']}")
    if parsed["revenue_buckets"] and company["revenue_range"] not in parsed["revenue_buckets"]:
        problems.append(f"revenue {company['revenue_range']} not in {parsed['revenue_buckets']}")

    for field_name, key in (("founded", "founded_year"), ("employees", "employee_count")):
        bound = parsed.get(field_name)
        value = company.get(key)
        if not bound or value is None:
            continue
        if bound["min"] is not None:
            ok = value >= bound["min"] if bound["min_inclusive"] else value > bound["min"]
            if not ok:
                problems.append(f"{key}={value} violates min {bound['min']}")
        if bound["max"] is not None:
            ok = value <= bound["max"] if bound["max_inclusive"] else value < bound["max"]
            if not ok:
                problems.append(f"{key}={value} violates max {bound['max']}")
    return problems


class Harness:
    def __init__(self, client: httpx.AsyncClient) -> None:
        self.client = client

    async def search(self, query: str, **kwargs: Any) -> tuple[dict[str, Any], float]:
        t0 = time.perf_counter()
        response = await self.client.post("/search", json={"query": query, **kwargs})
        elapsed = (time.perf_counter() - t0) * 1000
        if response.status_code == 429:
            # Only reachable against a remote deployment, where the harness
            # cannot turn the limiter off. Say so plainly instead of surfacing a
            # bare HTTPStatusError that reads like a service fault.
            raise RuntimeError(
                "rate limited by the target deployment — raise RATE_LIMIT_RPM "
                "there, or lower --load-requests/--concurrency here"
            )
        response.raise_for_status()
        return response.json(), elapsed

    async def run_queries(self, queries: list[dict], repeats: int) -> list[QueryResult]:
        results = []
        for case in queries:
            # Warm the embedding cache *before* the instrumented call. Taking
            # stage timings from the first request would attribute the one-off
            # ~25ms encode to every warm number in the report and make the
            # breakdown disagree with the p50 alongside it.
            await self.search(case["text"], limit=10)
            body, _ = await self.search(case["text"], limit=10, explain=True)
            result = QueryResult(
                id=case["id"],
                text=case["text"],
                total=body["total"],
                returned=len(body["results"]),
                relaxed=body["relaxation"]["applied"],
                dropped=body["relaxation"]["dropped"],
                top_name=body["results"][0]["name"] if body["results"] else "—",
                stage_ms=body.get("timings_ms", {}),
            )
            if not result.relaxed:
                for company in body["results"]:
                    result.violations.extend(_check(body["parsed"], company))

            # Cache is already primed above, so these are warm-path samples.
            # Cold is measured separately, against unseen query strings.
            for _ in range(repeats):
                _, elapsed = await self.search(case["text"], limit=10)
                result.latency_ms.append(elapsed)
            results.append(result)
        return results

    async def cold_latency(self, queries: list[dict]) -> list[float]:
        """One measurement per unique, never-before-seen query string.

        Suffixing each query defeats the embedding cache, which is what makes
        this the worst case: every request pays the full ~25ms encode.
        """
        out = []
        for i, case in enumerate(queries):
            _, elapsed = await self.search(f"{case['text']} #{i}-{time.time_ns()}", limit=10)
            out.append(elapsed)
        return out

    async def ablation(self, queries: list[dict]) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for mode in ("keyword", "semantic", "hybrid"):
            samples = []
            for case in queries:
                await self.search(case["text"], limit=10, mode=mode)  # warm
                for _ in range(3):
                    _, elapsed = await self.search(case["text"], limit=10, mode=mode)
                    samples.append(elapsed)
            out[mode] = {"p50": _pct(samples, 50), "p95": _pct(samples, 95),
                         "p99": _pct(samples, 99)}
        return out

    async def throughput(self, queries: list[dict], concurrency: int, total: int) -> dict[str, Any]:
        """Sustained load with a fixed number of in-flight requests."""
        semaphore = asyncio.Semaphore(concurrency)
        latencies: list[float] = []
        errors = 0

        async def one(i: int) -> None:
            nonlocal errors
            async with semaphore:
                case = queries[i % len(queries)]
                try:
                    _, elapsed = await self.search(case["text"], limit=10)
                    latencies.append(elapsed)
                except Exception:
                    errors += 1

        started = time.perf_counter()
        await asyncio.gather(*(one(i) for i in range(total)))
        wall = time.perf_counter() - started

        return {
            "concurrency": concurrency,
            "requests": total,
            "errors": errors,
            "wall_s": round(wall, 2),
            "rps": round(total / wall, 1) if wall else 0.0,
            "p50": _pct(latencies, 50),
            "p95": _pct(latencies, 95),
            "p99": _pct(latencies, 99),
        }

    async def similar_latency(self, ids: list[int], repeats: int = 10) -> dict[str, float]:
        samples = []
        for company_id in ids:
            for _ in range(repeats):
                t0 = time.perf_counter()
                response = await self.client.get(f"/companies/{company_id}/similar?limit=10")
                samples.append((time.perf_counter() - t0) * 1000)
                response.raise_for_status()
        return {"p50": _pct(samples, 50), "p95": _pct(samples, 95), "p99": _pct(samples, 99)}


def render(report: dict[str, Any]) -> str:
    lines: list[str] = []
    add = lines.append

    env = report["environment"]
    add("# Self-Assessment Report")
    add("")
    add(f"Generated {report['generated_at']} · transport: **{env['transport']}**")
    add("")
    add("## Environment")
    add("")
    add("| | |")
    add("|---|---|")
    for key, value in env.items():
        add(f"| {key} | {value} |")
    add("")

    add("## Per-query results")
    add("")
    add("All queries are the examples given in the assessment brief. Latency is "
        "warm (embedding cache primed), which is the steady state for repeated "
        "traffic; cold figures are below.")
    add("")
    add("| # | Matches | p50 ms | p95 ms | Relaxed | Filter violations | Top result |")
    add("|---:|---:|---:|---:|:---:|---:|---|")
    for q in report["queries"]:
        relaxed = "yes (" + ", ".join(q["dropped"]) + ")" if q["relaxed"] else "—"
        add(
            f"| {q['id']} | {q['total']:,} | {q['p50']} | {q['p95']} | {relaxed} "
            f"| {len(q['violations'])} | {q['top_name']} |"
        )
    add("")

    totals = report["summary"]
    add("## Latency")
    add("")
    add("| Case | p50 | p95 | p99 |")
    add("|---|---:|---:|---:|")
    add(f"| Warm (cache hit) | {totals['warm_p50']} | {totals['warm_p95']} "
        f"| {totals['warm_p99']} |")
    add(f"| Cold (unseen query) | {totals['cold_p50']} | {totals['cold_p95']} "
        f"| {totals['cold_p99']} |")
    s = report["similar"]
    add(f"| `/similar` | {s['p50']} | {s['p95']} | {s['p99']} |")
    add("")

    if report.get("stages"):
        add("### Where the time goes (warm, median across queries)")
        add("")
        add("| Stage | ms |")
        add("|---|---:|")
        for stage, value in report["stages"].items():
            add(f"| {stage} | {value} |")
        add("")

    add("## Retriever ablation")
    add("")
    add("Same queries, one retriever at a time. Isolates what each contributes "
        "to cost.")
    add("")
    add("| Mode | p50 | p95 | p99 |")
    add("|---|---:|---:|---:|")
    for mode, stats in report["ablation"].items():
        add(f"| {mode} | {stats['p50']} | {stats['p95']} | {stats['p99']} |")
    add("")

    add("## Throughput")
    add("")
    add("| Concurrency | Requests | RPS | p50 | p95 | p99 | Errors |")
    add("|---:|---:|---:|---:|---:|---:|---:|")
    for row in report["throughput"]:
        add(f"| {row['concurrency']} | {row['requests']} | {row['rps']} | {row['p50']} "
            f"| {row['p95']} | {row['p99']} | {row['errors']} |")
    add("")

    add("## Correctness")
    add("")
    c = report["correctness"]
    add(f"- Queries run: **{c['queries']}**")
    add(f"- Queries returning results: **{c['non_empty']}/{c['queries']}**")
    add(f"- Filter violations across all non-relaxed results: **{c['violations']}**")
    add(f"- Queries requiring relaxation: **{c['relaxed']}** "
        f"({', '.join('Q' + str(i) for i in c['relaxed_ids']) or 'none'})")
    add("")
    return "\n".join(lines)


def _write_report(path: Path, markdown: str, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown, encoding="utf-8")
    path.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")


async def main_async(args: argparse.Namespace) -> int:
    queries = json.loads(QUERIES_PATH.read_text(encoding="utf-8"))["queries"]

    if args.base_url:
        transport_label = f"HTTP → {args.base_url}"
        client_kwargs: dict[str, Any] = {"base_url": args.base_url, "timeout": 60.0}
    else:
        transport_label = "in-process ASGI (no network)"
        from app.main import app

        client_kwargs = {
            "transport": httpx.ASGITransport(app=app),
            "base_url": "http://testserver",
            "timeout": 60.0,
        }

    started_at = time.perf_counter()
    local_runtime = False
    async with httpx.AsyncClient(**client_kwargs) as client:
        if not args.base_url:
            # httpx's ASGI transport does not run the lifespan, so the index is
            # loaded explicitly here — and torn down at the end. Skipping the
            # teardown leaves aiosqlite's non-daemon connection threads running,
            # and the process hangs at exit after all the work is finished.
            from app.search import bootstrap

            await bootstrap.startup()
            local_runtime = True
        startup_s = time.perf_counter() - started_at

        health = (await client.get("/health")).json()
        index = health.get("index", {})

        harness = Harness(client)

        print("running per-query pass…", file=sys.stderr)
        results = await harness.run_queries(queries, repeats=args.repeats)

        print("measuring cold latency…", file=sys.stderr)
        cold = await harness.cold_latency(queries)

        print("running ablation…", file=sys.stderr)
        ablation = await harness.ablation(queries)

        print("measuring /similar…", file=sys.stderr)
        similar = await harness.similar_latency([1, 2, 3, 4, 5])

        print("running throughput…", file=sys.stderr)
        throughput = []
        for concurrency in args.concurrency:
            throughput.append(
                await harness.throughput(queries, concurrency, args.load_requests)
            )

    if local_runtime:
        from app.search import bootstrap

        await bootstrap.shutdown()

    warm = [ms for r in results for ms in r.latency_ms]
    stage_totals: dict[str, list[float]] = {}
    for r in results:
        for stage, value in r.stage_ms.items():
            stage_totals.setdefault(stage, []).append(value)

    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "environment": {
            "transport": transport_label,
            "documents": index.get("documents", "?"),
            "semantic search": index.get("semantic_search", "?"),
            "embedding model": index.get("embedding_model") or "disabled",
            "index load time": f"{startup_s:.2f}s" if not args.base_url else "n/a (remote)",
            "python": sys.version.split()[0],
        },
        "queries": [
            {
                "id": r.id, "text": r.text, "total": r.total, "returned": r.returned,
                "p50": r.p50, "p95": r.p95, "relaxed": r.relaxed, "dropped": r.dropped,
                "violations": r.violations, "top_name": r.top_name,
            }
            for r in results
        ],
        "summary": {
            "warm_p50": _pct(warm, 50), "warm_p95": _pct(warm, 95), "warm_p99": _pct(warm, 99),
            "cold_p50": _pct(cold, 50), "cold_p95": _pct(cold, 95), "cold_p99": _pct(cold, 99),
        },
        "stages": {
            stage: round(statistics.median(values), 3)
            for stage, values in sorted(stage_totals.items())
        },
        "ablation": ablation,
        "similar": similar,
        "throughput": throughput,
        "correctness": {
            "queries": len(results),
            "non_empty": sum(1 for r in results if r.returned > 0),
            "violations": sum(len(r.violations) for r in results),
            "relaxed": sum(1 for r in results if r.relaxed),
            "relaxed_ids": [r.id for r in results if r.relaxed],
        },
    }

    markdown = render(report)
    if args.output:
        # Off the event loop: this is a one-shot write at the end of the run,
        # but blocking file I/O in a coroutine is the habit worth not forming.
        await asyncio.to_thread(_write_report, Path(args.output), markdown, report)
        print(f"wrote {args.output}", file=sys.stderr)
    else:
        print(markdown)

    violations = report["correctness"]["violations"]
    if violations:
        print(f"FAIL: {violations} filter violations", file=sys.stderr)
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark the search service.")
    parser.add_argument("--base-url", help="Benchmark a deployed instance over HTTP")
    parser.add_argument("--repeats", type=int, default=10, help="Warm samples per query")
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 8, 16])
    parser.add_argument("--load-requests", type=int, default=200)
    parser.add_argument("--output", help="Write a Markdown report here (plus .json)")
    args = parser.parse_args()

    from app.core.logging import configure_logging

    configure_logging()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
