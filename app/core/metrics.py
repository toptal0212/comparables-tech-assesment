"""In-process metrics.

Scope is deliberately narrow: enough to answer "is it up, is it fast, what is it
spending time on" from `GET /metrics`, with no external collector required. That
matters for a single free-tier container where running Prometheus alongside the
service is not realistic.

Latency percentiles come from a bounded ring buffer per metric rather than a
true streaming histogram. The tradeoff: percentiles describe a recent window
(the last `_WINDOW` observations) instead of all time, and memory stays flat.
For steering a service that is the more useful statistic anyway.

A multi-replica deployment would swap this for `prometheus_client` and scrape
each pod; the call sites (`timer`, `incr`) would not change.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from contextlib import contextmanager
from typing import Any, Iterator

_WINDOW = 4096


class Metrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, float] = defaultdict(float)
        self._timings: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=_WINDOW))
        self._gauges: dict[str, float] = {}
        self._started = time.time()

    def incr(self, name: str, value: float = 1.0, **labels: Any) -> None:
        with self._lock:
            self._counters[_key(name, labels)] += value

    def gauge(self, name: str, value: float, **labels: Any) -> None:
        with self._lock:
            self._gauges[_key(name, labels)] = value

    def observe_ms(self, name: str, ms: float, **labels: Any) -> None:
        with self._lock:
            self._timings[_key(name, labels)].append(ms)

    @contextmanager
    def timer(self, name: str, **labels: Any) -> Iterator[None]:
        """Time a block and record it in milliseconds.

        Uses perf_counter so it is unaffected by wall-clock adjustments.
        """
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.observe_ms(name, (time.perf_counter() - t0) * 1000.0, **labels)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            counters = dict(self._counters)
            gauges = dict(self._gauges)
            timings = {k: sorted(v) for k, v in self._timings.items() if v}

        return {
            "uptime_s": round(time.time() - self._started, 1),
            "counters": counters,
            "gauges": gauges,
            "latency_ms": {k: _percentiles(v) for k, v in timings.items()},
        }

    def reset(self) -> None:
        """Only used by tests, so one case cannot see another's observations."""
        with self._lock:
            self._counters.clear()
            self._timings.clear()
            self._gauges.clear()


def _key(name: str, labels: dict[str, Any]) -> str:
    if not labels:
        return name
    rendered = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
    return f"{name}{{{rendered}}}"


def _percentiles(sorted_values: list[float]) -> dict[str, float]:
    def pct(p: float) -> float:
        if not sorted_values:
            return 0.0
        # Nearest-rank on an already-sorted list; exact for the window we hold.
        idx = min(len(sorted_values) - 1, int(round(p / 100.0 * (len(sorted_values) - 1))))
        return round(sorted_values[idx], 3)

    return {
        "count": len(sorted_values),
        "p50": pct(50),
        "p90": pct(90),
        "p95": pct(95),
        "p99": pct(99),
        "max": round(sorted_values[-1], 3),
        "mean": round(sum(sorted_values) / len(sorted_values), 3),
    }


metrics = Metrics()
