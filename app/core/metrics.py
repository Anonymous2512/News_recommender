"""Prometheus metrics (one registry per app instance, so tests can build many apps)."""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest


class Metrics:
    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.requests = Counter("newsrec_http_requests_total", "HTTP requests", ["route", "method", "status"],
                                registry=self.registry)
        self.latency = Histogram("newsrec_http_request_seconds", "HTTP request latency", ["route"],
                                 buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5),
                                 registry=self.registry)
        self.cache = Counter("newsrec_cache_events_total", "Cache events", ["cache", "event"],
                             registry=self.registry)
        self.cache_latency = Histogram("newsrec_cache_op_seconds", "Cache operation latency", ["op"],
                                       buckets=(0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.1, 0.25),
                                       registry=self.registry)
        self.candidates = Histogram("newsrec_candidates_retrieved", "Candidates retrieved per request",
                                    buckets=(0, 10, 25, 50, 100, 200, 400), registry=self.registry)

    def render(self) -> bytes:
        return generate_latest(self.registry)

    def cache_count(self, cache: str, event: str) -> float:
        return self.cache.labels(cache=cache, event=event)._value.get()
