"""Reproducible latency benchmark for GET /v1/recommendations.

Cold = first request per user (recommendation cache miss); warm = immediate repeat (cache hit when a
cache is enabled). Reports p50/p95/mean latency, observed cache hit rate and the environment.

Modes:
  --url http://127.0.0.1:8000          measure a running server over HTTP (realistic; uses its Redis)
  (default) in-process ASGI app        --cache fakeredis | disabled | redis (uses NEWSREC_REDIS_URL)

In-process numbers exclude network/server overhead and, with ``fakeredis``, use an in-process Redis
emulation - they are NOT production latencies. The output JSON records exactly what was measured.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def _pct(xs: list[float], q: float) -> float:
    s = sorted(xs)
    return s[min(len(s) - 1, max(0, round(q * (len(s) - 1))))]


def _summary(ms: list[float]) -> dict:
    return {"n": len(ms), "p50_ms": round(_pct(ms, 0.5), 3), "p95_ms": round(_pct(ms, 0.95), 3),
            "mean_ms": round(statistics.fmean(ms), 3), "max_ms": round(max(ms), 3)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=None)
    ap.add_argument("--cache", choices=["fakeredis", "disabled", "redis"], default="fakeredis")
    ap.add_argument("--requests", type=int, default=200, help="number of distinct users (cold) = warm repeats")
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--out", default="benchmarks")
    a = ap.parse_args()

    from newsrec.models.artifacts import resolve_artifact_dir

    art = resolve_artifact_dir(os.environ.get("NEWSREC_ARTIFACTS_DIR", "artifacts"),
                               os.environ.get("NEWSREC_ARTIFACT_VERSION") or None)
    # distinct users so every first request is a true cache miss; pad with never-seen (cold-start) ids
    known = list(dict.fromkeys(pd.read_parquet(art / "user_histories.parquet")["user_id"].tolist()))
    users = known[: a.requests] + [f"bench-cold-{i}" for i in range(max(0, a.requests - len(known)))]

    if a.url:
        import httpx

        client = httpx.Client(base_url=a.url, timeout=10)
        mode = f"http {a.url}"
    else:
        from fastapi.testclient import TestClient

        from app.cache.redis_cache import NullCache, RedisCache
        from app.core.config import Settings
        from app.core.metrics import Metrics
        from app.main import create_app

        settings = Settings(json_logs=False, log_level="WARNING")
        metrics = Metrics()
        if a.cache == "fakeredis":
            import fakeredis

            cache = RedisCache(fakeredis.FakeAsyncRedis(), metrics)
        elif a.cache == "disabled":
            cache = NullCache(metrics)
        else:
            cache = None  # real Redis from NEWSREC_REDIS_URL
        client = TestClient(create_app(settings, cache=cache))
        client.__enter__()
        mode = f"in-process ASGI, cache={a.cache}"

    ready = client.get("/health/ready").json()
    if ready.get("status") not in ("ready", "degraded"):
        raise SystemExit(f"service not ready: {ready}")
    client.get(f"/v1/recommendations/warmup-user?limit={a.limit}")  # first-request import/JIT effects

    def run(us: list[str]) -> tuple[list[float], list[str]]:
        lat, cache_status = [], []
        for u in us:
            t0 = time.perf_counter()
            r = client.get(f"/v1/recommendations/{u}?limit={a.limit}")
            lat.append((time.perf_counter() - t0) * 1000)
            r.raise_for_status()
            cache_status.append(r.json()["meta"]["cache"])
        return lat, cache_status

    cold, cold_cache = run(users)
    warm, warm_cache = run(users)
    all_status = cold_cache + warm_cache
    result = {
        "measured_at": datetime.now(timezone.utc).isoformat(), "mode": mode,
        "artifact": ready["checks"]["model"].get("artifact_version"),
        "data_source": ready["checks"]["model"].get("data_source"),
        "catalog_articles": ready["checks"]["database"].get("articles"),
        "limit": a.limit, "cold": _summary(cold), "warm": _summary(warm),
        "cache_hit_rate_overall": round(all_status.count("hit") / len(all_status), 3),
        "cache_hit_rate_warm": round(warm_cache.count("hit") / len(warm_cache), 3),
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "cpu_count": os.cpu_count(), "processor": platform.processor()},
        "caveat": ("in-process measurement; excludes network and uvicorn overhead"
                   + ("; fakeredis is an in-process emulation, not a network Redis" if "fakeredis" in mode else "")
                   if not a.url else "HTTP measurement against a running server"),
    }
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"benchmark-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{a.cache if not a.url else 'http'}.json"
    path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"saved -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
