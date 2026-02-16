"""Async Redis cache with timeouts, metrics and graceful degradation.

If Redis is unreachable, every operation is counted as an error and treated as a miss/no-op, so the
service keeps serving *uncached but correct* results from the model/catalogue/store. Stale or fabricated
values are never returned. ``NullCache`` is used when no ``REDIS_URL`` is configured.

Key scheme (all prefixed with ``{prefix}:``):
* ``profile:{artifact_version}:{user_id}:{history_version}`` — user profile vector (TTL)
* ``rec:{artifact_version}:{catalog_version}:{user_id}:{history_version}:{params}`` — response (TTL)
* ``userkeys:{user_id}`` — set of the user's profile/rec keys, used for explicit invalidation on click
* ``emb:{model_id}:{news_id}:{text_checksum}`` — article embeddings, written by the offline embed job
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Protocol

from app.core.metrics import Metrics

logger = logging.getLogger(__name__)


class AsyncCache(Protocol):
    name: str

    async def get(self, key: str, kind: str) -> bytes | None: ...

    async def set(self, key: str, value: bytes, ttl_s: int, kind: str, index_key: str | None = None) -> None: ...

    async def invalidate_index(self, index_key: str) -> int: ...

    async def ping(self) -> bool: ...

    async def close(self) -> None: ...


class NullCache:
    name = "disabled"

    def __init__(self, metrics: Metrics | None = None):
        self.metrics = metrics

    async def get(self, key: str, kind: str) -> bytes | None:
        if self.metrics:
            self.metrics.cache.labels(cache=kind, event="bypass").inc()
        return None

    async def set(self, key: str, value: bytes, ttl_s: int, kind: str, index_key: str | None = None) -> None:
        return None

    async def invalidate_index(self, index_key: str) -> int:
        return 0

    async def ping(self) -> bool:
        return False

    async def close(self) -> None:
        return None


class RedisCache:
    name = "redis"

    def __init__(self, client, metrics: Metrics, timeout_s: float = 0.25):
        self.client = client  # redis.asyncio.Redis (one pool per process)
        self.metrics = metrics
        self.timeout = timeout_s
        self.last_error: str | None = None

    @classmethod
    def from_url(cls, url: str, metrics: Metrics, timeout_s: float = 0.25) -> RedisCache:
        import redis.asyncio as aioredis

        client = aioredis.Redis.from_url(url, socket_timeout=timeout_s, socket_connect_timeout=timeout_s,
                                         health_check_interval=30)
        return cls(client, metrics, timeout_s)

    async def _call(self, op: str, kind: str, coro_fn):
        t0 = time.perf_counter()
        try:
            result = await asyncio.wait_for(coro_fn(), timeout=self.timeout * 2)
            self.last_error = None
            return result, True
        except Exception as e:  # redis errors, timeouts, connection refused
            self.last_error = f"{type(e).__name__}: {e}"
            self.metrics.cache.labels(cache=kind, event="error").inc()
            logger.warning("cache %s failed", op, extra={"fields": {"cache_kind": kind, "error": type(e).__name__}})
            return None, False
        finally:
            self.metrics.cache_latency.labels(op=op).observe(time.perf_counter() - t0)

    async def get(self, key: str, kind: str) -> bytes | None:
        value, ok = await self._call("get", kind, lambda: self.client.get(key))
        if ok:
            self.metrics.cache.labels(cache=kind, event="hit" if value is not None else "miss").inc()
        return value

    async def set(self, key: str, value: bytes, ttl_s: int, kind: str, index_key: str | None = None) -> None:
        async def _do():
            pipe = self.client.pipeline(transaction=True)
            pipe.set(key, value, ex=ttl_s)
            if index_key:
                pipe.sadd(index_key, key)
                pipe.expire(index_key, max(ttl_s, 3600))
            return await pipe.execute()

        _, ok = await self._call("set", kind, _do)
        if ok:
            self.metrics.cache.labels(cache=kind, event="write").inc()

    async def invalidate_index(self, index_key: str) -> int:
        async def _do():
            keys = await self.client.smembers(index_key)
            pipe = self.client.pipeline(transaction=True)
            if keys:
                pipe.delete(*keys)
            pipe.delete(index_key)
            await pipe.execute()
            return len(keys)

        n, ok = await self._call("invalidate", "user_index", _do)
        if ok:
            self.metrics.cache.labels(cache="user_index", event="invalidated").inc(n or 0)
        return int(n or 0)

    async def ping(self) -> bool:
        res, ok = await self._call("ping", "health", lambda: self.client.ping())
        return bool(ok and res)

    async def close(self) -> None:
        try:
            await self.client.aclose()
        except Exception:  # pragma: no cover
            pass


class Keys:
    def __init__(self, prefix: str):
        self.p = prefix

    def profile(self, artifact_version: str, user_id: str, history_version: str) -> str:
        return f"{self.p}:profile:{artifact_version}:{user_id}:{history_version}"

    def recommendation(self, artifact_version: str, catalog_version: str, user_id: str, history_version: str,
                       limit: int, category: str | None, exclude_seen: bool, include_ctr: bool) -> str:
        params = f"l{limit}:c{category or '*'}:x{int(exclude_seen)}:p{int(include_ctr)}"
        return f"{self.p}:rec:{artifact_version}:{catalog_version}:{user_id}:{history_version}:{params}"

    def user_index(self, user_id: str) -> str:
        return f"{self.p}:userkeys:{user_id}"
