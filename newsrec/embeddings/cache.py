"""Embedding caches keyed by (model_id, news_id, text checksum).

Used by the offline embedding job so vectors are reused across runs/workers. ``DiskEmbeddingCache`` is a
local npz file per model; ``RedisEmbeddingCache`` shares vectors through Redis (raw float32 bytes).
Vectors are never shared across model ids because the model id is part of every key.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol

import numpy as np

logger = logging.getLogger(__name__)

CacheKey = tuple[str, str, str]  # (model_id, news_id, checksum)


class EmbeddingCache(Protocol):
    def get_many(self, keys: list[CacheKey], dim: int) -> dict[CacheKey, np.ndarray]: ...

    def set_many(self, items: dict[CacheKey, np.ndarray]) -> None: ...


def redis_key(key: CacheKey, prefix: str = "newsrec") -> str:
    model_id, news_id, checksum = key
    return f"{prefix}:emb:{model_id}:{news_id}:{checksum}"


class RedisEmbeddingCache:
    def __init__(self, client, ttl_seconds: int | None = 30 * 24 * 3600, prefix: str = "newsrec",
                 batch: int = 1000):
        self.client = client  # redis.Redis (sync)
        self.ttl = ttl_seconds
        self.prefix = prefix
        self.batch = batch

    def get_many(self, keys: list[CacheKey], dim: int) -> dict[CacheKey, np.ndarray]:
        out: dict[CacheKey, np.ndarray] = {}
        for start in range(0, len(keys), self.batch):
            chunk = keys[start:start + self.batch]
            values = self.client.mget([redis_key(k, self.prefix) for k in chunk])
            for k, raw in zip(chunk, values):
                if raw is None:
                    continue
                vec = np.frombuffer(raw, dtype=np.float32)
                if vec.shape == (dim,) and np.isfinite(vec).all():
                    out[k] = vec.copy()
                else:
                    logger.warning("ignoring malformed cached embedding for %s", k[1])
        return out

    def set_many(self, items: dict[CacheKey, np.ndarray]) -> None:
        pipe = self.client.pipeline(transaction=False)
        for n, (k, v) in enumerate(items.items(), 1):
            pipe.set(redis_key(k, self.prefix), np.asarray(v, dtype=np.float32).tobytes(), ex=self.ttl)
            if n % self.batch == 0:
                pipe.execute()
        pipe.execute()


class DiskEmbeddingCache:
    """One ``<model_id>.npz`` per encoder in ``cache_dir``; call :meth:`flush` to persist new vectors."""

    def __init__(self, cache_dir: str | Path):
        self.dir = Path(cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._data: dict[str, dict[tuple[str, str], np.ndarray]] = {}
        self._dirty: set[str] = set()

    def _path(self, model_id: str) -> Path:
        safe = "".join(c if c.isalnum() or c in "-_.@" else "_" for c in model_id)
        return self.dir / f"{safe}.npz"

    def _load(self, model_id: str) -> dict[tuple[str, str], np.ndarray]:
        if model_id not in self._data:
            table: dict[tuple[str, str], np.ndarray] = {}
            p = self._path(model_id)
            if p.exists():
                with np.load(p, allow_pickle=False) as z:
                    if str(z["model_id"]) != model_id:
                        raise ValueError(f"cache file {p} belongs to model {z['model_id']}")
                    for nid, cs, vec in zip(z["news_ids"], z["checksums"], z["vectors"]):
                        table[(str(nid), str(cs))] = vec
            self._data[model_id] = table
        return self._data[model_id]

    def get_many(self, keys: list[CacheKey], dim: int) -> dict[CacheKey, np.ndarray]:
        out: dict[CacheKey, np.ndarray] = {}
        for k in keys:
            vec = self._load(k[0]).get((k[1], k[2]))
            if vec is not None and vec.shape == (dim,):
                out[k] = vec
        return out

    def set_many(self, items: dict[CacheKey, np.ndarray]) -> None:
        for (model_id, nid, cs), v in items.items():
            self._load(model_id)[(nid, cs)] = np.asarray(v, dtype=np.float32)
            self._dirty.add(model_id)

    def flush(self) -> None:
        for model_id in sorted(self._dirty):
            table = self._data[model_id]
            keys = list(table)
            np.savez(self._path(model_id), model_id=np.array(model_id),
                     news_ids=np.array([k[0] for k in keys], dtype=str),
                     checksums=np.array([k[1] for k in keys], dtype=str),
                     vectors=np.stack([table[k] for k in keys]).astype(np.float32))
        self._dirty.clear()
