"""Post-ranking constraints applied *within one candidate pool* (one request or one impression).

Scores are only compared inside the pool they were produced for. Steps (deterministic):
1. order by ranking score desc, ties by news_id;
2. greedily skip near-duplicates (cosine >= ``dedup_threshold`` to an already selected item);
3. enforce ``max_per_category``;
4. stop at ``limit``. Fewer than ``limit`` items are returned only if constraints leave fewer candidates.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PostRankConfig:
    dedup_threshold: float | None = 0.9
    max_per_category: int | None = None


def within_pool_minmax(scores: np.ndarray) -> np.ndarray:
    """Min-max normalize scores of ONE pool (constant pools map to 1.0)."""
    s = np.asarray(scores, dtype=float)
    if s.size == 0:
        return s
    lo, hi = s.min(), s.max()
    return np.ones_like(s) if hi - lo < 1e-12 else (s - lo) / (hi - lo)


def select_with_constraints(news_ids: list[str], scores: np.ndarray, categories: list[str],
                            vectors: np.ndarray, limit: int, cfg: PostRankConfig) -> list[int]:
    """Return selected positions (into the input lists) in final rank order."""
    if not (len(news_ids) == len(scores) == len(categories) == len(vectors)):
        raise ValueError("inputs must align")
    order = sorted(range(len(news_ids)), key=lambda i: (-float(scores[i]), news_ids[i]))
    v = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    v = np.where(norms > 0, v / np.maximum(norms, 1e-12), 0.0)
    selected: list[int] = []
    seen_ids: set[str] = set()
    per_cat: dict[str, int] = {}
    for i in order:
        if len(selected) >= limit:
            break
        if news_ids[i] in seen_ids:
            continue
        if cfg.max_per_category is not None and per_cat.get(categories[i], 0) >= cfg.max_per_category:
            continue
        if cfg.dedup_threshold is not None and selected and norms[i, 0] > 0:
            if float((v[selected] @ v[i]).max()) >= cfg.dedup_threshold:
                continue
        selected.append(i)
        seen_ids.add(news_ids[i])
        per_cat[categories[i]] = per_cat.get(categories[i], 0) + 1
    return selected
