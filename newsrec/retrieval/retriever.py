"""Candidate retrieval: narrows the article catalogue to a candidate set for ranking.

Retrieval is separate from ranking: the retriever returns candidates (no final order); the LambdaRank
model orders them; the post-ranker applies diversity/deduplication.

* ``ExactCosineIndex`` — brute-force inner product over L2-normalized vectors (exact; fine for the
  ~100K-article MIND catalogue). ``FaissFlatIndex`` is used instead when ``faiss`` is installed and
  requested; both implement :class:`VectorIndex`, so an ANN backend can be swapped in later.
* Popularity pool — articles ordered by smoothed past CTR among sufficiently exposed articles; this is
  the cold-start path for users without known history.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import pandas as pd

from newsrec.features.popularity import BetaPrior, PopularitySnapshot, smoothed_ctr


class VectorIndex(Protocol):
    def search(self, query: np.ndarray, k: int, allowed: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Return (row positions, cosine scores) of the top-k allowed rows, best first."""


class ExactCosineIndex:
    def __init__(self, vectors: np.ndarray):
        v = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        self.valid = norms[:, 0] > 0
        self.vectors = np.where(norms > 0, v / np.maximum(norms, 1e-12), 0.0).astype(np.float32)

    def search(self, query: np.ndarray, k: int, allowed: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        q = np.asarray(query, dtype=np.float32)
        qn = float(np.linalg.norm(q))
        if qn == 0 or k <= 0:
            return np.zeros(0, np.int64), np.zeros(0, np.float32)
        scores = self.vectors @ (q / qn)
        mask = self.valid if allowed is None else (self.valid & allowed)
        cand = np.flatnonzero(mask)
        if cand.size == 0:
            return np.zeros(0, np.int64), np.zeros(0, np.float32)
        k = min(k, cand.size)
        s = scores[cand]
        top = np.argpartition(-s, k - 1)[:k]
        top = top[np.lexsort((cand[top], -s[top]))]  # deterministic: score desc, position asc
        return cand[top], s[top]


class FaissFlatIndex:
    """Exact inner-product search via faiss (optional dependency)."""

    def __init__(self, vectors: np.ndarray):
        import faiss  # type: ignore[import-not-found]

        self._exact = ExactCosineIndex(vectors)
        self._index = faiss.IndexFlatIP(self._exact.vectors.shape[1])
        self._index.add(self._exact.vectors)
        self._n = len(vectors)

    def search(self, query: np.ndarray, k: int, allowed: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        if allowed is not None:  # filtered search: fall back to exact masking
            return self._exact.search(query, k, allowed)
        q = np.asarray(query, dtype=np.float32)
        qn = float(np.linalg.norm(q))
        if qn == 0 or k <= 0:
            return np.zeros(0, np.int64), np.zeros(0, np.float32)
        scores, idx = self._index.search((q / qn)[None, :], min(k, self._n))
        keep = (idx[0] >= 0) & self._exact.valid[np.clip(idx[0], 0, None)]
        return idx[0][keep].astype(np.int64), scores[0][keep]


def make_index(vectors: np.ndarray, backend: str = "exact") -> VectorIndex:
    if backend == "faiss":
        return FaissFlatIndex(vectors)
    if backend == "exact":
        return ExactCosineIndex(vectors)
    raise ValueError(f"unknown index backend {backend!r}")


@dataclass
class RetrievedCandidate:
    news_id: str
    sources: list[str] = field(default_factory=list)  # "similar" and/or "popular"
    similarity: float | None = None


class CandidateRetriever:
    def __init__(self, news_ids: np.ndarray, vectors: np.ndarray, categories: Iterable[str],
                 popularity: PopularitySnapshot, prior: BetaPrior, index_backend: str = "exact",
                 min_exposures_for_popular: int = 20, eligible: np.ndarray | None = None):
        self.news_ids = np.asarray(news_ids).astype(str)
        self.categories = np.asarray(list(categories)).astype(str)
        if len(self.categories) != len(self.news_ids):
            raise ValueError("categories must align with news_ids")
        self.index = make_index(vectors, index_backend)
        self.eligible = np.ones(len(self.news_ids), bool) if eligible is None else np.asarray(eligible, bool)
        self._pos = {n: i for i, n in enumerate(self.news_ids.tolist())}
        counts = popularity.lookup(self.news_ids.tolist(), popularity.as_of)
        ctr = smoothed_ctr(counts["past_clicks"], counts["past_impressions"], prior)
        exposed = counts["past_impressions"] >= min_exposures_for_popular
        # popular order: well-exposed articles by smoothed CTR, then the rest by exposures; ties by id
        key_ctr = np.where(exposed, ctr, -1.0)
        order = np.lexsort((self.news_ids, -counts["past_impressions"], -key_ctr))
        self.popular_order = order[self.eligible[order]]

    def retrieve(self, profile_vector: np.ndarray | None, exclude: set[str], n_similar: int = 100,
                 n_popular: int = 50, category: str | None = None) -> list[RetrievedCandidate]:
        allowed = self.eligible.copy()
        if category:
            allowed &= self.categories == category
        if exclude:
            ex = [self._pos[n] for n in exclude if n in self._pos]
            allowed[ex] = False
        out: dict[str, RetrievedCandidate] = {}
        if profile_vector is not None and np.linalg.norm(profile_vector) > 0 and n_similar > 0:
            pos, sims = self.index.search(profile_vector, n_similar, allowed)
            for p, s in zip(pos.tolist(), sims.tolist()):
                nid = self.news_ids[p]
                out[nid] = RetrievedCandidate(nid, ["similar"], float(s))
        taken = 0
        for p in self.popular_order:
            if taken >= n_popular:
                break
            if not allowed[p]:
                continue
            nid = self.news_ids[p]
            if nid in out:
                out[nid].sources.append("popular")
            else:
                out[nid] = RetrievedCandidate(nid, ["popular"])
            taken += 1
        return list(out.values())


def eligible_by_recency(news_ids: np.ndarray, popularity: PopularitySnapshot, max_age_hours: float | None,
                        reference_time: pd.Timestamp) -> np.ndarray:
    """Optionally restrict candidates to articles exposed within ``max_age_hours`` of the reference time
    (last exposure in the log; an exposure-based freshness proxy, not publish time)."""
    if not max_age_hours:
        return np.ones(len(news_ids), bool)
    last = popularity.table.set_index("news_id")["last_seen_time"].reindex(list(news_ids))
    age = (reference_time - last).dt.total_seconds() / 3600.0
    return (age <= max_age_hours).fillna(False).to_numpy()
