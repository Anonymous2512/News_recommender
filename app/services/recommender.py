"""Recommendation orchestration.

Flow per request: validate -> load durable history -> response cache -> profile (cached) -> retrieve
candidates -> point-in-time-safe features -> LambdaRank scores -> within-request constraints -> cache.
No training, dataset scans or encoder loading happen here.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field

from starlette.concurrency import run_in_threadpool

from app.cache.redis_cache import AsyncCache, Keys
from app.core.logging import log_event, user_hash
from app.core.metrics import Metrics
from app.repositories.store import Repository, UserHistory
from app.services.feature_builder import ServingFeatureBuilder
from app.services.profiles import ProfileService
from newsrec.models.artifacts import ModelBundle
from newsrec.postrank import PostRankConfig, select_with_constraints
from newsrec.retrieval.retriever import CandidateRetriever

logger = logging.getLogger(__name__)

SCORE_NOTE = ("ranking_score is a raw LambdaRank score: comparable only within this response, "
              "not a probability or confidence")


class RecommendationError(Exception):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


@dataclass
class RecommendationResult:
    user_id: str
    items: list[dict]
    meta: dict = field(default_factory=dict)


class Recommender:
    def __init__(self, bundle: ModelBundle, repo: Repository, cache: AsyncCache, keys: Keys,
                 retriever: CandidateRetriever, features: ServingFeatureBuilder, profiles: ProfileService,
                 metrics: Metrics, *, catalog_version: str, categories: set[str], rec_ttl_s: int,
                 n_similar: int, n_popular: int, max_history: int, postrank: PostRankConfig):
        self.bundle, self.repo, self.cache, self.keys = bundle, repo, cache, keys
        self.retriever, self.features, self.profiles, self.metrics = retriever, features, profiles, metrics
        self.catalog_version, self.categories, self.rec_ttl = catalog_version, categories, rec_ttl_s
        self.n_similar, self.n_popular, self.max_history, self.postrank = n_similar, n_popular, max_history, postrank

    async def recommend(self, user_id: str, limit: int, category: str | None, exclude_seen: bool,
                        include_ctr: bool, request_id: str) -> RecommendationResult:
        t0 = time.perf_counter()
        if category is not None and category not in self.categories:
            raise RecommendationError(422, "unknown_category", f"unknown category {category!r}")
        history: UserHistory = await run_in_threadpool(self.repo.get_user_history, user_id, self.max_history)
        key = self.keys.recommendation(self.bundle.version, self.catalog_version, user_id, history.version,
                                       limit, category, exclude_seen, include_ctr)
        raw = await self.cache.get(key, "recommendation")
        if raw is not None:
            try:
                cached = json.loads(raw)
                meta = {**cached["meta"], "cache": "hit", "profile_cache": "not_used", "request_id": request_id}
                self._log(user_id, "hit", meta, t0)
                return RecommendationResult(user_id=user_id, items=cached["items"], meta=meta)
            except (ValueError, KeyError):
                pass
        profile, profile_cache = await self.profiles.get(history)
        items, n_candidates = await run_in_threadpool(self._rank, history, profile, limit, category,
                                                      exclude_seen, include_ctr)
        articles = await run_in_threadpool(self.repo.get_articles, [i["news_id"] for i in items])
        for it in items:
            a = articles.get(it["news_id"]) or {}
            it.update(title=a.get("title", ""), category=a.get("category", it["category"]),
                      subcategory=a.get("subcategory", ""))
        meta = {
            "request_id": request_id, "artifact_version": self.bundle.version,
            "catalog_version": self.catalog_version, "data_source": self.bundle.manifest["data_source"],
            "cold_start": profile.is_empty, "history_items_used": profile.n_known,
            "candidates_retrieved": n_candidates, "profile_cache": profile_cache,
            "cache": "miss" if self.cache.name == "redis" else "bypass", "score_note": SCORE_NOTE,
        }
        await self.cache.set(key, json.dumps({"items": items, "meta": meta}).encode(), self.rec_ttl,
                             "recommendation", index_key=self.keys.user_index(user_id))
        self.metrics.candidates.observe(n_candidates)
        self._log(user_id, meta["cache"], meta, t0)
        return RecommendationResult(user_id=user_id, items=items, meta=meta)

    def _rank(self, history: UserHistory, profile, limit: int, category: str | None, exclude_seen: bool,
              include_ctr: bool) -> tuple[list[dict], int]:
        exclude = set(history.news_ids) if exclude_seen else set()
        cands = self.retriever.retrieve(None if profile.is_empty else profile.vector, exclude,
                                        self.n_similar, self.n_popular, category)
        if not cands:
            return [], 0
        ids = [c.news_id for c in cands]
        X = self.features.build(ids, history.news_ids, profile)
        scores = self.bundle.ranker.score(X).values
        probs = self.bundle.ctr_lightgbm.predict_proba(X).values if include_ctr else None
        vecs, _ = self.bundle.embeddings.lookup(ids)
        meta_cats = self.features.builder.meta.categories
        cats = [meta_cats.get(n, ("", ""))[0] for n in ids]
        chosen = select_with_constraints(ids, scores, cats, vecs, limit, self.postrank)
        items = []
        for rank, i in enumerate(chosen, start=1):
            item = {"rank": rank, "news_id": ids[i], "category": cats[i],
                    "ranking_score": round(float(scores[i]), 6), "retrieval_sources": cands[i].sources}
            if probs is not None:
                item["click_probability_uncalibrated"] = round(float(probs[i]), 6)
            items.append(item)
        return items, len(cands)

    def _log(self, user_id: str, cache: str, meta: dict, t0: float) -> None:
        log_event(logger, "recommendation", user=user_hash(user_id), cache=cache,
                  model_version=self.bundle.version, candidates=meta.get("candidates_retrieved"),
                  cold_start=meta.get("cold_start"), duration_ms=round((time.perf_counter() - t0) * 1000, 2))

    async def invalidate_user(self, user_id: str) -> int:
        return await self.cache.invalidate_index(self.keys.user_index(user_id))
