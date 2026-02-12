"""Dependency container: everything loaded once per process during app lifespan."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd
from starlette.requests import Request

from app.cache.redis_cache import AsyncCache, Keys, NullCache, RedisCache
from app.core.config import Settings
from app.core.metrics import Metrics
from app.repositories.store import Repository
from app.services.feature_builder import ServingFeatureBuilder
from app.services.profiles import ProfileService
from app.services.recommender import RecommendationError, Recommender
from app.services.retrieval import build_retriever
from newsrec.models.artifacts import ArtifactError, ModelBundle, load_bundle, resolve_artifact_dir
from newsrec.postrank import PostRankConfig

logger = logging.getLogger(__name__)


@dataclass
class Container:
    settings: Settings
    metrics: Metrics
    repo: Repository
    cache: AsyncCache
    bundle: ModelBundle | None = None
    recommender: Recommender | None = None
    startup_error: str | None = None

    def require_recommender(self) -> Recommender:
        if self.recommender is None:
            raise RecommendationError(503, "not_ready", self.startup_error or "model not loaded")
        return self.recommender

    async def close(self) -> None:
        await self.cache.close()
        self.repo.close()


def build_container(settings: Settings, cache: AsyncCache | None = None, bundle: ModelBundle | None = None,
                    metrics: Metrics | None = None) -> Container:
    metrics = metrics or Metrics()
    if cache is None:
        cache = (RedisCache.from_url(settings.redis_url, metrics, settings.redis_timeout_s)
                 if settings.redis_url else NullCache(metrics))
    repo = Repository(settings.database_url)
    c = Container(settings=settings, metrics=metrics, repo=repo, cache=cache)
    try:
        if bundle is None:
            bundle = load_bundle(resolve_artifact_dir(settings.artifacts_dir, settings.artifact_version))
        c.bundle = bundle
        catalog_version = repo.meta("catalog_version")
        imported_artifact = repo.meta("artifact_version")
        if catalog_version is None or repo.article_count() == 0:
            raise ArtifactError("article catalogue not imported; run `python -m newsrec import-catalog`")
        if imported_artifact != bundle.version:
            raise ArtifactError(f"catalogue was imported from artifact {imported_artifact!r} but the service "
                                f"loaded {bundle.version!r}; re-run import-catalog")
        ref = pd.Timestamp(settings.reference_time) if settings.reference_time else bundle.popularity.as_of
        builder = bundle.feature_builder()
        keys = Keys(settings.cache_prefix)
        c.recommender = Recommender(
            bundle, repo, cache, keys,
            retriever=build_retriever(bundle, ref, settings.index_backend, settings.candidate_max_age_hours),
            features=ServingFeatureBuilder(builder, bundle.popularity, ref),
            profiles=ProfileService(builder, cache, keys, bundle.version, settings.profile_cache_ttl_s),
            metrics=metrics, catalog_version=catalog_version, categories=repo.categories(),
            rec_ttl_s=settings.recommendation_cache_ttl_s, n_similar=settings.retrieval_similar,
            n_popular=settings.retrieval_popular, max_history=settings.max_history,
            postrank=PostRankConfig(settings.dedup_threshold, settings.max_per_category),
        )
    except (ArtifactError, FileNotFoundError) as e:
        c.startup_error = str(e)
        logger.error("service not ready: %s", e)
    return c


def get_container(request: Request) -> Container:
    return request.app.state.container
