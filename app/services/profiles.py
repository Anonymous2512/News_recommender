"""User profiles for serving: built from durable history, cached in Redis keyed by history version."""

from __future__ import annotations

import json

from starlette.concurrency import run_in_threadpool

from app.cache.redis_cache import AsyncCache, Keys
from app.repositories.store import UserHistory
from newsrec.features.builder import FeatureBuilder
from newsrec.features.profiles import UserProfile


class ProfileService:
    def __init__(self, builder: FeatureBuilder, cache: AsyncCache, keys: Keys, artifact_version: str, ttl_s: int):
        self.builder, self.cache, self.keys = builder, cache, keys
        self.artifact_version, self.ttl = artifact_version, ttl_s

    async def get(self, history: UserHistory) -> tuple[UserProfile, str]:
        """Return (profile, cache status). The key contains the history version, so a new click can never
        be answered with a profile computed before it."""
        key = self.keys.profile(self.artifact_version, history.user_id, history.version)
        raw = await self.cache.get(key, "profile")
        if raw is not None:
            try:
                return UserProfile.from_dict(json.loads(raw)), "hit"
            except (ValueError, KeyError, json.JSONDecodeError):
                pass  # corrupt entry -> rebuild
        profile = await run_in_threadpool(self.builder.profile, history.news_ids)
        await self.cache.set(key, json.dumps(profile.to_dict()).encode(), self.ttl, "profile",
                             index_key=self.keys.user_index(history.user_id))
        return profile, "miss" if self.cache.name == "redis" else "bypass"
