"""Service configuration from environment variables (prefix ``NEWSREC_``); see ``.env.example``."""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Empty env values fall back to defaults; set NEWSREC_REDIS_URL=disabled to turn caching off.
    model_config = SettingsConfigDict(env_prefix="NEWSREC_", env_file=".env", extra="ignore", env_ignore_empty=True)

    artifacts_dir: str = "artifacts"
    artifact_version: str | None = None  # None -> artifacts/CURRENT
    database_url: str = "sqlite:///data/newsrec.db"

    # "disabled"/"none" turns caching off (requests are served uncached; readiness reports "degraded").
    redis_url: str | None = "redis://localhost:6379/0"
    redis_timeout_s: float = Field(0.25, gt=0)
    cache_prefix: str = "newsrec"
    profile_cache_ttl_s: int = Field(3600, gt=0)
    recommendation_cache_ttl_s: int = Field(300, gt=0)

    default_limit: int = Field(5, ge=1)
    max_limit: int = Field(50, ge=1)
    max_history: int = Field(200, ge=1)  # history items loaded from the store per request
    retrieval_similar: int = Field(150, ge=0)
    retrieval_popular: int = Field(50, ge=0)
    candidate_max_age_hours: float | None = None  # exposure-recency eligibility window (None = all)
    index_backend: str = "exact"  # exact | faiss
    dedup_threshold: float | None = Field(0.9, ge=0, le=1)
    max_per_category: int | None = Field(None, ge=1)
    # Logical "now" for the exposure-age feature. Defaults to the popularity snapshot time, because the
    # model was trained on 2019 logs; using the wall clock would put every article out of distribution.
    reference_time: str | None = None

    log_level: str = "INFO"
    json_logs: bool = True

    @field_validator("redis_url")
    @classmethod
    def _redis_disabled(cls, v: str | None) -> str | None:
        return None if v is None or v.strip().lower() in {"", "none", "disabled", "off"} else v

    @field_validator("index_backend")
    @classmethod
    def _backend(cls, v: str) -> str:
        if v not in {"exact", "faiss"}:
            raise ValueError("index_backend must be 'exact' or 'faiss'")
        return v


@lru_cache
def get_settings() -> Settings:
    return Settings()
