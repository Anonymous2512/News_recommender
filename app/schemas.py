"""Pydantic v2 request/response contracts."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

ID_PATTERN = r"^[A-Za-z0-9_\-:.]+$"


class RecommendationItem(BaseModel):
    rank: int = Field(ge=1)
    news_id: str
    title: str
    category: str
    subcategory: str = ""
    ranking_score: float = Field(description="Raw LambdaRank score; only comparable within this response. "
                                             "Not a probability or confidence.")
    retrieval_sources: list[Literal["similar", "popular"]]
    click_probability_uncalibrated: float | None = Field(
        None, ge=0, le=1, description="LightGBM binary CTR model output (only if include_ctr=true). "
                                      "Not calibrated.")


class RecommendationMeta(BaseModel):
    request_id: str
    artifact_version: str
    catalog_version: str
    data_source: Literal["mind", "synthetic"]
    cold_start: bool
    history_items_used: int
    candidates_retrieved: int
    cache: Literal["hit", "miss", "bypass"]
    profile_cache: Literal["hit", "miss", "bypass", "not_used"]
    score_note: str


class RecommendationResponse(BaseModel):
    user_id: str
    items: list[RecommendationItem]
    meta: RecommendationMeta


class NewsResponse(BaseModel):
    news_id: str
    title: str
    category: str
    subcategory: str


class ClickEventIn(BaseModel):
    event_id: str = Field(min_length=8, max_length=128, pattern=ID_PATTERN,
                          description="Client-generated idempotency key (e.g. a UUID)")
    user_id: str = Field(min_length=1, max_length=64, pattern=ID_PATTERN)
    news_id: str = Field(min_length=1, max_length=32, pattern=ID_PATTERN)
    occurred_at: datetime | None = None


class ClickEventOut(BaseModel):
    event_id: str
    status: Literal["recorded", "duplicate"]
    history_version: str
    cache_keys_invalidated: int


class ErrorBody(BaseModel):
    code: str
    message: str
    request_id: str


class ErrorResponse(BaseModel):
    error: ErrorBody


class HealthResponse(BaseModel):
    status: Literal["ok", "ready", "degraded", "not_ready"]
    checks: dict[str, dict] = {}
