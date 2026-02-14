from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Response
from starlette.concurrency import run_in_threadpool

from app.cache.redis_cache import Keys
from app.container import Container, get_container
from app.core.logging import log_event, user_hash
from app.schemas import ClickEventIn, ClickEventOut, ErrorResponse
from app.services.recommender import RecommendationError

router = APIRouter(prefix="/v1/events", tags=["events"])
logger = logging.getLogger(__name__)


@router.post("/click", response_model=ClickEventOut, status_code=201,
             responses={200: {"model": ClickEventOut, "description": "duplicate event_id (idempotent replay)"},
                        409: {"model": ErrorResponse}, 422: {"model": ErrorResponse}})
async def record_click(event: ClickEventIn, response: Response,
                       c: Container = Depends(get_container)) -> ClickEventOut:
    if await run_in_threadpool(c.repo.get_article, event.news_id) is None:
        raise RecommendationError(422, "unknown_news_id", f"unknown news_id {event.news_id!r}")
    try:
        result = await run_in_threadpool(c.repo.record_click, event.event_id, event.user_id, event.news_id,
                                         event.occurred_at)
    except ValueError as e:
        raise RecommendationError(409, "event_id_conflict", str(e)) from e
    # Durable write is committed; now drop this user's cached profile/recommendations. Keys also embed the
    # history version, so even a failed invalidation cannot serve pre-click results.
    invalidated = await c.cache.invalidate_index(Keys(c.settings.cache_prefix).user_index(event.user_id))
    if not result.created:
        response.status_code = 200
    log_event(logger, "click", user=user_hash(event.user_id), created=result.created, invalidated=invalidated)
    return ClickEventOut(event_id=event.event_id, status="recorded" if result.created else "duplicate",
                         history_version=result.history_version, cache_keys_invalidated=invalidated)
