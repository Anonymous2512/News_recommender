from __future__ import annotations

from fastapi import APIRouter, Depends, Path
from starlette.concurrency import run_in_threadpool

from app.container import Container, get_container
from app.schemas import ErrorResponse, NewsResponse
from app.services.recommender import RecommendationError

router = APIRouter(prefix="/v1/news", tags=["news"])


@router.get("/{news_id}", response_model=NewsResponse, responses={404: {"model": ErrorResponse}})
async def get_news(news_id: str = Path(min_length=1, max_length=32, pattern=r"^[A-Za-z0-9_\-:.]+$"),
                   c: Container = Depends(get_container)) -> NewsResponse:
    article = await run_in_threadpool(c.repo.get_article, news_id)
    if article is None:
        raise RecommendationError(404, "news_not_found", f"unknown news_id {news_id!r}")
    return NewsResponse(**article)
