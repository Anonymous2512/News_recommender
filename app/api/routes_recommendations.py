from __future__ import annotations

from fastapi import APIRouter, Depends, Path, Query, Request

from app.container import Container, get_container
from app.core.logging import request_id_var
from app.schemas import ErrorResponse, RecommendationResponse
from app.services.recommender import RecommendationError

router = APIRouter(prefix="/v1/recommendations", tags=["recommendations"])


@router.get("/{user_id}", response_model=RecommendationResponse,
            responses={422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}})
async def get_recommendations(
    request: Request,
    user_id: str = Path(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_\-:.]+$"),
    limit: int | None = Query(None, ge=1, description="number of items (default 5, bounded by MAX_LIMIT)"),
    category: str | None = Query(None, min_length=1, max_length=64),
    exclude_seen: bool = Query(True, description="exclude articles already in the user's history"),
    include_ctr: bool = Query(False, description="also return the uncalibrated LightGBM click probability"),
    c: Container = Depends(get_container),
) -> RecommendationResponse:
    limit = limit or c.settings.default_limit
    if limit > c.settings.max_limit:
        raise RecommendationError(422, "limit_too_large", f"limit must be <= {c.settings.max_limit}")
    rec = c.require_recommender()
    result = await rec.recommend(user_id, limit, category, exclude_seen, include_ctr, request_id_var.get())
    return RecommendationResponse.model_validate({"user_id": result.user_id, "items": result.items,
                                                  "meta": result.meta})
