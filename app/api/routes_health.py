from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse, PlainTextResponse
from starlette.concurrency import run_in_threadpool

from app.container import Container, get_container
from app.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health/live", response_model=HealthResponse)
async def live() -> HealthResponse:
    return HealthResponse(status="ok")


@router.get("/health/ready", response_model=HealthResponse, responses={503: {"model": HealthResponse}})
async def ready(c: Container = Depends(get_container)):
    checks: dict[str, dict] = {}
    if c.bundle is not None:
        checks["model"] = {"ok": True, "artifact_version": c.bundle.version,
                           "data_source": c.bundle.manifest["data_source"],
                           "embedding_model_id": c.bundle.embeddings.model_id}
    else:
        checks["model"] = {"ok": False, "error": c.startup_error}
    try:
        await run_in_threadpool(c.repo.ping)
        checks["database"] = {"ok": True, "articles": await run_in_threadpool(c.repo.article_count)}
    except Exception as e:  # report, don't crash
        checks["database"] = {"ok": False, "error": type(e).__name__}
    checks["recommender"] = {"ok": c.recommender is not None, "error": None if c.recommender else c.startup_error}
    if c.cache.name == "redis":
        ok = await c.cache.ping()
        checks["cache"] = {"ok": ok, "backend": "redis",
                           "note": None if ok else "Redis unavailable: serving uncached"}
    else:
        checks["cache"] = {"ok": False, "backend": "disabled", "note": "caching disabled (NEWSREC_REDIS_URL=disabled): serving uncached"}
    core_ok = checks["model"]["ok"] and checks["database"]["ok"] and checks["recommender"]["ok"]
    status: Literal["ready", "degraded", "not_ready"] = (
        ("ready" if checks["cache"]["ok"] else "degraded") if core_ok else "not_ready")
    body = HealthResponse(status=status, checks=checks)
    return JSONResponse(body.model_dump(), status_code=200 if core_ok else 503)


@router.get("/metrics", include_in_schema=False)
async def metrics(c: Container = Depends(get_container)) -> PlainTextResponse:
    return PlainTextResponse(c.metrics.render().decode(), media_type="text/plain; version=0.0.4")
