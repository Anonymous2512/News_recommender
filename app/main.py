"""FastAPI application factory. Run: ``uvicorn app.main:app``."""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api import routes_events, routes_health, routes_news, routes_recommendations
from app.cache.redis_cache import AsyncCache
from app.container import build_container
from app.core.config import Settings, get_settings
from app.core.logging import configure_logging, log_event, request_id_var
from app.core.metrics import Metrics
from app.services.recommender import RecommendationError
from newsrec.models.artifacts import ModelBundle

logger = logging.getLogger("app.access")


def _error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse({"error": {"code": code, "message": message, "request_id": request_id_var.get()}},
                        status_code=status)


def create_app(settings: Settings | None = None, cache: AsyncCache | None = None,
               bundle: ModelBundle | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level, settings.json_logs)
    metrics = Metrics()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.container = build_container(settings, cache=cache, bundle=bundle, metrics=metrics)
        try:
            yield
        finally:
            await app.state.container.close()

    app = FastAPI(title="newsrec", version="0.2.0", lifespan=lifespan,
                  description="MIND news recommendations: retrieval + LightGBM LambdaRank ranking.")

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex
        token = request_id_var.set(rid[:64])
        t0 = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            response.headers["x-request-id"] = request_id_var.get()
            return response
        finally:
            route = request.scope.get("route")
            path = getattr(route, "path", "unmatched")
            dt = time.perf_counter() - t0
            metrics.requests.labels(route=path, method=request.method, status=str(status)).inc()
            metrics.latency.labels(route=path).observe(dt)
            log_event(logger, "request", method=request.method, route=path, status=status,
                      duration_ms=round(dt * 1000, 2))
            request_id_var.reset(token)

    @app.exception_handler(RecommendationError)
    async def _rec_error(_: Request, exc: RecommendationError):
        return _error(exc.status, exc.code, exc.message)

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError):
        details = "; ".join(f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors())
        return _error(422, "validation_error", details)

    @app.exception_handler(StarletteHTTPException)
    async def _http(_: Request, exc: StarletteHTTPException):
        return _error(exc.status_code, "http_error", str(exc.detail))

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception):
        logger.exception("unhandled error")
        return _error(500, "internal_error", "internal server error")

    for r in (routes_health, routes_recommendations, routes_news, routes_events):
        app.include_router(r.router)
    return app


def _default_app() -> FastAPI:
    return create_app()


app = _default_app()
