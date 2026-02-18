"""API, caching and durable-event behavior (fakeredis + temporary SQLite; no live services)."""

from __future__ import annotations

import time
import uuid

import fakeredis
import pytest
import redis.asyncio as aioredis
from fastapi.testclient import TestClient

from app.cache.redis_cache import Keys, NullCache, RedisCache
from app.core.config import Settings
from app.core.metrics import Metrics
from app.main import create_app
from app.repositories.store import import_catalog


def _settings(tmp_path, built, **kw) -> Settings:
    return Settings(artifacts_dir=str(built["artifacts"]), artifact_version="testv1",
                    database_url=f"sqlite:///{tmp_path / 'app.db'}", redis_url=None, json_logs=False,
                    log_level="WARNING", **kw)


@pytest.fixture()
def fake_redis():
    return fakeredis.FakeAsyncRedis()


@pytest.fixture()
def client(tmp_path, built, bundle, fake_redis):
    s = _settings(tmp_path, built)
    import_catalog(s.database_url, built["artifact_dir"])
    metrics = Metrics()
    app = create_app(s, cache=RedisCache(fake_redis, metrics), bundle=bundle)
    with TestClient(app) as c:
        c.fake_redis = fake_redis
        yield c


def _warm_user(bundle) -> str:
    import pandas as pd

    hist = pd.read_parquet(bundle.path / "user_histories.parquet")
    return hist[hist.history.map(len) >= 5].user_id.iloc[0]


def test_health(client):
    assert client.get("/health/live").json() == {"status": "ok", "checks": {}}
    r = client.get("/health/ready")
    assert r.status_code == 200 and r.json()["status"] == "ready"
    assert r.json()["checks"]["model"]["data_source"] == "synthetic"


def test_cold_start_user(client):
    r = client.get("/v1/recommendations/brand-new-user")
    assert r.status_code == 200
    body = r.json()
    assert body["meta"]["cold_start"] is True and len(body["items"]) == 5
    assert all(i["retrieval_sources"] == ["popular"] for i in body["items"])
    assert [i["rank"] for i in body["items"]] == [1, 2, 3, 4, 5]


def test_known_user_is_personalized_unique_and_excludes_seen(client, bundle):
    user = _warm_user(bundle)
    body = client.get(f"/v1/recommendations/{user}?limit=10").json()
    assert body["meta"]["cold_start"] is False and body["meta"]["history_items_used"] > 0
    ids = [i["news_id"] for i in body["items"]]
    assert len(ids) == len(set(ids)) == 10
    assert any("similar" in i["retrieval_sources"] for i in body["items"])
    import pandas as pd

    seen = set(pd.read_parquet(bundle.path / "user_histories.parquet").set_index("user_id").loc[user, "history"])
    assert not seen & set(ids)
    scores = [i["ranking_score"] for i in body["items"]]
    assert all("click_probability_uncalibrated" not in i or i["click_probability_uncalibrated"] is None
               for i in body["items"])
    assert "not a probability" in body["meta"]["score_note"]
    again = client.get(f"/v1/recommendations/{user}?limit=10").json()
    assert [i["news_id"] for i in again["items"]] == ids  # deterministic
    assert scores == [i["ranking_score"] for i in again["items"]]


def test_category_filter_limit_validation_and_ctr_field(client, bundle):
    user = _warm_user(bundle)
    cat = bundle.articles.category.iloc[0]
    body = client.get(f"/v1/recommendations/{user}?category={cat}&limit=3&include_ctr=true").json()
    assert body["items"] and all(i["category"] == cat for i in body["items"])
    assert all(0 <= i["click_probability_uncalibrated"] <= 1 for i in body["items"])
    assert client.get("/v1/recommendations/u1?limit=0").status_code == 422
    assert client.get("/v1/recommendations/u1?limit=51").json()["error"]["code"] == "limit_too_large"
    assert client.get("/v1/recommendations/u1?category=nope").json()["error"]["code"] == "unknown_category"
    assert client.get("/v1/recommendations/bad%20user").status_code == 422


def test_recommendation_cache_hit_and_invalidation_on_click(client, bundle):
    user = _warm_user(bundle)
    first = client.get(f"/v1/recommendations/{user}").json()
    assert first["meta"]["cache"] == "miss" and first["meta"]["profile_cache"] == "miss"
    second = client.get(f"/v1/recommendations/{user}").json()
    assert second["meta"]["cache"] == "hit" and second["items"] == first["items"]
    clicked = first["items"][0]["news_id"]
    ev = {"event_id": str(uuid.uuid4()), "user_id": user, "news_id": clicked}
    r = client.post("/v1/events/click", json=ev)
    assert r.status_code == 201 and r.json()["status"] == "recorded"
    assert r.json()["cache_keys_invalidated"] >= 2  # profile + recommendation keys
    third = client.get(f"/v1/recommendations/{user}").json()
    assert third["meta"]["cache"] == "miss" and third["meta"]["profile_cache"] == "miss"
    assert clicked not in [i["news_id"] for i in third["items"]]  # newly seen item excluded
    assert third["meta"]["history_items_used"] >= first["meta"]["history_items_used"]


def test_click_idempotency_conflict_and_validation(client, bundle):
    nid = bundle.articles.news_id.iloc[1]
    ev = {"event_id": "evt-abcdef12", "user_id": "U-x", "news_id": nid}
    assert client.post("/v1/events/click", json=ev).status_code == 201
    dup = client.post("/v1/events/click", json=ev)
    assert dup.status_code == 200 and dup.json()["status"] == "duplicate"
    conflict = client.post("/v1/events/click", json={**ev, "news_id": bundle.articles.news_id.iloc[2]})
    assert conflict.status_code == 409
    assert client.post("/v1/events/click", json={**ev, "event_id": "evt-zz000001", "news_id": "NOPE"}).status_code == 422
    assert client.post("/v1/events/click", json={"user_id": "U"}).status_code == 422
    bad = client.post("/v1/events/click", content=b"{not json", headers={"content-type": "application/json"})
    assert bad.status_code == 422 and "error" in bad.json()
    # the user is no longer cold-start after one recorded click
    assert client.get("/v1/recommendations/U-x").json()["meta"]["cold_start"] is False


def test_news_endpoint_and_openapi(client, bundle):
    nid = bundle.articles.news_id.iloc[0]
    assert client.get(f"/v1/news/{nid}").json()["news_id"] == nid
    r = client.get("/v1/news/N000000")
    assert r.status_code == 404 and r.json()["error"]["code"] == "news_not_found"
    spec = client.get("/openapi.json").json()
    assert {"/v1/recommendations/{user_id}", "/v1/news/{news_id}", "/v1/events/click", "/health/ready"} <= set(spec["paths"])
    assert "newsrec_http_requests_total" in client.get("/metrics").text


def test_redis_unavailable_serves_uncached_and_reports_degraded(tmp_path, built, bundle):
    s = _settings(tmp_path, built)
    import_catalog(s.database_url, built["artifact_dir"])
    metrics = Metrics()
    dead = aioredis.Redis(host="127.0.0.1", port=1, socket_timeout=0.05, socket_connect_timeout=0.05)
    with TestClient(create_app(s, cache=RedisCache(dead, metrics, timeout_s=0.05), bundle=bundle)) as c:
        r = c.get("/v1/recommendations/someone")
        assert r.status_code == 200 and len(r.json()["items"]) == 5
        ready = c.get("/health/ready").json()
        assert ready["status"] == "degraded" and ready["checks"]["cache"]["ok"] is False
        assert metrics.cache_count("recommendation", "error") >= 1


def test_cache_disabled_mode(tmp_path, built, bundle):
    s = _settings(tmp_path, built)
    import_catalog(s.database_url, built["artifact_dir"])
    with TestClient(create_app(s, cache=NullCache(), bundle=bundle)) as c:
        body = c.get("/v1/recommendations/someone").json()
        assert body["meta"]["cache"] == "bypass"
        assert c.get("/health/ready").json()["status"] == "degraded"


def test_not_ready_without_artifacts_or_catalog(tmp_path, built):
    s = Settings(artifacts_dir=str(tmp_path / "none"), database_url=f"sqlite:///{tmp_path / 'x.db'}",
                 redis_url=None, json_logs=False, log_level="WARNING")
    with TestClient(create_app(s, cache=NullCache())) as c:
        assert c.get("/health/ready").status_code == 503
        r = c.get("/v1/recommendations/u")
        assert r.status_code == 503 and r.json()["error"]["code"] == "not_ready"
        assert c.get("/health/live").status_code == 200
    # artifacts present but catalogue never imported -> not ready with an actionable message
    s2 = _settings(tmp_path / "other", built)
    with TestClient(create_app(s2, cache=NullCache())) as c:
        msg = c.get("/v1/recommendations/u").json()["error"]["message"]
        assert "import-catalog" in msg


# ------------------------------------------------------------------------------- cache unit tests
@pytest.mark.anyio
async def test_redis_cache_hit_miss_expiry_and_version_separation():
    metrics = Metrics()
    cache = RedisCache(fakeredis.FakeAsyncRedis(), metrics)
    keys = Keys("t")
    k1 = keys.recommendation("v1", "cat1", "u", "s1-c0", 5, None, True, False)
    k2 = keys.recommendation("v2", "cat1", "u", "s1-c0", 5, None, True, False)
    assert k1 != k2 and keys.profile("v1", "u", "h1") != keys.profile("v1", "u", "h2")
    assert await cache.get(k1, "recommendation") is None
    await cache.set(k1, b"x", 1, "recommendation", index_key=keys.user_index("u"))
    assert await cache.get(k1, "recommendation") == b"x"
    assert await cache.get(k2, "recommendation") is None  # new model version never reads old entries
    assert metrics.cache_count("recommendation", "hit") == 1 and metrics.cache_count("recommendation", "miss") == 2
    time.sleep(1.1)
    assert await cache.get(k1, "recommendation") is None  # TTL expired
    await cache.set(k1, b"y", 60, "recommendation", index_key=keys.user_index("u"))
    assert await cache.invalidate_index(keys.user_index("u")) == 1
    assert await cache.get(k1, "recommendation") is None


@pytest.fixture
def anyio_backend():
    return "asyncio"
