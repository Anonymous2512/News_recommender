"""Optional integration tests against a REAL Redis (e.g. `docker compose up -d redis`).

    REDIS_URL=redis://localhost:6379/15 pytest -m integration
"""

from __future__ import annotations

import os
import uuid

import pytest

pytestmark = pytest.mark.integration
REDIS_URL = os.environ.get("REDIS_URL")


@pytest.fixture()
def live_client(tmp_path, built, bundle):
    if not REDIS_URL:
        pytest.skip("REDIS_URL not set")
    from fastapi.testclient import TestClient

    from app.core.config import Settings
    from app.main import create_app
    from app.repositories.store import import_catalog

    s = Settings(artifacts_dir=str(built["artifacts"]), artifact_version="testv1",
                 database_url=f"sqlite:///{tmp_path / 'app.db'}", redis_url=REDIS_URL,
                 cache_prefix=f"it-{uuid.uuid4().hex[:8]}", json_logs=False, log_level="WARNING")
    import_catalog(s.database_url, built["artifact_dir"])
    with TestClient(create_app(s, bundle=bundle)) as c:
        yield c


def test_live_cache_roundtrip_and_invalidation(live_client, bundle):
    assert live_client.get("/health/ready").json()["status"] == "ready"
    user = "it-user"
    nid = bundle.articles.news_id.iloc[0]
    live_client.post("/v1/events/click", json={"event_id": uuid.uuid4().hex, "user_id": user, "news_id": nid})
    assert live_client.get(f"/v1/recommendations/{user}").json()["meta"]["cache"] == "miss"
    assert live_client.get(f"/v1/recommendations/{user}").json()["meta"]["cache"] == "hit"
    r = live_client.post("/v1/events/click", json={"event_id": uuid.uuid4().hex, "user_id": user,
                                                   "news_id": bundle.articles.news_id.iloc[1]})
    assert r.json()["cache_keys_invalidated"] >= 2
    assert live_client.get(f"/v1/recommendations/{user}").json()["meta"]["cache"] == "miss"


def test_redis_embedding_cache_live(tmp_path):
    if not REDIS_URL:
        pytest.skip("REDIS_URL not set")
    import pandas as pd
    import redis

    from newsrec.embeddings.cache import RedisEmbeddingCache
    from newsrec.embeddings.encoder import HashingEncoder, build_article_embeddings

    cache = RedisEmbeddingCache(redis.Redis.from_url(REDIS_URL), prefix=f"it-{uuid.uuid4().hex[:8]}")
    news = pd.DataFrame({"news_id": ["A", "B"], "title": ["one", "two"]})
    build_article_embeddings(news, HashingEncoder(dim=8), cache)
    _, stats = build_article_embeddings(news, HashingEncoder(dim=8), cache)
    assert stats["cache_hits"] == 2
