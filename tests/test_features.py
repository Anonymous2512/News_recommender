"""Embeddings (versioned caching, projection), profiles and the shared feature builder."""

from __future__ import annotations

import fakeredis
import numpy as np
import pandas as pd
import pytest

from newsrec.config import ProfileConfig
from newsrec.embeddings.cache import DiskEmbeddingCache, RedisEmbeddingCache
from newsrec.embeddings.encoder import ArticleEmbeddings, HashingEncoder, build_article_embeddings
from newsrec.embeddings.projection import EmbeddingProjection, fit_projection
from newsrec.features.builder import ArticleMeta, FeatureBuilder, group_boundaries
from newsrec.features.popularity import BetaPrior
from newsrec.features.profiles import UserProfile, build_profile, profile_candidate_features, recency_weights
from newsrec.features.schema import FeatureSpec, make_feature_spec


class CountingEncoder(HashingEncoder):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.calls = 0
        self.texts_encoded = 0

    def encode(self, texts):
        self.calls += 1
        self.texts_encoded += len(texts)
        return super().encode(texts)


NEWS = pd.DataFrame({"news_id": ["N1", "N2", "N3", "N1"], "title": ["storm warning", "market rally", "", "storm warning"],
                     "category": ["weather", "finance", "finance", "weather"], "subcategory": ["w", "f", "f", "w"]})


@pytest.mark.parametrize("make_cache", ["disk", "redis"])
def test_embeddings_cached_per_model_and_text(tmp_path, make_cache):
    cache = DiskEmbeddingCache(tmp_path) if make_cache == "disk" else RedisEmbeddingCache(fakeredis.FakeRedis())
    enc = CountingEncoder(dim=16)
    emb, stats = build_article_embeddings(NEWS, enc, cache)
    assert len(emb) == 3 and stats["encoded"] == 2 and stats["empty_text"] == 1  # one vector per unique id
    assert np.allclose(emb.lookup(["N3"])[0], 0)  # empty title -> zero vector
    if make_cache == "disk":
        cache.flush()
        cache = DiskEmbeddingCache(tmp_path)
    emb2, stats2 = build_article_embeddings(NEWS, enc, cache)
    assert stats2["cache_hits"] == 2 and stats2["encoded"] == 0 and np.allclose(emb.vectors, emb2.vectors)
    # changed text -> recomputed; different model id -> never reuses vectors
    changed = NEWS.assign(title=["storm warning issued", "market rally", "", "storm warning issued"])
    _, s3 = build_article_embeddings(changed, enc, cache)
    assert s3["encoded"] == 1 and s3["cache_hits"] == 1
    _, s4 = build_article_embeddings(NEWS, CountingEncoder(dim=16, seed=1), cache)
    assert s4["cache_hits"] == 0 and s4["encoded"] == 2


def test_embedding_table_roundtrip(tmp_path):
    emb, _ = build_article_embeddings(NEWS, HashingEncoder(dim=8))
    emb.save(tmp_path / "e.npz")
    back = ArticleEmbeddings.load(tmp_path / "e.npz")
    assert back.model_id == emb.model_id and np.array_equal(back.vectors, emb.vectors)
    vecs, known = back.lookup(["N2", "UNKNOWN"])
    assert known.tolist() == [True, False] and np.allclose(vecs[1], 0)


def test_pca_projection_fit_on_train_only_and_persisted(tmp_path):
    rng = np.random.default_rng(0)
    train = rng.normal(size=(50, 10)).astype(np.float32)
    proj = fit_projection(train, "pca", 4)
    assert proj.output_dim == 4
    proj.save(tmp_path / "p.npz")
    back = EmbeddingProjection.load(tmp_path / "p.npz")
    x = rng.normal(size=(5, 10)).astype(np.float32)
    assert np.allclose(proj.transform(x), back.transform(x))
    assert np.allclose(proj.transform(np.zeros((1, 10))), 0)
    assert fit_projection(train, "full").transform(x).shape == (5, 10)
    assert fit_projection(train, "none").output_dim == 0


def test_recency_weights_and_profile():
    w = recency_weights(3, half_life=1.0)
    assert w.tolist() == [0.25, 0.5, 1.0]
    emb = ArticleEmbeddings(news_ids=np.array(["A", "B", "C"]),
                            vectors=np.array([[1, 0], [0, 1], [0, 0]], np.float32), model_id="m",
                            checksums=np.array(["a", "b", "c"]))
    cats = {"A": ("x", "x1"), "B": ("y", "y1")}
    p = build_profile(["A", "B", "C", "ZZ"], emb, cats, max_history=10, half_life=1.0)
    assert p.n_known == 2  # C (zero vector) and ZZ (unknown) skipped
    assert np.isclose(np.linalg.norm(p.vector), 1.0)
    assert p.vector[1] > p.vector[0]  # most recent item (B) weighs more
    assert p.category_weights["y"] > p.category_weights["x"]
    empty = build_profile([], emb, cats)
    assert empty.is_empty and not np.any(empty.vector)
    feats = profile_candidate_features(empty, emb.vectors, ["x", "y", ""], ["", "", ""])
    assert all(np.all(v == 0) for v in feats.values())
    back = UserProfile.from_dict(p.to_dict())
    assert np.array_equal(back.vector, p.vector) and back.category_weights == p.category_weights


def test_group_boundaries_requires_contiguity():
    assert group_boundaries(np.array([1, 1, 2, 3, 3])) == [(0, 2), (2, 3), (3, 5)]
    with pytest.raises(ValueError):
        group_boundaries(np.array([1, 2, 1]))


def _builder(bundle) -> FeatureBuilder:
    return bundle.feature_builder()


def test_feature_order_and_values_identical_offline_vs_serving(bundle):
    from app.services.feature_builder import ServingFeatureBuilder

    b = _builder(bundle)
    ids = bundle.embeddings.news_ids[:6].tolist()
    history = bundle.embeddings.news_ids[10:20].tolist()
    serving = ServingFeatureBuilder(b, bundle.popularity, bundle.popularity.as_of)
    X_serve = serving.build(ids, history, b.profile(history))
    frame = serving.frame(ids)  # same raw counters passed through the offline entry point
    X_offline = b.build(frame, {0: history})
    assert X_serve.shape == (6, len(bundle.spec.names))
    assert np.array_equal(X_serve, X_offline)
    assert bundle.ranker.booster.feature_name() == bundle.spec.names


def test_builder_rejects_mismatched_embeddings_and_bad_counters(bundle):
    other = ArticleEmbeddings(news_ids=np.array(["A"]), vectors=np.ones((1, bundle.embeddings.dim), np.float32),
                              model_id="other-model", checksums=np.array(["x"]))
    with pytest.raises(ValueError, match="expects embeddings"):
        FeatureBuilder(bundle.spec, other, bundle.projection, bundle.prior,
                       ArticleMeta({}, {}), ProfileConfig())
    frame = pd.DataFrame({"group_id": [0], "news_id": ["N1"], "past_impressions": [1], "past_clicks": [5],
                          "hours_since_first_seen": [0.0]})
    with pytest.raises(ValueError, match="invalid popularity"):
        _builder(bundle).build(frame, {0: []})


def test_feature_spec_fingerprint_detects_changes():
    spec = make_feature_spec("m", "pca", 3)
    d = spec.to_dict()
    assert FeatureSpec.from_dict(d).names == spec.names
    d["features"] = d["features"][::-1]
    with pytest.raises(ValueError):
        FeatureSpec.from_dict(d)
    assert "global_ctr" not in spec.names  # the leaky whole-dataset feature is gone


def test_prior_is_saved(bundle):
    assert isinstance(bundle.prior, BetaPrior) and bundle.prior.alpha > 0
