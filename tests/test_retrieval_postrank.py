"""Retrieval, within-pool post-ranking and the corrected front-page optimizer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from newsrec.features.popularity import BetaPrior, PopularitySnapshot
from newsrec.postrank import PostRankConfig, select_with_constraints, within_pool_minmax
from newsrec.retrieval.retriever import CandidateRetriever, ExactCosineIndex
from optimizer import DEFAULT_LAYOUT, LayoutSlot, OptimizerConfig, optimize_front_page


def _catalog():
    ids = np.array(["A", "B", "C", "D", "E"])
    vecs = np.array([[1, 0], [0.9, 0.1], [0, 1], [0.1, 0.9], [0, 0]], np.float32)
    cats = ["x", "x", "y", "y", "z"]
    snap = PopularitySnapshot(table=pd.DataFrame({
        "news_id": ["A", "B", "C", "D"], "impressions": [100, 100, 100, 5], "clicks": [10, 30, 20, 5],
        "first_seen_time": [pd.Timestamp("2019-11-09")] * 4, "last_seen_time": [pd.Timestamp("2019-11-10")] * 4,
    }), as_of=pd.Timestamp("2019-11-10"))
    return ids, vecs, cats, snap


def test_exact_index_topk_deterministic():
    idx = ExactCosineIndex(np.array([[1, 0], [0, 1], [1, 1], [0, 0]], np.float32))
    pos, sims = idx.search(np.array([1, 0.1]), 2)
    assert pos.tolist() == [0, 2] and sims[0] > sims[1]
    pos2, _ = idx.search(np.array([1, 0.1]), 10)
    assert 3 not in pos2.tolist()  # zero vector never retrieved
    assert idx.search(np.zeros(2), 3)[0].size == 0


def test_retriever_similarity_popularity_exclude_and_category():
    ids, vecs, cats, snap = _catalog()
    r = CandidateRetriever(ids, vecs, cats, snap, BetaPrior(1, 9), min_exposures_for_popular=20)
    out = r.retrieve(np.array([1, 0], np.float32), exclude={"A"}, n_similar=2, n_popular=2)
    got = {c.news_id: c.sources for c in out}
    assert "A" not in got and len(got) == len(out)  # excluded, no duplicates
    assert "similar" in got["B"]
    assert "popular" in got["B"] or "popular" in got["C"]
    cold = r.retrieve(None, exclude=set(), n_similar=10, n_popular=3)
    assert [c.news_id for c in cold] == ["B", "C", "A"]  # highest smoothed CTR among well-exposed first
    assert all(c.sources == ["popular"] for c in cold)
    only_y = r.retrieve(np.array([1, 0], np.float32), exclude=set(), n_similar=5, n_popular=5, category="y")
    assert {c.news_id for c in only_y} <= {"C", "D"}


def test_select_with_constraints_dedup_and_caps():
    ids = ["A", "B", "C", "D"]
    vecs = np.array([[1, 0], [1, 0.01], [0, 1], [0.7, 0.7]], np.float32)
    sel = select_with_constraints(ids, np.array([4, 3, 2, 1.0]), ["x", "x", "y", "x"], vecs, 4,
                                  PostRankConfig(dedup_threshold=0.95, max_per_category=None))
    assert [ids[i] for i in sel] == ["A", "C", "D"]  # B is a near-duplicate of A
    sel2 = select_with_constraints(ids, np.array([4, 3, 2, 1.0]), ["x", "x", "y", "x"], vecs, 4,
                                   PostRankConfig(dedup_threshold=None, max_per_category=1))
    assert [ids[i] for i in sel2] == ["A", "C"]
    tie = select_with_constraints(["Z", "Y"], np.array([1.0, 1.0]), ["x", "x"], np.zeros((2, 2)), 2,
                                  PostRankConfig(None, None))
    assert tie == [1, 0]  # ties broken by news_id


def test_within_pool_minmax():
    assert within_pool_minmax(np.array([-2.0, 0.0, 2.0])).tolist() == [0.0, 0.5, 1.0]
    assert within_pool_minmax(np.array([3.0, 3.0])).tolist() == [1.0, 1.0]


def _pool(n=40):
    rng = np.random.default_rng(1)
    cats = ["a", "b", "c", "d", "e", "f", "g", "entertainment"]  # 8 categories x cap 3 >= 20 slots
    return pd.DataFrame({"news_id": [f"N{i}" for i in range(n)], "ranking_score": rng.normal(size=n),
                         "category": [cats[i % len(cats)] for i in range(n)],
                         "title": [f"t{i}" for i in range(n)]})


def test_optimizer_visibility_does_not_change_selection_and_articles_unique():
    pool = _pool()
    dup = pd.concat([pool, pool.assign(ranking_score=pool.ranking_score - 10)])  # duplicate rows of same ids
    m1 = optimize_front_page(dup, OptimizerConfig(dedup_threshold=None))
    assert m1.news_id.is_unique and len(m1) == len(DEFAULT_LAYOUT)
    flat = [LayoutSlot(s.slot_id, s.zone, s.position, 1.0) for s in DEFAULT_LAYOUT]
    m2 = optimize_front_page(dup, OptimizerConfig(dedup_threshold=None), flat)
    assert m1.news_id.tolist() == m2.news_id.tolist()
    assert (m1.groupby("category").size() <= 3).all()


def test_optimizer_fills_fewer_slots_only_when_constraints_bind():
    pool = _pool(6)
    m = optimize_front_page(pool, OptimizerConfig(dedup_threshold=None, max_per_category=1))
    assert len(m) == 6  # 6 categories x 1
    with pytest.raises(KeyError):
        optimize_front_page(pool.drop(columns="ranking_score"))
