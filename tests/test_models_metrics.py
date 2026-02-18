"""Metrics on hand-calculated cases, output types, group construction and artifact validation."""

from __future__ import annotations

import json
import shutil

import numpy as np
import pytest

from newsrec.evaluation.metrics import (
    classification_metrics,
    ndcg_at_k,
    per_impression_metrics,
    recall_at_k,
    reciprocal_rank,
    summarize_ranking,
)
from newsrec.models.artifacts import ArtifactError, load_bundle
from newsrec.models.outputs import ClickProbabilities, RankingScores
from newsrec.models.training import group_sizes


def test_hand_calculated_ranking_metrics():
    labels = np.array([1, 0, 0, 1, 0])  # clicks at ranks 1 and 4
    assert ndcg_at_k(labels, 5) == pytest.approx((1 + 1 / np.log2(5)) / (1 + 1 / np.log2(3)))
    assert ndcg_at_k(labels, 5) == pytest.approx(0.8772, abs=1e-4)
    assert reciprocal_rank(np.array([0, 0, 1])) == pytest.approx(1 / 3)
    assert recall_at_k(labels, 2) == 0.5
    assert ndcg_at_k(np.array([0, 0]), 5) == 0.0


def test_per_impression_excludes_no_click_groups_and_reports_share():
    scores = RankingScores(np.array([3.0, 2.0, 1.0, 5.0, 4.0, 0.5, 0.4]))
    labels = np.array([0, 1, 0, 0, 0, 1, 0])
    groups = np.array([1, 1, 1, 2, 2, 3, 3])
    per = per_impression_metrics(scores, labels, groups, ks=(5,))
    assert per["_has_click"].tolist() == [True, False, True]
    assert per["mrr"].tolist() == [0.5, 1.0]
    s = summarize_ranking(per, n_boot=20)
    assert s["no_click_impression_share"] == pytest.approx(1 / 3) and s["impressions_evaluated"] == 2


def test_ties_are_broken_randomly_but_deterministically():
    scores = RankingScores(np.zeros(4))
    labels, groups = np.array([0, 0, 0, 1]), np.zeros(4, int)
    a = per_impression_metrics(scores, labels, groups, seed=1)["mrr"]
    b = per_impression_metrics(scores, labels, groups, seed=1)["mrr"]
    assert a.tolist() == b.tolist()


def test_probabilities_and_scores_cannot_be_confused():
    with pytest.raises(ValueError):
        ClickProbabilities(np.array([0.2, 1.3]))
    with pytest.raises(TypeError):
        classification_metrics(RankingScores(np.array([0.1, 0.9])), np.array([0, 1]))
    with pytest.raises(TypeError):
        per_impression_metrics(np.array([0.1, 0.2]), np.array([0, 1]), np.array([0, 0]))
    m = classification_metrics(ClickProbabilities(np.array([0.1, 0.8, 0.3, 0.6])), np.array([0, 1, 0, 1]))
    assert m["roc_auc"] == 1.0 and "calibration_note" in m


def test_group_sizes_contiguous_and_sum():
    assert group_sizes(np.array([5, 5, 7, 9, 9, 9])).tolist() == [2, 1, 3]
    with pytest.raises(ValueError):
        group_sizes(np.array([5, 7, 5]))


def test_trained_models_output_types(bundle, built, processed):
    from newsrec.pipeline import split_frame

    rows, hist = split_frame(processed["impressions"], processed["candidates"], "test")
    X = bundle.feature_builder().build(rows.head(50), hist)
    p = bundle.ctr_lightgbm.predict_proba(X)
    q = bundle.ctr_logreg.predict_proba(X)
    s = bundle.ranker.score(X)
    assert isinstance(p, ClickProbabilities) and isinstance(q, ClickProbabilities)
    assert isinstance(s, RankingScores) and not isinstance(s, ClickProbabilities)
    assert ((p.values >= 0) & (p.values <= 1)).all()


def test_report_contains_baselines_boundaries_and_synthetic_warning(built):
    rep = json.loads((built["artifact_dir"] / "evaluation.json").read_text())
    assert rep["data_source"] == "synthetic" and "SYNTHETIC" in rep["warning"]
    for name in ("baseline_random", "baseline_past_popularity", "lambdarank_full",
                 "lambdarank_popularity_only", "lambdarank_content_only", "lambdarank_full_minus_profile"):
        r = rep["rankers"][name]
        assert {"ndcg@5", "ndcg@10", "mrr", "recall@10", "no_click_impression_share"} <= set(r)
        assert len(r["ndcg@10"]["ci95"]) == 2
    assert {"val_start", "test_start"} <= set(rep["split_boundaries"])
    for clf in ("logistic_regression", "lightgbm_binary"):
        assert {"roc_auc", "pr_auc", "log_loss", "brier", "ece_10bin"} <= set(rep["classifiers"][clf]["test"])
    # on synthetic data with a planted signal the learned ranker must beat random (software sanity check)
    assert rep["rankers"]["lambdarank_full"]["ndcg@10"]["mean"] > rep["rankers"]["baseline_random"]["ndcg@10"]["mean"]


def test_load_bundle_fails_loudly(built, tmp_path):
    with pytest.raises(ArtifactError, match="missing"):
        load_bundle(tmp_path)
    broken = tmp_path / "broken"
    shutil.copytree(built["artifact_dir"], broken)
    spec = json.loads((broken / "feature_spec.json").read_text())
    spec["features"] = spec["features"][1:] + spec["features"][:1]
    spec.pop("fingerprint")
    (broken / "feature_spec.json").write_text(json.dumps(spec))
    with pytest.raises(ArtifactError):
        load_bundle(broken)
