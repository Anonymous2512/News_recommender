"""Ranking and classification metrics.

Ranking conventions (also written into every evaluation report):
* metrics are computed per held-out impression over its **complete** candidate slate;
* relevance is binary (clicked); DCG = sum (2^rel - 1) / log2(rank + 1);
* impressions with zero clicks are **excluded** from the ranking-metric means (NDCG/MRR/Recall are
  undefined for them) and their share is reported separately as ``no_click_impression_share``;
* ties are broken by a seeded random permutation, so no ordering information leaks from input order.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from newsrec.features.builder import group_boundaries
from newsrec.models.outputs import ClickProbabilities, RankingScores, ordering_values, require_probabilities

METRIC_DEFINITIONS = {
    "ndcg@k": "per impression: DCG@k / IDCG@k with gain 2^rel-1 and discount log2(rank+1); binary rel",
    "mrr": "per impression: 1 / rank of the first clicked candidate",
    "recall@k": "per impression: (#clicked in top k) / (#clicked)",
    "aggregation": "unweighted mean over held-out impressions with >= 1 click",
    "no_click_impression_share": "share of held-out impressions with zero clicks (excluded from means)",
    "ties": "broken by a seeded random permutation",
    "ci": "95% percentile bootstrap over impressions",
}


def dcg_at_k(sorted_labels: np.ndarray, k: int) -> float:
    rel = np.asarray(sorted_labels[:k], dtype=float)
    if rel.size == 0:
        return 0.0
    return float(np.sum((2.0**rel - 1.0) / np.log2(np.arange(2, rel.size + 2))))


def ndcg_at_k(sorted_labels: np.ndarray, k: int) -> float:
    ideal = dcg_at_k(np.sort(np.asarray(sorted_labels))[::-1], k)
    return 0.0 if ideal == 0 else dcg_at_k(sorted_labels, k) / ideal


def reciprocal_rank(sorted_labels: np.ndarray) -> float:
    hits = np.flatnonzero(np.asarray(sorted_labels) > 0)
    return 0.0 if hits.size == 0 else 1.0 / (hits[0] + 1)


def recall_at_k(sorted_labels: np.ndarray, k: int) -> float:
    labels = np.asarray(sorted_labels)
    total = labels.sum()
    return 0.0 if total == 0 else float(labels[:k].sum() / total)


def per_impression_metrics(
    scores: ClickProbabilities | RankingScores,
    labels: np.ndarray,
    group_ids: np.ndarray,
    ks: tuple[int, ...] = (5, 10),
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Per-impression metric arrays for impressions with >= 1 click, plus the clicked mask."""
    values = ordering_values(scores)
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    tiebreak = rng.random(len(values))
    out: dict[str, list[float]] = {f"ndcg@{k}": [] for k in ks}
    out.update({f"recall@{k}": [] for k in ks})
    out["mrr"] = []
    has_click: list[bool] = []
    for s, e in group_boundaries(np.asarray(group_ids)):
        y = labels[s:e]
        if y.sum() == 0:
            has_click.append(False)
            continue
        has_click.append(True)
        order = np.lexsort((tiebreak[s:e], -values[s:e]))  # primary: score desc; secondary: random
        ys = y[order]
        for k in ks:
            out[f"ndcg@{k}"].append(ndcg_at_k(ys, k))
            out[f"recall@{k}"].append(recall_at_k(ys, k))
        out["mrr"].append(reciprocal_rank(ys))
    res = {k: np.asarray(v, dtype=float) for k, v in out.items()}
    res["_has_click"] = np.asarray(has_click, dtype=bool)
    return res


def bootstrap_ci(values: np.ndarray, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if values.size == 0 or n_boot <= 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    means = values[idx].mean(axis=1)
    return (float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))


def summarize_ranking(per_imp: dict[str, np.ndarray], n_boot: int = 1000, seed: int = 0) -> dict[str, Any]:
    has_click = per_imp["_has_click"]
    summary: dict[str, Any] = {
        "impressions_total": int(has_click.size),
        "impressions_evaluated": int(has_click.sum()),
        "no_click_impression_share": float(1 - has_click.mean()) if has_click.size else float("nan"),
    }
    for name, vals in per_imp.items():
        if name.startswith("_"):
            continue
        lo, hi = bootstrap_ci(vals, n_boot=n_boot, seed=seed)
        summary[name] = {"mean": float(vals.mean()) if vals.size else float("nan"), "ci95": [lo, hi]}
    return summary


def expected_calibration_error(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> tuple[float, list[dict]]:
    """Equal-width-bin ECE and the reliability table."""
    bins = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    ece, table = 0.0, []
    for b in range(n_bins):
        m = bins == b
        if not m.any():
            continue
        conf, acc = float(p[m].mean()), float(y[m].mean())
        ece += m.mean() * abs(conf - acc)
        table.append({"bin": b, "count": int(m.sum()), "mean_predicted": conf, "observed_rate": acc})
    return float(ece), table


def classification_metrics(probs: ClickProbabilities, labels: np.ndarray) -> dict[str, Any]:
    p = require_probabilities(probs)
    y = np.asarray(labels).astype(int)
    out: dict[str, Any] = {"n": int(y.size), "positive_rate": float(y.mean()), "mean_predicted": float(p.mean())}
    if 0 < y.sum() < y.size:
        out["roc_auc"] = float(roc_auc_score(y, p))
        out["pr_auc"] = float(average_precision_score(y, p))
    else:
        out["roc_auc"] = out["pr_auc"] = None
    pc = np.clip(p, 1e-7, 1 - 1e-7)
    out["log_loss"] = float(log_loss(y, pc, labels=[0, 1]))
    out["brier"] = float(brier_score_loss(y, p))
    ece, table = expected_calibration_error(p, y)
    out["ece_10bin"] = ece
    out["reliability"] = table
    out["calibration_note"] = "diagnostic only; no calibration model was fitted"
    return out
