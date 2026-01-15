"""Point-in-time (past-only) article popularity.

For a candidate shown in an impression at time ``t``, the counters use only impressions with time
**strictly before** ``t``. Impressions sharing the same timestamp never see each other's labels, and an
impression never sees its own labels. This replaces the old whole-dataset ``global_ctr`` feature, which
included the row's own click label and future/validation labels.

Smoothing: ``past_ctr_smoothed = (past_clicks + alpha) / (past_impressions + alpha + beta)`` with a Beta
prior fitted (method of moments) on training-period counts only and stored in the artifact.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def point_in_time_counts(candidates: pd.DataFrame, impression_times: pd.Series) -> pd.DataFrame:
    """Add ``past_impressions``, ``past_clicks``, ``hours_since_first_seen`` to candidate rows.

    Args:
        candidates: rows with ``impression_id``, ``news_id``, ``clicked``.
        impression_times: time per impression, indexed by ``impression_id``.

    ``hours_since_first_seen`` is the age of the article's first exposure in the log relative to ``t``
    (0 if this is its first exposure). It is an exposure-age proxy, NOT a publication time - MIND has no
    publish timestamps.
    """
    rows = candidates[["impression_id", "news_id", "clicked"]].copy()
    rows["time"] = rows["impression_id"].map(impression_times)
    if rows["time"].isna().any():
        raise ValueError("every candidate's impression_id must have a timestamp")
    # Aggregate exposures/clicks per (article, timestamp) bucket, then take strictly-earlier cumsums.
    buckets = (
        rows.groupby(["news_id", "time"], sort=True)["clicked"]
        .agg(n="size", c="sum")
        .reset_index()
    )
    g = buckets.groupby("news_id", sort=False)
    buckets["past_impressions"] = g["n"].cumsum() - buckets["n"]
    buckets["past_clicks"] = g["c"].cumsum() - buckets["c"]
    first_time = g["time"].transform("min")
    age = (buckets["time"] - first_time).dt.total_seconds() / 3600.0
    buckets["hours_since_first_seen"] = age.where(buckets["past_impressions"] > 0, 0.0)
    cols = ["past_impressions", "past_clicks", "hours_since_first_seen"]
    joined = rows.reset_index(drop=True).merge(
        buckets[["news_id", "time", *cols]], on=["news_id", "time"], how="left", sort=False,
    )  # a left merge keeps the left row order
    if len(joined) != len(candidates) or joined[cols].isna().any().any():
        raise AssertionError("point-in-time join lost or duplicated rows")
    out = candidates.copy()
    out["past_impressions"] = joined["past_impressions"].to_numpy(dtype="int64")
    out["past_clicks"] = joined["past_clicks"].to_numpy(dtype="int64")
    out["hours_since_first_seen"] = joined["hours_since_first_seen"].to_numpy(dtype="float64")
    return out


@dataclass(frozen=True)
class BetaPrior:
    alpha: float
    beta: float
    fitted_on: str = "train"
    n_articles_used: int = 0

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> BetaPrior:
        return cls(**d)


def fit_beta_prior(
    impressions: np.ndarray,
    clicks: np.ndarray,
    min_impressions: int = 20,
    min_strength: float = 2.0,
    max_strength: float = 10_000.0,
) -> BetaPrior:
    """Method-of-moments Beta prior from per-article (impressions, clicks) totals of the training period."""
    impressions = np.asarray(impressions, dtype=float)
    clicks = np.asarray(clicks, dtype=float)
    total_n = impressions.sum()
    if total_n <= 0:
        raise ValueError("cannot fit a CTR prior without training impressions")
    mu = float(np.clip(clicks.sum() / total_n, 1e-6, 1 - 1e-6))
    mask = impressions >= min_impressions
    strength = max_strength
    if mask.sum() >= 2:
        rates = clicks[mask] / impressions[mask]
        w = impressions[mask]
        var = float(np.average((rates - mu) ** 2, weights=w))
        # subtract expected binomial sampling variance so the prior reflects between-article variance
        sampling_var = float(np.average(mu * (1 - mu) / impressions[mask], weights=w))
        between = var - sampling_var
        if between > 0:
            strength = mu * (1 - mu) / between - 1
    strength = float(np.clip(strength, min_strength, max_strength))
    return BetaPrior(alpha=mu * strength, beta=(1 - mu) * strength, n_articles_used=int(mask.sum()))


def smoothed_ctr(past_clicks: np.ndarray, past_impressions: np.ndarray, prior: BetaPrior) -> np.ndarray:
    c = np.asarray(past_clicks, dtype=float)
    n = np.asarray(past_impressions, dtype=float)
    return (c + prior.alpha) / (n + prior.alpha + prior.beta)


@dataclass
class PopularitySnapshot:
    """Cumulative per-article counters as of ``as_of`` (end of the processed log) for serving."""

    table: pd.DataFrame  # news_id, impressions, clicks, first_seen_time, last_seen_time
    as_of: pd.Timestamp

    @classmethod
    def from_log(cls, candidates: pd.DataFrame, impression_times: pd.Series) -> PopularitySnapshot:
        rows = candidates[["impression_id", "news_id", "clicked"]].copy()
        rows["time"] = rows["impression_id"].map(impression_times)
        table = rows.groupby("news_id").agg(
            impressions=("clicked", "size"), clicks=("clicked", "sum"),
            first_seen_time=("time", "min"), last_seen_time=("time", "max"),
        ).reset_index()
        return cls(table=table, as_of=pd.Timestamp(rows["time"].max()))

    def lookup(self, news_ids: list[str], reference_time: pd.Timestamp) -> dict[str, np.ndarray]:
        """Counters for news_ids; unseen articles get zeros (the prior supplies their CTR)."""
        t = self.table.set_index("news_id").reindex(news_ids)
        imps = t["impressions"].fillna(0).to_numpy(dtype=float)
        clicks = t["clicks"].fillna(0).to_numpy(dtype=float)
        first = t["first_seen_time"]
        age = ((reference_time - first).dt.total_seconds() / 3600.0).fillna(0.0).clip(lower=0.0)
        return {"past_impressions": imps, "past_clicks": clicks,
                "hours_since_first_seen": age.to_numpy(dtype=float)}

    def save(self, path: str | Path) -> None:
        df = self.table.copy()
        df.attrs = {}
        df.to_parquet(path, index=False)
        Path(str(path) + ".as_of").write_text(str(self.as_of))

    @classmethod
    def load(cls, path: str | Path) -> PopularitySnapshot:
        table = pd.read_parquet(path)
        as_of = pd.Timestamp(Path(str(path) + ".as_of").read_text().strip())
        return cls(table=table, as_of=as_of)
