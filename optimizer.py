"""
optimizer.py
Front-page slot filling over ONE candidate pool (one impression or one API request).

Corrections versus the original version:
* Ranker scores are normalized *within the pool*; LambdaRank scores from different impressions are never
  compared (they are only meaningful within a slate).
* Each article appears once; the pool is deduplicated by ``news_id`` before anything else.
* No "recency" from impression-log timestamps: MIND has no publish time. An optional, default-off
  ``exposure_age_weight`` uses ``hours_since_first_seen`` and is labeled as an exposure-age proxy.
* Slot ``visibility`` is display metadata. Assigning items to slots so that sum(visibility * utility) is
  maximal is achieved by putting higher-utility items in more visible slots (rearrangement inequality), so
  visibility can never change *which* articles are selected — only per-slot constraints can. It is
  therefore not used as a ranking signal; ``display_weighted_utility`` is reported for inspection only.
* Deterministic: ties are broken by ``news_id``. Fewer slots are filled only if constraints leave fewer
  eligible articles.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

from newsrec.postrank import PostRankConfig, select_with_constraints, within_pool_minmax

logger = logging.getLogger(__name__)


class SlotZone(str, Enum):
    HERO = "hero"
    TOP_RAIL = "top_rail"
    SIDEBAR = "sidebar"
    MID_RAIL = "mid_rail"
    BOTTOM_RAIL = "bottom_rail"
    BELOW_FOLD = "below_fold"


@dataclass(frozen=True)
class LayoutSlot:
    slot_id: int
    zone: SlotZone
    position: int
    visibility: float  # hand-set display weight; NOT estimated from data (MIND has no positions)


def _zone(zone: SlotZone, start: int, weights: list[float]) -> list[LayoutSlot]:
    return [LayoutSlot(start + i, zone, i + 1, w) for i, w in enumerate(weights)]


DEFAULT_LAYOUT: list[LayoutSlot] = (
    _zone(SlotZone.HERO, 0, [1.0])
    + _zone(SlotZone.TOP_RAIL, 1, [0.82, 0.76, 0.70, 0.64])
    + _zone(SlotZone.SIDEBAR, 5, [0.58, 0.52, 0.46, 0.40])
    + _zone(SlotZone.MID_RAIL, 9, [0.38, 0.34, 0.30, 0.26])
    + _zone(SlotZone.BOTTOM_RAIL, 13, [0.20, 0.175, 0.15, 0.125])
    + _zone(SlotZone.BELOW_FOLD, 17, [0.08, 0.06, 0.04])
)


@dataclass
class OptimizerConfig:
    score_col: str = "ranking_score"
    category_col: str = "category"
    # optional editorial adjustments, applied to within-pool normalized components
    sentiment_weight: float = 0.0  # uses `sent_polarity` if present (feature_engineering.py output)
    clickbait_penalty: float = 0.0  # uses `cb_composite_score` if present
    exposure_age_weight: float = 0.0  # uses `hours_since_first_seen` (proxy; newer exposure = higher)
    max_per_category: int | None = 3
    dedup_threshold: float | None = 0.82
    embedding_prefix: str = "emb_"
    hero_categories_allowed: tuple[str, ...] | None = None
    sidebar_max_entertainment: int | None = 1


def composite_scores(pool: pd.DataFrame, cfg: OptimizerConfig) -> np.ndarray:
    """Utility within this pool: normalized ranker score plus optional, explicitly weighted adjustments."""
    u = within_pool_minmax(pool[cfg.score_col].to_numpy(dtype=float))
    if cfg.sentiment_weight and "sent_polarity" in pool:
        u = u + cfg.sentiment_weight * within_pool_minmax(pool["sent_polarity"].to_numpy(dtype=float))
    if cfg.clickbait_penalty and "cb_composite_score" in pool:
        u = u - cfg.clickbait_penalty * pool["cb_composite_score"].clip(0, 1).to_numpy(dtype=float)
    if cfg.exposure_age_weight and "hours_since_first_seen" in pool:
        age = np.log1p(pool["hours_since_first_seen"].clip(lower=0).to_numpy(dtype=float))
        u = u + cfg.exposure_age_weight * (1 - within_pool_minmax(age))
    return u


def _embeddings(pool: pd.DataFrame, prefix: str) -> np.ndarray:
    cols = [c for c in pool.columns if c.startswith(prefix) and c[len(prefix):].isdigit()]
    cols.sort(key=lambda c: int(c[len(prefix):]))
    if not cols:
        return np.zeros((len(pool), 1), dtype=np.float32)  # zero vectors disable dedup
    return pool[cols].to_numpy(dtype=np.float32)


def _slot_allows(slot: LayoutSlot, category: str, cfg: OptimizerConfig, placed: dict[str, int]) -> bool:
    if slot.zone == SlotZone.HERO and cfg.hero_categories_allowed and category not in cfg.hero_categories_allowed:
        return False
    if (slot.zone == SlotZone.SIDEBAR and cfg.sidebar_max_entertainment is not None and category == "entertainment"
            and placed.get("sidebar:entertainment", 0) >= cfg.sidebar_max_entertainment):
        return False
    return True


def optimize_front_page(pool: pd.DataFrame, cfg: OptimizerConfig | None = None,
                        layout: list[LayoutSlot] | None = None) -> pd.DataFrame:
    """Fill ``layout`` from a single candidate pool. Returns the slot manifest (one row per filled slot)."""
    cfg = cfg or OptimizerConfig()
    layout = sorted(layout or DEFAULT_LAYOUT, key=lambda s: s.slot_id)
    if cfg.score_col not in pool:
        raise KeyError(f"pool must contain the ranker score column {cfg.score_col!r}")
    pool = (pool.sort_values([cfg.score_col, "news_id"], ascending=[False, True], kind="mergesort")
            .drop_duplicates("news_id").reset_index(drop=True))
    utility = composite_scores(pool, cfg)
    cats = pool[cfg.category_col].fillna("").astype(str).tolist() if cfg.category_col in pool else [""] * len(pool)
    ids = pool["news_id"].astype(str).tolist()
    order = select_with_constraints(ids, utility, cats, _embeddings(pool, cfg.embedding_prefix), len(pool),
                                    PostRankConfig(cfg.dedup_threshold, cfg.max_per_category))
    remaining = list(order)
    placed: dict[str, int] = {}
    rows = []
    for slot in layout:
        pick = next((i for i in remaining if _slot_allows(slot, cats[i], cfg, placed)), None)
        if pick is None:
            logger.info("slot %d (%s) left empty: no eligible article", slot.slot_id, slot.zone.value)
            continue
        remaining.remove(pick)
        if slot.zone == SlotZone.SIDEBAR:
            placed[f"sidebar:{cats[pick]}"] = placed.get(f"sidebar:{cats[pick]}", 0) + 1
        rows.append({
            "slot_id": slot.slot_id, "zone": slot.zone.value, "position": slot.position,
            "visibility": slot.visibility, "news_id": ids[pick], "category": cats[pick],
            "title": pool.at[pick, "title"] if "title" in pool else "",
            "ranking_score": float(pool.at[pick, cfg.score_col]),
            "composite_score": float(utility[pick]),
            "display_weighted_utility": float(utility[pick] * slot.visibility),
        })
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    p = argparse.ArgumentParser(description="Front-page slot filling for ONE ranked candidate pool")
    p.add_argument("--input", required=True, help="ranked parquet (e.g. from `ranker.py predict`)")
    p.add_argument("--output", required=True)
    p.add_argument("--group-col", default="group_id")
    p.add_argument("--group-id", default=None, help="pool to use when the input holds several groups")
    p.add_argument("--max-per-category", type=int, default=3)
    p.add_argument("--dedup-threshold", type=float, default=0.82)
    p.add_argument("--sentiment-w", type=float, default=0.0)
    p.add_argument("--clickbait-pen", type=float, default=0.0)
    p.add_argument("--exposure-age-w", type=float, default=0.0)
    a = p.parse_args(argv)
    df = pd.read_parquet(a.input)
    if a.group_col in df and df[a.group_col].nunique() > 1:
        if a.group_id is None:
            p.error(f"input has {df[a.group_col].nunique()} pools; pass --group-id (scores are only "
                    "comparable within one pool)")
        df = df[df[a.group_col].astype(str) == str(a.group_id)]
    cfg = OptimizerConfig(max_per_category=a.max_per_category, dedup_threshold=a.dedup_threshold,
                          sentiment_weight=a.sentiment_w, clickbait_penalty=a.clickbait_pen,
                          exposure_age_weight=a.exposure_age_w)
    manifest = optimize_front_page(df, cfg)
    manifest.to_parquet(a.output, index=False)
    print(manifest.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
