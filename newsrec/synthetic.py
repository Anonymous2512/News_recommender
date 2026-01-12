"""Deterministic SYNTHETIC data in MIND's TSV format — for tests and the demo only.

Clicks are simulated from latent user category preferences and latent article quality, so models trained
on it can be checked for *software correctness* (they should beat random). Metrics on this data say
nothing about performance on real MIND and must never be reported as such. A ``SYNTHETIC_DATA`` marker
file is written next to the TSVs; the pipeline propagates ``data_source="synthetic"`` into every artifact.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

MARKER = "SYNTHETIC_DATA"
# Curated neutral topical vocabularies (readable titles; real words give a real encoder something to use).
_TOPICS: dict[str, list[str]] = {
    "sports": ["league", "coach", "season", "playoff", "striker", "goal", "stadium", "tournament", "rookie",
               "championship", "match", "referee", "draft", "injury", "victory", "quarterback", "marathon",
               "trophy", "derby", "olympic", "pitcher", "tennis", "golf", "cycling", "swimmer"],
    "finance": ["market", "stocks", "earnings", "inflation", "bank", "investors", "bond", "rates", "budget",
                "dividend", "merger", "startup", "revenue", "economy", "shares", "forecast", "tariff",
                "pension", "lending", "currency", "profit", "fund", "trading", "mortgage", "exports"],
    "health": ["vaccine", "clinic", "nutrition", "sleep", "heart", "doctors", "study", "fitness", "hospital",
               "therapy", "diet", "allergy", "patients", "wellness", "screening", "vitamin", "nurses",
               "recovery", "exercise", "hydration", "dental", "posture", "immunity", "checkup", "stretching"],
    "travel": ["airline", "beach", "passport", "resort", "itinerary", "museum", "island", "railway", "hotel",
               "cruise", "mountain", "festival", "luggage", "airport", "hiking", "village", "coastline",
               "landmark", "roadtrip", "camping", "skyline", "lagoon", "ferry", "canyon", "tourists"],
    "technology": ["software", "smartphone", "chip", "robotics", "cloud", "battery", "browser", "startup",
                   "laptop", "satellite", "algorithm", "sensor", "network", "gadget", "update", "processor",
                   "drone", "wearable", "database", "keyboard", "camera", "streaming", "router", "tablet",
                   "console"],
    "food": ["recipe", "bakery", "chef", "pasta", "harvest", "dessert", "kitchen", "coffee", "restaurant",
             "spices", "breakfast", "vegetables", "soup", "bread", "orchard", "grill", "noodles", "cheese",
             "salad", "market", "tea", "pastry", "dinner", "flavors", "picnic"],
    "weather": ["storm", "rainfall", "forecast", "heatwave", "snow", "wind", "flooding", "drought", "frost",
                "sunshine", "humidity", "hurricane", "temperatures", "clouds", "season", "thunder", "fog",
                "monsoon", "breeze", "blizzard", "tide", "hail", "rainbow", "drizzle", "outlook"],
    "science": ["telescope", "fossil", "galaxy", "laboratory", "genome", "planet", "physics", "climate",
                "researchers", "species", "comet", "molecule", "volcano", "ocean", "experiment", "orbit",
                "particles", "glacier", "microscope", "asteroid", "reef", "dinosaur", "eclipse", "magnet",
                "quantum"],
}
_GENERIC = ["new", "report", "local", "week", "update", "today", "guide", "plans", "review", "big", "first",
            "latest", "city", "year", "best", "team", "story", "change", "record", "season", "event", "list",
            "how", "why", "inside", "major", "small", "next", "early", "late"]
_SUB_SUFFIX = ["roundup", "analysis", "spotlight", "briefing", "explainer", "feature"]


@dataclass(frozen=True)
class SyntheticConfig:
    n_categories: int = 6
    subcats_per_category: int = 3
    n_articles: int = 400
    n_users: int = 300
    n_impressions: int = 2500
    days: int = 6
    min_candidates: int = 8
    max_candidates: int = 20
    cold_start_user_share: float = 0.15
    seed: int = 7


def generate(out_dir: str | Path, cfg: SyntheticConfig = SyntheticConfig()) -> dict:
    rng = np.random.default_rng(cfg.seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if not 1 <= cfg.n_categories <= len(_TOPICS) or cfg.subcats_per_category > len(_SUB_SUFFIX):
        raise ValueError(f"n_categories must be 1..{len(_TOPICS)}, subcats_per_category <= {len(_SUB_SUFFIX)}")
    cats = list(_TOPICS)[: cfg.n_categories]
    cat_vocab = {c: _TOPICS[c] for c in cats}
    # each subcategory gets a disjoint slice of its category's words
    sub_vocab = {(c, s): _TOPICS[c][s::cfg.subcats_per_category] for c in cats
                 for s in range(cfg.subcats_per_category)}
    generic = _GENERIC
    start = pd.Timestamp("2019-11-09 00:00:00")
    end = start + pd.Timedelta(days=cfg.days)

    # articles
    a_cat = rng.integers(0, cfg.n_categories, cfg.n_articles)
    a_sub = rng.integers(0, cfg.subcats_per_category, cfg.n_articles)
    quality = rng.normal(0, 1, cfg.n_articles)
    publish = start - pd.Timedelta(days=2) + pd.to_timedelta(
        rng.uniform(0, (cfg.days + 2) * 86400, cfg.n_articles), unit="s")
    rows = []
    for i in range(cfg.n_articles):
        c = cats[a_cat[i]]
        words = list(dict.fromkeys(  # no repeated words in a title
            list(rng.choice(cat_vocab[c], 3, replace=False)) + list(rng.choice(sub_vocab[(c, a_sub[i])], 2, replace=False))
            + list(rng.choice(generic, rng.integers(1, 4), replace=False))))
        rng.shuffle(words)
        title = " ".join(words).capitalize()
        if i % 97 == 0:
            title = ""  # a few empty titles exercise the missing-title path
        rows.append((f"N{10000 + i}", c, f"{c}_{_SUB_SUFFIX[a_sub[i]]}", title, "", "", "[]", "[]"))
    news = pd.DataFrame(rows)
    news.to_csv(out / "news.tsv", sep="\t", header=False, index=False)

    # users: concentrated category preferences, base history from articles published before start
    prefs = rng.dirichlet(np.full(cfg.n_categories, 0.35), cfg.n_users)
    old = np.flatnonzero(publish < start)
    histories: list[list[str]] = []
    for u in range(cfg.n_users):
        if rng.random() < cfg.cold_start_user_share or old.size == 0:
            histories.append([])
            continue
        w = prefs[u][a_cat[old]] + 1e-3
        k = int(rng.integers(3, 25))
        picks = rng.choice(old, size=min(k, old.size), replace=False, p=w / w.sum())
        picks = picks[np.argsort(publish[picks])]  # ordered by time, oldest first
        histories.append([f"N{10000 + p}" for p in picks])

    # impressions
    times = np.sort(rng.uniform(0, cfg.days * 86400, cfg.n_impressions))
    beh = []
    for imp, ts in enumerate(times, start=1):
        t = start + pd.Timedelta(seconds=float(ts))
        live = np.flatnonzero((publish <= t) & (publish > t - pd.Timedelta(hours=48)))
        if live.size < cfg.min_candidates:
            live = np.flatnonzero(publish <= t)
        u = int(rng.integers(0, cfg.n_users))
        n = min(int(live.size), int(rng.integers(cfg.min_candidates, cfg.max_candidates + 1)))
        cand = rng.choice(live, size=n, replace=False)
        logit = -3.0 + 4.0 * prefs[u][a_cat[cand]] + 0.8 * quality[cand]
        clicked = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
        imp_str = " ".join(f"N{10000 + c}-{y}" for c, y in zip(cand, clicked))
        beh.append((imp, f"U{u}", t.strftime("%m/%d/%Y %I:%M:%S %p"), " ".join(histories[u]), imp_str))
    pd.DataFrame(beh).to_csv(out / "behaviors.tsv", sep="\t", header=False, index=False)
    (out / MARKER).write_text(
        "This directory contains SYNTHETIC data generated by newsrec.synthetic for tests/demo only.\n"
        "It is not Microsoft MIND data. Do not report metrics computed on it as real results.\n")
    return {"news": str(out / "news.tsv"), "behaviors": str(out / "behaviors.tsv"), "end": str(end),
            "config": cfg.__dict__}


def is_synthetic(news_path: str | Path) -> bool:
    return (Path(news_path).parent / MARKER).exists()
