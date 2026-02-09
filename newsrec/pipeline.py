"""Offline pipeline stages: prepare -> embed -> train (+ evaluate) -> artifacts for the API."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from newsrec import __version__
from newsrec.config import PipelineConfig, ProfileConfig
from newsrec.data.mind import LoadReport, explode_candidates, load_behaviors, load_news
from newsrec.data.sampling import chronological_split, sample_impressions, split_statistics
from newsrec.embeddings.cache import DiskEmbeddingCache, EmbeddingCache, RedisEmbeddingCache
from newsrec.embeddings.encoder import ArticleEmbeddings, build_article_embeddings, make_encoder
from newsrec.embeddings.projection import fit_projection
from newsrec.evaluation.metrics import (
    METRIC_DEFINITIONS,
    classification_metrics,
    per_impression_metrics,
    summarize_ranking,
)
from newsrec.features.builder import ArticleMeta, FeatureBuilder
from newsrec.features.popularity import PopularitySnapshot, fit_beta_prior, point_in_time_counts
from newsrec.features.profiles import append_unique, augment_histories
from newsrec.features.schema import ABLATIONS, make_feature_spec
from newsrec.models.artifacts import save_bundle
from newsrec.models.outputs import RankingScores
from newsrec.models.training import (
    train_lambdarank,
    train_lgbm_ctr,
    train_logreg,
)
from newsrec.synthetic import is_synthetic

logger = logging.getLogger(__name__)


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str))


# --------------------------------------------------------------------------------------------- prepare
def prepare(news_path: str | Path, behaviors_path: str | Path, out_dir: str | Path,
            cfg: PipelineConfig = PipelineConfig()) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    load_report = LoadReport()
    news = load_news(news_path)
    behaviors = load_behaviors(behaviors_path, load_report)
    candidates = explode_candidates(behaviors)
    behaviors, candidates, sample_report = sample_impressions(
        behaviors, candidates, cfg.sampling.max_impressions, cfg.sampling.max_candidate_rows, cfg.sampling.seed)
    split, split_bounds = chronological_split(behaviors, cfg.split.val_frac, cfg.split.test_frac)
    behaviors["split"] = split
    times = behaviors.set_index("impression_id")["time"]
    candidates = point_in_time_counts(candidates, times)
    if cfg.profile.augment_with_log_clicks:
        behaviors["model_history"] = augment_histories(behaviors, candidates)
    else:
        behaviors["model_history"] = behaviors["history"]
    behaviors = behaviors.drop(columns=["impressions"])
    # canonical order: by split, then time, then impression id -> contiguous, chronological groups
    behaviors = behaviors.sort_values(["time", "impression_id"], kind="mergesort").reset_index(drop=True)
    rank = pd.Series(np.arange(len(behaviors)), index=behaviors["impression_id"])
    candidates = candidates.assign(_r=candidates["impression_id"].map(rank)).sort_values(
        ["_r"], kind="mergesort").drop(columns="_r").reset_index(drop=True)
    unknown = set(candidates["news_id"]) - set(news["news_id"])
    if unknown:
        logger.warning("%d candidate news_ids are missing from news.tsv", len(unknown))
    behaviors.to_parquet(out / "impressions.parquet", index=False)
    candidates.to_parquet(out / "candidates.parquet", index=False)
    news.to_parquet(out / "news.parquet", index=False)
    report = {
        "data_source": "synthetic" if is_synthetic(news_path) else "mind",
        "news_path": str(news_path), "behaviors_path": str(behaviors_path),
        "load": asdict(load_report), "sampling": sample_report, "split_boundaries": split_bounds,
        "split_statistics": split_statistics(behaviors, candidates),
        "candidate_news_missing_from_catalog": len(unknown),
        "config": cfg.to_dict(), "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(out / "prepare_report.json", report)
    return report


# ----------------------------------------------------------------------------------------------- embed
def embed(processed_dir: str | Path, backend: str = "minilm", cache_dir: str | Path | None = None,
          redis_url: str | None = None, **encoder_kwargs) -> dict[str, Any]:
    processed = Path(processed_dir)
    news = pd.read_parquet(processed / "news.parquet")
    encoder = make_encoder(backend, **encoder_kwargs)
    cache: EmbeddingCache | None = None
    disk: DiskEmbeddingCache | None = None
    if redis_url:
        import redis

        cache = RedisEmbeddingCache(redis.Redis.from_url(redis_url, socket_timeout=5))
    elif cache_dir:
        disk = DiskEmbeddingCache(cache_dir)
        cache = disk
    t0 = time.perf_counter()
    emb, stats = build_article_embeddings(news, encoder, cache)
    if disk is not None:
        disk.flush()
    emb.save(processed / "article_embeddings.npz")
    report = {**stats, "model_id": emb.model_id, "dim": emb.dim, "backend": backend,
              "seconds": round(time.perf_counter() - t0, 2),
              "cache": "redis" if redis_url else ("disk" if cache_dir else "none")}
    _write_json(processed / "embed_report.json", report)
    return report


# ----------------------------------------------------------------------------------------------- train
def split_frame(impressions: pd.DataFrame, candidates: pd.DataFrame, name: str) -> tuple[pd.DataFrame, dict]:
    imp = impressions[impressions["split"] == name]
    rows = candidates[candidates["impression_id"].isin(imp["impression_id"])].rename(
        columns={"impression_id": "group_id"})
    hist = dict(zip(imp["impression_id"], imp["model_history"].map(list)))
    return rows.reset_index(drop=True), hist


def _serving_histories(impressions: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    """Latest known history per user at the end of the log (seed for the API's durable store)."""
    last = impressions.sort_values(["time", "impression_id"]).groupby("user_id").tail(1)
    clicks = candidates[candidates["clicked"] == 1].groupby("impression_id")["news_id"].agg(list)
    out = []
    for row in last.itertuples(index=False):
        out.append((row.user_id, append_unique(list(row.model_history), clicks.get(row.impression_id, []))))
    return pd.DataFrame(out, columns=["user_id", "history"])


def _rank_eval(scores, y, g, cfg) -> dict[str, Any]:
    per = per_impression_metrics(scores, y, g, ks=tuple(sorted({5, 10, *cfg.train.ndcg_eval_at})),
                                 seed=cfg.train.seed)
    return summarize_ranking(per, n_boot=cfg.train.bootstrap_samples, seed=cfg.train.seed)


def train(processed_dir: str | Path, artifacts_root: str | Path, version: str | None = None,
          cfg: PipelineConfig = PipelineConfig()) -> dict[str, Any]:
    processed = Path(processed_dir)
    prep = json.loads((processed / "prepare_report.json").read_text())
    impressions = pd.read_parquet(processed / "impressions.parquet")
    candidates = pd.read_parquet(processed / "candidates.parquet")
    news = pd.read_parquet(processed / "news.parquet")
    emb_path = processed / "article_embeddings.npz"
    if not emb_path.exists():
        raise FileNotFoundError(f"{emb_path} not found; run `python -m newsrec embed` first")
    embeddings = ArticleEmbeddings.load(emb_path)
    version = version or datetime.now(timezone.utc).strftime("v%Y%m%dT%H%M%SZ")
    t0 = time.perf_counter()

    # Fit everything that is "learned from data" on the TRAINING period only.
    tr_rows, tr_hist = split_frame(impressions, candidates, "train")
    va_rows, va_hist = split_frame(impressions, candidates, "val")
    te_rows, te_hist = split_frame(impressions, candidates, "test")
    if min(len(tr_rows), len(va_rows), len(te_rows)) == 0:
        raise ValueError("each split needs at least one impression")
    train_articles = set(tr_rows["news_id"]) | {n for h in tr_hist.values() for n in h}
    vecs, known = embeddings.lookup(sorted(train_articles))
    projection = fit_projection(vecs[known], cfg.embedding_features.mode, cfg.embedding_features.pca_components)
    totals = tr_rows.groupby("news_id")["clicked"].agg(["size", "sum"])
    prior = fit_beta_prior(totals["size"].to_numpy(), totals["sum"].to_numpy(),
                           cfg.prior.min_impressions_for_fit, cfg.prior.min_strength, cfg.prior.max_strength)
    spec = make_feature_spec(embeddings.model_id, projection.mode, projection.output_dim)
    builder = FeatureBuilder(spec, embeddings, projection, prior, ArticleMeta.from_news(news), cfg.profile)

    X_tr = builder.build(tr_rows, tr_hist)
    X_va = builder.build(va_rows, va_hist)
    X_te = builder.build(te_rows, te_hist)
    y_tr, y_va, y_te = (r["clicked"].to_numpy(dtype=int) for r in (tr_rows, va_rows, te_rows))
    g_tr, g_va, g_te = (r["group_id"].to_numpy() for r in (tr_rows, va_rows, te_rows))
    names = spec.names
    tc = cfg.train

    logreg = train_logreg(X_tr, y_tr, names, tc)
    ctr_lgb = train_lgbm_ctr(X_tr, y_tr, X_va, y_va, names, tc)
    ranker = train_lambdarank(X_tr, y_tr, g_tr, X_va, y_va, g_va, names, tc)

    # ---------------- evaluation on the untouched test period ----------------
    rankers: dict[str, dict[str, Any]] = {}
    rng = np.random.default_rng(tc.seed)
    rankers["baseline_random"] = _rank_eval(RankingScores(rng.random(len(y_te))), y_te, g_te, cfg)
    pop_col = names.index("past_ctr_smoothed")
    rankers["baseline_past_popularity"] = _rank_eval(RankingScores(X_te[:, pop_col]), y_te, g_te, cfg)
    rankers["lambdarank_full"] = _rank_eval(ranker.score(X_te), y_te, g_te, cfg)
    rankers["ctr_logreg_as_ranker"] = _rank_eval(logreg.predict_proba(X_te), y_te, g_te, cfg)
    rankers["ctr_lightgbm_as_ranker"] = _rank_eval(ctr_lgb.predict_proba(X_te), y_te, g_te, cfg)
    ablation_info = {}
    if tc.run_ablations:
        for ab_name, groups in ABLATIONS.items():
            if ab_name == "full":
                continue
            idx = spec.indices(groups)
            if not idx:
                continue
            sub = [names[i] for i in idx]
            m = train_lambdarank(X_tr[:, idx], y_tr, g_tr, X_va[:, idx], y_va, g_va, sub, tc)
            rankers[f"lambdarank_{ab_name}"] = _rank_eval(m.score(X_te[:, idx]), y_te, g_te, cfg)
            ablation_info[ab_name] = {"features": sub, "best_iteration": m.booster.best_iteration}

    classifiers = {
        "logistic_regression": {"val": classification_metrics(logreg.predict_proba(X_va), y_va),
                                "test": classification_metrics(logreg.predict_proba(X_te), y_te)},
        "lightgbm_binary": {"val": classification_metrics(ctr_lgb.predict_proba(X_va), y_va),
                            "test": classification_metrics(ctr_lgb.predict_proba(X_te), y_te),
                            "best_iteration": ctr_lgb.booster.best_iteration},
    }
    importance = pd.DataFrame({
        "feature": names,
        "gain": ranker.booster.feature_importance("gain"),
        "split": ranker.booster.feature_importance("split"),
    }).sort_values("gain", ascending=False)

    data_source = prep["data_source"]
    report = {
        "artifact_version": version,
        "data_source": data_source,
        "warning": ("SYNTHETIC DATA: software smoke test only - not a model evaluation on MIND."
                    if data_source == "synthetic" else None),
        "split_boundaries": prep["split_boundaries"],
        "split_statistics": prep["split_statistics"],
        "sampling": prep["sampling"],
        "embedding_model_id": embeddings.model_id,
        "projection": {"mode": projection.mode, "output_dim": projection.output_dim,
                       "explained_variance": (float(projection.explained_variance_ratio.sum())
                                              if projection.explained_variance_ratio is not None else None)},
        "popularity_prior": prior.to_dict(),
        "metric_definitions": METRIC_DEFINITIONS,
        "evaluated_on": "test split (chronologically last); validation used only for early stopping",
        "rankers": rankers,
        "classifiers": classifiers,
        "lambdarank_best_iteration": ranker.booster.best_iteration,
        "ablations": ablation_info,
        "feature_importance_gain_top15": importance.head(15).to_dict(orient="records"),
        "train_seconds": round(time.perf_counter() - t0, 2),
    }

    snapshot = PopularitySnapshot.from_log(candidates, impressions.set_index("impression_id")["time"])
    manifest = {
        "artifact_version": version, "newsrec_version": __version__, "data_source": data_source,
        "created_at": datetime.now(timezone.utc).isoformat(), "config": cfg.to_dict(),
        "split_boundaries": prep["split_boundaries"], "popularity_prior": prior.to_dict(),
        "popularity_snapshot_as_of": str(snapshot.as_of), "lightgbm_version": lgb.__version__,
        "models": {"lambdarank": {"objective": "lambdarank", "output": "raw ranking score",
                                  "best_iteration": ranker.booster.best_iteration},
                   "ctr_lightgbm": {"objective": "binary", "output": "uncalibrated click probability"},
                   "ctr_logreg": {"output": "uncalibrated click probability"}},
    }
    art_dir = Path(artifacts_root) / version
    save_bundle(art_dir, manifest=manifest, spec=spec, embeddings=embeddings, projection=projection,
                articles=news, popularity=snapshot, ranker=ranker, ctr_lightgbm=ctr_lgb, ctr_logreg=logreg,
                user_histories=_serving_histories(impressions, candidates))
    _write_json(art_dir / "evaluation.json", report)
    _ranker_table(rankers).to_csv(art_dir / "evaluation_rankers.csv", index=False)
    (Path(artifacts_root) / "CURRENT").write_text(version)
    return report


def _ranker_table(rankers: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for name, r in rankers.items():
        row = {"model": name, "impressions_evaluated": r["impressions_evaluated"],
               "no_click_impression_share": r["no_click_impression_share"]}
        for k, v in r.items():
            if isinstance(v, dict) and "mean" in v:
                row[k] = v["mean"]
                row[f"{k}_ci_low"], row[f"{k}_ci_high"] = v["ci95"]
        rows.append(row)
    return pd.DataFrame(rows)


# -------------------------------------------------------------------------------------------- evaluate
def evaluate(artifact_dir: str | Path, processed_dir: str | Path, split: str = "test") -> dict[str, Any]:
    """Re-score a split with a SAVED bundle through the serving feature path (load_bundle)."""
    from newsrec.models.artifacts import load_bundle

    bundle = load_bundle(artifact_dir)
    processed = Path(processed_dir)
    impressions = pd.read_parquet(processed / "impressions.parquet")
    candidates = pd.read_parquet(processed / "candidates.parquet")
    rows, hist = split_frame(impressions, candidates, split)
    X = bundle.feature_builder().build(rows, hist)
    y, g = rows["clicked"].to_numpy(dtype=int), rows["group_id"].to_numpy()
    cfg = PipelineConfig()
    out = {
        "artifact_version": bundle.version, "data_source": bundle.manifest["data_source"], "split": split,
        "lambdarank": _rank_eval(bundle.ranker.score(X), y, g, cfg),
        "lightgbm_binary": classification_metrics(bundle.ctr_lightgbm.predict_proba(X), y),
        "logistic_regression": classification_metrics(bundle.ctr_logreg.predict_proba(X), y),
    }
    _write_json(Path(artifact_dir) / f"evaluation_rerun_{split}.json", out)
    return out


def profile_config_from_manifest(manifest: dict) -> ProfileConfig:
    return ProfileConfig(**manifest["config"]["profile"])
