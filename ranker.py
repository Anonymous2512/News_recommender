"""
ranker.py — backward-compatible entry point for the LightGBM ranker.

The original version sampled individual rows (breaking impression slates), split randomly, reused the
validation set for reporting, and used the first 64 embedding coordinates. ``train`` now delegates to
``newsrec.pipeline.train`` (complete slates, chronological split, LambdaRank + separate CTR classifiers,
baselines, ablations, test-period evaluation). ``predict`` scores one prepared split with a saved bundle
and writes per-pool raw ranking scores for ``optimizer.py``.

    python ranker.py train   --processed data/processed --artifacts artifacts
    python ranker.py predict --artifacts artifacts --processed data/processed --split test --output ranked.parquet
"""

from __future__ import annotations

import argparse
import json
import logging

import pandas as pd

from newsrec.evaluation.metrics import dcg_at_k, ndcg_at_k, reciprocal_rank  # noqa: F401  (re-exported)
from newsrec.models.artifacts import load_bundle, resolve_artifact_dir
from newsrec.pipeline import split_frame, train


def predict(artifacts: str, processed: str, split: str, output: str, version: str | None = None) -> pd.DataFrame:
    bundle = load_bundle(resolve_artifact_dir(artifacts, version))
    impressions = pd.read_parquet(f"{processed}/impressions.parquet")
    candidates = pd.read_parquet(f"{processed}/candidates.parquet")
    rows, hist = split_frame(impressions, candidates, split)
    X = bundle.feature_builder().build(rows, hist)
    meta = bundle.articles.set_index("news_id")
    out = rows[["group_id", "news_id", "clicked", "hours_since_first_seen"]].copy()
    out["ranking_score"] = bundle.ranker.score(X).values
    out["category"] = meta["category"].reindex(out["news_id"]).fillna("").to_numpy()
    out["title"] = meta["title"].reindex(out["news_id"]).fillna("").to_numpy()
    vecs, _ = bundle.embeddings.lookup(out["news_id"].tolist())
    emb = pd.DataFrame(vecs, columns=[f"emb_{i}" for i in range(vecs.shape[1])])
    out = pd.concat([out.reset_index(drop=True), emb], axis=1)
    out["rank_position"] = out.groupby("group_id")["ranking_score"].rank(method="first", ascending=False).astype(int)
    out.to_parquet(output, index=False)
    return out


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    ap = argparse.ArgumentParser(description="LightGBM LambdaRank — MIND")
    sub = ap.add_subparsers(dest="cmd", required=True)
    tr = sub.add_parser("train")
    tr.add_argument("--processed", default="data/processed")
    tr.add_argument("--artifacts", default="artifacts")
    tr.add_argument("--version", default=None)
    pr = sub.add_parser("predict")
    pr.add_argument("--artifacts", default="artifacts")
    pr.add_argument("--version", default=None)
    pr.add_argument("--processed", default="data/processed")
    pr.add_argument("--split", choices=["train", "val", "test"], default="test")
    pr.add_argument("--output", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "train":
        rep = train(a.processed, a.artifacts, a.version)
        print(json.dumps({k: rep[k] for k in ("artifact_version", "data_source", "warning", "rankers")},
                         indent=2, default=str))
    else:
        out = predict(a.artifacts, a.processed, a.split, a.output, a.version)
        print(out[["group_id", "news_id", "ranking_score", "rank_position"]].head(20).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
