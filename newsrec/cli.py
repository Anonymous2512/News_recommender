"""Command line: ``python -m newsrec <command>``.

Commands: synth, prepare, embed, train, evaluate, import-catalog, demo.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from newsrec.config import (
    EmbeddingFeatureConfig,
    PipelineConfig,
    ProfileConfig,
    SamplingConfig,
    SplitConfig,
    TrainConfig,
)


def _cfg(args: argparse.Namespace) -> PipelineConfig:
    return PipelineConfig(
        split=SplitConfig(val_frac=args.val_frac, test_frac=args.test_frac),
        sampling=SamplingConfig(max_impressions=args.max_impressions, max_candidate_rows=args.max_candidate_rows,
                                seed=args.seed),
        profile=ProfileConfig(max_history=args.max_history, half_life=args.half_life,
                              augment_with_log_clicks=not args.no_log_click_history),
        embedding_features=EmbeddingFeatureConfig(mode=args.emb_features, pca_components=args.pca_components),
        train=TrainConfig(seed=args.seed, bootstrap_samples=args.bootstrap, run_ablations=not args.no_ablations),
    )


def _add_cfg_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--test-frac", type=float, default=0.2)
    p.add_argument("--max-impressions", type=int, default=None, help="sample whole impressions")
    p.add_argument("--max-candidate-rows", type=int, default=None, help="row budget; slates kept whole")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-history", type=int, default=50)
    p.add_argument("--half-life", type=float, default=10.0)
    p.add_argument("--no-log-click-history", action="store_true")
    p.add_argument("--emb-features", choices=["pca", "full", "none"], default="pca")
    p.add_argument("--pca-components", type=int, default=32)
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--no-ablations", action="store_true")


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(prog="newsrec")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("synth", help="write SYNTHETIC MIND-format data (tests/demo only)")
    s.add_argument("--out", default="data/synthetic")
    s.add_argument("--impressions", type=int, default=2500)
    s.add_argument("--seed", type=int, default=7)

    p = sub.add_parser("prepare", help="parse MIND, sample whole impressions, split by time, past-only stats")
    p.add_argument("--news", required=True)
    p.add_argument("--behaviors", required=True)
    p.add_argument("--out", default="data/processed")
    _add_cfg_args(p)

    e = sub.add_parser("embed", help="one embedding per unique article (versioned cache)")
    e.add_argument("--processed", default="data/processed")
    e.add_argument("--backend", choices=["minilm", "hashing"], default="minilm")
    e.add_argument("--device", default=None)
    e.add_argument("--batch-size", type=int, default=128)
    e.add_argument("--cache-dir", default="data/cache/embeddings")
    e.add_argument("--redis-url", default=None, help="use Redis as the shared embedding cache")

    t = sub.add_parser("train", help="train LogReg + LightGBM CTR classifiers and LambdaRank; evaluate")
    t.add_argument("--processed", default="data/processed")
    t.add_argument("--artifacts", default="artifacts")
    t.add_argument("--version", default=None)
    _add_cfg_args(t)

    v = sub.add_parser("evaluate", help="re-score a split with a saved artifact bundle")
    v.add_argument("--artifact-dir", required=True)
    v.add_argument("--processed", default="data/processed")
    v.add_argument("--split", choices=["val", "test"], default="test")

    i = sub.add_parser("import-catalog", help="load articles + seed user histories into the API database")
    i.add_argument("--artifacts", default="artifacts")
    i.add_argument("--version", default=None)
    i.add_argument("--database-url", default="sqlite:///data/newsrec.db")

    d = sub.add_parser("demo", help="synth -> prepare -> embed(hashing) -> train -> import (SYNTHETIC)")
    d.add_argument("--workdir", default="data/demo")
    d.add_argument("--artifacts", default="artifacts")
    d.add_argument("--database-url", default="sqlite:///data/newsrec.db")
    d.add_argument("--backend", choices=["minilm", "hashing"], default="hashing")
    d.add_argument("--impressions", type=int, default=2500)

    args = ap.parse_args(argv)
    from newsrec import pipeline

    if args.cmd == "synth":
        from newsrec.synthetic import SyntheticConfig, generate

        res = generate(args.out, SyntheticConfig(n_impressions=args.impressions, seed=args.seed))
    elif args.cmd == "prepare":
        res = pipeline.prepare(args.news, args.behaviors, args.out, _cfg(args))
    elif args.cmd == "embed":
        kw = {"device": args.device, "batch_size": args.batch_size} if args.backend == "minilm" else {}
        res = pipeline.embed(args.processed, args.backend, cache_dir=args.cache_dir, redis_url=args.redis_url, **kw)
    elif args.cmd == "train":
        res = pipeline.train(args.processed, args.artifacts, args.version, _cfg(args))
        res = {k: res[k] for k in ("artifact_version", "data_source", "warning", "rankers", "split_boundaries")}
    elif args.cmd == "evaluate":
        res = pipeline.evaluate(args.artifact_dir, args.processed, args.split)
    elif args.cmd == "import-catalog":
        from app.repositories.store import import_catalog
        from newsrec.models.artifacts import resolve_artifact_dir

        res = import_catalog(args.database_url, resolve_artifact_dir(args.artifacts, args.version))
    elif args.cmd == "demo":
        from app.repositories.store import import_catalog
        from newsrec.models.artifacts import resolve_artifact_dir
        from newsrec.synthetic import SyntheticConfig, generate

        work = Path(args.workdir)
        paths = generate(work / "raw", SyntheticConfig(n_impressions=args.impressions))
        cfg = PipelineConfig(train=TrainConfig(bootstrap_samples=200))
        pipeline.prepare(paths["news"], paths["behaviors"], work / "processed", cfg)
        pipeline.embed(work / "processed", args.backend, cache_dir=work / "cache")
        rep = pipeline.train(work / "processed", args.artifacts, None, cfg)
        imp = import_catalog(args.database_url, resolve_artifact_dir(args.artifacts))
        res = {"warning": rep["warning"], "artifact_version": rep["artifact_version"], "import": imp,
               "lambdarank_full_ndcg@10": rep["rankers"]["lambdarank_full"]["ndcg@10"],
               "baseline_random_ndcg@10": rep["rankers"]["baseline_random"]["ndcg@10"]}
    else:  # pragma: no cover
        ap.error(f"unknown command {args.cmd}")
        return 2
    json.dump(res, sys.stdout, indent=2, default=str)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
