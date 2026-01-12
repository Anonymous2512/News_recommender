"""
mind_preprocessing.py — backward-compatible entry point for MIND preprocessing.

The original implementation computed article popularity over the whole sample (leaking each row's own
label and future/validation labels) and dropped abstracts/histories. It now delegates to the corrected
pipeline (``newsrec.pipeline.prepare``): whole-impression sampling, chronological train/val/test split,
past-only popularity counters and history augmentation from strictly earlier clicks.

    python mind_preprocessing.py --news MINDlarge_train/news.tsv --behaviors MINDlarge_train/behaviors.tsv \
        --output data/processed [--max-impressions 300000]
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from newsrec.config import PipelineConfig, SamplingConfig, SplitConfig
from newsrec.data.mind import explode_candidates, load_behaviors, load_news, parse_impressions  # noqa: F401
from newsrec.pipeline import prepare


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    p = argparse.ArgumentParser(description="MIND preprocessing (corrected pipeline)")
    p.add_argument("--news", required=True)
    p.add_argument("--behaviors", required=True)
    p.add_argument("--output", default="data/processed", help="output DIRECTORY")
    p.add_argument("--max-impressions", type=int, default=None)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--test-frac", type=float, default=0.2)
    a = p.parse_args(argv)
    out = Path(a.output)
    if out.suffix == ".parquet":
        logging.warning("--output is now a directory; using %s", out.parent)
        out = out.parent
    cfg = PipelineConfig(split=SplitConfig(a.val_frac, a.test_frac),
                         sampling=SamplingConfig(max_impressions=a.max_impressions))
    report = prepare(a.news, a.behaviors, out, cfg)
    print(json.dumps(report["split_statistics"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
