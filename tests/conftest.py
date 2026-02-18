"""Shared fixtures. Everything runs on SYNTHETIC data with the hashing test encoder: no MIND files, no
model downloads and no live Redis are needed. Results here test software behavior only."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from newsrec import pipeline
from newsrec.config import PipelineConfig, TrainConfig
from newsrec.models.artifacts import load_bundle
from newsrec.synthetic import SyntheticConfig, generate


@pytest.fixture(scope="session")
def built(tmp_path_factory) -> dict:
    root = tmp_path_factory.mktemp("newsrec")
    paths = generate(root / "raw", SyntheticConfig(n_articles=250, n_users=150, n_impressions=1200, seed=3))
    cfg = PipelineConfig(train=TrainConfig(bootstrap_samples=50))
    pipeline.prepare(paths["news"], paths["behaviors"], root / "processed", cfg)
    pipeline.embed(root / "processed", "hashing", cache_dir=root / "cache")
    report = pipeline.train(root / "processed", root / "artifacts", "testv1", cfg)
    return {"root": root, "raw": paths, "processed": root / "processed", "artifacts": root / "artifacts",
            "artifact_dir": root / "artifacts" / "testv1", "report": report}


@pytest.fixture(scope="session")
def bundle(built):
    return load_bundle(built["artifact_dir"])


@pytest.fixture(scope="session")
def processed(built) -> dict[str, pd.DataFrame]:
    p: Path = built["processed"]
    return {"impressions": pd.read_parquet(p / "impressions.parquet"),
            "candidates": pd.read_parquet(p / "candidates.parquet"),
            "news": pd.read_parquet(p / "news.parquet")}
