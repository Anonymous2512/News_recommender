"""Versioned model bundle: everything the API needs, saved together and validated on load.

Layout of ``artifacts/<version>/``::

    manifest.json              versions, config, split boundaries, data source, fingerprints
    feature_spec.json          ordered feature definitions
    models/lambdarank.txt      LightGBM ranker (raw scores)
    models/ctr_lightgbm.txt    LightGBM binary classifier (click probabilities)
    models/ctr_logreg.joblib   scikit-learn pipeline (click probabilities)
    article_embeddings.npz     one vector per news_id (+ encoder model id, text checksums)
    projection.npz             fitted embedding projection
    articles.parquet           article metadata (news_id, title, category, subcategory)
    popularity_snapshot.parquet(+.as_of)  serving counters as of the end of the processed log
    user_histories.parquet     latest MIND history per user (seed for the API's durable store)
    evaluation.json            offline evaluation report
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import pandas as pd

from newsrec.config import ProfileConfig
from newsrec.embeddings.encoder import ArticleEmbeddings
from newsrec.embeddings.projection import EmbeddingProjection
from newsrec.features.builder import ArticleMeta, FeatureBuilder
from newsrec.features.popularity import BetaPrior, PopularitySnapshot
from newsrec.features.schema import FeatureSpec
from newsrec.models.training import LambdaRanker, LightGBMCTR, LogRegCTR

ARTIFACT_FORMAT = 1
REQUIRED_FILES = (
    "manifest.json", "feature_spec.json", "models/lambdarank.txt", "models/ctr_lightgbm.txt",
    "models/ctr_logreg.joblib", "article_embeddings.npz", "projection.npz", "articles.parquet",
    "popularity_snapshot.parquet", "popularity_snapshot.parquet.as_of",
)


class ArtifactError(RuntimeError):
    """Missing, incomplete or incompatible model artifacts."""


@dataclass
class ModelBundle:
    path: Path
    manifest: dict[str, Any]
    spec: FeatureSpec
    embeddings: ArticleEmbeddings
    projection: EmbeddingProjection
    prior: BetaPrior
    articles: pd.DataFrame
    popularity: PopularitySnapshot
    ranker: LambdaRanker
    ctr_lightgbm: LightGBMCTR
    ctr_logreg: LogRegCTR
    profile_config: ProfileConfig

    @property
    def version(self) -> str:
        return str(self.manifest["artifact_version"])

    def feature_builder(self) -> FeatureBuilder:
        return FeatureBuilder(self.spec, self.embeddings, self.projection, self.prior,
                              ArticleMeta.from_news(self.articles), self.profile_config)


def save_bundle(path: Path, *, manifest: dict, spec: FeatureSpec, embeddings: ArticleEmbeddings,
                projection: EmbeddingProjection, articles: pd.DataFrame, popularity: PopularitySnapshot,
                ranker: LambdaRanker, ctr_lightgbm: LightGBMCTR, ctr_logreg: LogRegCTR,
                user_histories: pd.DataFrame | None = None) -> None:
    (path / "models").mkdir(parents=True, exist_ok=True)
    ranker.booster.save_model(str(path / "models/lambdarank.txt"), num_iteration=ranker.booster.best_iteration)
    ctr_lightgbm.booster.save_model(str(path / "models/ctr_lightgbm.txt"),
                                    num_iteration=ctr_lightgbm.booster.best_iteration)
    joblib.dump({"pipeline": ctr_logreg.pipeline, "feature_names": ctr_logreg.feature_names},
                path / "models/ctr_logreg.joblib")
    embeddings.save(path / "article_embeddings.npz")
    projection.save(path / "projection.npz")
    articles[["news_id", "title", "category", "subcategory"]].to_parquet(path / "articles.parquet", index=False)
    popularity.save(path / "popularity_snapshot.parquet")
    if user_histories is not None:
        user_histories.to_parquet(path / "user_histories.parquet", index=False)
    (path / "feature_spec.json").write_text(json.dumps(spec.to_dict(), indent=2))
    manifest = {**manifest, "artifact_format": ARTIFACT_FORMAT, "feature_fingerprint": spec.fingerprint(),
                "embedding_model_id": embeddings.model_id}
    (path / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))


def resolve_artifact_dir(root: str | Path, version: str | None = None) -> Path:
    root = Path(root)
    if version:
        return root / version
    current = root / "CURRENT"
    if current.exists():
        return root / current.read_text().strip()
    if (root / "manifest.json").exists():
        return root
    raise ArtifactError(f"no artifact version given and {current} not found")


def load_bundle(path: str | Path) -> ModelBundle:
    path = Path(path)
    missing = [f for f in REQUIRED_FILES if not (path / f).exists()]
    if missing:
        raise ArtifactError(f"artifact directory {path} is missing: {', '.join(missing)}. "
                            "Run `python -m newsrec train ...` first.")
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest.get("artifact_format") != ARTIFACT_FORMAT:
        raise ArtifactError(f"unsupported artifact_format {manifest.get('artifact_format')}")
    try:
        spec = FeatureSpec.from_dict(json.loads((path / "feature_spec.json").read_text()))
    except ValueError as e:
        raise ArtifactError(str(e)) from e
    if spec.fingerprint() != manifest.get("feature_fingerprint"):
        raise ArtifactError("feature spec does not match manifest fingerprint")
    embeddings = ArticleEmbeddings.load(path / "article_embeddings.npz")
    if embeddings.model_id != spec.embedding_model_id:
        raise ArtifactError(f"embeddings were built with {embeddings.model_id!r} but the model expects "
                            f"{spec.embedding_model_id!r}")
    projection = EmbeddingProjection.load(path / "projection.npz")
    ranker = LambdaRanker(lgb.Booster(model_file=str(path / "models/lambdarank.txt")))
    ctr = LightGBMCTR(lgb.Booster(model_file=str(path / "models/ctr_lightgbm.txt")))
    lr_obj = joblib.load(path / "models/ctr_logreg.joblib")
    logreg = LogRegCTR(pipeline=lr_obj["pipeline"], feature_names=lr_obj["feature_names"])
    for name, names in (("lambdarank", ranker.booster.feature_name()),
                        ("ctr_lightgbm", ctr.booster.feature_name()),
                        ("ctr_logreg", logreg.feature_names)):
        if list(names) != spec.names:
            raise ArtifactError(f"{name} feature order does not match feature_spec.json")
    bundle = ModelBundle(
        path=path, manifest=manifest, spec=spec, embeddings=embeddings, projection=projection,
        prior=BetaPrior.from_dict(manifest["popularity_prior"]),
        articles=pd.read_parquet(path / "articles.parquet"),
        popularity=PopularitySnapshot.load(path / "popularity_snapshot.parquet"),
        ranker=ranker, ctr_lightgbm=ctr, ctr_logreg=logreg,
        profile_config=ProfileConfig(**manifest["config"]["profile"]),
    )
    bundle.feature_builder()  # validates spec/projection/embedding compatibility
    return bundle
