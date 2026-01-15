"""Central feature schema shared by training, offline evaluation and the API.

Each feature documents meaning, when it is computed, its source, dtype, missing/default behavior and
leakage status. Models are saved with the exact ordered feature list and refuse to load on mismatch.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

FEATURE_SCHEMA_VERSION = "2"


@dataclass(frozen=True)
class FeatureDef:
    name: str
    group: str  # popularity | freshness | profile | content | embedding
    description: str
    computed_at: str
    source: str
    default: float = 0.0
    dtype: str = "float32"
    leakage: str = "safe: uses only information available before the impression time"


BASE_FEATURES: tuple[FeatureDef, ...] = (
    FeatureDef("past_impressions_log1p", "popularity",
               "log1p(#exposures of the article in impressions strictly before t)",
               "impression time t (offline replay) / snapshot as_of (serving)", "impression log", 0.0),
    FeatureDef("past_clicks_log1p", "popularity",
               "log1p(#clicks on the article in impressions strictly before t)",
               "impression time t / snapshot as_of", "impression log", 0.0),
    FeatureDef("past_ctr_smoothed", "popularity",
               "(past_clicks + alpha) / (past_impressions + alpha + beta); Beta prior fit on train period",
               "impression time t / snapshot as_of", "impression log + saved prior",
               default=0.0,  # never missing: unseen articles have 0 counts -> prior mean
               leakage="safe: past-only counts; prior = 2 global scalars fitted on the training period"),
    FeatureDef("hours_since_first_seen_log1p", "freshness",
               "log1p(hours since the article's first exposure in the log; 0 on first exposure). "
               "Exposure-age proxy, NOT publish time (MIND has none)",
               "impression time t / reference time", "impression log", 0.0),
    FeatureDef("history_length_log1p", "profile",
               "log1p(#known history items used for the profile)",
               "impression time t", "user click history", 0.0),
    FeatureDef("profile_cosine", "profile",
               "cosine(recency-weighted mean of history title embeddings, candidate title embedding)",
               "impression time t", "history + article embeddings", 0.0),
    FeatureDef("profile_max_cosine", "profile",
               "max cosine between the candidate and any of the last N history items",
               "impression time t", "history + article embeddings", 0.0),
    FeatureDef("category_affinity", "profile",
               "recency-weighted share of history items in the candidate's category",
               "impression time t", "history + article metadata", 0.0),
    FeatureDef("subcategory_affinity", "profile",
               "recency-weighted share of history items in the candidate's subcategory",
               "impression time t", "history + article metadata", 0.0),
    FeatureDef("title_word_count", "content", "number of whitespace tokens in the title",
               "article ingestion", "news metadata", 0.0),
)


@dataclass(frozen=True)
class FeatureSpec:
    features: tuple[FeatureDef, ...]
    embedding_model_id: str
    projection_mode: str
    version: str = FEATURE_SCHEMA_VERSION

    @property
    def names(self) -> list[str]:
        return [f.name for f in self.features]

    def indices(self, groups: set[str]) -> list[int]:
        return [i for i, f in enumerate(self.features) if f.group in groups]

    def fingerprint(self) -> str:
        payload = json.dumps({"names": self.names, "emb": self.embedding_model_id,
                              "proj": self.projection_mode, "v": self.version}, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def to_dict(self) -> dict:
        return {"version": self.version, "embedding_model_id": self.embedding_model_id,
                "projection_mode": self.projection_mode, "fingerprint": self.fingerprint(),
                "features": [asdict(f) for f in self.features]}

    @classmethod
    def from_dict(cls, d: dict) -> FeatureSpec:
        spec = cls(features=tuple(FeatureDef(**f) for f in d["features"]),
                   embedding_model_id=d["embedding_model_id"], projection_mode=d["projection_mode"],
                   version=d["version"])
        if d.get("fingerprint") and d["fingerprint"] != spec.fingerprint():
            raise ValueError("feature spec fingerprint mismatch")
        return spec


def make_feature_spec(embedding_model_id: str, projection_mode: str, n_embedding_features: int) -> FeatureSpec:
    emb = tuple(
        FeatureDef(f"emb_{projection_mode}_{i}", "embedding",
                   f"component {i} of the title embedding after the saved '{projection_mode}' projection",
                   "article ingestion", "article embeddings + projection fitted on train-period articles")
        for i in range(n_embedding_features)
    )
    return FeatureSpec(features=BASE_FEATURES + emb, embedding_model_id=embedding_model_id,
                       projection_mode=projection_mode)


# Ablation feature groups (used by the training report)
ABLATIONS: dict[str, set[str]] = {
    "full": {"popularity", "freshness", "profile", "content", "embedding"},
    "popularity_only": {"popularity", "freshness"},
    "content_only": {"content", "embedding"},
    "full_minus_profile": {"popularity", "freshness", "content", "embedding"},
}
