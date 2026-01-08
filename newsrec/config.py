"""Pipeline configuration. Every knob that affects features or models is recorded in the artifact manifest."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class SplitConfig:
    # Fractions of impressions (ordered by time) assigned to validation and test.
    val_frac: float = 0.1
    test_frac: float = 0.2

    def __post_init__(self) -> None:
        if not (0 < self.val_frac < 1 and 0 < self.test_frac < 1 and self.val_frac + self.test_frac < 1):
            raise ValueError("val_frac and test_frac must be in (0,1) and sum to < 1")


@dataclass(frozen=True)
class SamplingConfig:
    # Whole impressions are sampled; None disables the respective budget.
    max_impressions: int | None = None
    max_candidate_rows: int | None = None
    seed: int = 42


@dataclass(frozen=True)
class ProfileConfig:
    # Number of most recent history items used for the profile vector.
    max_history: int = 50
    # Recency weight halves every `half_life` history positions (sequence order is the recency proxy;
    # MIND history has no per-click timestamps).
    half_life: float = 10.0
    # Append clicks from the same user's earlier impressions (strictly earlier timestamps) to the
    # impression's MIND history string. Mirrors serving, where recorded clicks update the history.
    augment_with_log_clicks: bool = True


@dataclass(frozen=True)
class EmbeddingFeatureConfig:
    # "pca": PCA fitted on training-period article vectors only; "full": all dims; "none": no raw dims.
    mode: str = "pca"
    pca_components: int = 32

    def __post_init__(self) -> None:
        if self.mode not in {"pca", "full", "none"}:
            raise ValueError(f"unknown embedding feature mode {self.mode!r}")
        if self.mode == "pca" and self.pca_components < 1:
            raise ValueError("pca_components must be >= 1")


@dataclass(frozen=True)
class PriorConfig:
    # Articles with at least this many training-period impressions inform the Beta prior fit.
    min_impressions_for_fit: int = 20
    # Bounds for the prior strength alpha+beta (method-of-moments can be unstable on small data).
    min_strength: float = 2.0
    max_strength: float = 10_000.0


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 42
    lgbm_num_leaves: int = 31
    lgbm_learning_rate: float = 0.05
    lgbm_num_boost_round: int = 1000
    lgbm_early_stopping_rounds: int = 50
    lgbm_min_data_in_leaf: int = 50
    lgbm_feature_fraction: float = 0.8
    lgbm_bagging_fraction: float = 0.8
    lgbm_bagging_freq: int = 5
    lgbm_lambda_l2: float = 1.0
    logreg_c: float = 1.0
    logreg_max_iter: int = 2000
    ndcg_eval_at: tuple[int, ...] = (5, 10)
    bootstrap_samples: int = 1000
    run_ablations: bool = True


@dataclass(frozen=True)
class PipelineConfig:
    split: SplitConfig = field(default_factory=SplitConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    profile: ProfileConfig = field(default_factory=ProfileConfig)
    embedding_features: EmbeddingFeatureConfig = field(default_factory=EmbeddingFeatureConfig)
    prior: PriorConfig = field(default_factory=PriorConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
