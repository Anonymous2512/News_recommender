"""Model training: Logistic Regression CTR baseline, LightGBM CTR classifier, LightGBM LambdaRank.

All three use the same chronological split and feature matrix. Validation is used only for early
stopping; the test period is touched only by evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from newsrec.config import TrainConfig
from newsrec.features.builder import group_boundaries
from newsrec.models.outputs import ClickProbabilities, RankingScores


def group_sizes(group_ids: np.ndarray) -> np.ndarray:
    """Contiguous group sizes for LightGBM ranking; asserts contiguity and that sizes sum to n rows."""
    bounds = group_boundaries(np.asarray(group_ids))
    sizes = np.array([e - s for s, e in bounds], dtype=np.int32)
    if int(sizes.sum()) != len(group_ids):
        raise AssertionError("group sizes do not sum to the number of rows")
    return sizes


@dataclass
class LogRegCTR:
    pipeline: Pipeline
    feature_names: list[str]

    def predict_proba(self, X: np.ndarray) -> ClickProbabilities:
        return ClickProbabilities(self.pipeline.predict_proba(X)[:, 1])


@dataclass
class LightGBMCTR:
    booster: lgb.Booster

    def predict_proba(self, X: np.ndarray) -> ClickProbabilities:
        return ClickProbabilities(np.asarray(self.booster.predict(X, num_iteration=self.booster.best_iteration or None)))


@dataclass
class LambdaRanker:
    booster: lgb.Booster

    def score(self, X: np.ndarray) -> RankingScores:
        return RankingScores(np.asarray(self.booster.predict(X, num_iteration=self.booster.best_iteration or None)))


def _lgb_common(cfg: TrainConfig) -> dict:
    return {
        "num_leaves": cfg.lgbm_num_leaves, "learning_rate": cfg.lgbm_learning_rate,
        "min_data_in_leaf": cfg.lgbm_min_data_in_leaf, "feature_fraction": cfg.lgbm_feature_fraction,
        "bagging_fraction": cfg.lgbm_bagging_fraction, "bagging_freq": cfg.lgbm_bagging_freq,
        "lambda_l2": cfg.lgbm_lambda_l2, "seed": cfg.seed, "deterministic": True,
        "force_row_wise": True, "num_threads": 1, "verbosity": -1,
    }


def train_logreg(X: np.ndarray, y: np.ndarray, feature_names: list[str], cfg: TrainConfig) -> LogRegCTR:
    # No class weighting: keeps outputs interpretable as (uncalibrated) click probabilities.
    pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(C=cfg.logreg_c, max_iter=cfg.logreg_max_iter, random_state=cfg.seed)),
    ])
    pipe.fit(X, y)
    return LogRegCTR(pipeline=pipe, feature_names=feature_names)


def train_lgbm_ctr(X_tr, y_tr, X_va, y_va, feature_names: list[str], cfg: TrainConfig) -> LightGBMCTR:
    params = {**_lgb_common(cfg), "objective": "binary", "metric": "binary_logloss"}
    dtr = lgb.Dataset(X_tr, label=y_tr, feature_name=feature_names, free_raw_data=False)
    dva = lgb.Dataset(X_va, label=y_va, feature_name=feature_names, reference=dtr, free_raw_data=False)
    booster = lgb.train(params, dtr, num_boost_round=cfg.lgbm_num_boost_round, valid_sets=[dva],
                        valid_names=["val"],
                        callbacks=[lgb.early_stopping(cfg.lgbm_early_stopping_rounds, verbose=False)])
    return LightGBMCTR(booster=booster)


def train_lambdarank(X_tr, y_tr, g_tr, X_va, y_va, g_va, feature_names: list[str],
                     cfg: TrainConfig) -> LambdaRanker:
    params = {**_lgb_common(cfg), "objective": "lambdarank", "metric": "ndcg",
              "ndcg_eval_at": list(cfg.ndcg_eval_at)}
    dtr = lgb.Dataset(X_tr, label=y_tr, group=group_sizes(g_tr), feature_name=feature_names,
                      free_raw_data=False)
    dva = lgb.Dataset(X_va, label=y_va, group=group_sizes(g_va), feature_name=feature_names,
                      reference=dtr, free_raw_data=False)
    booster = lgb.train(params, dtr, num_boost_round=cfg.lgbm_num_boost_round, valid_sets=[dva],
                        valid_names=["val"],
                        callbacks=[lgb.early_stopping(cfg.lgbm_early_stopping_rounds, verbose=False,
                                                      first_metric_only=True)])
    return LambdaRanker(booster=booster)
