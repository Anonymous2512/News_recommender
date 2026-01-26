"""Single feature-construction code path for training, evaluation and serving.

Input is a candidate frame with contiguous groups (an offline impression or one API request):

    group_id, news_id, past_impressions, past_clicks, hours_since_first_seen

plus one history list (oldest -> newest) per group. Raw counters come from the offline point-in-time
replay or from the serving snapshot; every transform below is shared.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from newsrec.config import ProfileConfig
from newsrec.embeddings.encoder import ArticleEmbeddings
from newsrec.embeddings.projection import EmbeddingProjection
from newsrec.features.popularity import BetaPrior, smoothed_ctr
from newsrec.features.profiles import UserProfile, build_profile, profile_candidate_features
from newsrec.features.schema import FeatureSpec

REQUIRED_COLUMNS = ("group_id", "news_id", "past_impressions", "past_clicks", "hours_since_first_seen")


@dataclass
class ArticleMeta:
    categories: dict[str, tuple[str, str]]  # news_id -> (category, subcategory)
    title_word_count: dict[str, int]

    @classmethod
    def from_news(cls, news: pd.DataFrame) -> ArticleMeta:
        ids = news["news_id"].astype(str).tolist()
        return cls(
            categories=dict(zip(ids, zip(news["category"].tolist(), news["subcategory"].tolist()))),
            title_word_count=dict(zip(ids, news["title"].fillna("").str.split().str.len().fillna(0)
                                      .astype(int).tolist())),
        )


def group_boundaries(group_ids: np.ndarray) -> list[tuple[int, int]]:
    """(start, end) of each contiguous run; raises if a group id appears in two separate runs."""
    n = len(group_ids)
    if n == 0:
        return []
    change = np.flatnonzero(group_ids[1:] != group_ids[:-1]) + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [n]])
    if len(set(group_ids[starts].tolist())) != len(starts):
        raise ValueError("candidate rows of a group must be contiguous")
    return list(zip(starts.tolist(), ends.tolist()))


class FeatureBuilder:
    def __init__(self, spec: FeatureSpec, embeddings: ArticleEmbeddings, projection: EmbeddingProjection,
                 prior: BetaPrior, meta: ArticleMeta, profile_config: ProfileConfig):
        if spec.embedding_model_id != embeddings.model_id:
            raise ValueError(f"feature spec expects embeddings {spec.embedding_model_id!r}, "
                             f"got {embeddings.model_id!r}")
        n_emb = sum(1 for f in spec.features if f.group == "embedding")
        if n_emb != projection.output_dim or projection.input_dim != embeddings.dim:
            raise ValueError("projection does not match feature spec / embedding dimension")
        self.spec, self.embeddings, self.projection = spec, embeddings, projection
        self.prior, self.meta, self.profile_config = prior, meta, profile_config
        self._col = {name: i for i, name in enumerate(spec.names)}
        self._emb_start = len(spec.names) - n_emb

    def profile(self, history_ids: list[str]) -> UserProfile:
        return build_profile(history_ids, self.embeddings, self.meta.categories,
                             self.profile_config.max_history, self.profile_config.half_life)

    def build(
        self,
        frame: pd.DataFrame,
        histories: Mapping[object, list[str]],
        profiles: Mapping[object, UserProfile] | None = None,
    ) -> np.ndarray:
        """Return a float32 matrix (rows of ``frame``, columns in ``spec.names`` order)."""
        missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
        if missing:
            raise KeyError(f"candidate frame missing columns {missing}")
        n = len(frame)
        X = np.zeros((n, len(self.spec.names)), dtype=np.float32)
        news_ids = frame["news_id"].astype(str).tolist()
        imps = frame["past_impressions"].to_numpy(dtype=float)
        clicks = frame["past_clicks"].to_numpy(dtype=float)
        if (imps < 0).any() or (clicks < 0).any() or (clicks > imps).any():
            raise ValueError("invalid popularity counters (negative or clicks > impressions)")
        col = self._col
        X[:, col["past_impressions_log1p"]] = np.log1p(imps)
        X[:, col["past_clicks_log1p"]] = np.log1p(clicks)
        X[:, col["past_ctr_smoothed"]] = smoothed_ctr(clicks, imps, self.prior)
        hours = np.clip(frame["hours_since_first_seen"].to_numpy(dtype=float), 0, None)
        X[:, col["hours_since_first_seen_log1p"]] = np.log1p(hours)
        X[:, col["title_word_count"]] = [self.meta.title_word_count.get(i, 0) for i in news_ids]

        cand_vecs, _ = self.embeddings.lookup(news_ids)
        if self.projection.output_dim:
            X[:, self._emb_start:] = self.projection.transform(cand_vecs)

        cats = [self.meta.categories.get(i, ("", "")) for i in news_ids]
        cat_list = [c[0] for c in cats]
        sub_list = [c[1] for c in cats]
        gids = frame["group_id"].to_numpy()
        for start, end in group_boundaries(gids):
            gid = gids[start]
            prof = profiles.get(gid) if profiles is not None else None
            if prof is None:
                prof = self.profile(list(histories.get(gid, [])))
            feats = profile_candidate_features(prof, cand_vecs[start:end], cat_list[start:end],
                                               sub_list[start:end])
            X[start:end, col["history_length_log1p"]] = np.log1p(prof.n_known)
            for name, values in feats.items():
                X[start:end, col[name]] = values
        if not np.isfinite(X).all():
            raise ValueError("feature matrix contains non-finite values")
        return X
