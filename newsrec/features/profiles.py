"""User-history profiles built from article embeddings of previously clicked items.

profile vector ``u = sum_j w_j e_j / ||sum_j w_j e_j||`` over the last ``max_history`` *known* history
items, with ``w_j = 0.5 ** (d_j / half_life)`` where ``d_j`` is the item's distance from the most recent
history position (0 = most recent). MIND's history is documented as "ordered by time"; we treat the last
element as most recent. Sequence position is a recency *proxy* - there are no per-click timestamps.

This is a deterministic aggregate, not a trained user tower.
"""

from __future__ import annotations

import base64
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from newsrec.embeddings.encoder import ArticleEmbeddings


@dataclass
class UserProfile:
    vector: np.ndarray  # (dim,) unit vector, or zeros if no known history
    recent_vectors: np.ndarray  # (k, dim) known history vectors used, oldest -> newest
    n_known: int
    category_weights: dict[str, float] = field(default_factory=dict)  # normalized, sums to 1
    subcategory_weights: dict[str, float] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return self.n_known == 0

    def to_dict(self) -> dict:
        return {
            "vector": base64.b64encode(self.vector.astype(np.float32).tobytes()).decode(),
            "recent": base64.b64encode(self.recent_vectors.astype(np.float32).tobytes()).decode(),
            "dim": int(self.vector.shape[0]),
            "n_known": self.n_known,
            "cat": self.category_weights,
            "subcat": self.subcategory_weights,
        }

    @classmethod
    def from_dict(cls, d: dict) -> UserProfile:
        dim = int(d["dim"])
        vec = np.frombuffer(base64.b64decode(d["vector"]), dtype=np.float32).copy()
        recent = np.frombuffer(base64.b64decode(d["recent"]), dtype=np.float32).copy().reshape(-1, dim)
        if vec.shape != (dim,) or not np.isfinite(vec).all():
            raise ValueError("corrupt cached profile")
        return cls(vector=vec, recent_vectors=recent, n_known=int(d["n_known"]),
                   category_weights=dict(d["cat"]), subcategory_weights=dict(d["subcat"]))


def append_unique(base: list[str], extra: list[str]) -> list[str]:
    """``base`` followed by items of ``extra`` not already present (order preserved)."""
    seen = set(base)
    out = list(base)
    for n in extra:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def recency_weights(n: int, half_life: float) -> np.ndarray:
    """Weights for n items ordered oldest -> newest; the newest has weight 1."""
    distance = np.arange(n - 1, -1, -1, dtype=float)
    return 0.5 ** (distance / half_life)


def build_profile(
    history_ids: list[str],
    embeddings: ArticleEmbeddings,
    categories: Mapping[str, tuple[str, str]],
    max_history: int = 50,
    half_life: float = 10.0,
) -> UserProfile:
    """Profile from history (oldest -> newest). Unknown ids and empty-title articles are skipped."""
    vecs, known = embeddings.lookup(list(history_ids))
    known_ids = [h for h, k in zip(history_ids, known) if k]
    vecs = vecs[known][-max_history:]
    known_ids = known_ids[-max_history:]
    if len(known_ids) == 0:
        dim = embeddings.dim
        return UserProfile(vector=np.zeros(dim, np.float32), recent_vectors=np.zeros((0, dim), np.float32),
                           n_known=0)
    w = recency_weights(len(known_ids), half_life)
    agg = (w[:, None] * vecs).sum(axis=0)
    norm = float(np.linalg.norm(agg))
    vector = (agg / norm).astype(np.float32) if norm > 1e-12 else np.zeros_like(agg, dtype=np.float32)
    cat_w: dict[str, float] = {}
    sub_w: dict[str, float] = {}
    for nid, wi in zip(known_ids, w):
        cat, sub = categories.get(nid, ("", ""))
        if cat:
            cat_w[cat] = cat_w.get(cat, 0.0) + float(wi)
        if sub:
            sub_w[sub] = sub_w.get(sub, 0.0) + float(wi)
    total = float(w.sum())
    return UserProfile(
        vector=vector, recent_vectors=vecs.astype(np.float32), n_known=len(known_ids),
        category_weights={k: v / total for k, v in cat_w.items()},
        subcategory_weights={k: v / total for k, v in sub_w.items()},
    )


def profile_candidate_features(
    profile: UserProfile,
    cand_vectors: np.ndarray,
    cand_categories: list[str],
    cand_subcategories: list[str],
) -> dict[str, np.ndarray]:
    """Candidate-specific profile features. All are 0 for an empty profile or an unknown candidate."""
    n = len(cand_vectors)
    if profile.is_empty:
        z = np.zeros(n, dtype=np.float32)
        return {"profile_cosine": z, "profile_max_cosine": z.copy(), "category_affinity": z.copy(),
                "subcategory_affinity": z.copy()}
    # vectors are unit-norm (or zero), so dot products are cosines
    cos = cand_vectors @ profile.vector
    max_cos = (cand_vectors @ profile.recent_vectors.T).max(axis=1) if len(profile.recent_vectors) else np.zeros(n)
    cat = np.array([profile.category_weights.get(c, 0.0) for c in cand_categories], dtype=np.float32)
    sub = np.array([profile.subcategory_weights.get(s, 0.0) for s in cand_subcategories], dtype=np.float32)
    return {"profile_cosine": cos.astype(np.float32), "profile_max_cosine": max_cos.astype(np.float32),
            "category_affinity": cat, "subcategory_affinity": sub}


def augment_histories(impressions: pd.DataFrame, candidates: pd.DataFrame) -> pd.Series:
    """History per impression = MIND history string + this user's clicks from strictly earlier impressions.

    Impressions sharing a timestamp do not see each other's clicks; an impression never sees its own.
    Returns a Series of lists indexed like ``impressions``.
    """
    clicks = (candidates[candidates["clicked"] == 1]
              .groupby("impression_id", sort=False)["news_id"].agg(list))
    order = impressions.sort_values(["user_id", "time", "impression_id"], kind="mergesort")
    result: dict[int, list[str]] = {}
    for _, user_rows in order.groupby("user_id", sort=False):
        log_clicks: list[str] = []
        pending: list[str] = []
        prev_time = None
        for idx, imp_id, t, hist in zip(user_rows.index, user_rows["impression_id"], user_rows["time"],
                                        user_rows["history"]):
            if prev_time is not None and t > prev_time:
                log_clicks.extend(pending)
                pending = []
            result[idx] = append_unique(list(hist), log_clicks)
            pending.extend(clicks.get(imp_id, []))
            prev_time = t
    return pd.Series(result).reindex(impressions.index)
