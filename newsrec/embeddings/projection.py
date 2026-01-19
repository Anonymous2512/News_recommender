"""Embedding-to-feature projection.

Replaces the old "first 64 coordinates" truncation. Modes:
* ``pca``  — PCA fitted on *training-period* article vectors only; the same fitted transform is applied to
  validation/test and at serving.
* ``full`` — all embedding dimensions as features.
* ``none`` — no raw embedding dimensions (similarity features still use full vectors).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class EmbeddingProjection:
    mode: str
    input_dim: int
    mean: np.ndarray | None = None  # (input_dim,)
    components: np.ndarray | None = None  # (k, input_dim)
    explained_variance_ratio: np.ndarray | None = None

    @property
    def output_dim(self) -> int:
        if self.mode == "none":
            return 0
        if self.mode == "full":
            return self.input_dim
        assert self.components is not None
        return int(self.components.shape[0])

    def transform(self, vectors: np.ndarray) -> np.ndarray:
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != self.input_dim:
            raise ValueError(f"expected (n, {self.input_dim}) vectors, got {vectors.shape}")
        if self.mode == "none":
            return np.zeros((len(vectors), 0), dtype=np.float32)
        if self.mode == "full":
            return vectors
        assert self.mean is not None and self.components is not None
        out = (vectors - self.mean) @ self.components.T
        # zero vectors (unknown/empty-title articles) map to the projection of the mean -> make them 0
        out[np.abs(vectors).sum(axis=1) == 0] = 0.0
        return out.astype(np.float32)

    def save(self, path: str | Path) -> None:
        arrays: dict[str, np.ndarray] = {"mode": np.array(self.mode), "input_dim": np.array(self.input_dim)}
        if self.mode == "pca":
            if self.mean is None or self.components is None or self.explained_variance_ratio is None:
                raise ValueError("PCA projection is not fitted")
            arrays.update(mean=self.mean, components=self.components,
                          explained_variance_ratio=self.explained_variance_ratio)
        np.savez(str(path), **arrays)  # type: ignore[arg-type]

    @classmethod
    def load(cls, path: str | Path) -> EmbeddingProjection:
        with np.load(path, allow_pickle=False) as z:
            mode = str(z["mode"])
            if mode == "pca":
                return cls(mode=mode, input_dim=int(z["input_dim"]), mean=z["mean"],
                           components=z["components"], explained_variance_ratio=z["explained_variance_ratio"])
            return cls(mode=mode, input_dim=int(z["input_dim"]))


def fit_projection(train_vectors: np.ndarray, mode: str, n_components: int = 32) -> EmbeddingProjection:
    """Fit on training-period article vectors (zero vectors are ignored)."""
    train_vectors = np.asarray(train_vectors, dtype=np.float64)
    dim = int(train_vectors.shape[1])
    if mode in ("none", "full"):
        return EmbeddingProjection(mode=mode, input_dim=dim)
    if mode != "pca":
        raise ValueError(mode)
    x = train_vectors[np.abs(train_vectors).sum(axis=1) > 0]
    k = min(n_components, dim, max(len(x) - 1, 1))
    if len(x) < 2:
        raise ValueError("need at least 2 non-empty training article vectors to fit PCA")
    mean = x.mean(axis=0)
    _, s, vt = np.linalg.svd(x - mean, full_matrices=False)
    comps = vt[:k]
    # deterministic sign: largest-magnitude loading of each component is positive
    signs = np.sign(comps[np.arange(k), np.abs(comps).argmax(axis=1)])
    comps = comps * signs[:, None]
    var = s**2
    return EmbeddingProjection(mode="pca", input_dim=dim, mean=mean.astype(np.float32),
                               components=comps.astype(np.float32),
                               explained_variance_ratio=(var[:k] / var.sum()).astype(np.float32))
