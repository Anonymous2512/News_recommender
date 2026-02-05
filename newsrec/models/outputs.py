"""Distinct output types so CTR probabilities and ranking scores cannot be confused.

* ``ClickProbabilities`` — outputs of a binary classifier in [0, 1]. Not calibrated unless a calibration
  step is fitted and evaluated (none is fitted in this project; calibration is only *diagnosed*).
* ``RankingScores`` — raw LambdaRank scores: unbounded, meaningful only for ordering candidates within
  the same impression/request. Never a probability or a confidence.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ClickProbabilities:
    values: np.ndarray

    def __post_init__(self) -> None:
        v = np.asarray(self.values, dtype=float)
        if not np.isfinite(v).all() or (v < 0).any() or (v > 1).any():
            raise ValueError("click probabilities must be finite and within [0, 1]")
        object.__setattr__(self, "values", v)


@dataclass(frozen=True)
class RankingScores:
    values: np.ndarray

    def __post_init__(self) -> None:
        v = np.asarray(self.values, dtype=float)
        if not np.isfinite(v).all():
            raise ValueError("ranking scores must be finite")
        object.__setattr__(self, "values", v)


def ordering_values(scores: ClickProbabilities | RankingScores) -> np.ndarray:
    """Both kinds can *order* candidates; only ClickProbabilities may be used as probabilities."""
    if not isinstance(scores, (ClickProbabilities, RankingScores)):
        raise TypeError(f"expected ClickProbabilities or RankingScores, got {type(scores).__name__}")
    return scores.values


def require_probabilities(scores: object) -> np.ndarray:
    if not isinstance(scores, ClickProbabilities):
        raise TypeError(f"probability metrics require ClickProbabilities, got {type(scores).__name__}")
    return scores.values
