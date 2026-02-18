"""Point-in-time correctness: popularity and history features never see current or future labels."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from newsrec.features.popularity import (
    BetaPrior,
    PopularitySnapshot,
    fit_beta_prior,
    point_in_time_counts,
    smoothed_ctr,
)
from newsrec.features.profiles import augment_histories

T0 = pd.Timestamp("2019-11-09 08:00")


def _log():
    # impression 3 shares a timestamp with 2
    times = pd.Series({1: T0, 2: T0 + pd.Timedelta(hours=1), 3: T0 + pd.Timedelta(hours=1),
                       4: T0 + pd.Timedelta(hours=5)})
    cands = pd.DataFrame([
        (1, "A", 1), (1, "B", 0),
        (2, "A", 0), (2, "B", 1),
        (3, "A", 1),
        (4, "A", 0), (4, "B", 0), (4, "C", 1),
    ], columns=["impression_id", "news_id", "clicked"])
    return cands, times


def _feats(cands, times):
    return point_in_time_counts(cands, times)[["past_impressions", "past_clicks", "hours_since_first_seen"]]


def test_hand_computed_counts():
    cands, times = _log()
    f = point_in_time_counts(cands, times)
    got = list(zip(f.impression_id, f.news_id, f.past_impressions, f.past_clicks, f.hours_since_first_seen))
    assert got == [
        (1, "A", 0, 0, 0.0), (1, "B", 0, 0, 0.0),
        (2, "A", 1, 1, 1.0), (2, "B", 1, 0, 1.0),
        (3, "A", 1, 1, 1.0),            # same timestamp as impression 2: does not see it
        (4, "A", 3, 2, 5.0), (4, "B", 2, 1, 5.0), (4, "C", 0, 0, 0.0),
    ]


@pytest.mark.parametrize("row", range(8))
def test_flipping_any_label_never_changes_that_rows_or_earlier_features(row):
    cands, times = _log()
    base = _feats(cands, times)
    flipped = cands.copy()
    flipped.loc[row, "clicked"] = 1 - flipped.loc[row, "clicked"]
    new = _feats(flipped, times)
    t_row = times[cands.loc[row, "impression_id"]]
    row_times = cands["impression_id"].map(times)
    unaffected = row_times <= t_row  # the row itself, its own impression, same-time and earlier rows
    pd.testing.assert_frame_equal(base[unaffected], new[unaffected])


def test_random_future_label_perturbation_property():
    rng = np.random.default_rng(0)
    n_imp = 60
    times = pd.Series({i: T0 + pd.Timedelta(minutes=int(m)) for i, m in enumerate(rng.integers(0, 500, n_imp))})
    cands = pd.DataFrame([(i, f"N{a}", int(rng.random() < 0.3)) for i in range(n_imp)
                          for a in rng.choice(15, 5, replace=False)], columns=["impression_id", "news_id", "clicked"])
    base = _feats(cands, times)
    cutoff = T0 + pd.Timedelta(minutes=250)
    later = cands["impression_id"].map(times) >= cutoff
    perturbed = cands.copy()
    perturbed.loc[later, "clicked"] = 1 - perturbed.loc[later, "clicked"]  # "validation/test" labels changed
    new = _feats(perturbed, times)
    pd.testing.assert_frame_equal(base[~later], new[~later])
    assert (new["past_clicks"] <= new["past_impressions"]).all()


def test_smoothing_is_finite_for_unseen_articles_and_prior_fit():
    prior = fit_beta_prior(np.array([100, 200, 50, 400]), np.array([5, 30, 1, 20]), min_impressions=10)
    assert prior.alpha > 0 and prior.beta > 0
    ctr = smoothed_ctr(np.array([0.0, 3.0]), np.array([0.0, 10.0]), prior)
    assert np.isfinite(ctr).all() and ctr[0] == pytest.approx(prior.mean)
    with pytest.raises(ValueError):
        fit_beta_prior(np.array([0]), np.array([0]))


def test_snapshot_lookup_unseen_article_is_zero():
    cands, times = _log()
    snap = PopularitySnapshot.from_log(cands, times)
    out = snap.lookup(["A", "ZZZ"], snap.as_of)
    assert out["past_impressions"].tolist() == [4.0, 0.0] and out["past_clicks"].tolist() == [2.0, 0.0]
    assert out["hours_since_first_seen"][1] == 0.0


def test_augmented_history_excludes_current_same_time_and_future_clicks():
    imps = pd.DataFrame({
        "impression_id": [1, 2, 3, 4], "user_id": ["U", "U", "U", "V"],
        "time": [T0, T0 + pd.Timedelta(hours=1), T0 + pd.Timedelta(hours=1), T0],
        "history": [["H1"], ["H1"], ["H1"], []],
    })
    cands = pd.DataFrame([(1, "A", 1), (2, "B", 1), (3, "C", 1), (4, "D", 1)],
                         columns=["impression_id", "news_id", "clicked"])
    h = augment_histories(imps, cands)
    assert h[0] == ["H1"]                # own click A not included
    assert h[1] == ["H1", "A"]          # earlier click included, own click B excluded
    assert h[2] == ["H1", "A"]          # same-timestamp impression 2's click B excluded
    assert h[3] == []                    # other users' clicks never included


def test_training_features_use_prior_from_train_only(built):
    rep = built["report"]
    assert rep["popularity_prior"]["fitted_on"] == "train"
    assert BetaPrior.from_dict(rep["popularity_prior"]).mean < 1
