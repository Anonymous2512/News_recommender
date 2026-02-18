"""Parsing, group-preserving sampling and chronological splitting."""

from __future__ import annotations

import pandas as pd
import pytest

from newsrec.data.mind import (
    LoadReport,
    MindFormatError,
    explode_candidates,
    load_behaviors,
    parse_impression_token,
    parse_impressions,
)
from newsrec.data.sampling import chronological_split, sample_impressions


def test_parse_tokens():
    assert parse_impression_token("N123-1") == ("N123", 1)
    assert parse_impression_token("N456-0") == ("N456", 0)
    assert parse_impressions("N1-1 N2-0") == [("N1", 1), ("N2", 0)]
    assert parse_impressions("") == []
    for bad in ("N123", "N1-2", "-1", "N1-"):
        with pytest.raises(MindFormatError):
            parse_impression_token(bad)


def _behaviors(rows):
    return pd.DataFrame(rows, columns=["impression_id", "user_id", "time", "history", "impressions"])


def test_explode_vectorized_and_strict():
    b = _behaviors([(1, "U1", pd.Timestamp("2019-11-09 10:00"), [], "N1-1 N2-0 N3-0"),
                    (2, "U2", pd.Timestamp("2019-11-09 11:00"), [], "N2-1")])
    c = explode_candidates(b)
    assert c.to_dict("list") == {"impression_id": [1, 1, 1, 2], "news_id": ["N1", "N2", "N3", "N2"],
                                 "clicked": [1, 0, 0, 1]}
    with pytest.raises(MindFormatError):
        explode_candidates(_behaviors([(3, "U", pd.Timestamp("2019-11-09"), [], "N1 N2-0")]))


def test_explode_merges_duplicate_tokens():
    c = explode_candidates(_behaviors([(1, "U", pd.Timestamp("2019-11-09"), [], "N1-0 N1-1 N2-0")]))
    assert sorted(zip(c.news_id, c.clicked)) == [("N1", 1), ("N2", 0)]


def test_load_behaviors_counts_bad_times(tmp_path):
    p = tmp_path / "behaviors.tsv"
    p.write_text("1\tU1\t11/15/2019 8:55:22 AM\tN1 N2\tN3-1 N4-0\n"
                 "2\tU2\tnot-a-time\t\tN3-0\n"
                 "3\tU3\t11/15/2019 9:00:00 PM\t\tN5-0\n")
    rep = LoadReport()
    b = load_behaviors(p, rep)
    assert rep.rows_dropped_bad_time == 1 and len(b) == 2
    assert b.loc[0, "history"] == ["N1", "N2"] and b.loc[1, "history"] == []
    assert b.loc[1, "time"] == pd.Timestamp("2019-11-15 21:00:00")


def _toy(n=40, seed=0):
    import numpy as np

    rng = np.random.default_rng(seed)
    rows, cands = [], []
    for i in range(n):
        k = int(rng.integers(2, 9))
        rows.append((i, f"U{i % 7}", pd.Timestamp("2019-11-09") + pd.Timedelta(minutes=int(rng.integers(0, 5000))),
                     [], ""))
        cands += [(i, f"N{j}", int(rng.random() < 0.2)) for j in rng.choice(50, k, replace=False)]
    return (_behaviors(rows), pd.DataFrame(cands, columns=["impression_id", "news_id", "clicked"]))


def test_sampling_keeps_whole_impressions_and_is_deterministic():
    b, c = _toy()
    sizes = c.groupby("impression_id").size()
    for kwargs in ({"max_impressions": 10}, {"max_candidate_rows": 37}):
        b1, c1, rep = sample_impressions(b, c, seed=5, **kwargs)
        b2, c2, _ = sample_impressions(b, c, seed=5, **kwargs)
        assert b1.equals(b2) and c1.equals(c2)
        assert (c1.groupby("impression_id").size() == sizes.loc[c1.impression_id.unique()]).all()
        assert set(c1.impression_id) == set(b1.impression_id)
        assert rep["candidate_rows_after"] == len(c1)
    _, c3, _ = sample_impressions(b, c, max_candidate_rows=37, seed=5)
    # budget crossing impression is kept whole -> rows >= budget, and dropping it would go below
    assert len(c3) >= 37


def test_chronological_split_is_strict_and_exhaustive():
    b, _ = _toy(200)
    split, rep = chronological_split(b, 0.1, 0.2)
    b = b.assign(split=split)
    assert set(split) == {"train", "val", "test"} and len(split) == len(b)
    assert b[b.split == "train"].time.max() < b[b.split == "val"].time.min()
    assert b[b.split == "val"].time.max() < b[b.split == "test"].time.min()
    assert pd.Timestamp(rep["val_start"]) == b[b.split == "val"].time.min()


def test_prepared_data_invariants(built, processed):
    imp, cand = processed["impressions"], processed["candidates"]
    # every candidate belongs to exactly one impression, each impression in one split
    assert cand.impression_id.isin(imp.impression_id).all()
    assert imp.impression_id.is_unique
    # groups are contiguous in the stored table
    runs = (cand.impression_id != cand.impression_id.shift()).sum()
    assert runs == cand.impression_id.nunique()
    stats = built["report"]["split_statistics"]
    assert sum(stats[s]["candidates"] for s in ("train", "val", "test")) == len(cand)
    assert built["report"]["data_source"] == "synthetic"
