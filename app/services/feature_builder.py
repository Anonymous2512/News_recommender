"""Serving-side candidate frame: same columns/transforms as training (``newsrec.features.builder``).

Popularity counters come from the artifact's popularity snapshot (as of the end of the processed log).
Recorded API clicks update user history but not these counters; impressions served by the API are not
logged, so a CTR denominator for live traffic does not exist yet (documented limitation).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from newsrec.features.builder import FeatureBuilder
from newsrec.features.popularity import PopularitySnapshot
from newsrec.features.profiles import UserProfile


class ServingFeatureBuilder:
    def __init__(self, builder: FeatureBuilder, popularity: PopularitySnapshot, reference_time: pd.Timestamp):
        self.builder = builder
        self.popularity = popularity
        self.reference_time = reference_time

    def frame(self, news_ids: list[str]) -> pd.DataFrame:
        counts = self.popularity.lookup(news_ids, self.reference_time)
        return pd.DataFrame({
            "group_id": np.zeros(len(news_ids), dtype=np.int64),
            "news_id": news_ids,
            "past_impressions": counts["past_impressions"],
            "past_clicks": counts["past_clicks"],
            "hours_since_first_seen": counts["hours_since_first_seen"],
        })

    def build(self, news_ids: list[str], history_ids: list[str], profile: UserProfile) -> np.ndarray:
        return self.builder.build(self.frame(news_ids), {0: history_ids}, profiles={0: profile})
