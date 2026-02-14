"""Builds the serving CandidateRetriever from a loaded artifact bundle (once, at startup)."""

from __future__ import annotations

import pandas as pd

from newsrec.models.artifacts import ModelBundle
from newsrec.retrieval.retriever import CandidateRetriever, eligible_by_recency


def build_retriever(bundle: ModelBundle, reference_time: pd.Timestamp, index_backend: str = "exact",
                    max_age_hours: float | None = None) -> CandidateRetriever:
    emb = bundle.embeddings
    cats = bundle.articles.set_index("news_id")["category"].reindex(emb.news_ids).fillna("").tolist()
    eligible = eligible_by_recency(emb.news_ids, bundle.popularity, max_age_hours, reference_time)
    return CandidateRetriever(emb.news_ids, emb.vectors, cats, bundle.popularity, bundle.prior,
                              index_backend=index_backend, eligible=eligible)
