"""Text encoders and the per-article embedding table.

* ``MiniLMEncoder`` — frozen ``sentence-transformers/all-MiniLM-L6-v2`` (384-d, mean pooling,
  L2-normalized). No fine-tuning is performed anywhere in this project.
* ``HashingEncoder`` — deterministic hashed bag-of-words + random projection. **For tests and the
  synthetic demo only**: it is lexical, not semantic, and its vectors are not MiniLM vectors. Its
  ``model_id`` differs, so cached vectors can never be mixed with MiniLM vectors.

One embedding is computed per unique ``news_id``; cache keys contain the encoder ``model_id``, the
``news_id`` and a checksum of the exact input text.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from newsrec.embeddings.cache import EmbeddingCache

logger = logging.getLogger(__name__)

MINILM_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class TextEncoder(Protocol):
    model_id: str
    dim: int

    def encode(self, texts: list[str]) -> np.ndarray:  # (n, dim) float32, L2-normalized rows
        ...


class MiniLMEncoder:
    def __init__(self, model_name: str = MINILM_MODEL_NAME, device: str | None = None, batch_size: int = 128):
        from sentence_transformers import SentenceTransformer  # heavy import only when used

        self._model = SentenceTransformer(model_name, device=device)
        self._model.eval()
        self.batch_size = batch_size
        dim = self._model.get_sentence_embedding_dimension()
        if dim is None:
            raise RuntimeError(f"{model_name} does not report a sentence embedding dimension")
        self.dim = int(dim)
        self.model_id = f"{model_name.split('/')[-1]}@st-l2norm-d{self.dim}"

    def encode(self, texts: list[str]) -> np.ndarray:
        import torch

        with torch.inference_mode():
            vecs = self._model.encode(
                texts, batch_size=self.batch_size, normalize_embeddings=True,
                convert_to_numpy=True, show_progress_bar=len(texts) > 5000,
            )
        return np.asarray(vecs, dtype=np.float32)


_TOKEN_RE = re.compile(r"[a-z0-9]+")


class HashingEncoder:
    """TEST/DEMO ONLY. Hashed unigram bag-of-words projected to ``dim`` dims, L2-normalized."""

    def __init__(self, dim: int = 64, n_buckets: int = 4096, seed: int = 0):
        self.dim = dim
        self.n_buckets = n_buckets
        self.model_id = f"hashing-bow-v1-b{n_buckets}-d{dim}-s{seed}"
        rng = np.random.default_rng(seed)
        self._proj = rng.standard_normal((n_buckets, dim)).astype(np.float32) / np.sqrt(dim)

    def _bucket(self, token: str) -> int:
        return int.from_bytes(hashlib.md5(token.encode()).digest()[:4], "little") % self.n_buckets

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            toks = _TOKEN_RE.findall(text.lower())
            if not toks:
                continue
            idx = [self._bucket(t) for t in toks]
            v = self._proj[idx].sum(axis=0)
            norm = np.linalg.norm(v)
            if norm > 0:
                out[i] = v / norm
        return out


def make_encoder(backend: str, **kwargs) -> TextEncoder:
    if backend == "minilm":
        return MiniLMEncoder(**kwargs)
    if backend == "hashing":
        logger.warning("Using HashingEncoder: test/demo encoder, NOT MiniLM. Do not report its results as MiniLM.")
        return HashingEncoder(**kwargs)
    raise ValueError(f"unknown embedding backend {backend!r} (expected 'minilm' or 'hashing')")


def embedding_text(title: str) -> str:
    """The exact text fed to the encoder (title only)."""
    return (title or "").strip()


def text_checksum(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


@dataclass
class ArticleEmbeddings:
    """Side table of one L2-normalized vector per article. Empty titles get a zero vector."""

    news_ids: np.ndarray  # (n,) str
    vectors: np.ndarray  # (n, dim) float32
    model_id: str
    checksums: np.ndarray  # (n,) str

    def __post_init__(self) -> None:
        self.vectors = np.ascontiguousarray(self.vectors, dtype=np.float32)
        if not np.isfinite(self.vectors).all():
            raise ValueError("embedding table contains non-finite values")
        self._index = {nid: i for i, nid in enumerate(self.news_ids.tolist())}

    @property
    def dim(self) -> int:
        return int(self.vectors.shape[1])

    def __len__(self) -> int:
        return len(self.news_ids)

    def positions(self, news_ids: list[str]) -> np.ndarray:
        """Row index per id, -1 if unknown."""
        return np.fromiter((self._index.get(n, -1) for n in news_ids), dtype=np.int64, count=len(news_ids))

    def lookup(self, news_ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """(vectors, known_mask). Unknown ids get zero vectors and known_mask False."""
        pos = self.positions(news_ids)
        known = pos >= 0
        out = np.zeros((len(news_ids), self.dim), dtype=np.float32)
        out[known] = self.vectors[pos[known]]
        return out, known & (np.abs(out).sum(axis=1) > 0)

    def save(self, path: str | Path) -> None:
        np.savez(path, news_ids=self.news_ids.astype(str), vectors=self.vectors,
                 checksums=self.checksums.astype(str), model_id=np.array(self.model_id))

    @classmethod
    def load(cls, path: str | Path) -> ArticleEmbeddings:
        with np.load(path, allow_pickle=False) as z:
            return cls(news_ids=z["news_ids"], vectors=z["vectors"], model_id=str(z["model_id"]),
                       checksums=z["checksums"])


def build_article_embeddings(
    news: pd.DataFrame,
    encoder: TextEncoder,
    cache: EmbeddingCache | None = None,
) -> tuple[ArticleEmbeddings, dict[str, int]]:
    """Embed each unique article title once, reusing cached vectors for the same model/id/text."""
    arts = news.drop_duplicates("news_id")
    ids = arts["news_id"].astype(str).tolist()
    texts = [embedding_text(t) for t in arts["title"].tolist()]
    sums = [text_checksum(t) for t in texts]
    vectors = np.zeros((len(ids), encoder.dim), dtype=np.float32)
    stats = {"articles": len(ids), "empty_text": 0, "cache_hits": 0, "encoded": 0}

    todo = [i for i, t in enumerate(texts) if t]
    stats["empty_text"] = len(ids) - len(todo)
    if cache is not None and todo:
        keys = [(encoder.model_id, ids[i], sums[i]) for i in todo]
        found = cache.get_many(keys, encoder.dim)
        remaining = []
        for i, k in zip(todo, keys):
            if k in found:
                vectors[i] = found[k]
            else:
                remaining.append(i)
        stats["cache_hits"] = len(todo) - len(remaining)
        todo = remaining
    if todo:
        encoded = encoder.encode([texts[i] for i in todo])
        if encoded.shape != (len(todo), encoder.dim):
            raise ValueError(f"encoder returned shape {encoded.shape}, expected {(len(todo), encoder.dim)}")
        vectors[todo] = encoded
        stats["encoded"] = len(todo)
        if cache is not None:
            cache.set_many({(encoder.model_id, ids[i], sums[i]): encoded[j] for j, i in enumerate(todo)})
    emb = ArticleEmbeddings(news_ids=np.array(ids), vectors=vectors, model_id=encoder.model_id,
                            checksums=np.array(sums))
    return emb, stats
