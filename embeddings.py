"""
embeddings.py — backward-compatible entry point for article embeddings.

Delegates to ``newsrec.pipeline.embed``: frozen all-MiniLM-L6-v2 title embeddings, one per unique
news_id, cached under keys that include the encoder model id and a text checksum (disk or Redis).
The old ``--fuse-abstract`` option silently fell back to titles because abstracts were dropped upstream;
it is now rejected explicitly rather than pretending to work.

    python embeddings.py --processed data/processed [--device cuda] [--redis-url redis://localhost:6379/0]
"""

from __future__ import annotations

import argparse
import json
import logging

from newsrec.embeddings.encoder import ArticleEmbeddings, HashingEncoder, MiniLMEncoder  # noqa: F401
from newsrec.pipeline import embed


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    p = argparse.ArgumentParser(description="Article title embeddings (all-MiniLM-L6-v2, frozen)")
    p.add_argument("--processed", default="data/processed")
    p.add_argument("--backend", choices=["minilm", "hashing"], default="minilm")
    p.add_argument("--device", default=None)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--cache-dir", default="data/cache/embeddings")
    p.add_argument("--redis-url", default=None)
    p.add_argument("--fuse-abstract", action="store_true", help=argparse.SUPPRESS)
    a = p.parse_args(argv)
    if a.fuse_abstract:
        p.error("--fuse-abstract is not implemented in the corrected pipeline (title-only embeddings)")
    kw = {"device": a.device, "batch_size": a.batch_size} if a.backend == "minilm" else {}
    print(json.dumps(embed(a.processed, a.backend, a.cache_dir, a.redis_url, **kw), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
