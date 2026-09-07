"""Loaders for the shared corpus and the shared embedding file."""

import json
import sys
from typing import Iterator

from src import config


def _read_jsonl(path) -> Iterator[dict]:
    if not path.exists():
        sys.exit(f"Missing {path}. See the README for the step that generates it.")
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_corpus() -> list[dict]:
    """Raw documents: {id, text, source}."""
    return list(_read_jsonl(config.CORPUS_PATH))


def load_embeddings() -> list[dict]:
    """Embedded documents: {id, text, source, embedding}.

    Both stores load from this same file on purpose. Generating vectors
    separately per store would make any difference in results noise rather
    than a property of the store being measured.
    """
    records = list(_read_jsonl(config.EMBEDDINGS_PATH))
    if records and len(records[0]["embedding"]) != config.EMBED_DIMENSION:
        sys.exit(
            f"Dimension mismatch: embeddings are {len(records[0]['embedding'])}-d "
            f"but EMBED_DIMENSION is {config.EMBED_DIMENSION}. "
            "Fix .env and regenerate, or the OpenSearch index mapping will not match."
        )
    return records


def load_queries() -> list[str]:
    if not config.QUERIES_PATH.exists():
        sys.exit(f"Missing {config.QUERIES_PATH}.")
    lines = config.QUERIES_PATH.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.startswith("#")]
