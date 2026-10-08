"""Embeddings for both stores.

We compute every vector ourselves (instead of letting each store do it) so that
Chroma and OpenSearch receive exactly the same numbers. Vectors are cached on
disk in data/cache/ so re-running notebooks does not call Bedrock again.

Providers (choose with EMBEDDING_PROVIDER in .env):
  titan  (default) Amazon Titan Text Embeddings V2 via Bedrock, 1024 dims.
  local  Chroma's built-in all-MiniLM-L6-v2, 384 dims. Offline smoke tests only;
         numbers produced with it are NOT comparable with the article.
"""

from __future__ import annotations

import hashlib
import json
import os

from src.common import CACHE_DIR, EMBEDDING_DIM, load_env

TITAN_MODEL_ID = "amazon.titan-embed-text-v2:0"


class TitanEmbedder:
    name = "titan-v2"
    dimension = EMBEDDING_DIM

    def __init__(self, region: str | None = None):
        import boto3

        self.region = region or os.environ.get("AWS_REGION", "us-east-1")
        self.client = boto3.client("bedrock-runtime", region_name=self.region)

    def embed_one(self, text: str) -> list[float]:
        body = json.dumps({"inputText": text, "dimensions": self.dimension, "normalize": True})
        response = self.client.invoke_model(modelId=TITAN_MODEL_ID, body=body)
        return json.loads(response["body"].read())["embedding"]


class LocalEmbedder:
    name = "local-minilm"
    dimension = 384

    def __init__(self):
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

        self._fn = DefaultEmbeddingFunction()

    def embed_one(self, text: str) -> list[float]:
        return [float(x) for x in self._fn([text])[0]]


class CachedEmbedder:
    """Wraps a provider and stores vectors in data/cache/<provider>.json."""

    def __init__(self, provider):
        self.provider = provider
        self.name = provider.name
        self.dimension = provider.dimension
        self.path = CACHE_DIR / f"{provider.name}.json"
        self._cache: dict[str, list[float]] = {}
        if self.path.exists():
            self._cache = json.loads(self.path.read_text())

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def embed(self, texts: list[str]) -> list[list[float]]:
        missing = [t for t in texts if self._key(t) not in self._cache]
        for text in missing:
            self._cache[self._key(text)] = self.provider.embed_one(text)
        if missing:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self._cache))
        return [self._cache[self._key(t)] for t in texts]


def get_embedder() -> CachedEmbedder:
    load_env()
    choice = os.environ.get("EMBEDDING_PROVIDER", "titan").lower()
    if choice == "titan":
        return CachedEmbedder(TitanEmbedder())
    if choice == "local":
        return CachedEmbedder(LocalEmbedder())
    raise ValueError(f"Unknown EMBEDDING_PROVIDER={choice!r} (use 'titan' or 'local')")
