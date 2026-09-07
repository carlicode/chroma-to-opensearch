"""Chroma side of the comparison: local, embedded, no infrastructure.

Run:  python -m src.chroma_store
"""

import json

import chromadb

from src import config
from src.corpus import load_embeddings


def get_collection():
    client = chromadb.PersistentClient(path=str(config.CHROMA_PATH))
    return client.get_or_create_collection(
        name=config.CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )


def build() -> None:
    """Load the shared embeddings into a persistent Chroma collection."""
    records = load_embeddings()
    collection = get_collection()

    collection.add(
        ids=[record["id"] for record in records],
        embeddings=[record["embedding"] for record in records],
        documents=[record["text"] for record in records],
        metadatas=[{"source": record["source"]} for record in records],
    )
    print(f"Loaded {len(records)} documents into Chroma at {config.CHROMA_PATH}")


def search(query_vector: list[float], k: int = config.TOP_K) -> dict:
    return get_collection().query(query_embeddings=[query_vector], n_results=k)


if __name__ == "__main__":
    from src.embed import bedrock_client, embed_text

    build()

    question = "How does Bedrock handle tool permissions for agents?"
    vector = embed_text(bedrock_client(), question)
    print(f"\nQuery: {question}\n")
    print(json.dumps(search(vector), indent=2)[:2000])
