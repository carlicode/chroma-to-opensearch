"""Benchmark: same queries, same vectors, both stores.

Reports p50/p95 latency (ms) and mean recall@3, split by query kind.

Latency covers only the call to the store (query embeddings are computed and
cached before timing starts), so we compare the databases, not Bedrock.
For OpenSearch it includes the network round trip from this machine to the
collection endpoint; Chroma runs in-process, so the two are not like-for-like.

Usage:
    python benchmark.py                     # Chroma + OpenSearch
    python benchmark.py --stores chroma     # Chroma only (no AWS needed for
                                            # the store; Titan still needs Bedrock
                                            # unless vectors are cached)
"""

from __future__ import annotations

import argparse
import time

from tabulate import tabulate

from src.common import (
    CHROMA_COLLECTION,
    CHROMA_PATH,
    INDEX_NAME,
    QUERY_KINDS,
    SEARCH_PIPELINE,
    TOP_K,
    load_queries,
    min_max_fuse,
    percentile,
    recall_at_k,
)
from src.embeddings import get_embedder

CANDIDATES = 10  # how many hits each sub-query returns before client-side fusion


# ---------------------------------------------------------------------------
# One function per "mode": each takes (query_text, query_vector) and returns
# the top-k doc ids.
# ---------------------------------------------------------------------------


def chroma_modes() -> dict:
    import chromadb

    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = client.get_collection(CHROMA_COLLECTION, embedding_function=None)

    def vector(text, vec):
        return collection.query(query_embeddings=[vec], n_results=TOP_K)["ids"][0]

    return {"chroma / k-NN": vector}


def opensearch_modes(probe_vector: list[float]) -> dict:
    from opensearchpy.exceptions import OpenSearchException

    from src.common import get_opensearch_client

    client = get_opensearch_client()

    def ids(response):
        return [h["_id"] for h in response["hits"]["hits"]]

    def bm25_hits(text, size):
        body = {"size": size, "_source": False, "query": {"match": {"text": text}}}
        return client.search(index=INDEX_NAME, body=body)["hits"]["hits"]

    def knn_hits(vec, size):
        body = {"size": size, "_source": False, "query": {"knn": {"embedding": {"vector": vec, "k": size}}}}
        return client.search(index=INDEX_NAME, body=body)["hits"]["hits"]

    def bm25(text, vec):
        return [h["_id"] for h in bm25_hits(text, TOP_K)]

    def knn(text, vec):
        return [h["_id"] for h in knn_hits(vec, TOP_K)]

    def hybrid_server(text, vec):
        body = {
            "size": TOP_K,
            "_source": False,
            "query": {
                "hybrid": {
                    "queries": [
                        {"match": {"text": text}},
                        {"knn": {"embedding": {"vector": vec, "k": CANDIDATES}}},
                    ]
                }
            },
        }
        return ids(client.search(index=INDEX_NAME, body=body, params={"search_pipeline": SEARCH_PIPELINE}))

    def hybrid_client(text, vec):
        lexical = [(h["_id"], h["_score"]) for h in bm25_hits(text, CANDIDATES)]
        semantic = [(h["_id"], h["_score"]) for h in knn_hits(vec, CANDIDATES)]
        return [doc_id for doc_id, _ in min_max_fuse([lexical, semantic])]

    modes = {"opensearch / BM25": bm25, "opensearch / k-NN": knn}

    # Use the server-side hybrid query only if this collection supports it
    # (notebook 03 creates the pipeline and tells us whether it works).
    try:
        hybrid_server("probe", probe_vector)
        modes["opensearch / hybrid (server)"] = hybrid_server
    except OpenSearchException as exc:
        print(f"Server-side hybrid not available ({type(exc).__name__}: {exc}); skipping it.")
    modes["opensearch / hybrid (client)"] = hybrid_client
    return modes


# ---------------------------------------------------------------------------


def run(modes: dict, queries: list[dict], vectors: list[list[float]], repeats: int, warmup: int) -> list[list]:
    rows = []
    for mode_name, search in modes.items():
        # Warm-up: wakes a scaled-to-zero collection and fills caches.
        for _ in range(warmup):
            for q, vec in zip(queries, vectors):
                search(q["text"], vec)

        latencies = {kind: [] for kind in QUERY_KINDS}
        recalls = {kind: [] for kind in QUERY_KINDS}
        for q, vec in zip(queries, vectors):
            retrieved = []
            for _ in range(repeats):
                start = time.perf_counter()
                retrieved = search(q["text"], vec)
                latencies[q["kind"]].append((time.perf_counter() - start) * 1000)
            recalls[q["kind"]].append(recall_at_k(retrieved, q["relevant_ids"]))

        for kind in (*QUERY_KINDS, "all"):
            lat = [x for v in latencies.values() for x in v] if kind == "all" else latencies[kind]
            rec = [x for v in recalls.values() for x in v] if kind == "all" else recalls[kind]
            rows.append(
                [mode_name, kind, len(rec), f"{percentile(lat, 50):.2f}", f"{percentile(lat, 95):.2f}", f"{sum(rec) / len(rec):.2f}"]
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stores", default="chroma,opensearch", help="comma-separated: chroma, opensearch")
    parser.add_argument("--repeats", type=int, default=20, help="timed runs per query (default 20)")
    parser.add_argument("--warmup", type=int, default=2, help="untimed passes over all queries (default 2)")
    args = parser.parse_args()

    queries = load_queries()
    embedder = get_embedder()
    vectors = embedder.embed([q["text"] for q in queries])

    modes: dict = {}
    stores = {s.strip() for s in args.stores.split(",")}
    if "chroma" in stores:
        modes.update(chroma_modes())
    if "opensearch" in stores:
        modes.update(opensearch_modes(vectors[0]))

    rows = run(modes, queries, vectors, args.repeats, args.warmup)
    print(f"\nembeddings: {embedder.name} ({embedder.dimension} dims) | top-k: {TOP_K} | "
          f"repeats: {args.repeats} | warm-up passes: {args.warmup}\n")
    print(tabulate(rows, headers=["mode", "kind", "queries", "p50 ms", "p95 ms", "recall@3"], tablefmt="github"))


if __name__ == "__main__":
    main()
