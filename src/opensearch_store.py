"""OpenSearch side of the comparison: distributed, authenticated, hybrid-capable.

Run:  python -m src.opensearch_store
"""

import json
import sys

import boto3
from opensearchpy import OpenSearch, RequestsHttpConnection, helpers
from requests_aws4auth import AWS4Auth

from src import config
from src.corpus import load_embeddings

# Serverless collections authenticate against the "aoss" service name.
# A managed OpenSearch domain uses "es" instead — using the wrong one is the
# most common cause of an unexplained 403.
AWS_SERVICE = "aoss"


def get_client() -> OpenSearch:
    if not config.AOSS_ENDPOINT:
        sys.exit("AOSS_ENDPOINT is not set. Run infra/create_collection.py first.")

    credentials = boto3.Session().get_credentials()
    auth = AWS4Auth(
        credentials.access_key,
        credentials.secret_key,
        config.AWS_REGION,
        AWS_SERVICE,
        session_token=credentials.token,
    )

    host = config.AOSS_ENDPOINT.replace("https://", "").rstrip("/")
    return OpenSearch(
        hosts=[{"host": host, "port": 443}],
        http_auth=auth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        pool_maxsize=20,
    )


def create_index(client: OpenSearch) -> None:
    if client.indices.exists(index=config.OPENSEARCH_INDEX):
        print(f"Index {config.OPENSEARCH_INDEX} already exists")
        return

    client.indices.create(
        index=config.OPENSEARCH_INDEX,
        body={
            "settings": {"index.knn": True},
            "mappings": {
                "properties": {
                    "embedding": {
                        "type": "knn_vector",
                        "dimension": config.EMBED_DIMENSION,
                        "method": {
                            "name": "hnsw",
                            "space_type": "cosinesimil",
                            "engine": "faiss",
                        },
                    },
                    "text": {"type": "text"},
                    "doc_id": {"type": "keyword"},
                    "source": {"type": "keyword"},
                }
            },
        },
    )
    print(f"Created index {config.OPENSEARCH_INDEX}")


def bulk_load(client: OpenSearch) -> None:
    """Load the same vectors Chroma received.

    The original identifier is stored as a `doc_id` field rather than as the
    document `_id`, so this works the same on Serverless and on a managed
    domain without depending on custom-ID support.
    """
    records = load_embeddings()
    actions = [
        {
            "_index": config.OPENSEARCH_INDEX,
            "_source": {
                "doc_id": record["id"],
                "text": record["text"],
                "source": record["source"],
                "embedding": record["embedding"],
            },
        }
        for record in records
    ]
    helpers.bulk(client, actions)
    client.indices.refresh(index=config.OPENSEARCH_INDEX)
    print(f"Indexed {len(records)} documents into OpenSearch")


def search_knn(client: OpenSearch, query_vector: list[float], k: int = config.TOP_K) -> dict:
    return client.search(
        index=config.OPENSEARCH_INDEX,
        body={
            "size": k,
            "query": {"knn": {"embedding": {"vector": query_vector, "k": k}}},
        },
    )


def search_bm25(client: OpenSearch, query_text: str, k: int = config.TOP_K) -> dict:
    return client.search(
        index=config.OPENSEARCH_INDEX,
        body={"size": k, "query": {"match": {"text": query_text}}},
    )


def create_search_pipeline(client: OpenSearch) -> bool:
    """Register the normalization pipeline the `hybrid` query clause requires.

    Sending a `hybrid` query without this pipeline fails: there is nothing to
    reconcile the two score scales. Returns False when the deployment does not
    support search pipelines, in which case use `search_hybrid_rrf` instead.
    """
    body = {
        "description": "Normalize and combine BM25 and k-NN scores",
        "phase_results_processors": [
            {
                "normalization-processor": {
                    "normalization": {"technique": "min_max"},
                    "combination": {
                        "technique": "arithmetic_mean",
                        "parameters": {"weights": [0.3, 0.7]},
                    },
                }
            }
        ],
    }
    try:
        client.transport.perform_request(
            "PUT", f"/_search/pipeline/{config.SEARCH_PIPELINE}", body=body
        )
        print(f"Created search pipeline {config.SEARCH_PIPELINE}")
        return True
    except Exception as error:  # noqa: BLE001 - surfaced to the caller as a fallback signal
        print(f"Search pipeline unavailable ({error}); falling back to client-side fusion")
        return False


def search_hybrid_pipeline(
    client: OpenSearch, query_text: str, query_vector: list[float], k: int = config.TOP_K
) -> dict:
    return client.search(
        index=config.OPENSEARCH_INDEX,
        params={"search_pipeline": config.SEARCH_PIPELINE},
        body={
            "size": k,
            "query": {
                "hybrid": {
                    "queries": [
                        {"match": {"text": query_text}},
                        {"knn": {"embedding": {"vector": query_vector, "k": k}}},
                    ]
                }
            },
        },
    )


def search_hybrid_rrf(
    client: OpenSearch,
    query_text: str,
    query_vector: list[float],
    k: int = config.TOP_K,
    rrf_k: int = 60,
) -> list[dict]:
    """Reciprocal Rank Fusion, computed client-side.

    The portable fallback when search pipelines are unavailable: rank position,
    not raw score, so the two incompatible score scales never have to be
    reconciled.
    """
    keyword_hits = search_bm25(client, query_text, k)["hits"]["hits"]
    vector_hits = search_knn(client, query_vector, k)["hits"]["hits"]

    scores: dict[str, float] = {}
    documents: dict[str, dict] = {}
    for hits in (keyword_hits, vector_hits):
        for rank, hit in enumerate(hits, start=1):
            doc_id = hit["_source"]["doc_id"]
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank)
            documents[doc_id] = hit["_source"]

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:k]
    return [{"doc_id": doc_id, "score": score, **documents[doc_id]} for doc_id, score in ranked]


if __name__ == "__main__":
    from src.embed import bedrock_client, embed_text

    os_client = get_client()
    create_index(os_client)
    bulk_load(os_client)
    has_pipeline = create_search_pipeline(os_client)

    question = "How does Bedrock handle tool permissions for agents?"
    vector = embed_text(bedrock_client(), question)

    print(f"\nQuery: {question}\n")
    print("--- k-NN ---")
    print(json.dumps(search_knn(os_client, vector), indent=2)[:2000])

    print("\n--- hybrid ---")
    if has_pipeline:
        print(json.dumps(search_hybrid_pipeline(os_client, question, vector), indent=2)[:2000])
    else:
        for row in search_hybrid_rrf(os_client, question, vector):
            print(f"{row['score']:.5f}  {row['doc_id']}  {row['text'][:80]}")
