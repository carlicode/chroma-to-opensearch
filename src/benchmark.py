"""Run the same query set against both stores and report latency.

Run:  python -m src.benchmark

Note on fairness: Chroma runs in-process, so its numbers carry no network hop.
OpenSearch runs over HTTPS against AWS. That gap is real and is part of the
result, not a flaw in the measurement — but it has to be stated alongside it.
"""

import statistics
import time

from src import config
from src.corpus import load_queries
from src.embed import bedrock_client, embed_text
from src import chroma_store, opensearch_store

WARMUP_QUERIES = 3


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(int(len(ordered) * fraction), len(ordered) - 1)
    return ordered[index]


def time_calls(label: str, call, payloads: list) -> dict:
    for payload in payloads[:WARMUP_QUERIES]:
        call(*payload)

    durations = []
    for payload in payloads:
        started = time.perf_counter()
        call(*payload)
        durations.append((time.perf_counter() - started) * 1000)

    return {
        "label": label,
        "runs": len(durations),
        "p50": percentile(durations, 0.50),
        "p95": percentile(durations, 0.95),
        "mean": statistics.fmean(durations),
    }


def main() -> None:
    queries = load_queries()
    print(f"Embedding {len(queries)} queries once, reused by every store\n")

    bedrock = bedrock_client()
    embedded = [(query, embed_text(bedrock, query)) for query in queries]

    os_client = opensearch_store.get_client()
    has_pipeline = opensearch_store.create_search_pipeline(os_client)

    results = [
        time_calls(
            "Chroma (k-NN, in-process)",
            lambda _text, vector: chroma_store.search(vector),
            embedded,
        ),
        time_calls(
            "OpenSearch (k-NN, HTTPS)",
            lambda _text, vector: opensearch_store.search_knn(os_client, vector),
            embedded,
        ),
    ]

    if has_pipeline:
        results.append(
            time_calls(
                "OpenSearch (hybrid, pipeline)",
                lambda text, vector: opensearch_store.search_hybrid_pipeline(
                    os_client, text, vector
                ),
                embedded,
            )
        )
    else:
        results.append(
            time_calls(
                "OpenSearch (hybrid, client-side RRF)",
                lambda text, vector: opensearch_store.search_hybrid_rrf(os_client, text, vector),
                embedded,
            )
        )

    print(f"\n{len(queries)} queries per store, top_k={config.TOP_K}, "
          f"{WARMUP_QUERIES} warmup runs discarded\n")
    print("| Store | Runs | p50 (ms) | p95 (ms) | mean (ms) |")
    print("|---|---|---|---|---|")
    for row in results:
        print(
            f"| {row['label']} | {row['runs']} | {row['p50']:.1f} | "
            f"{row['p95']:.1f} | {row['mean']:.1f} |"
        )


if __name__ == "__main__":
    main()
