"""Shared constants, loaders and small helpers used by every module.

Run `python -m src.common` from the repo root to validate the dataset.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DOCS_PATH = DATA_DIR / "docs.jsonl"
QUERIES_PATH = DATA_DIR / "queries.jsonl"
CACHE_DIR = DATA_DIR / "cache"
CHROMA_PATH = ROOT / "chroma_db"

# Names we reuse in both stores so results line up.
CHROMA_COLLECTION = "kb_docs"
INDEX_NAME = "kb-docs"
SEARCH_PIPELINE = "kb-hybrid-minmax"

TOP_K = 3
EMBEDDING_DIM = 1024  # Titan Text Embeddings V2, default output size
QUERY_KINDS = ("keyword", "semantic", "mixed")

# Weights for [BM25, k-NN] when we fuse scores (server-side or client-side).
HYBRID_WEIGHTS = (0.5, 0.5)


def load_env() -> None:
    """Load variables from the repo's .env file (if present) into os.environ."""
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_docs() -> list[dict]:
    return _read_jsonl(DOCS_PATH)


def load_queries() -> list[dict]:
    return _read_jsonl(QUERIES_PATH)


# ---------------------------------------------------------------------------
# OpenSearch Serverless client (used by notebooks 02/03 and benchmark.py)
# ---------------------------------------------------------------------------


def get_opensearch_client():
    """Return an OpenSearch client that signs every request with SigV4.

    Credentials come from the default boto3 chain (env vars, ~/.aws, SSO, role).
    The service name for OpenSearch *Serverless* is "aoss" (not "es").
    """
    import boto3
    from opensearchpy import OpenSearch, RequestsHttpConnection
    from requests_aws4auth import AWS4Auth

    load_env()
    region = os.environ.get("AWS_REGION", "us-east-1")
    host = os.environ.get("OPENSEARCH_HOST", "").replace("https://", "").strip("/")
    if not host:
        raise RuntimeError(
            "OPENSEARCH_HOST is empty. Copy .env.example to .env and paste the "
            "collection endpoint printed by infra/create_collection.sh."
        )

    credentials = boto3.Session().get_credentials()
    if credentials is None:
        raise RuntimeError("No AWS credentials found in the default boto3 chain.")

    auth = AWS4Auth(refreshable_credentials=credentials, region=region, service="aoss")
    return OpenSearch(
        hosts=[{"host": host, "port": 443}],
        http_auth=auth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        # The first request after the collection scaled to zero can take 10-30 s.
        timeout=60,
    )


# ---------------------------------------------------------------------------
# Metrics and score fusion
# ---------------------------------------------------------------------------


def recall_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int = TOP_K) -> float:
    """Fraction of the relevant docs that appear in the top-k results."""
    hits = len(set(retrieved_ids[:k]) & set(relevant_ids))
    return hits / len(relevant_ids)


def percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile (no interpolation), easy to verify by hand."""
    ordered = sorted(values)
    rank = max(1, round(pct / 100 * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


def min_max_fuse(
    result_lists: list[list[tuple[str, float]]],
    weights: tuple[float, ...] = HYBRID_WEIGHTS,
    k: int = TOP_K,
) -> list[tuple[str, float]]:
    """Client-side version of OpenSearch's normalization processor.

    Each input is a list of (doc_id, raw_score). Scores are min-max normalized
    per list, then combined with a weighted arithmetic mean. A doc missing from
    a list contributes 0 for that list, like the server-side processor.
    """
    fused: dict[str, float] = {}
    for results, weight in zip(result_lists, weights):
        if not results:
            continue
        scores = [s for _, s in results]
        lo, hi = min(scores), max(scores)
        for doc_id, score in results:
            norm = 1.0 if hi == lo else (score - lo) / (hi - lo)
            fused[doc_id] = fused.get(doc_id, 0.0) + weight * norm
    total = sum(weights)
    ranked = sorted(((d, s / total) for d, s in fused.items()), key=lambda x: -x[1])
    return ranked[:k]


# ---------------------------------------------------------------------------
# Dataset validation
# ---------------------------------------------------------------------------

_STOPWORDS = {
    # English
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "can", "do", "does",
    "for", "from", "get", "got", "how", "i", "if", "in", "into", "is", "it", "me",
    "my", "no", "not", "of", "on", "or", "our", "so", "that", "the", "their",
    "them", "then", "there", "this", "to", "too", "up", "us", "was", "we", "what",
    "when", "where", "which", "who", "why", "will", "with", "you", "your",
    # Spanish
    "al", "con", "de", "del", "el", "en", "es", "la", "las", "lo", "los", "mi",
    "mis", "para", "por", "que", "se", "su", "sus", "tu", "un", "una", "y", "cómo",
    "qué",
}


def content_tokens(text: str) -> set[str]:
    words = re.findall(r"[\wáéíóúñü]+", text.lower())
    return {w for w in words if w not in _STOPWORDS and len(w) > 2}


def validate_dataset(verbose: bool = True) -> None:
    docs = load_docs()
    queries = load_queries()
    doc_by_id = {d["id"]: d for d in docs}

    assert len(doc_by_id) == len(docs), "duplicate doc ids"
    assert all(d["text"].strip() for d in docs), "empty doc text"
    assert len({q["id"] for q in queries}) == len(queries), "duplicate query ids"

    for q in queries:
        assert q["kind"] in QUERY_KINDS, f"{q['id']}: bad kind {q['kind']}"
        assert q["relevant_ids"], f"{q['id']}: no relevant_ids"
        for rid in q["relevant_ids"]:
            assert rid in doc_by_id, f"{q['id']}: unknown doc id {rid}"

    # Semantic queries must not share content words with their relevant docs,
    # otherwise BM25 could solve them and the comparison would be unfair.
    for q in queries:
        relevant_tokens = set().union(*(content_tokens(doc_by_id[r]["text"]) for r in q["relevant_ids"]))
        overlap = content_tokens(q["text"]) & relevant_tokens
        if q["kind"] == "semantic":
            assert not overlap, f"{q['id']}: semantic query shares words {overlap}"
        if q["kind"] == "keyword":
            assert overlap, f"{q['id']}: keyword query shares no words with its docs"
        if verbose:
            print(f"{q['id']} [{q['kind']:<8}] shared words with relevant docs: {sorted(overlap) or '-'}")

    if verbose:
        langs: dict[str, int] = {}
        for d in docs:
            langs[d["lang"]] = langs.get(d["lang"], 0) + 1
        kinds: dict[str, int] = {}
        for q in queries:
            kinds[q["kind"]] = kinds.get(q["kind"], 0) + 1
        codes = sorted({c for d in docs for c in re.findall(r"E-\d{4}", d["text"])})
        print(f"\ndocs: {len(docs)}  languages: {langs}")
        print(f"queries: {len(queries)}  kinds: {kinds}")
        print(f"error codes in corpus: {codes}")
        print("dataset OK")


if __name__ == "__main__":
    validate_dataset()
