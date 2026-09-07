# chroma-to-opensearch

The same corpus, the same embeddings, and the same queries, running in **Chroma**
(local, embedded) and in **Amazon OpenSearch Serverless** (distributed, on AWS) —
so the difference between prototyping a RAG layer and running one in production
can be measured instead of asserted.

Companion repository for the article *From Chroma to OpenSearch: A Practical
Guide to Taking Vector Search to Production on AWS*.

## Why it is built this way

Embeddings are generated **once** with Amazon Bedrock and written to
`data/embeddings.jsonl`. Both stores load from that same file.

If each store generated its own vectors, any difference in results would be
noise from the embedding step rather than a property of the store, and the
comparison would say nothing. This constraint is the reason the repo exists in
this shape.

## What is measured

| Question | Where it is answered |
|---|---|
| How different is the code, really? | `src/chroma_store.py` vs `src/opensearch_store.py` |
| How long until the first query works? | Wall-clock time through the steps below |
| How does retrieval latency compare? | `src/benchmark.py` |
| What does hybrid search add? | `search_knn` vs `search_hybrid_*` |
| What does an idle collection cost? | AWS Cost Explorer, after the collection has been running |

## Requirements

- Python 3.10+
- An AWS account with credentials configured (`aws configure` or environment variables)
- Bedrock model access enabled for the embedding model you choose

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

`EMBED_DIMENSION` in `.env` must match the model in `BEDROCK_EMBED_MODEL`.
Titan V2 returns 1024 dimensions by default; Titan V1 returns 1536. A mismatch
is rejected at index time by OpenSearch, not at embedding time.

## Running it

**1. Generate the shared embeddings**

```bash
python -m src.embed
```

**2. Load and query Chroma** — no infrastructure required, no AWS cost beyond the
embedding calls already made.

```bash
python -m src.chroma_store
```

**3. Create the OpenSearch Serverless collection** — this is where AWS billing
starts.

```bash
python infra/create_collection.py
```

Copy the printed `AOSS_ENDPOINT` into `.env`.

**4. Load and query OpenSearch**

```bash
python -m src.opensearch_store
```

**5. Compare latency**

```bash
python -m src.benchmark
```

**6. Tear everything down**

```bash
python infra/teardown.py
```

## Cost warning

An OpenSearch Serverless collection is billed per hour for as long as it exists,
whether or not it serves a single query — capacity has a floor, so there is no
scale-to-zero. Deleting the index does not stop billing; deleting the collection
does. Run `infra/teardown.py` when you are finished, and confirm in Cost Explorer.

Chroma, by contrast, costs nothing beyond the machine it already runs on.

## Known failure modes

| Symptom | Cause |
|---|---|
| `AccessDeniedException` from Bedrock | The model is not enabled under Model access for this account and region |
| 403 from the collection | Missing data access policy, or SigV4 signed against `es` instead of `aoss` |
| Documents rejected at index time | `EMBED_DIMENSION` does not match the mapping's `dimension` |
| `hybrid` query fails | No search pipeline with a normalization processor; use the client-side RRF fallback |
| Charges continue after finishing | The collection still exists — deleting the index is not enough |

## Layout

```
data/corpus.jsonl        Source documents
data/queries.txt         Query set for the benchmark
data/embeddings.jsonl    Generated once, shared by both stores (gitignored)
src/embed.py             Bedrock embeddings
src/chroma_store.py      Chroma: build and query
src/opensearch_store.py  OpenSearch: index, k-NN, BM25, hybrid, RRF fallback
src/benchmark.py         Latency comparison
infra/create_collection.py  Collection plus its three required policies
infra/teardown.py           Delete everything and stop billing
```
