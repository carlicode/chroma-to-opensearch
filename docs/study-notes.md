# Study notes

Our companion notes for the repo: a glossary first, then the "why" behind each step.
We keep them short on purpose; the notebooks show the "how".

## Glossary

| Term | What it means for us |
|---|---|
| **Chunk / document** | One short piece of text (one line in `data/docs.jsonl`). OpenSearch calls each stored item a *document*. |
| **Embedding / vector** | A list of numbers that represents the meaning of a text. Titan Text Embeddings V2 gives us 1024 numbers per text. |
| **Embedding function (Chroma)** | The function Chroma calls to turn text into vectors. If we do not pass one, Chroma uses `all-MiniLM-L6-v2` (384 dims) behind the scenes. |
| **Collection (Chroma)** | A named set of documents + vectors inside a Chroma database. Roughly an "index" in OpenSearch terms. |
| **Index (OpenSearch)** | A named set of documents with a *mapping*. We use one index, `kb-docs`. |
| **Mapping** | The schema of an index: which fields exist and how each one is stored and searched. |
| **`text` field type** | Analyzed full text: split into terms so BM25 can score it. |
| **`keyword` field type** | Stored as-is, for exact match, filters and aggregations (`id`, `lang`, `topic`). |
| **`knn_vector` field type** | A vector field that supports approximate nearest-neighbour search. Needs a fixed `dimension`. |
| **BM25** | The classic lexical ranking function. Scores documents by how often the query terms appear, weighted by how rare they are. Great for exact tokens like `E-1042`. |
| **k-NN** | k-nearest-neighbours: find the k vectors closest to the query vector. Great for paraphrases. |
| **HNSW** | Hierarchical Navigable Small World: the graph algorithm that makes k-NN fast (approximate, not exhaustive). Chroma uses it too. |
| **Faiss** | A vector-search library. In OpenSearch it is one of the *engines* that can build the HNSW graph. |
| **`cosinesimil`** | Cosine similarity, OpenSearch's name for the space type. Chroma calls the same idea `cosine`. |
| **Hybrid search** | Running BM25 and k-NN together and merging the two result lists. |
| **Search pipeline** | A named chain of processors that OpenSearch applies around a search request. |
| **Normalization processor** | A search-pipeline processor that rescales the BM25 and k-NN scores to a common range and combines them. |
| **Min-max normalization** | `(score - min) / (max - min)`: maps each list to the 0..1 range. |
| **recall@3** | Of the relevant documents for a query, the share that appears in the top 3 results. |
| **p50 / p95 latency** | The median and the 95th-percentile response time. p95 shows the slow tail. |
| **OpenSearch Serverless** | AWS-managed OpenSearch without clusters to size. We pay for compute in OCUs plus storage. |
| **OCU** | OpenSearch Compute Unit, the billing unit for Serverless compute (indexing and search are counted separately). |
| **Collection (Serverless)** | The Serverless equivalent of a domain/cluster. It has an endpoint and holds indexes. Type `VECTORSEARCH` for us. |
| **NextGen collection** | The newer generation of Serverless collections. It lives in a *collection group* and can scale to zero. |
| **Collection group** | A container for NextGen collections that sets min and max OCUs. With min = 0, idle collections release all compute. |
| **Encryption policy** | Says which KMS key encrypts the collection. Required before the collection can be created. |
| **Network policy** | Says whether the endpoint is reachable from the internet or only from a VPC. |
| **Data-access policy** | Says which IAM principals may do what on collections and indexes. Separate from IAM. |
| **SigV4** | AWS Signature Version 4: every HTTP request is signed with our credentials. `aoss` is the service name for Serverless. |
| **Default credential chain** | The order in which boto3 looks for credentials: env vars, `~/.aws/config` / SSO, instance or container roles. |

## Why each step

### Module 1: Chroma

- **Why show the default embedding first?** Because Chroma hides it. `collection.add(documents=...)`
  silently downloads and runs a local model. Moving to OpenSearch, nothing does that for us:
  we must compute vectors ourselves.
- **Why re-run with Titan vectors?** A comparison is only fair if both stores search the
  *same* vectors. Otherwise we would be comparing two embedding models, not two databases.
- **Why `embedding_function=None` and `query_embeddings`?** So Chroma never computes its own
  vectors behind our back, neither at insert time nor at query time.
- **Why the cache in `data/cache/`?** Bedrock calls cost money and time. Embedding the corpus once
  and reusing the vectors keeps every later run identical and cheap.

### Module 2: OpenSearch Serverless

- **Why three policies before the collection?** Serverless separates concerns: encryption
  (at rest), network (who can reach the endpoint) and data access (who can read/write). The
  collection cannot be created without a matching encryption policy, and we cannot call it
  without the other two.
- **Why a collection group with min OCU = 0?** It lets NextGen collections release all
  compute when idle, so a study project does not keep billing while we sleep. The trade-off
  is a cold start on the first request after idle.
- **Why an explicit mapping?** Without it, OpenSearch guesses field types from the first
  document, and a list of floats would become a plain number array, not a vector.
  `dimension` must match the embedding model exactly (1024 for Titan V2).
- **Why does notebook 02 retry without `engine`?** AWS documents that NextGen vector
  collections pick the engine internally. We ask for `faiss` explicitly as the task requires,
  and if the service rejects it we show the error and fall back. We have not yet seen which
  branch the real service takes.
- **Why poll the document count?** Serverless indexing is not instantly searchable; documents
  appear after a short refresh delay.

### Module 3: Hybrid search

- **Why hybrid at all?** BM25 finds exact tokens (`E-1042`, `PayFlow`); k-NN finds meaning
  ("I forgot my login secret" → the password reset article). Neither wins everywhere.
- **Why normalize?** BM25 scores are unbounded (2.97, 1.68, ...); cosine scores live
  roughly in 0..1. Adding them raw would let BM25 dominate. Min-max puts both on 0..1 first.
- **Why a client-side fallback?** If a store does not support the `hybrid` query or search
  pipelines, we can still run both queries and fuse the lists in Python (`min_max_fuse` in
  `src/common.py`). Same math, two round trips instead of one.
- **Why can server-side and client-side hybrid differ?** They fuse different candidate
  lists: the server fuses the hits each sub-query returns for the requested size, while our
  client fuses 10 candidates per list. Ties (score 0.5 for "top of one list, absent from the
  other") can also be ordered differently.

### Benchmark

- **Why warm up?** The first requests pay one-off costs (connections, caches, and on
  Serverless possibly a cold start from zero OCUs). We exclude them from the timings.
- **Why time only the store call?** Query vectors are computed and cached before timing, so
  we measure the database, not Bedrock.
- **Why are Chroma and OpenSearch latencies not like-for-like?** Chroma runs in-process on our
  machine; OpenSearch Serverless adds a network round trip and SigV4 signing.
- **Why split by query kind?** The averages hide the story: BM25 should shine on `keyword`,
  vectors on `semantic`, hybrid on `mixed`.
