# chroma-to-opensearch

Companion repo for the article **"From Chroma to OpenSearch: A Practical Guide to Taking Vector
Search to Production on AWS"**.

We run the same retrieval flow (no LLM generation step) on two stores:

- **Chroma**, locally, the way most of us start.
- **Amazon OpenSearch Serverless**, a NextGen `VECTORSEARCH` collection in `us-east-1`.

Both stores get the **same 51 documents**, the **same 15 queries** and the **same Titan Text
Embeddings V2 vectors (1024 dims)**, so the numbers we compare come from the databases, not from
different embedding models. Every number in the article is produced by the code here; nothing is
hardcoded.

> **Verification status.** Everything was executed for real on 2026-10-08, locally and against an
> OpenSearch Serverless NextGen collection in `us-east-1` with Titan V2 embeddings. Results and the
> things only a real run could reveal are in [docs/aws-run-2026-10-08.md](docs/aws-run-2026-10-08.md).
> The IAM policy below was exercised by a broader user, so it is still not validated as minimal.

```
data/docs.jsonl          51 short IT/HR chunks (English + Spanish)
data/queries.jsonl       15 queries: 5 keyword, 5 semantic, 5 mixed, with relevant_ids
src/common.py            loaders, constants, SigV4 client, recall@k, score fusion, dataset validator
src/embeddings.py        Titan V2 via Bedrock (+ offline fallback), cached in data/cache/
notebooks/01..03         the three modules
benchmark.py             p50/p95 latency + recall@3, per query kind, both stores
infra/                   create_collection.sh / teardown.sh (AWS CLI v2)
docs/study-notes.md      glossary + the "why" behind each step
article/article.md        the article this repo accompanies (images in article/images/)
```

---

## 0. Prerequisites

- **Python 3.11+** (we tested with 3.12; `ipykernel` 7 does not install on 3.10).
- **AWS CLI v2**, recent enough to have `aws opensearchserverless create-collection-group`
  (check with `aws opensearchserverless create-collection-group help`).
- **An IAM user or role** (never root) in the default credential chain: environment variables,
  `aws configure`, or `aws sso login`. Check with `aws sts get-caller-identity`.
- **Bedrock access to `amazon.titan-embed-text-v2:0`** in `us-east-1`.
- Modules 2, 3 and the OpenSearch half of the benchmark create **billable** resources
  (see [Cost](#cost-and-scale-to-zero)). Module 1 is free and fully local, except for the
  Bedrock calls that embed 51 short documents and 15 queries once (then cached).

### Minimal IAM permissions

This is the smallest set we derived from the API calls the repo makes. We have not yet
validated it in a real account, so if a call is denied, the error message names the missing
action.

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "TitanEmbeddings",
      "Effect": "Allow",
      "Action": "bedrock:InvokeModel",
      "Resource": "arn:aws:bedrock:us-east-1::foundation-model/amazon.titan-embed-text-v2:0"
    },
    {
      "Sid": "ServerlessControlPlane",
      "Effect": "Allow",
      "Action": [
        "aoss:CreateSecurityPolicy", "aoss:GetSecurityPolicy", "aoss:DeleteSecurityPolicy",
        "aoss:CreateAccessPolicy", "aoss:GetAccessPolicy", "aoss:DeleteAccessPolicy",
        "aoss:CreateCollectionGroup", "aoss:BatchGetCollectionGroup",
        "aoss:ListCollectionGroups", "aoss:DeleteCollectionGroup",
        "aoss:AddCollectionToCollectionGroup",
        "aoss:CreateCollection", "aoss:BatchGetCollection",
        "aoss:ListCollections", "aoss:DeleteCollection"
      ],
      "Resource": "*"
    },
    {
      "Sid": "ServerlessDataPlane",
      "Effect": "Allow",
      "Action": "aoss:APIAccessAll",
      "Resource": "*"
    },
    {
      "Sid": "FirstCollectionInAccountOnly",
      "Effect": "Allow",
      "Action": "iam:CreateServiceLinkedRole",
      "Resource": "*",
      "Condition": {"StringEquals": {"iam:AWSServiceName": "observability.aoss.amazonaws.com"}}
    }
  ]
}
```

- `aoss:APIAccessAll` lets IAM *allow* data-plane calls (index, search). The collection's
  **data-access policy** must *also* allow them; `create_collection.sh` creates that policy for
  the principal running the script.
- `iam:CreateServiceLinkedRole` is only needed when the account creates its first Serverless
  collection (AWS documents this in "Configuring permissions for collections").
- `sts:GetCallerIdentity` needs no permission.

### Cost and scale-to-zero

- We use a **NextGen collection group** with `minIndexingCapacityInOCU=0` and
  `minSearchCapacityInOCU=0`. AWS documents that NextGen OCUs scale to zero after about
  **10 minutes** without traffic, so an idle collection should not bill compute.
- The **first request after idle** wakes the collection up; expect it to take **10 to 30 s**.
  The notebooks print how long that first request took, and the benchmark warms up before timing.
- The group caps compute at `MAX_OCU` (default `2`). While it runs, we pay OCU-hours for
  indexing and search plus storage. Check current prices at
  <https://aws.amazon.com/opensearch-service/pricing/> before running.
- Bedrock Titan V2 bills per input token. Our corpus is small and vectors are cached in
  `data/cache/`, so re-runs do not call Bedrock again.
- **Always run [Teardown](#6-teardown) when done.**

---

## 1. Setup

```bash
git clone <this repo> && cd chroma-to-opensearch
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # AWS_REGION=us-east-1, EMBEDDING_PROVIDER=titan
python -m src.common            # validates the dataset, must end with "dataset OK"
```

`.env` is in `.gitignore`. It never holds credentials: boto3 reads those from the default chain.

**Offline practice mode.** Setting `EMBEDDING_PROVIDER=local` uses Chroma's built-in
`all-MiniLM-L6-v2` (384 dims) instead of Titan, so Module 1 and the Chroma benchmark run without
AWS. Those numbers are **not** comparable with the article (different model, different
dimension), and notebook 02 prints a warning when the dimension is not 1024.

---

## 2. Module 1: Chroma (`notebooks/01_chroma_setup.ipynb`)

```bash
cd notebooks && jupyter lab        # or: jupyter nbconvert --to notebook --execute --inplace 01_chroma_setup.ipynb
```

What the notebook does:

1. Opens a `chromadb.PersistentClient` in `chroma_db/`.
2. Creates a collection **without** an embedding function, adds the raw texts and shows that
   Chroma stored a **384-dim** vector it computed on its own (`all-MiniLM-L6-v2`).
3. Embeds the docs with Titan V2 (1024 dims, cached), recreates the collection with
   `embedding_function=None` and cosine space, and adds our vectors explicitly.
4. Queries with `query_embeddings` and prints a `results` dict: per query, the retrieved ids,
   the relevant ids and recall@3, then recall@3 per query kind.

> **What you learned**
> - Chroma quietly embeds text for us with a local model; OpenSearch will not.
> - For a fair comparison, both stores must search the *same* vectors, so we compute them once
>   and pass them in (`embeddings=` / `query_embeddings=`).
> - `embedding_function=None` guarantees Chroma never mixes in its own model.
> - recall@3 per query kind already shows where pure vector search struggles: exact codes like
>   `E-1042`.

**Common errors**
- *403 from SigV4*: not applicable to Chroma, but the Bedrock call can fail with
  `AccessDeniedException`; check `bedrock:InvokeModel` and model access in `us-east-1`.
- *Wrong vector dimension*: `Collection expecting embedding with dimension of 384, got 1024`
  means an old collection built with another model is still on disk. The notebook deletes and
  recreates it; if we switched providers by hand, delete `chroma_db/`.
- *Collection not ACTIVE yet* / *data-access policy missing*: not applicable to Chroma (no AWS
  resources yet).
- `NoCredentialsError`: no credentials in the default chain; run `aws configure` or
  `aws sso login`.

---

## 3. Module 2: OpenSearch Serverless (`notebooks/02_opensearch_setup.ipynb`)

### 3.1 Create the collection (billable)

```bash
aws sts get-caller-identity          # must NOT end in :root
bash infra/create_collection.sh      # prints OPENSEARCH_HOST=... when ACTIVE
```

The script creates, in order (each step is skipped if it already exists):

| Resource | Name | Billable |
|---|---|---|
| Encryption policy (AWS-owned key) | `c2o-vectors-enc` | no |
| Network policy (public endpoint) | `c2o-vectors-net` | no |
| Data-access policy (our principal only) | `c2o-vectors-access` | no |
| NextGen collection group, min OCU 0, max `MAX_OCU` | `c2o-group` | no by itself |
| NextGen `VECTORSEARCH` collection in that group | `c2o-vectors` | **yes** (OCU-hours + storage) |

Names, region and `MAX_OCU` can be overridden with `COLLECTION_NAME`, `COLLECTION_GROUP_NAME`,
`AWS_REGION` and `MAX_OCU`. If we work through an assumed role whose ARN has a path, we set
`PRINCIPAL_ARN` to the full role ARN.

Copy the printed endpoint into `.env`, **without** `https://`:

```
OPENSEARCH_HOST=abc123xyz.us-east-1.aoss.amazonaws.com
```

### 3.2 Run the notebook

The notebook builds a client signed with **SigV4** (`AWS4Auth`, service **`aoss`**), creates the
`kb-docs` index with an explicit mapping, bulk-indexes the 51 docs with their Titan vectors,
waits until they are searchable and runs k-NN queries. The mapping:

| Field | Type | Why |
|---|---|---|
| `id` | `keyword` | exact id, used to score recall |
| `text` | `text` | analyzed for BM25 (Module 3) |
| `lang`, `topic` | `keyword` | exact filters |
| `embedding` | `knn_vector`, `dimension: 1024`, `method: hnsw`, `engine: faiss`, `space_type: cosinesimil` | the vector; dimension must match Titan V2 |
| index setting | `index.knn: true` | turns on k-NN for the index |

AWS documents that NextGen vector collections choose the engine internally. The notebook first
sends the explicit `hnsw` + `faiss` mapping; if the service rejects it, it prints the error and
retries without `engine`. We have not yet seen which branch a real NextGen collection takes.

> **What you learned**
> - Serverless needs three policies (encryption, network, data access) before a collection is usable.
> - Every request is signed with SigV4 using service name `aoss`, with credentials from the default chain.
> - An explicit mapping makes `embedding` a real `knn_vector`; its `dimension` must equal the model's.
> - New documents become searchable after a short delay, so we poll the count before querying.
> - With the same vectors, k-NN in OpenSearch should return the same top 3 as Chroma; the
>   notebook counts how many queries match.

**Common errors**
- **403 from SigV4** (`AuthorizationException` / `security_exception`): wrong region in `.env`,
  service name not `aoss`, expired SSO session (`aws sso login`), clock skew, or `https://` left
  in `OPENSEARCH_HOST`.
- **Wrong vector dimension** (`mapper_parsing_exception` / `Vector dimension mismatch`): the index
  was created for another dimension (for example with `EMBEDDING_PROVIDER=local`). Delete the
  index (the notebook recreates it) and make sure `EMBEDDING_PROVIDER=titan`.
- **Collection not ACTIVE yet**: `create_collection.sh` waits for `ACTIVE`; if it timed out,
  re-run it (safe) or check `aws opensearchserverless batch-get-collection --names c2o-vectors`.
  Calling the endpoint earlier fails with DNS or 5xx errors.
- **Data-access policy missing** (403 even with valid credentials): the policy does not list our
  principal. For SSO/assumed roles it must contain the *role* ARN, not the `sts` session ARN; set
  `PRINCIPAL_ARN` and re-run `create_collection.sh` after deleting `c2o-vectors-access`.
- **First request is slow (10 to 30 s)**: expected after scale-to-zero, not an error.

---

## 4. Module 3: Hybrid search (`notebooks/03_hybrid_search.ipynb`)

1. Runs **BM25** (`match` on `text`) and **k-NN** separately for the same query, so we can see
   which one finds what.
2. Creates the search pipeline `kb-hybrid-minmax` with a `normalization-processor`
   (`min_max` normalization, `arithmetic_mean` with weights `[0.5, 0.5]`).
3. Sends a `hybrid` query (BM25 + k-NN) through that pipeline and prints **clearly** whether it
   works on the collection we are connected to.
4. Always also runs the **client-side workaround**: two queries (BM25 and k-NN, 10 candidates
   each), min-max normalize each list in Python, weighted mean, top 3 (`min_max_fuse` in
   `src/common.py`). If the server-side query fails, the rest of the notebook uses this.
5. Prints recall@3 per query and per kind for BM25, k-NN, client hybrid and (if supported)
   server hybrid.

### Is the `hybrid` query supported on OpenSearch Serverless?

**Not verified yet.** We ran this notebook against a local OpenSearch 3.3.0 node, where the
pipeline and the `hybrid` query work. We have not run it on a Serverless NextGen collection,
so we do not claim support there. The notebook prints `RESULT: ... WORKS on this collection`
or `RESULT: ... NOT supported` with the server's error message. If it is not supported, the
client-side fusion above is the workaround: same math, two round trips instead of one.

> **What you learned**
> - BM25 wins on exact tokens (error codes, product names); vectors win on paraphrases.
> - BM25 and cosine scores live on different scales, so they must be normalized before combining.
> - A search pipeline with a `normalization-processor` does the fusion server-side in one request.
> - When a store lacks hybrid support, two queries plus client-side min-max fusion give the same idea.

**Common errors**
- **403 from SigV4**: as in Module 2. Creating a search pipeline may need broader data-access
  permissions than indexing documents; `create_collection.sh` grants index and collection-item
  permissions on our collection.
- **Wrong vector dimension**: the query vector must have the index's dimension; do not mix
  providers between Module 2 and Module 3.
- **Collection not ACTIVE yet**: re-run Module 2 first; Module 3 reuses its index.
- **Data-access policy missing**: same fix as Module 2.
- `illegal_argument_exception` / unknown query `hybrid` or pipeline API errors: the server-side
  path is not available on that collection; use the client-side result.

---

## 5. Benchmark (`benchmark.py`)

```bash
python benchmark.py                    # Chroma + OpenSearch
python benchmark.py --stores chroma    # Chroma only
python benchmark.py --repeats 50 --warmup 3
```

- Runs the 15 queries against every mode: `chroma / k-NN`, `opensearch / BM25`,
  `opensearch / k-NN`, `opensearch / hybrid (server)` (only if supported) and
  `opensearch / hybrid (client)`.
- Query vectors are computed (or loaded from cache) **before** timing, so latency covers only
  the store call. OpenSearch latency includes the network round trip from our machine; Chroma
  is in-process, so the latencies are not like-for-like.
- Warm-up passes run first (untimed), then each query is timed `--repeats` times.
- Prints p50/p95 latency (ms) and mean recall@3 per kind (`keyword`, `semantic`, `mixed`, `all`).
- Requires Modules 1 and 2 to have been run (it reads `chroma_db/` and the `kb-docs` index).

---

## 6. Teardown

```bash
bash infra/teardown.sh
bash infra/teardown.sh     # second run should only report "not found, skipping"
```

Deletes, in order: the collection (and with it the index and search pipeline), the collection
group, the data-access policy, the network policy and the encryption policy. It is idempotent:
anything already gone is skipped. At the end it prints `aws` commands to confirm nothing is left.
Local leftovers can be removed with `rm -rf chroma_db data/cache`.

---

## What we ran locally

All commands below were run while building the repo, without AWS credentials, so with
`EMBEDDING_PROVIDER=local` (MiniLM, 384 dims). **These are not the article's numbers**; they
only prove the code paths run. Latencies are from a single development VM.

**Dataset validation** (`python -m src.common`, last lines):

```
docs: 51  languages: {'en': 33, 'es': 18}
queries: 15  kinds: {'keyword': 5, 'semantic': 5, 'mixed': 5}
error codes in corpus: ['E-1042', 'E-1043', 'E-2001', 'E-2210', 'E-3007', 'E-4115', 'E-5003', 'E-6020', 'E-7001']
dataset OK
```

**Module 1**: `notebooks/01_chroma_setup.ipynb` is committed with the outputs of that local run
(Chroma's default embedding: 384 dims; recall@3 by kind with MiniLM: keyword 0.8, semantic 0.6,
mixed 1.0).

**Benchmark, Chroma only** (`EMBEDDING_PROVIDER=local python benchmark.py --stores chroma`):

```
| mode          | kind     |   queries |   p50 ms |   p95 ms |   recall@3 |
|---------------|----------|-----------|----------|----------|------------|
| chroma / k-NN | keyword  |         5 |     0.52 |     0.96 |        0.8 |
| chroma / k-NN | semantic |         5 |     0.67 |     1.15 |        0.6 |
| chroma / k-NN | mixed    |         5 |     0.49 |     0.77 |        1   |
| chroma / k-NN | all      |        15 |     0.52 |     1.01 |        0.8 |
```

**Modules 2 and 3 and the full benchmark against a local OpenSearch 3.3.0 node** (security
disabled, plain HTTP, a test-only shim replaced the SigV4 client; this checks our query and
mapping code, **not** Serverless): the explicit `hnsw` + `faiss` mapping was accepted, k-NN
returned the same top 3 as Chroma for 15/15 queries, and the `hybrid` query with the
normalization pipeline worked. Full benchmark on that node:

```
| mode                         | kind     |   queries |   p50 ms |   p95 ms |   recall@3 |
|------------------------------|----------|-----------|----------|----------|------------|
| chroma / k-NN                | all      |        15 |     0.43 |     0.51 |       0.8  |
| opensearch / BM25            | keyword  |         5 |     2.36 |     4.01 |       1    |
| opensearch / BM25            | semantic |         5 |     2.06 |     2.7  |       0    |
| opensearch / BM25            | mixed    |         5 |     1.95 |     2.28 |       0.8  |
| opensearch / k-NN            | all      |        15 |     2.05 |     2.52 |       0.8  |
| opensearch / hybrid (server) | all      |        15 |     2.21 |     2.66 |       0.83 |
| opensearch / hybrid (client) | all      |        15 |     3.03 |     4.13 |       0.87 |
```

(Excerpt; per-kind rows for every mode are printed by the script.)

**Infra scripts**: `shellcheck` passes on both scripts. Without an AWS account we ran
`create_collection.sh` twice and `teardown.sh` twice against a local mock of the
`opensearchserverless` and `sts` CLI calls: the second create skipped every existing resource,
and the second teardown found nothing to delete. This tests the script logic, not the real API.

## What still needs a real AWS run

```bash
cp .env.example .env
aws sts get-caller-identity
bash infra/create_collection.sh                 # then put OPENSEARCH_HOST in .env
cd notebooks
jupyter nbconvert --to notebook --execute --inplace \
  01_chroma_setup.ipynb 02_opensearch_setup.ipynb 03_hybrid_search.ipynb
cd ..
python benchmark.py
bash infra/teardown.sh && bash infra/teardown.sh
```

Open questions those runs will answer: whether NextGen accepts the explicit `faiss` engine,
whether the `hybrid` query and search pipelines work on Serverless, whether the IAM policy
above is complete, and whether the data-access policy permissions are accepted as written.
