"""Central configuration, loaded from environment variables."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CORPUS_PATH = DATA_DIR / "corpus.jsonl"
EMBEDDINGS_PATH = DATA_DIR / "embeddings.jsonl"
QUERIES_PATH = DATA_DIR / "queries.txt"
CHROMA_PATH = ROOT / "chroma_data"

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

# Embedding model. EMBED_DIMENSION must match the model's actual output size,
# or OpenSearch rejects every document at index time.
#   amazon.titan-embed-text-v2:0 -> 1024 (also supports 256 and 512)
#   amazon.titan-embed-text-v1   -> 1536 (fixed)
BEDROCK_EMBED_MODEL = os.getenv("BEDROCK_EMBED_MODEL", "amazon.titan-embed-text-v2:0")
EMBED_DIMENSION = int(os.getenv("EMBED_DIMENSION", "1024"))

AOSS_COLLECTION_NAME = os.getenv("AOSS_COLLECTION_NAME", "chroma-to-opensearch")
AOSS_ENDPOINT = os.getenv("AOSS_ENDPOINT", "")
OPENSEARCH_INDEX = os.getenv("OPENSEARCH_INDEX", "docs")
SEARCH_PIPELINE = os.getenv("SEARCH_PIPELINE", "hybrid-normalization")

CHROMA_COLLECTION = "docs"
TOP_K = int(os.getenv("TOP_K", "5"))
