"""Generate embeddings once with Amazon Bedrock; both stores read the result.

Run:  python -m src.embed
"""

import json

import boto3

from src import config
from src.corpus import load_corpus


def bedrock_client():
    return boto3.client("bedrock-runtime", region_name=config.AWS_REGION)


def embed_text(client, text: str) -> list[float]:
    if "titan-embed-text-v2" in config.BEDROCK_EMBED_MODEL:
        payload = {
            "inputText": text,
            "dimensions": config.EMBED_DIMENSION,
            "normalize": True,
        }
    else:
        # Titan v1 has a fixed output size and takes no dimension argument.
        payload = {"inputText": text}

    response = client.invoke_model(
        modelId=config.BEDROCK_EMBED_MODEL,
        body=json.dumps(payload),
    )
    return json.loads(response["body"].read())["embedding"]


def main() -> None:
    client = bedrock_client()
    documents = load_corpus()
    print(f"Embedding {len(documents)} documents with {config.BEDROCK_EMBED_MODEL}")

    with config.EMBEDDINGS_PATH.open("w", encoding="utf-8") as out:
        for position, document in enumerate(documents, start=1):
            vector = embed_text(client, document["text"])

            if position == 1 and len(vector) != config.EMBED_DIMENSION:
                raise SystemExit(
                    f"Model returned {len(vector)}-d vectors but EMBED_DIMENSION "
                    f"is {config.EMBED_DIMENSION}. Update .env before continuing."
                )

            out.write(json.dumps({**document, "embedding": vector}) + "\n")
            print(f"  [{position}/{len(documents)}] {document['id']}")

    print(f"Wrote {config.EMBEDDINGS_PATH}")


if __name__ == "__main__":
    main()
