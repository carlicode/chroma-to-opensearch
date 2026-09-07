"""Create an OpenSearch Serverless vector collection and the three policies it needs.

Run:  python infra/create_collection.py

A collection will not serve traffic unless all three exist:
  1. encryption policy  - required before the collection can be created
  2. network policy     - who can reach the endpoint and the Dashboards UI
  3. data access policy - which principal may read and write; missing or
                          misconfigured, this is the usual cause of a 403 that
                          looks like a credentials problem but is not

Billing note: a Serverless collection is billed per hour while it exists,
whether or not it serves a single query. Run infra/teardown.py when finished.
"""

import json
import os
import sys
import time

import boto3

REGION = os.getenv("AWS_REGION", "us-east-1")
NAME = os.getenv("AOSS_COLLECTION_NAME", "chroma-to-opensearch")

aoss = boto3.client("opensearchserverless", region_name=REGION)


def caller_principal() -> str:
    """Return an ARN usable in a data access policy.

    An assumed-role session ARN (arn:aws:sts::...:assumed-role/Role/session) is
    not accepted; it has to be expressed as the underlying IAM role ARN.
    """
    identity = boto3.client("sts", region_name=REGION).get_caller_identity()
    arn = identity["Arn"]
    if ":assumed-role/" in arn:
        account = identity["Account"]
        role_name = arn.split(":assumed-role/")[1].split("/")[0]
        return f"arn:aws:iam::{account}:role/{role_name}"
    return arn


def create_security_policy(policy_type: str, policy: list | dict) -> None:
    try:
        aoss.create_security_policy(
            name=f"{NAME}-{policy_type}",
            type=policy_type,
            policy=json.dumps(policy),
        )
        print(f"Created {policy_type} policy")
    except aoss.exceptions.ConflictException:
        print(f"{policy_type} policy already exists")


def create_access_policy(principal: str) -> None:
    policy = [
        {
            "Rules": [
                {
                    "ResourceType": "collection",
                    "Resource": [f"collection/{NAME}"],
                    "Permission": ["aoss:*"],
                },
                {
                    "ResourceType": "index",
                    "Resource": [f"index/{NAME}/*"],
                    "Permission": ["aoss:*"],
                },
            ],
            "Principal": [principal],
        }
    ]
    try:
        aoss.create_access_policy(
            name=f"{NAME}-data", type="data", policy=json.dumps(policy)
        )
        print(f"Created data access policy for {principal}")
    except aoss.exceptions.ConflictException:
        print("data access policy already exists")


def create_collection() -> None:
    try:
        aoss.create_collection(name=NAME, type="VECTORSEARCH")
        print(f"Creating collection {NAME}")
    except aoss.exceptions.ConflictException:
        print(f"Collection {NAME} already exists")


def wait_for_active(timeout_seconds: int = 900) -> dict:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        details = aoss.batch_get_collection(names=[NAME])["collectionDetails"]
        if not details:
            sys.exit(f"Collection {NAME} not found")
        status = details[0]["status"]
        if status == "ACTIVE":
            return details[0]
        if status == "FAILED":
            sys.exit(f"Collection {NAME} failed to create")
        print(f"  status={status}, waiting...")
        time.sleep(20)
    sys.exit("Timed out waiting for the collection to become ACTIVE")


def main() -> None:
    principal = caller_principal()

    create_security_policy(
        "encryption",
        {
            "Rules": [{"ResourceType": "collection", "Resource": [f"collection/{NAME}"]}],
            "AWSOwnedKey": True,
        },
    )
    create_security_policy(
        "network",
        [
            {
                "Rules": [
                    {"ResourceType": "collection", "Resource": [f"collection/{NAME}"]},
                    {"ResourceType": "dashboard", "Resource": [f"collection/{NAME}"]},
                ],
                "AllowFromPublic": True,
            }
        ],
    )
    create_access_policy(principal)
    create_collection()

    collection = wait_for_active()
    print("\nCollection is ACTIVE.")
    print(f"  AOSS_ENDPOINT={collection['collectionEndpoint']}")
    print(f"  Dashboards:    {collection['dashboardEndpoint']}")
    print("\nCopy the AOSS_ENDPOINT line into your .env file.")


if __name__ == "__main__":
    main()
