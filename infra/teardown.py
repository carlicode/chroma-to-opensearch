"""Delete the collection and its policies so billing stops.

Run:  python infra/teardown.py

A Serverless collection accrues cost for as long as it exists. Deleting the
index is not enough — the collection itself has to go.
"""

import os

import boto3

REGION = os.getenv("AWS_REGION", "us-east-1")
NAME = os.getenv("AOSS_COLLECTION_NAME", "chroma-to-opensearch")

aoss = boto3.client("opensearchserverless", region_name=REGION)


def main() -> None:
    confirmation = input(f"Delete collection '{NAME}' and its policies? [y/N] ")
    if confirmation.strip().lower() != "y":
        print("Aborted.")
        return

    details = aoss.batch_get_collection(names=[NAME])["collectionDetails"]
    if details:
        aoss.delete_collection(id=details[0]["id"])
        print(f"Deleted collection {NAME}")
    else:
        print(f"Collection {NAME} not found")

    for policy_type, suffix in (("encryption", "encryption"), ("network", "network")):
        try:
            aoss.delete_security_policy(name=f"{NAME}-{suffix}", type=policy_type)
            print(f"Deleted {policy_type} policy")
        except aoss.exceptions.ResourceNotFoundException:
            print(f"{policy_type} policy not found")

    try:
        aoss.delete_access_policy(name=f"{NAME}-data", type="data")
        print("Deleted data access policy")
    except aoss.exceptions.ResourceNotFoundException:
        print("data access policy not found")

    print("\nConfirm in Cost Explorer that charges stop within the next billing period.")


if __name__ == "__main__":
    main()
