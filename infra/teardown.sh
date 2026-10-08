#!/usr/bin/env bash
# Deletes EVERYTHING created by create_collection.sh:
#   collection (with its indexes and search pipelines), collection group,
#   data-access policy, network policy, encryption policy.
#
# Idempotent: resources that are already gone are reported and skipped, so it is
# safe to run twice. Order matters: the group must be empty before it can be
# deleted, and the encryption policy cannot be removed while a collection uses it.
set -euo pipefail

REGION="${AWS_REGION:-us-east-1}"
COLLECTION="${COLLECTION_NAME:-c2o-vectors}"
GROUP="${COLLECTION_GROUP_NAME:-c2o-group}"
POLL_SECONDS="${POLL_SECONDS:-20}"

export AWS_REGION="$REGION"
export AWS_PAGER=""
aoss() { aws opensearchserverless "$@" --region "$REGION"; }

# --- 1. Collection -------------------------------------------------------------------
COLLECTION_ID="$(aoss batch-get-collection --names "$COLLECTION" --query 'collectionDetails[0].id' --output text)"
if [[ -n "$COLLECTION_ID" && "$COLLECTION_ID" != "None" ]]; then
  STATUS="$(aoss batch-get-collection --ids "$COLLECTION_ID" --query 'collectionDetails[0].status' --output text)"
  if [[ "$STATUS" != "DELETING" ]]; then
    aoss delete-collection --id "$COLLECTION_ID" >/dev/null
  fi
  echo "deleting collection $COLLECTION ($COLLECTION_ID)..."
  for _ in $(seq 1 60); do
    STATUS="$(aoss batch-get-collection --ids "$COLLECTION_ID" --query 'collectionDetails[0].status' --output text)"
    [[ -z "$STATUS" || "$STATUS" == "None" ]] && break
    echo "  status: $STATUS, waiting ${POLL_SECONDS} s..."
    sleep "$POLL_SECONDS"
  done
  if [[ -n "$STATUS" && "$STATUS" != "None" ]]; then
    echo "Collection still $STATUS; re-run teardown later." >&2
    exit 1
  fi
  echo "collection deleted"
else
  echo "collection $COLLECTION not found, skipping"
fi

# --- 2. Collection group (must be empty) ---------------------------------------------
GROUP_ID="$(aoss batch-get-collection-group --names "$GROUP" --query 'collectionGroupDetails[0].id' --output text)"
if [[ -n "$GROUP_ID" && "$GROUP_ID" != "None" ]]; then
  aoss delete-collection-group --id "$GROUP_ID" >/dev/null
  echo "collection group $GROUP deleted"
else
  echo "collection group $GROUP not found, skipping"
fi

# --- 3. Policies -----------------------------------------------------------------------
if aoss get-access-policy --type data --name "${COLLECTION}-access" >/dev/null 2>&1; then
  aoss delete-access-policy --type data --name "${COLLECTION}-access" >/dev/null
  echo "data-access policy deleted"
else
  echo "data-access policy not found, skipping"
fi

for TYPE in network encryption; do
  SHORT="$([[ "$TYPE" == network ]] && echo net || echo enc)"
  NAME="${COLLECTION}-${SHORT}"
  if aoss get-security-policy --type "$TYPE" --name "$NAME" >/dev/null 2>&1; then
    aoss delete-security-policy --type "$TYPE" --name "$NAME" >/dev/null
    echo "$TYPE policy deleted"
  else
    echo "$TYPE policy not found, skipping"
  fi
done

echo
echo "Teardown complete. Verify in the console that no collections or collection groups remain:"
echo "  aws opensearchserverless list-collections --region $REGION"
echo "  aws opensearchserverless list-collection-groups --region $REGION"
