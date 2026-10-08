#!/usr/bin/env bash
# Creates everything the OpenSearch modules need, in this order:
#   1. encryption policy   (free)
#   2. network policy      (free)
#   3. data-access policy  (free)
#   4. NextGen collection group with min OCU = 0   (free by itself; caps spend)
#   5. NextGen VECTORSEARCH collection in that group (billable: OCU-hours + storage)
#
# Safe to re-run: every step skips resources that already exist.
# Requires AWS CLI v2 recent enough to have `aws opensearchserverless create-collection-group`.
# Credentials come from the default chain (env vars, ~/.aws, SSO). Never use root.
set -euo pipefail

REGION="${AWS_REGION:-us-east-1}"
COLLECTION="${COLLECTION_NAME:-c2o-vectors}"
GROUP="${COLLECTION_GROUP_NAME:-c2o-group}"
MAX_OCU="${MAX_OCU:-2}"   # allowed values: 2, 4, 8, 16 or multiples of 16
POLL_SECONDS="${POLL_SECONDS:-20}"

export AWS_REGION="$REGION"
export AWS_PAGER=""
aoss() { aws opensearchserverless "$@" --region "$REGION"; }

# --- Who are we? The data-access policy needs an IAM principal ARN. -------------
CALLER_ARN="$(aws sts get-caller-identity --query Arn --output text)"
ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
if [[ "$CALLER_ARN" == *":root" ]]; then
  echo "Refusing to run with root credentials. Use an IAM user or role." >&2
  exit 1
fi
# An SSO / assumed-role session looks like arn:aws:sts::123:assumed-role/RoleName/session.
# Data-access policies need the role ARN instead. Roles with a path need PRINCIPAL_ARN set by hand.
if [[ "$CALLER_ARN" == *":assumed-role/"* ]]; then
  ROLE_NAME="$(cut -d/ -f2 <<<"$CALLER_ARN")"
  DEFAULT_PRINCIPAL="arn:aws:iam::${ACCOUNT_ID}:role/${ROLE_NAME}"
else
  DEFAULT_PRINCIPAL="$CALLER_ARN"
fi
PRINCIPAL="${PRINCIPAL_ARN:-$DEFAULT_PRINCIPAL}"
echo "Region: $REGION | collection: $COLLECTION | group: $GROUP | principal: $PRINCIPAL"

policy_exists() {  # $1 = security policy type (encryption|network), $2 = name
  aoss get-security-policy --type "$1" --name "$2" >/dev/null 2>&1
}

# --- 1. Encryption policy (free) --------------------------------------------------
# Encrypts the collection at rest with an AWS owned KMS key (no KMS key cost).
if policy_exists encryption "${COLLECTION}-enc"; then
  echo "encryption policy exists, skipping"
else
  aoss create-security-policy --type encryption --name "${COLLECTION}-enc" \
    --policy "{\"Rules\":[{\"ResourceType\":\"collection\",\"Resource\":[\"collection/${COLLECTION}\"]}],\"AWSOwnedKey\":true}" >/dev/null
  echo "created encryption policy ${COLLECTION}-enc"
fi

# --- 2. Network policy (free) -----------------------------------------------------
# Public endpoint so we can reach it from a laptop. Requests still need SigV4 + data access.
if policy_exists network "${COLLECTION}-net"; then
  echo "network policy exists, skipping"
else
  aoss create-security-policy --type network --name "${COLLECTION}-net" \
    --policy "[{\"Rules\":[{\"ResourceType\":\"collection\",\"Resource\":[\"collection/${COLLECTION}\"]}],\"AllowFromPublic\":true}]" >/dev/null
  echo "created network policy ${COLLECTION}-net"
fi

# --- 3. Data-access policy (free) -------------------------------------------------
# Lets our principal create indexes, read/write documents and manage search pipelines.
if aoss get-access-policy --type data --name "${COLLECTION}-access" >/dev/null 2>&1; then
  echo "data-access policy exists, skipping"
else
  aoss create-access-policy --type data --name "${COLLECTION}-access" --policy "[{
    \"Rules\":[
      {\"ResourceType\":\"index\",\"Resource\":[\"index/${COLLECTION}/*\"],
       \"Permission\":[\"aoss:CreateIndex\",\"aoss:DeleteIndex\",\"aoss:UpdateIndex\",\"aoss:DescribeIndex\",\"aoss:ReadDocument\",\"aoss:WriteDocument\"]},
      {\"ResourceType\":\"collection\",\"Resource\":[\"collection/${COLLECTION}\"],
       \"Permission\":[\"aoss:CreateCollectionItems\",\"aoss:DescribeCollectionItems\",\"aoss:UpdateCollectionItems\",\"aoss:DeleteCollectionItems\"]}
    ],
    \"Principal\":[\"${PRINCIPAL}\"]}]" >/dev/null
  echo "created data-access policy ${COLLECTION}-access"
fi

# --- 4. NextGen collection group, min OCU = 0 -------------------------------------
# COST: the group itself is not billed. Its capacity limits bound what the
# collections inside it can spend: min 0 OCU (scales to zero after ~10 min idle),
# max $MAX_OCU OCU for indexing and for search.
GROUP_ID="$(aoss batch-get-collection-group --names "$GROUP" --query 'collectionGroupDetails[0].id' --output text 2>/dev/null || true)"
if [[ -n "$GROUP_ID" && "$GROUP_ID" != "None" ]]; then
  echo "collection group exists ($GROUP_ID), skipping"
else
  aoss create-collection-group --name "$GROUP" --generation NEXTGEN --standby-replicas DISABLED \
    --capacity-limits "minIndexingCapacityInOCU=0,maxIndexingCapacityInOCU=${MAX_OCU},minSearchCapacityInOCU=0,maxSearchCapacityInOCU=${MAX_OCU}" >/dev/null
  echo "created collection group $GROUP"
fi

# --- 5. NextGen vector search collection ------------------------------------------
# COST: billable. Indexing and search OCU-hours (per-second billing, scale to zero
# when idle because min OCU = 0) + managed storage per GB-month (tiny for 51 docs).
# See https://aws.amazon.com/opensearch-service/pricing/ for current us-east-1 rates.
COLLECTION_STATUS="$(aoss batch-get-collection --names "$COLLECTION" --query 'collectionDetails[0].status' --output text 2>/dev/null || true)"
if [[ -n "$COLLECTION_STATUS" && "$COLLECTION_STATUS" != "None" ]]; then
  echo "collection exists (status $COLLECTION_STATUS), skipping create"
else
  aoss create-collection --name "$COLLECTION" --type VECTORSEARCH --collection-group-name "$GROUP" >/dev/null
  echo "created collection $COLLECTION"
fi

# --- Wait until ACTIVE (usually a few minutes) --------------------------------------
for _ in $(seq 1 60); do
  COLLECTION_STATUS="$(aoss batch-get-collection --names "$COLLECTION" --query 'collectionDetails[0].status' --output text)"
  [[ "$COLLECTION_STATUS" == "ACTIVE" ]] && break
  if [[ "$COLLECTION_STATUS" == "FAILED" ]]; then echo "collection FAILED" >&2; exit 1; fi
  echo "status: $COLLECTION_STATUS, waiting ${POLL_SECONDS} s..."
  sleep "$POLL_SECONDS"
done
if [[ "$COLLECTION_STATUS" != "ACTIVE" ]]; then
  echo "Collection still $COLLECTION_STATUS after 20 min; re-run this script later." >&2
  exit 1
fi

ENDPOINT="$(aoss batch-get-collection --names "$COLLECTION" --query 'collectionDetails[0].collectionEndpoint' --output text)"
echo
echo "Collection is ACTIVE. Put this in your .env:"
echo "OPENSEARCH_HOST=${ENDPOINT#https://}"
