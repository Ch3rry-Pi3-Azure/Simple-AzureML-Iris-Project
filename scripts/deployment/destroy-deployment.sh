#!/bin/bash

# Delete the managed online deployment from the configured endpoint and
# wait until Azure has fully removed it before returning.

set -e

ENDPOINT_NAME="${ENDPOINT_NAME:-roger-iris-endpoint-01}"
DEPLOYMENT_NAME="${DEPLOYMENT_NAME:-blue}"
POLL_SECONDS="${POLL_SECONDS:-10}"
MAX_POLLS="${MAX_POLLS:-60}"

if ! az ml online-endpoint show --name "$ENDPOINT_NAME" >/dev/null 2>&1; then
  echo "Endpoint '$ENDPOINT_NAME' does not exist. Skipping deployment delete."
  exit 0
fi

if ! az ml online-deployment show \
  --endpoint-name "$ENDPOINT_NAME" \
  --name "$DEPLOYMENT_NAME" >/dev/null 2>&1; then
  echo "Deployment '$DEPLOYMENT_NAME' does not exist on endpoint '$ENDPOINT_NAME'."
  exit 0
fi

echo "Deleting deployment '$DEPLOYMENT_NAME' from endpoint '$ENDPOINT_NAME'..."

az ml online-deployment delete \
  --endpoint-name "$ENDPOINT_NAME" \
  --name "$DEPLOYMENT_NAME" \
  --yes

echo "Waiting for deployment '$DEPLOYMENT_NAME' to be fully deleted..."

for ((poll_count=1; poll_count<=MAX_POLLS; poll_count++)); do
  if ! az ml online-deployment show \
    --endpoint-name "$ENDPOINT_NAME" \
    --name "$DEPLOYMENT_NAME" >/dev/null 2>&1; then
    echo "Deployment '$DEPLOYMENT_NAME' has been deleted."
    exit 0
  fi

  echo "Deployment still exists. Poll $poll_count/$MAX_POLLS..."
  sleep "$POLL_SECONDS"
done

echo "Timed out waiting for deployment '$DEPLOYMENT_NAME' to delete."
exit 1
