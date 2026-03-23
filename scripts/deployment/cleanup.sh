#!/bin/bash

# Delete the managed online deployment first, then delete the endpoint.
# This makes teardown clearer and avoids races where the deployment is
# still visible while the endpoint delete is in progress.

set -e

ENDPOINT_NAME="${ENDPOINT_NAME:-roger-iris-endpoint-01}"
DEPLOYMENT_NAME="${DEPLOYMENT_NAME:-blue}"
POLL_SECONDS="${POLL_SECONDS:-10}"
MAX_POLLS="${MAX_POLLS:-60}"

if ! az ml online-endpoint show --name "$ENDPOINT_NAME" >/dev/null 2>&1; then
  echo "Endpoint '$ENDPOINT_NAME' does not exist. Nothing to delete."
  exit 0
fi

if az ml online-deployment show \
  --endpoint-name "$ENDPOINT_NAME" \
  --name "$DEPLOYMENT_NAME" >/dev/null 2>&1; then
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
      break
    fi

    echo "Deployment still exists. Poll $poll_count/$MAX_POLLS..."
    sleep "$POLL_SECONDS"
  done

  if az ml online-deployment show \
    --endpoint-name "$ENDPOINT_NAME" \
    --name "$DEPLOYMENT_NAME" >/dev/null 2>&1; then
    echo "Timed out waiting for deployment '$DEPLOYMENT_NAME' to delete."
    exit 1
  fi
else
  echo "Deployment '$DEPLOYMENT_NAME' does not exist. Skipping deployment delete."
fi

echo "Deleting endpoint '$ENDPOINT_NAME'..."

az ml online-endpoint delete \
  --name "$ENDPOINT_NAME" \
  --yes

echo "Waiting for endpoint '$ENDPOINT_NAME' to be fully deleted..."

for ((poll_count=1; poll_count<=MAX_POLLS; poll_count++)); do
  if ! az ml online-endpoint show --name "$ENDPOINT_NAME" >/dev/null 2>&1; then
    echo "Endpoint '$ENDPOINT_NAME' has been deleted."
    exit 0
  fi

  echo "Endpoint still exists. Poll $poll_count/$MAX_POLLS..."
  sleep "$POLL_SECONDS"
done

echo "Timed out waiting for endpoint '$ENDPOINT_NAME' to delete."
exit 1
