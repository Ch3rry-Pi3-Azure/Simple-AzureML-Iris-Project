#!/bin/bash

# Delete the managed online endpoint. The script also deletes the named
# deployment first so teardown happens in a predictable order.

set -e

ENDPOINT_NAME="${ENDPOINT_NAME:-roger-iris-endpoint-01}"
POLL_SECONDS="${POLL_SECONDS:-10}"
MAX_POLLS="${MAX_POLLS:-60}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! az ml online-endpoint show --name "$ENDPOINT_NAME" >/dev/null 2>&1; then
  echo "Endpoint '$ENDPOINT_NAME' does not exist. Nothing to delete."
  exit 0
fi

"$SCRIPT_DIR/destroy-deployment.sh"

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
