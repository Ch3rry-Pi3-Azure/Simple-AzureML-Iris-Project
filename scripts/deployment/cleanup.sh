#!/bin/bash

# Delete the managed online endpoint and all attached deployments.

ENDPOINT_NAME="${ENDPOINT_NAME:-roger-iris-endpoint-01}"

echo "Deleting endpoint '$ENDPOINT_NAME'..."

az ml online-endpoint delete \
  --name "$ENDPOINT_NAME" \
  --yes

echo "Endpoint deleted."
