#!/bin/bash

# Retrieve the endpoint authentication credentials for manual testing.

ENDPOINT_NAME="${ENDPOINT_NAME:-roger-iris-endpoint-01}"

echo "Retrieving credentials for endpoint '$ENDPOINT_NAME'..."
az ml online-endpoint get-credentials \
  --name "$ENDPOINT_NAME"
