#!/bin/bash

# Show the full managed online endpoint definition from Azure ML.

ENDPOINT_NAME="${ENDPOINT_NAME:-roger-iris-endpoint-01}"

echo "Describing endpoint '$ENDPOINT_NAME'..."
az ml online-endpoint show \
  --name "$ENDPOINT_NAME"
