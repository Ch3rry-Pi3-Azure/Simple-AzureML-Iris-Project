#!/bin/bash

# Fetch the most recent deployment container logs from Azure ML.

ENDPOINT_NAME="${ENDPOINT_NAME:-roger-iris-endpoint-01}"
DEPLOYMENT_NAME="${DEPLOYMENT_NAME:-blue}"

echo "Fetching logs for deployment '$DEPLOYMENT_NAME' on endpoint '$ENDPOINT_NAME'..."
az ml online-deployment get-logs \
  --endpoint-name "$ENDPOINT_NAME" \
  --name "$DEPLOYMENT_NAME" \
  --lines 200
