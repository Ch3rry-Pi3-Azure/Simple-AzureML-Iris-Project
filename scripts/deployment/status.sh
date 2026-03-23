#!/bin/bash

# Show the current provisioning state for the managed online deployment.

ENDPOINT_NAME="${ENDPOINT_NAME:-roger-iris-endpoint-01}"
DEPLOYMENT_NAME="${DEPLOYMENT_NAME:-blue}"

echo "Checking deployment status for '$DEPLOYMENT_NAME' on endpoint '$ENDPOINT_NAME'..."
az ml online-deployment show \
  --endpoint-name "$ENDPOINT_NAME" \
  --name "$DEPLOYMENT_NAME" \
  --query provisioning_state
