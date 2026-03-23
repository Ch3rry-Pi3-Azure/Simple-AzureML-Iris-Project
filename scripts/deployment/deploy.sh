#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

ENDPOINT_NAME="${ENDPOINT_NAME:-roger-iris-endpoint-01}"
DEPLOYMENT_NAME="${DEPLOYMENT_NAME:-blue}"

echo "Creating endpoint..."

az ml online-endpoint create \
  --file "$PROJECT_ROOT/deployment/endpoint.yml"


echo "Creating deployment..."

az ml online-deployment create \
  --file "$PROJECT_ROOT/deployment/deployment.yml" \
  --all-traffic


echo "Deployment complete."

echo "Endpoint status:"
az ml online-endpoint show \
  --name $ENDPOINT_NAME \
  --query provisioning_state
