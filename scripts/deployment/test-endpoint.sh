#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

ENDPOINT_NAME="${ENDPOINT_NAME:-roger-iris-endpoint-01}"
DEPLOYMENT_NAME="${DEPLOYMENT_NAME:-blue}"

az ml online-endpoint invoke \
  --name "$ENDPOINT_NAME" \
  --deployment-name "$DEPLOYMENT_NAME" \
  --request-file "$PROJECT_ROOT/deployment/sample-request.json"
