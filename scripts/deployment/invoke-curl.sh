#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

ENDPOINT_NAME="${ENDPOINT_NAME:-roger-iris-endpoint-01}"

KEY=$(az ml online-endpoint get-credentials \
  --name "$ENDPOINT_NAME" \
  --query primaryKey -o tsv)

SCORING_URI=$(az ml online-endpoint show \
  --name "$ENDPOINT_NAME" \
  --query scoring_uri -o tsv)

curl "$SCORING_URI" \
  -H "Authorization: Bearer $KEY" \
  -H "Content-Type: application/json" \
  -d @"$PROJECT_ROOT/deployment/sample-request.json"
