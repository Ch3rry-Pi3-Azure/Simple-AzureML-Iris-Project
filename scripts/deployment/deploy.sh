#!/bin/bash

# Create or update the managed online endpoint deployment.
# By default the script resolves and deploys the latest registered
# version of the configured model name.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

ENDPOINT_NAME="${ENDPOINT_NAME:-roger-iris-endpoint-01}"
DEPLOYMENT_NAME="${DEPLOYMENT_NAME:-blue}"
MODEL_NAME="${MODEL_NAME:-simple_iris_rf_model}"
MODEL_VERSION="${MODEL_VERSION:-latest}"

if [[ "$MODEL_VERSION" == "latest" ]]; then
  echo "Resolving latest registered version for model '$MODEL_NAME'..."
  MODEL_VERSION="$(az ml model list --name "$MODEL_NAME" -o json | python -c $'import json, sys\nmodels = json.load(sys.stdin)\nif not models:\n    raise SystemExit(1)\ndef sort_key(model):\n    version = model.get(\"version\")\n    try:\n        return (0, int(version))\n    except (TypeError, ValueError):\n        return (1, str(version))\nprint(max(models, key=sort_key)[\"version\"])')"
fi

MODEL_REF="azureml:${MODEL_NAME}:${MODEL_VERSION}"

echo "Using deployment name '$DEPLOYMENT_NAME' under endpoint '$ENDPOINT_NAME'."
echo "Creating endpoint if needed..."

az ml online-endpoint create \
  --file "$PROJECT_ROOT/deployment/endpoint.yml"


echo "Creating or updating deployment..."
echo "Deploying model: $MODEL_REF"

az ml online-deployment create \
  --file "$PROJECT_ROOT/deployment/deployment.yml" \
  --set model="$MODEL_REF" \
  --all-traffic


echo "Deployment complete."

echo "Endpoint status:"
az ml online-endpoint show \
  --name $ENDPOINT_NAME \
  --query provisioning_state
