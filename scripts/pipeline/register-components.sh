#!/bin/bash

# Register the Azure ML pipeline components used by the training and
# evaluation pipeline so the pipeline can reference them with
# `azureml:<component>@latest`.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "Registering training component..."
az ml component create \
  --file "$PROJECT_ROOT/pipelines/components/train.yml"

echo "Registering evaluation component..."
az ml component create \
  --file "$PROJECT_ROOT/pipelines/components/evaluate.yml"

echo "Component registration complete."
