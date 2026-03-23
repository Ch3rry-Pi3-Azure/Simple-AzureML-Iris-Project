#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

PIPELINE_FILE="${PIPELINE_FILE:-$PROJECT_ROOT/pipelines/train_evaluate.yml}"

az ml job create \
  --file "$PIPELINE_FILE"
