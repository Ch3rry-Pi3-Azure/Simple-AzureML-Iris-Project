#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

az ml component create \
  --file "$PROJECT_ROOT/pipelines/components/train.yml"

az ml component create \
  --file "$PROJECT_ROOT/pipelines/components/evaluate.yml"
