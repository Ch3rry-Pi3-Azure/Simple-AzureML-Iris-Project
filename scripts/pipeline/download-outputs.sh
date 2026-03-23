#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

JOB_NAME="${1:-${JOB_NAME:-}}"
DOWNLOAD_ROOT="${DOWNLOAD_ROOT:-$PROJECT_ROOT/outputs/azure_runs}"
INCLUDE_MODEL_OUTPUT="${INCLUDE_MODEL_OUTPUT:-false}"

if [[ -z "$JOB_NAME" ]]; then
  echo "No pipeline job name supplied. Looking up the latest pipeline job..."
  JOB_NAME="$(az ml job list --query "sort_by([?type=='pipeline'], &creation_context.created_at)[-1].name" -o tsv)"
fi

if [[ -z "$JOB_NAME" || "$JOB_NAME" == "null" ]]; then
  echo "Could not determine a pipeline job name."
  echo "Usage: ./scripts/pipeline/download-outputs.sh <pipeline_job_name>"
  echo "Or set JOB_NAME=<pipeline_job_name> before running."
  exit 1
fi

CREATED_AT="$(az ml job show --name "$JOB_NAME" --query creation_context.created_at -o tsv)"

if [[ -z "$CREATED_AT" || "$CREATED_AT" == "null" ]]; then
  DATE_FOLDER="$(date +%F)"
else
  DATE_FOLDER="${CREATED_AT%%T*}"
fi

TARGET_DIR="$DOWNLOAD_ROOT/$DATE_FOLDER/$JOB_NAME"
mkdir -p "$TARGET_DIR"

echo "Downloading pipeline outputs for job '$JOB_NAME' into '$TARGET_DIR'..."

az ml job download \
  --name "$JOB_NAME" \
  --output-name train_metrics \
  --download-path "$TARGET_DIR"

az ml job download \
  --name "$JOB_NAME" \
  --output-name evaluation_report \
  --download-path "$TARGET_DIR"

if [[ "$INCLUDE_MODEL_OUTPUT" == "true" ]]; then
  az ml job download \
    --name "$JOB_NAME" \
    --output-name trained_model \
    --download-path "$TARGET_DIR"
fi

echo "Download complete."
echo "Local pipeline artifact folder: $TARGET_DIR"
