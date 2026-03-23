#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

JOB_NAME="${1:-${JOB_NAME:-}}"
MODEL_NAME="${MODEL_NAME:-simple_iris_rf_model}"

if [[ -z "$JOB_NAME" ]]; then
  echo "Usage: ./scripts/model/register-from-job.sh <pipeline_job_name>"
  echo "Or set JOB_NAME=<pipeline_job_name> before running."
  exit 1
fi

MODEL_PATH="azureml://jobs/${JOB_NAME}/outputs/trained_model/paths/model_output"

echo "Registering model '${MODEL_NAME}' from pipeline job '${JOB_NAME}'..."

az ml model create \
  --file "$PROJECT_ROOT/models/register_from_job.yml" \
  --set name="$MODEL_NAME" \
  --set path="$MODEL_PATH" \
  --set tags.pipeline_job="$JOB_NAME"
