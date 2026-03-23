#!/bin/bash

# Submit the Azure ML train/evaluate pipeline. By default the script
# waits for completion and then downloads the output artifacts.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

PIPELINE_FILE="${PIPELINE_FILE:-$PROJECT_ROOT/pipelines/train_evaluate.yml}"
WAIT_FOR_COMPLETION="${WAIT_FOR_COMPLETION:-true}"
AUTO_DOWNLOAD_OUTPUTS="${AUTO_DOWNLOAD_OUTPUTS:-true}"
POLL_SECONDS="${POLL_SECONDS:-15}"
USE_DATA_ASSET="${USE_DATA_ASSET:-auto}"
DATA_ASSET_NAME="${DATA_ASSET_NAME:-iris_csv}"
DATA_ASSET_VERSION="${DATA_ASSET_VERSION:-1}"

if [[ "$AUTO_DOWNLOAD_OUTPUTS" == "true" ]]; then
  WAIT_FOR_COMPLETION="true"
fi

echo "Submitting pipeline file: $PIPELINE_FILE"

JOB_CREATE_ARGS=(--file "$PIPELINE_FILE")

if [[ "$USE_DATA_ASSET" != "false" ]]; then
  if az ml data show --name "$DATA_ASSET_NAME" --version "$DATA_ASSET_VERSION" >/dev/null 2>&1; then
    echo "Using Azure ML data asset: azureml:${DATA_ASSET_NAME}:${DATA_ASSET_VERSION}"
    JOB_CREATE_ARGS+=(--set "inputs.data_input.path=azureml:${DATA_ASSET_NAME}:${DATA_ASSET_VERSION}")
  elif [[ "$USE_DATA_ASSET" == "true" ]]; then
    echo "Required Azure ML data asset not found: azureml:${DATA_ASSET_NAME}:${DATA_ASSET_VERSION}"
    exit 1
  else
    echo "Azure ML data asset not found. Falling back to built-in Iris loading."
  fi
fi

JOB_NAME="$(az ml job create "${JOB_CREATE_ARGS[@]}" --query name -o tsv)"

if [[ -z "$JOB_NAME" || "$JOB_NAME" == "null" ]]; then
  echo "Pipeline submission did not return a job name."
  exit 1
fi

echo "Submitted pipeline job: $JOB_NAME"

if [[ "$WAIT_FOR_COMPLETION" != "true" ]]; then
  exit 0
fi

echo "Waiting for pipeline job '$JOB_NAME' to finish..."

while true; do
  STATUS="$(az ml job show --name "$JOB_NAME" --query status -o tsv)"
  echo "Current status: $STATUS"

  case "$STATUS" in
    Completed)
      break
      ;;
    Failed|Canceled|NotResponding)
      echo "Pipeline job '$JOB_NAME' ended with status: $STATUS"
      exit 1
      ;;
  esac

  sleep "$POLL_SECONDS"
done

echo "Pipeline job '$JOB_NAME' completed successfully."

if [[ "$AUTO_DOWNLOAD_OUTPUTS" == "true" ]]; then
  echo "Auto-downloading pipeline outputs..."
  "$SCRIPT_DIR/download-outputs.sh" "$JOB_NAME"
fi
