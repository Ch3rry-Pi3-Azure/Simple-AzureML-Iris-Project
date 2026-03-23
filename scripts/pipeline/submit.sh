#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

PIPELINE_FILE="${PIPELINE_FILE:-$PROJECT_ROOT/pipelines/train_evaluate.yml}"
WAIT_FOR_COMPLETION="${WAIT_FOR_COMPLETION:-true}"
AUTO_DOWNLOAD_OUTPUTS="${AUTO_DOWNLOAD_OUTPUTS:-true}"
POLL_SECONDS="${POLL_SECONDS:-15}"

if [[ "$AUTO_DOWNLOAD_OUTPUTS" == "true" ]]; then
  WAIT_FOR_COMPLETION="true"
fi

JOB_NAME="$(az ml job create --file "$PIPELINE_FILE" --query name -o tsv)"

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
  "$SCRIPT_DIR/download-outputs.sh" "$JOB_NAME"
fi
