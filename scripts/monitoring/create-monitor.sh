#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

ENDPOINT_NAME="${ENDPOINT_NAME:-roger-iris-endpoint-01}"
DEPLOYMENT_NAME="${DEPLOYMENT_NAME:-blue}"
MONITOR_SCHEDULE_NAME="${MONITOR_SCHEDULE_NAME:-iris-endpoint-monitor}"
MONITOR_DISPLAY_NAME="${MONITOR_DISPLAY_NAME:-Iris Endpoint Monitor}"
SCHEDULE_FILE="${SCHEDULE_FILE:-$PROJECT_ROOT/monitoring/endpoint_monitor.yml}"
MONITOR_TARGET="azureml:${ENDPOINT_NAME}:${DEPLOYMENT_NAME}"

echo "Creating or updating monitor schedule '$MONITOR_SCHEDULE_NAME'..."
echo "Monitoring target: $MONITOR_TARGET"

az ml schedule create \
  --file "$SCHEDULE_FILE" \
  --set name="$MONITOR_SCHEDULE_NAME" \
  --set display_name="$MONITOR_DISPLAY_NAME" \
  --set create_monitor.monitoring_target.endpoint_deployment_id="$MONITOR_TARGET"

echo "Monitor schedule created."
