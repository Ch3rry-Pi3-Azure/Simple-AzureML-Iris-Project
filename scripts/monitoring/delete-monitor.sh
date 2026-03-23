#!/bin/bash

# Delete the Azure ML monitor schedule for this project.

set -e

MONITOR_SCHEDULE_NAME="${MONITOR_SCHEDULE_NAME:-iris-endpoint-monitor}"

echo "Deleting monitor schedule '$MONITOR_SCHEDULE_NAME'..."
az ml schedule delete \
  --name "$MONITOR_SCHEDULE_NAME" \
  --yes

echo "Monitor schedule deleted."
