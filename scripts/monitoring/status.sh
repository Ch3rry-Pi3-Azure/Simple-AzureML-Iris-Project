#!/bin/bash

set -e

MONITOR_SCHEDULE_NAME="${MONITOR_SCHEDULE_NAME:-iris-endpoint-monitor}"

echo "Monitor schedule details:"
az ml schedule show \
  --name "$MONITOR_SCHEDULE_NAME"
