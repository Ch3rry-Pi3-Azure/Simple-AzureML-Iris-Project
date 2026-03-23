#!/bin/bash

# Configure a simple autoscale policy for the managed online deployment.
# The script creates an autoscale setting and adds basic CPU-based
# scale-out and scale-in rules.

set -e

RESOURCE_GROUP="${RESOURCE_GROUP:-your-resource-group}"
ENDPOINT_NAME="${ENDPOINT_NAME:-roger-iris-endpoint-01}"
DEPLOYMENT_NAME="${DEPLOYMENT_NAME:-blue}"
AUTOSCALE_NAME="autoscale-${ENDPOINT_NAME}-${DEPLOYMENT_NAME}"

echo "Getting deployment resource ID..."
DEPLOYMENT_ID=$(az ml online-deployment show \
  --endpoint-name "$ENDPOINT_NAME" \
  --name "$DEPLOYMENT_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --query id -o tsv)

echo "Creating autoscale setting..."
az monitor autoscale create \
  --resource-group "$RESOURCE_GROUP" \
  --name "$AUTOSCALE_NAME" \
  --resource "$DEPLOYMENT_ID" \
  --min-count 1 \
  --max-count 3 \
  --count 1

echo "Adding scale-out rule..."
az monitor autoscale rule create \
  --resource-group "$RESOURCE_GROUP" \
  --autoscale-name "$AUTOSCALE_NAME" \
  --condition "Percentage CPU > 70 avg 5m" \
  --scale out 1

echo "Adding scale-in rule..."
az monitor autoscale rule create \
  --resource-group "$RESOURCE_GROUP" \
  --autoscale-name "$AUTOSCALE_NAME" \
  --condition "Percentage CPU < 30 avg 10m" \
  --scale in 1

echo "Done. Current autoscale config:"
az monitor autoscale show \
  --resource-group "$RESOURCE_GROUP" \
  --name "$AUTOSCALE_NAME"
