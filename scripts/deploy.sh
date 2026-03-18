#!/bin/bash

set -e

ENDPOINT_NAME="roger-iris-endpoint-01"
DEPLOYMENT_NAME="blue"

echo "Creating endpoint..."

az ml online-endpoint create \
  --file deployment/endpoint.yml


echo "Creating deployment..."

az ml online-deployment create \
  --file deployment/deployment.yml \
  --all-traffic


echo "Deployment complete."

echo "Endpoint status:"
az ml online-endpoint show \
  --name $ENDPOINT_NAME \
  --query provisioning_state