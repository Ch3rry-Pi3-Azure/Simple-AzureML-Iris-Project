#!/bin/bash
set -e

ENDPOINT_NAME="${ENDPOINT_NAME:-roger-iris-endpoint-01}"
DEPLOYMENT_NAME="${DEPLOYMENT_NAME:-blue}"

echo "Azure CLI defaults:"
az configure --list-defaults

echo
echo "Active subscription:"
az account show --query "{name:name,id:id}" -o table

echo
echo "Azure ML workspace:"
az ml workspace show \
  --query "{name:name,location:location,resourceGroup:resource_group}" \
  -o table

echo
echo "Managed online endpoint:"
az ml online-endpoint show \
  --name "$ENDPOINT_NAME" \
  --query "{name:name,authMode:auth_mode,state:provisioning_state,scoringUri:scoring_uri}" \
  -o table

echo
echo "Managed online deployment:"
az ml online-deployment show \
  --endpoint-name "$ENDPOINT_NAME" \
  --name "$DEPLOYMENT_NAME" \
  --query "{name:name,endpoint:endpoint_name,state:provisioning_state}" \
  -o table

echo
echo "Traffic routing:"
az ml online-endpoint show \
  --name "$ENDPOINT_NAME" \
  --query traffic
