#!/bin/bash
set -e

ENDPOINT="roger-iris-endpoint-01"

az ml online-endpoint invoke \
  --name $ENDPOINT \
  --deployment-name blue \
  --request-file deployment/sample-request.json