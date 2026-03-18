#!/bin/bash

ENDPOINT="roger-iris-endpoint-01"

KEY=$(az ml online-endpoint get-credentials \
  --name $ENDPOINT \
  --query primaryKey -o tsv)

SCORING_URI=$(az ml online-endpoint show \
  --name $ENDPOINT \
  --query scoring_uri -o tsv)

curl $SCORING_URI \
  -H "Authorization: Bearer $KEY" \
  -H "Content-Type: application/json" \
  -d @deployment/sample-request.json