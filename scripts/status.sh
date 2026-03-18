#!/bin/bash

ENDPOINT="roger-iris-endpoint-01"

az ml online-deployment show \
  --endpoint-name $ENDPOINT \
  --name blue \
  --query provisioning_state