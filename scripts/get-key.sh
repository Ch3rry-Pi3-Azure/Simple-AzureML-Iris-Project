#!/bin/bash

ENDPOINT_NAME="${ENDPOINT_NAME:-roger-iris-endpoint-01}"

az ml online-endpoint get-credentials \
  --name "$ENDPOINT_NAME"
