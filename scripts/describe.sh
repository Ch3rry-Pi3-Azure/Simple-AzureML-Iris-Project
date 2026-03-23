#!/bin/bash

ENDPOINT_NAME="${ENDPOINT_NAME:-roger-iris-endpoint-01}"

az ml online-endpoint show \
  --name "$ENDPOINT_NAME"
