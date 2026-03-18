#!/bin/bash

ENDPOINT="roger-iris-endpoint-01"

az ml online-endpoint show \
  --name $ENDPOINT