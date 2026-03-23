#!/bin/bash

# Recreate the endpoint deployment from scratch by deleting the
# existing endpoint first and then running the standard deploy script.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Resetting endpoint deployment..."
"$SCRIPT_DIR/cleanup.sh"
"$SCRIPT_DIR/deploy.sh"
