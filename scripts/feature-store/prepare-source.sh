#!/bin/bash

# Prepare a derived Iris feature-source dataset and scaffold files for
# later Azure ML feature store registration.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"
python -m src.feature_store.prepare_source "$@"
