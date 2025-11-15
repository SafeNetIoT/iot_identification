#!/bin/bash
set -e  # Exit immediately on errors
set -u  # Treat unset vars as errors

echo "Setting up Python environment..."

# Ensure we're at the project root
cd "$(dirname "$0")/.." || exit 1

# set up git hooks
./scripts/.githooks/pre-push.sh
./scripts/raw_data.sh