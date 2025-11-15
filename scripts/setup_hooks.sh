#!/bin/bash
# setup_hooks.sh - one-time setup script for contributors

set -e

# Find repo root
REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null || echo "")

if [ -z "$REPO_ROOT" ]; then
  echo "Not inside a Git repository."
  exit 1
fi

# Configure Git to use .githooks as hook path
git config core.hooksPath .githooks

echo "Git hooks path set to .githooks"
echo "Verifying pre-push hook..."
if [ -x "$REPO_ROOT/scripts/.githooks/pre-push" ]; then
  echo "pre-push hook is executable."
else
  chmod +x "$REPO_ROOT/.githooks/pre-push"
  echo "Made pre-push hook executable."
fi

echo "Hook setup complete!"
