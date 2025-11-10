#!/bin/bash
set -e

# Move to repo root
REPO_ROOT=$(git rev-parse --show-toplevel)
cd "$REPO_ROOT" || exit 1

# Determine model directory from config
MODEL_DIR=$(python -c "from config import settings; print(settings.model_under_test)")
ARCHIVE="${MODEL_DIR%/}.tar.gz"

echo "Model directory: $MODEL_DIR"
echo "Archive target: $ARCHIVE"

# Check directory exists
if [ ! -d "$MODEL_DIR" ]; then
  echo "Model directory not found: $MODEL_DIR"
  exit 1
fi

# Create tarball
echo "Creating archive..."
tar -czf "$ARCHIVE" -C "$MODEL_DIR" .
echo "Created archive: $ARCHIVE"

# Ensure the GitHub release 'latest' exists
if ! gh release view latest &>/dev/null; then
  echo "🚀 Creating 'latest' release..."
  gh release create latest --title "Latest Model" --notes "Auto-uploaded model for CI"
fi

# Upload to GitHub release
echo "Uploading archive to release..."
gh release upload latest "$ARCHIVE" --clobber
echo "Upload complete!"
