#!/bin/bash
set -e

echo " Uploading Model Artifact"

# Get repo root
REPO_ROOT=$(git rev-parse --show-toplevel)
cd "$REPO_ROOT" || exit 1

# Load model path from config
MODEL_DIR=$(python - <<EOF
from config import settings
print(settings.model_under_test)
EOF
)

if [ -z "$MODEL_DIR" ]; then
  echo "Could not read settings.model_under_test"
  exit 1
fi

if [ ! -d "$MODEL_DIR" ]; then
  echo "Model directory not found: $MODEL_DIR"
  exit 1
fi

# Version = current git branch
BRANCH=$(git rev-parse --abbrev-ref HEAD)

# File name follows convention: model-<branch>.tar.gz
ARCHIVE="model-${BRANCH}.tar.gz"

echo "Model directory: $MODEL_DIR"
echo "Branch: $BRANCH"
echo "Archive: $ARCHIVE"
echo ""

# Create archive
echo "Creating TAR archive..."
tar -czf "$ARCHIVE" -C "$MODEL_DIR" .
echo "Created $ARCHIVE"
echo ""

# Ensure a release exists for this branch
if ! gh release view "$BRANCH" &>/dev/null; then
  echo "Creating GitHub Release for branch '$BRANCH'..."
  gh release create "$BRANCH" --title "$BRANCH" --notes "Model artifacts for branch $BRANCH"
else
  echo "Release '$BRANCH' already exists."
fi
echo ""

# Upload the archive (overwrite if exists)
echo "Uploading model to release '$BRANCH'..."
gh release upload "$BRANCH" "$ARCHIVE" --clobber
echo "Upload complete!"
echo ""

# Cleanup local archive
echo "Removing local archive..."
rm "$ARCHIVE"
echo "✔ Cleanup complete!"
echo ""

echo "Model upload finished successfully!"
echo " Asset: model-${BRANCH}.tar.gz"
echo " Release: $BRANCH"
