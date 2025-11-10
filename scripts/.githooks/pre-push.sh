#!/bin/bash
REPO_ROOT=$(git rev-parse --show-toplevel)
cd "$REPO_ROOT" || exit 1

MODEL_DIR=$(python -c "from config import settings; print(settings.model_under_test)")
ARCHIVE="${MODEL_DIR%/}.tar.gz"

if [ -d "$MODEL_DIR" ]; then
  echo "Creating archive $ARCHIVE..."
  tar -czf "$ARCHIVE" -C "$MODEL_DIR" .
  echo "Uploading to GitHub Release..."
  gh release upload latest "$ARCHIVE" --clobber
else
  echo "Model directory not found: $MODEL_DIR"
fi
