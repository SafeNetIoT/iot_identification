#!/bin/bash
set -e

echo "=== Uploading Model Artifact ==="

# Determine model directory
MODEL_DIR=$(python - <<'EOF'
from config import settings
print(settings.model_under_test)
EOF
)

if [ -z "$MODEL_DIR" ]; then
  echo "ERROR: model_under_test is empty in config."
  exit 1
fi

echo "Model directory: $MODEL_DIR"

# Get raw branch name
RAW_BRANCH=$(git rev-parse --abbrev-ref HEAD)
RAW_BRANCH="${RAW_BRANCH#heads/}"
echo "Raw branch name: $RAW_BRANCH"

# Sanitize branch for filenames (replace / with -)
SAFE_BRANCH=$(echo "$RAW_BRANCH" | sed 's|/|-|g')
echo "Sanitized branch: $SAFE_BRANCH"

# TAR file name
ARCHIVE="model-${SAFE_BRANCH}.tar.gz"
echo "Archive name: $ARCHIVE"

# Verify model dir
if [ ! -d "$MODEL_DIR" ]; then
  echo "ERROR: Model directory not found: $MODEL_DIR"
  exit 1
fi

# Create archive
echo "Creating TAR archive..."
tar -czf "$ARCHIVE" -C "$MODEL_DIR" .

echo "Archive created: $(pwd)/$ARCHIVE"

# Ensure the branch-specific GitHub release exists
if ! gh release view "$SAFE_BRANCH" &>/dev/null; then
  gh release create "$SAFE_BRANCH" \
    --title "Model for $RAW_BRANCH" \
    --notes "Auto-uploaded model artifact for branch $RAW_BRANCH" \
    --target "$RAW_BRANCH"
fi

# Upload (replace any existing archive)
echo "Uploading archive to GitHub Release..."
gh release upload "$SAFE_BRANCH" "$ARCHIVE" --clobber

# Clean up local file
echo "Cleaning up local archive..."
rm "$ARCHIVE"

echo "=== Upload Complete ==="
