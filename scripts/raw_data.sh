#!/bin/bash
set -euo pipefail

TARGET_DIR="data/raw"
TMP_DIR=$(mktemp -d)

# Clone just that repo (shallow clone for speed)
git clone --depth 1 https://github.com/SafeNetIoT/doh_iot.git "$TMP_DIR"

# Copy the relevant subdirectory
mkdir -p "$TARGET_DIR"
cp -r "$TMP_DIR/data/dns_only"/* "$TARGET_DIR/"

echo "✅ Copied data into $TARGET_DIR"
ls -R "$TARGET_DIR"
