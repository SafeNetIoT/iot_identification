#!/bin/bash

set -e  # Exit immediately if a command fails
set -u  # Treat unset variables as an error

# Target directory
TARGET_DIR="data/raw"

# Ensure target directory exists
mkdir -p "$TARGET_DIR"

# Base URL for raw GitHub files
BASE_URL="https://raw.githubusercontent.com/SafeNetIoT/doh_iot/main/data/dns_only"

# List of files to fetch (from the GitHub repo)
FILES=(
  "dns_iot.csv"
  "dns_non_iot.csv"
  "dns_iot_labels.csv"
  "dns_non_iot_labels.csv"
)

echo "Downloading DNS dataset files into $TARGET_DIR..."

for FILE in "${FILES[@]}"; do
  DEST="$TARGET_DIR/$FILE"
  if [ -f "$DEST" ]; then
    echo " $FILE already exists, skipping."
  else
    echo " Downloading $FILE..."
    curl -L "$BASE_URL/$FILE" -o "$DEST"
  fi
done

echo "All files ready in $TARGET_DIR"
