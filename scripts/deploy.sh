#!/bin/bash

# Define the output file and the exclusion list
OUTPUT_FILE="prod_deploy.tar.gz"
EXCLUDE_LIST=".deploy_exclude"

echo "Creating deployment artifact: $OUTPUT_FILE"

# Create the tarball, excluding files listed in .deploy_exclude
tar -czf "$OUTPUT_FILE" --exclude-from="$EXCLUDE_LIST" .

echo "Artifact created successfully!"