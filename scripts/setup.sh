#!/bin/bash
set -e  # Exit immediately on errors
set -u  # Treat unset vars as errors

echo "Setting up Python environment..."

# Ensure we're at the project root
cd "$(dirname "$0")/.." || exit 1

# Create venv if not exists
if [ ! -d "venv" ]; then
  echo "Creating virtual environment..."
  python3 -m venv venv
else
  echo "Virtual environment already exists."
fi

# Activate venv
source venv/bin/activate

# Install project in editable mode + dependencies
if [ -f "requirements.txt" ]; then
  echo "Installing dependencies..."
  pip install -r requirements.txt
else
  echo "No requirements.txt found, skipping dependency install."
fi

echo "Installing project in editable mode..."
pip install -e .

echo "Setup complete! Environment ready."
echo
echo "To activate it manually later, run:"
echo "  source venv/bin/activate"
