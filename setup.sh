#!/bin/bash
set -e

echo "Initializing AI Image Recognition Environment..."

if ! command -v uv &> /dev/null; then
    echo "uv not found. Please install it first: https://astral.sh/uv"
    exit 1
fi

echo "Syncing dependencies..."
uv sync

mkdir -p mlruns

echo "Setup complete!"
echo "Run the app: uv run streamlit run app/app.py"