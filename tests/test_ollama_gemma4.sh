#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${1:-http://localhost:11434/v1}"
MODEL="${2:-gemma4:latest}"
QUERY="${3:-the red rectangular object}"

PYTHONPATH=src .venv/bin/python examples/test_ollama_gemma4.py \
  --base-url "$BASE_URL" \
  --api-key ollama \
  --model "$MODEL" \
  --query "$QUERY"
