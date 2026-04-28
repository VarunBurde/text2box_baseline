#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${1:-Datasets/ycbv}"
SPLIT="${2:-val}"
LIMIT="${3:-10}"
OLLAMA_MODEL="${4:-gemma4:latest}"
OLLAMA_BASE_URL="${5:-http://localhost:11434/v1}"

export OLLAMA_MODEL
export OLLAMA_BASE_URL

PYTHONPATH=src .venv/bin/python -m text2box_infer.cli \
  --data-root "$DATA_ROOT" \
  --split "$SPLIT" \
  --mode baseline-2d3d \
  --limit "$LIMIT" \
  --provider ollama \
  --output-parquet "outputs/preds_ollama_${SPLIT}.parquet" \
  --manifest-jsonl "outputs/preds_ollama_${SPLIT}_manifest.jsonl"
