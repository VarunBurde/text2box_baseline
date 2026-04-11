#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${1:-/path/to/bop-text2box}"
SPLIT="${2:-test}"
LIMIT="${3:-100}"

PYTHONPATH=src .venv/bin/python -m text2box_infer.cli \
  --data-root "$DATA_ROOT" \
  --split "$SPLIT" \
  --mode baseline-2d3d \
  --limit "$LIMIT" \
  --output-parquet "outputs/preds_openai_${SPLIT}.parquet" \
  --manifest-jsonl "outputs/preds_openai_${SPLIT}_manifest.jsonl"
