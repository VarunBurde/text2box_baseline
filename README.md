# text2box inference baseline

This repository provides a CLI-first Text2Box inference and evaluation workflow for BOP-style datasets.

## Overview

The pipeline does the following:

- Loads dataset metadata from parquet tables.
- Reads RGB frames from dataset image shards.
- Calls a VLM provider (OpenAI-compatible endpoint or Ollama).
- Parses normalized 2D boxes and 3D projected cuboid corners.
- Writes model predictions incrementally during inference.
- Computes protocol metrics (AP2D, AP3D, AR2D, AR3D, ACD3D).
- Optionally generates per-image debug reports and detailed JSON files when `--debug` is enabled.

## Installation

```bash
pip install -r requirements.txt
```

## Environment Configuration

Create a `.env` file as needed. Common variables:

- `OPENAI_API_KEY`
- `OPENAI_MODEL` (default: `gpt-4.1-mini`)
- `OPENAI_BASE_URL` (optional)
- `NVIDIA_BASE_URL` (default: `https://integrate.api.nvidia.com/v1`)
- `OLLAMA_BASE_URL` (default: `http://localhost:11434/v1`)
- `OLLAMA_MODEL` (default: `gemma4:latest`)
- `TEMPERATURE` (default: `0.0`)
- `MAX_OUTPUT_TOKENS` (default: `1200`)

## Dataset Input

Use `--data-root` to point to a dataset folder such as:

- `Datasets/ycbv`
- `Datasets/tless`
- `Datasets/tudl`

The folder should include split parquet files like:

- `queries_<split>.parquet`
- `gts_<split>.parquet`
- `images_info_<split>.parquet`
- `objects_info.parquet`

## 1) Run Inference

### OpenAI-compatible provider

```bash
PYTHONPATH=src .venv/bin/python -m text2box_infer.cli \
  --data-root Datasets/ycbv \
  --split test \
  --mode baseline-2d3d \
  --provider openai \
  --debug
```

### Ollama provider

```bash
PYTHONPATH=src .venv/bin/python -m text2box_infer.cli \
  --data-root Datasets/ycbv \
  --split test \
  --mode baseline-2d3d \
  --provider ollama \
  --debug
```

### Optional smaller run

```bash
PYTHONPATH=src .venv/bin/python -m text2box_infer.cli \
  --data-root Datasets/ycbv \
  --split test \
  --mode baseline-2d3d \
  --provider ollama \
  --limit 20 \
  --debug
```

`--limit` is a query limit (not image count).

### Prompt profiles

Choose one of the three prompt styles with `--prompt-profile`:

- `direct-json`: asks the model to directly output final coordinates in the JSON schema.
- `normalized`: normalized-coordinate prompt without extra PnP-oriented geometry constraints.
- `normalized-pnp`: normalized prompt with stricter geometry constraints for downstream PnP (default).

Example:

```bash
PYTHONPATH=src .venv/bin/python -m text2box_infer.cli \
  --data-root Datasets/ycbv \
  --split test \
  --mode baseline-2d3d \
  --provider ollama \
  --prompt-profile normalized \
  --debug
```

If you want exactly N images, use `--limit-images`:

```bash
PYTHONPATH=src .venv/bin/python -m text2box_infer.cli \
  --data-root Datasets/ycbv \
  --split test \
  --mode baseline-2d3d \
  --provider ollama \
  --limit-images 10 \
  --debug
```

### Inference outputs (written incrementally)

For provider `ollama` and split `test`, defaults are under a run folder:

- `outputs/<dataset>/<model>/<timestamp__config>/predictions/preds_ollama_test_manifest.jsonl`
- `outputs/<dataset>/<model>/<timestamp__config>/predictions/preds_ollama_test.parquet`
- `outputs/<dataset>/<model>/<timestamp__config>/predictions/preds_ollama_test_manifest.summary.json`
- `outputs/<dataset>/<model>/<timestamp__config>/debug/<image_id>.json` (only with `--debug`)
- `outputs/<dataset>/<model>/<timestamp__config>/debug/<image_id>_report.png` (only with `--debug`)

`<timestamp__config>` includes generation settings (for example `temp0_maxTok1200`).
Older runs may still have `__default` from previous naming behavior.

Behavior:

- Manifest is appended query-by-query.
- Parquet is checkpointed per image and finalized at end.
- Debug JSON and report PNG are emitted per image only when `--debug` is set.
- Summary JSON contains inference timing and throughput counters.

## 2) Evaluate Metrics Only

Run protocol metrics from predictions + dataset GT:

```bash
PYTHONPATH=src .venv/bin/python -m text2box_infer.evaluation \
  --data-root Datasets/ycbv \
  --split test
```

Default behavior:

- Auto-discovers the latest `*_manifest.jsonl` under `outputs/`.
- Writes metrics to `outputs/metrics/final_metrics.json` when `--output-json` is omitted.

If you need a specific manifest file:

```bash
PYTHONPATH=src .venv/bin/python -m text2box_infer.evaluation \
  --manifest-jsonl outputs/<dataset>/<model>/<timestamp__config>/predictions/preds_ollama_test_manifest.jsonl \
  --data-root Datasets/ycbv \
  --split test
```

If you need a fixed metrics file path:

```bash
PYTHONPATH=src .venv/bin/python -m text2box_infer.evaluation \
  --data-root Datasets/ycbv \
  --split test \
  --output-json outputs/metrics/final_metrics.json
```

## 3) Inference + Visualization Together (Recommended)

Use `--debug` during inference to generate visualization artifacts live (per image):

```bash
PYTHONPATH=src .venv/bin/python -m text2box_infer.cli \
  --data-root Datasets/ycbv \
  --split test \
  --mode baseline-2d3d \
  --provider ollama \
  --debug
```

This is the simplest way to run inference and visualization at the same time.

Output run tree:

- `outputs/<dataset>/<model>/<timestamp__config>/predictions/preds_ollama_test_manifest.jsonl`
- `outputs/<dataset>/<model>/<timestamp__config>/predictions/preds_ollama_test.parquet`
- `outputs/<dataset>/<model>/<timestamp__config>/predictions/preds_ollama_test_manifest.summary.json`
- `outputs/<dataset>/<model>/<timestamp__config>/debug/*_report.png`
- `outputs/<dataset>/<model>/<timestamp__config>/debug/*.json`

## 4) Visualization Standalone (Post-hoc)

If you already ran inference, you can regenerate reports later.

Preferred replay path (use previously generated debug JSON files):

```bash
PYTHONPATH=src .venv/bin/python -m text2box_infer.visualization \
  --debug-json-dir outputs/<dataset>/<model>/<timestamp__config>/debug \
  --run-dir outputs/<dataset>/<model>/<timestamp__config> \
  --data-root Datasets/ycbv \
  --split test \
  --model-name gemma4
```

Manifest-enriched path (computes protocol metrics and writes fresh debug JSON/PNG):

```bash
PYTHONPATH=src .venv/bin/python -m text2box_infer.visualization \
  --manifest-jsonl outputs/<dataset>/<model>/<timestamp__config>/predictions/preds_ollama_test_manifest.jsonl \
  --run-dir outputs/<dataset>/<model>/<timestamp__config> \
  --data-root Datasets/ycbv \
  --split test \
  --model-name gemma4
```

## 5) Metrics + Visualization (Two-step Post-hoc)

If you want protocol metrics first and then visualization from manifest:

```bash
PYTHONPATH=src .venv/bin/python -m text2box_infer.evaluation \
  --manifest-jsonl outputs/<dataset>/<model>/<timestamp__config>/predictions/preds_ollama_test_manifest.jsonl \
  --data-root Datasets/ycbv \
  --split test

PYTHONPATH=src .venv/bin/python -m text2box_infer.visualization \
  --manifest-jsonl outputs/<dataset>/<model>/<timestamp__config>/predictions/preds_ollama_test_manifest.jsonl \
  --run-dir outputs/<dataset>/<model>/<timestamp__config> \
  --data-root Datasets/ycbv \
  --split test \
  --model-name gemma4
```

## Troubleshooting

### "No manifest found"

Run inference first, or pass a specific manifest path:

```bash
--manifest-jsonl outputs/<dataset>/<model>/<timestamp__config>/predictions/preds_ollama_test_manifest.jsonl
```

### Ollama connection errors

Check that Ollama server is running and `OLLAMA_BASE_URL` is correct.

### Long runtime on full split

Use `--limit` (query count) or `--limit-images` (unique image count) for quick validation before launching a full run.
