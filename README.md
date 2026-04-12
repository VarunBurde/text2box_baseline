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

Metric definitions and interpretation guide: see `metrics.md`.

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

Choose one of the two prompt styles with `--prompt-profile`:

- `direct-json`: asks the model to directly output final coordinates in the JSON schema.
- `simple`: canonical unified prompt profile.

Current default profile: `simple`.

Backward-compatible aliases: `normalized-pnp` and `direct-json` both map to the same unified prompt.

### Low-level math: how 3D is computed per profile

This section explains exactly where 3D coordinates come from in each profile.

#### 1) `direct-json`

- The model directly returns camera-frame 3D corners in millimeters as `bbox_3d_corners_cam_xyz_mm`.
- No PnP is needed to recover 3D corners.
- If needed, these camera-frame corners can be projected to normalized 2D using intrinsics.

Projection used in code for camera-frame XYZ to normalized 2D:

$$
x_{px} = f_x \cdot \frac{X}{Z} + c_x,\quad
y_{px} = f_y \cdot \frac{Y}{Z} + c_y
$$

$$
x_{norm} = 1000 \cdot \frac{x_{px}}{W},\quad
y_{norm} = 1000 \cdot \frac{y_{px}}{H}
$$

#### 2) `simple` (challenge-compliant)

- The model returns:
  - `bbox_2d_norm_1000`
  - `bbox_3d_corners_norm_1000` (8 projected corners)
  - `bbox_3d_size_mm` = `[length_mm, width_mm, height_mm]`
- These corners are 2D image points; metric 3D pose is recovered with PnP.
- In inference, this path uses only:
  - image,
  - query,
  - camera intrinsics,
  - model-predicted `bbox_3d_size_mm`.
- Dataset object metadata is not used in inference.

Denormalization to pixels:

$$
u_i = \frac{x_i^{norm}}{1000} W,\quad
v_i = \frac{y_i^{norm}}{1000} H
$$

Camera model:

$$
K =
\begin{bmatrix}
f_x & 0 & c_x \\
0 & f_y & c_y \\
0 & 0 & 1
\end{bmatrix}
$$

Build centered 3D cuboid corners from predicted size:

$$
X_i^{box} \in \left\{\left(\pm\frac{l}{2}, \pm\frac{w}{2}, \pm\frac{h}{2}\right)\right\}
$$

PnP solves for rotation $R$ and translation $t$:

$$
s_i
\begin{bmatrix}
u_i \\
v_i \\
1
\end{bmatrix}
=
K [R\ |\ t]
\begin{bmatrix}
X_i \\
Y_i \\
Z_i \\
1
\end{bmatrix}
$$

Equivalent optimization objective:

$$
\min_{R,t} \sum_{i=1}^{8} \left\|p_i - \pi\left(K(RX_i + t)\right)\right\|_2^2
$$

Then metric camera-frame corners are obtained by:

$$
X_i^{cam} = R X_i^{box} + t
$$

#### Reprojection error and acceptance

The solver tries multiple corner permutations, keeps the best one by mean reprojection error, and rejects poor fits above a threshold.

$$
e = \frac{1}{8} \sum_{i=1}^{8} \|\hat{p}_i - p_i\|_2
$$

Lower error means the recovered 3D pose projects back closer to predicted 2D corners.

#### Metadata policy

- Inference uses only image/query/intrinsics and model outputs.
- Object metadata (`objects_info.parquet`) is reserved for evaluation.

This keeps inference challenge-compliant while still allowing full metric evaluation offline.

Example:

```bash
PYTHONPATH=src .venv/bin/python -m text2box_infer.cli \
  --data-root Datasets/ycbv \
  --split test \
  --mode baseline-2d3d \
  --provider ollama \
  --prompt-profile simple \
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

### One-image profile comparison (both prompt profiles)

Run both prompt profiles on one image and render a side-by-side comparison panel:

```bash
PYTHONPATH=src .venv/bin/python run_one_image_all_profiles.py \
  --data-root Datasets/ycbv \
  --split test \
  --provider ollama \
  --output outputs/compare_all_profiles.png
```

Default behavior for this helper:

- Runs inference once per profile (`direct-json`, `simple`).
- Then runs post-hoc metric enrichment per profile.
- Rewrites `debug/<image_id>.json` and `debug/<image_id>_report.png` so reports include:
  - Per-instance metrics: `IoU2D`, `IoU3D`, `ACD3D`, `hit2D@50`, `hit3D@25`.
  - Per-image averages: `avg IoU2D`, `avg IoU3D`, `avg ACD3D`, plus hit-rate averages.

Useful flags:

- `--no-metric-enrichment`: keep inference-only debug reports.
- `--dmax 100`: `D_max` used by enrichment metrics.
- `--continuous-symmetry-steps 36`: symmetry discretization steps for 3D metrics.

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

For exact metric definitions, thresholds, and interpretation, see `metrics.md`.

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
