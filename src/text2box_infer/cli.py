from __future__ import annotations

import argparse
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

from .config import Settings
from .pipeline import run_inference
from .types import PromptProfile, RunMode


def _slugify(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", str(text).strip())
    cleaned = re.sub(r"-+", "-", cleaned).strip("-._")
    return cleaned or "unknown"


def _format_config_value(value: float | int) -> str:
    if isinstance(value, float):
        return _slugify(f"{value:.4g}".replace(".", "p"))
    return _slugify(str(value))


def _default_run_paths(
    data_root: Path,
    provider_slug: str,
    split: str,
    settings: Settings,
) -> tuple[Path, Path]:
    dataset_slug = _slugify(data_root.name.lower())
    if provider_slug == "ollama":
        model_name = settings.ollama_model
    else:
        model_name = settings.openai_model
    model_slug = _slugify(model_name)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    config_tokens = [
        f"temp{_format_config_value(settings.temperature)}",
        f"maxTok{_format_config_value(settings.max_output_tokens)}",
    ]
    run_slug = f"{timestamp}__{'_'.join(config_tokens)}"
    run_dir = Path("outputs") / dataset_slug / model_slug / run_slug
    predictions_dir = run_dir / "predictions"

    output_parquet = predictions_dir / f"preds_{provider_slug}_{split}.parquet"
    manifest_jsonl = predictions_dir / f"preds_{provider_slug}_{split}_manifest.jsonl"
    return output_parquet, manifest_jsonl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Text2Box inference and visualization CLI."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--mode",
        choices=[m.value for m in RunMode],
        default=RunMode.BASELINE_2D3D.value,
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-parquet", type=Path, default=None)
    parser.add_argument(
        "--limit-images",
        type=int,
        default=None,
        help="Optional limit on number of unique images to process",
    )
    parser.add_argument("--manifest-jsonl", type=Path, default=None)
    parser.add_argument("--provider", default="openai", choices=["openai", "ollama"])
    parser.add_argument("--env-file", type=Path, default=None)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable per-image debug JSON and PNG report generation during inference.",
    )
    parser.add_argument(
        "--prompt-profile",
        choices=[p.value for p in PromptProfile],
        default=PromptProfile.NORMALIZED_PNP.value,
        help=(
            "Prompt variant to use: direct-json, normalized, or normalized-pnp. "
            "Default: normalized-pnp."
        ),
    )

    # ── Visualize-mode arguments (ignored for inference modes) ────────
    viz = parser.add_argument_group(
        "visualize",
        "Options for --mode visualize (post-hoc report generation)",
    )
    viz.add_argument(
        "--debug-json-dir",
        type=Path,
        default=None,
        help="Replay reports from existing per-image debug JSON files in this directory.",
    )
    viz.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Output root for visualize mode (default: outputs/)",
    )
    viz.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Existing run directory; if set, debug/ and metrics/ are written into it.",
    )
    viz.add_argument(
        "--dataset-name",
        default=None,
        help="Dataset label override for output folder naming (default: inferred from --data-root).",
    )
    viz.add_argument(
        "--model-name",
        default="auto",
        help="Model label for report heading. Use 'auto' to infer from manifest (default: auto).",
    )
    viz.add_argument("--metrics-json", type=Path, default=None,
                     help="Explicit output path for the final metrics JSON.")
    viz.add_argument("--image-ids",    default=None,
                     help="Comma-separated image IDs to render (e.g. 1,36,47).")
    viz.add_argument("--max-detections", type=int, default=None,
                     help="Max detections per image to render.")
    viz.add_argument("--dmax",           type=int, default=100)
    viz.add_argument("--continuous-symmetry-steps", type=int, default=36)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    mode = RunMode(args.mode)

    # ── Visualize mode ────────────────────────────────────────────────
    if mode == RunMode.VISUALIZE:
        from .visualization import run_visualization

        manifest_jsonl = args.manifest_jsonl
        debug_json_dir = args.debug_json_dir
        if manifest_jsonl is None and debug_json_dir is None:
            parser.error("--manifest-jsonl or --debug-json-dir is required for --mode visualize")

        image_ids: set[int] | None = None
        if args.image_ids:
            image_ids = {int(v.strip()) for v in str(args.image_ids).split(",") if v.strip()}

        run_visualization(
            manifest_jsonl        = manifest_jsonl,
            debug_json_dir        = debug_json_dir,
            data_root             = args.data_root,
            split                 = args.split,
            output_root           = args.output_dir,
            model_name            = args.model_name,
            run_dir               = args.run_dir,
            metrics_json_path     = args.metrics_json,
            image_ids             = image_ids,
            limit                 = args.limit,
            max_detections        = args.max_detections,
            dmax                  = args.dmax,
            continuous_symmetry_steps = args.continuous_symmetry_steps,
            dataset_name          = args.dataset_name,
        )
        return

    # ── Inference modes ───────────────────────────────────────────────
    settings = Settings.from_env(args.env_file)
    provider_slug = str(args.provider).strip().lower()
    prompt_profile = PromptProfile(args.prompt_profile)

    output_parquet = args.output_parquet
    manifest_jsonl = args.manifest_jsonl
    if output_parquet is None or manifest_jsonl is None:
        default_parquet, default_manifest = _default_run_paths(
            data_root=args.data_root,
            provider_slug=provider_slug,
            split=str(args.split),
            settings=settings,
        )
        if output_parquet is None:
            output_parquet = default_parquet
        if manifest_jsonl is None:
            manifest_jsonl = default_manifest

    summary = run_inference(
        data_root=args.data_root,
        split=args.split,
        provider_name=args.provider,
        mode=mode,
        output_parquet=output_parquet,
        manifest_jsonl=manifest_jsonl,
        settings=settings,
        limit=args.limit,
        limit_images=args.limit_images,
        debug=bool(args.debug),
        prompt_profile=prompt_profile,
    )

    print("Inference finished.")
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
