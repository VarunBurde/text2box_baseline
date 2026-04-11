"""
Run inference on a single image with all 3 prompt profiles and produce a
side-by-side comparison PNG.

Usage:
    PYTHONPATH=src .venv/bin/python run_one_image_all_profiles.py \
        --data-root Datasets/ycbv \
        --split test \
        --provider openai   # or ollama
"""
from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


PROFILES = ["direct-json", "normalized", "normalized-pnp"]
PROFILE_COLORS = {
    "direct-json":     (220, 38,  38),   # red
    "normalized":      (37,  99,  235),  # blue
    "normalized-pnp":  (22,  163, 74),   # green
}

# ── cuboid topology ───────────────────────────────────────────────────────────
FRONT_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0)]
BACK_EDGES  = [(4, 5), (5, 6), (6, 7), (7, 4)]
CONN_EDGES  = [(0, 4), (1, 5), (2, 6), (3, 7)]


def load_font(size: int) -> ImageFont.ImageFont:
    for name in ("DejaVuSans.ttf", "Arial.ttf", "Helvetica.ttc"):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _corners_to_px(
    corners_norm: list[list[float]], img_w: int, img_h: int
) -> list[tuple[float, float]]:
    """Convert [[y,x], ...] norm-1000 corners to pixel (px, py) tuples."""
    return [
        (c[1] * img_w / 1000.0, c[0] * img_h / 1000.0)
        for c in corners_norm
    ]


def _draw_dashed(
    draw: ImageDraw.ImageDraw,
    p0: tuple[float, float],
    p1: tuple[float, float],
    color: tuple[int, int, int],
    width: int = 1,
    dash: int = 7,
    gap: int = 4,
) -> None:
    import math
    x0, y0 = p0
    x1, y1 = p1
    length = math.hypot(x1 - x0, y1 - y0)
    if length < 1:
        return
    dx, dy = (x1 - x0) / length, (y1 - y0) / length
    pos = 0.0
    while pos < length:
        end = min(pos + dash, length)
        draw.line(
            [(x0 + dx * pos, y0 + dy * pos), (x0 + dx * end, y0 + dy * end)],
            fill=color,
            width=width,
        )
        pos = end + gap


def draw_cuboid(
    draw: ImageDraw.ImageDraw,
    corners_norm: list[list[float]],
    img_w: int,
    img_h: int,
    color: tuple[int, int, int],
    front_w: int = 3,
    back_w: int = 1,
) -> None:
    if len(corners_norm) != 8:
        return
    pts = _corners_to_px(corners_norm, img_w, img_h)
    back_col = tuple(min(255, int(v * 1.6)) for v in color)  # lighter shade for back
    for i, j in BACK_EDGES:
        _draw_dashed(draw, pts[i], pts[j], back_col, width=back_w)  # type: ignore[arg-type]
    for i, j in CONN_EDGES:
        draw.line([pts[i], pts[j]], fill=back_col, width=back_w)
    for i, j in FRONT_EDGES:
        draw.line([pts[i], pts[j]], fill=color, width=front_w)


def draw_bbox(
    draw: ImageDraw.ImageDraw,
    bbox_xyxy: list[float],
    color: tuple[int, int, int],
    width: int = 2,
    dashed: bool = False,
) -> None:
    x0, y0, x1, y1 = bbox_xyxy
    if dashed:
        _draw_dashed(draw, (x0, y0), (x1, y0), color, width=width)
        _draw_dashed(draw, (x1, y0), (x1, y1), color, width=width)
        _draw_dashed(draw, (x1, y1), (x0, y1), color, width=width)
        _draw_dashed(draw, (x0, y1), (x0, y0), color, width=width)
    else:
        draw.rectangle([x0, y0, x1, y1], outline=color, width=width)


# ── minimal JSON-parse helpers ────────────────────────────────────────────────

def _last_debug_json(profile_run_dir: Path) -> Path | None:
    debug_dir = profile_run_dir / "debug"
    jsons = sorted(debug_dir.glob("*.json")) if debug_dir.exists() else []
    # exclude summary-like files; keep only 6-digit image id files
    candidates = [p for p in jsons if p.stem.isdigit() and len(p.stem) <= 8]
    return candidates[0] if candidates else None


def _find_latest_run(outputs_root: Path, dataset: str, model_slug: str) -> Path | None:
    base = outputs_root / dataset / model_slug
    if not base.exists():
        return None
    runs = sorted(base.iterdir(), reverse=True)
    return runs[0] if runs else None


def _load_debug_payload(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text("utf-8"))
    except Exception as exc:
        print(f"  [warn] could not load {path}: {exc}")
        return None
    return data if isinstance(data, dict) else None


# ── overlay rendering ─────────────────────────────────────────────────────────

def render_profile_overlay(
    base_image: Image.Image,
    payload: dict[str, Any],
    profile_name: str,
    color: tuple[int, int, int],
) -> Image.Image:
    img = base_image.copy()
    draw = ImageDraw.Draw(img)
    img_w, img_h = img.size

    instances = payload.get("instances", [])
    for inst in instances:
        # GT box (dashed white)
        gt_bbox = inst.get("gt_bbox_xyxy")
        if isinstance(gt_bbox, list) and len(gt_bbox) == 4:
            try:
                gt = [float(v) for v in gt_bbox]
                draw_bbox(draw, gt, (255, 255, 255), width=2, dashed=True)
            except Exception:
                pass

        # Pred 2D box
        pred_bbox = inst.get("pred_bbox_xyxy")
        if isinstance(pred_bbox, list) and len(pred_bbox) == 4:
            try:
                draw_bbox(draw, [float(v) for v in pred_bbox], color, width=3)
            except Exception:
                pass

        # 3D cuboid
        corners = inst.get("bbox_3d_corners_norm_1000")
        if isinstance(corners, list) and len(corners) == 8:
            try:
                draw_cuboid(draw, corners, img_w, img_h, color)
            except Exception:
                pass

    # Label in top-left
    font = load_font(18)
    label = profile_name
    tw = int(draw.textlength(label, font=font))
    draw.rounded_rectangle((6, 6, tw + 18, 32), radius=4, fill=(0, 0, 0, 180))
    draw.text((12, 8), label, fill=color, font=font)

    return img


def render_comparison(
    base_image: Image.Image,
    profile_payloads: dict[str, dict[str, Any]],
) -> Image.Image:
    """Build a 2×2 grid: top-left = original, then one panel per profile."""
    font_big = load_font(20)
    font_sm  = load_font(14)

    panels: list[tuple[str, Image.Image]] = [("original", base_image.copy())]
    for profile in PROFILES:
        payload = profile_payloads.get(profile)
        if payload is None:
            placeholder = Image.new("RGB", base_image.size, (30, 30, 30))
            d = ImageDraw.Draw(placeholder)
            d.text((10, 10), f"{profile}\n(no data)", fill=(200, 200, 200), font=font_sm)
            panels.append((profile, placeholder))
        else:
            overlay = render_profile_overlay(
                base_image,
                payload,
                profile,
                PROFILE_COLORS[profile],
            )
            panels.append((profile, overlay))

    # Scale all panels to same size
    max_w = max(p.size[0] for _, p in panels)
    max_h = max(p.size[1] for _, p in panels)

    MARGIN = 8
    LABEL_H = 30
    cell_w = max_w + 2 * MARGIN
    cell_h = max_h + LABEL_H + MARGIN

    # 2 columns × 2 rows
    grid_w = 2 * cell_w
    grid_h = 2 * cell_h + 50  # +50 for title bar

    canvas = Image.new("RGB", (grid_w, grid_h), (20, 20, 30))
    draw   = ImageDraw.Draw(canvas)

    # title bar
    title = "All 3 prompt profiles — single image comparison"
    draw.text((MARGIN, 10), title, fill=(240, 240, 240), font=font_big)

    positions = [(0, 0), (1, 0), (0, 1), (1, 1)]
    for idx, (label, panel) in enumerate(panels):
        col, row = positions[idx]
        ox = col * cell_w
        oy = 50 + row * cell_h

        # fit panel
        pw, ph = panel.size
        scale = min(max_w / pw, max_h / ph) if pw > 0 and ph > 0 else 1.0
        nw, nh = max(1, int(pw * scale)), max(1, int(ph * scale))
        resized = panel.resize((nw, nh), Image.Resampling.LANCZOS)

        canvas.paste(resized, (ox + MARGIN, oy + LABEL_H))

        # label
        color = PROFILE_COLORS.get(label, (230, 230, 230))
        draw.text((ox + MARGIN + 4, oy + 6), label, fill=color, font=font_sm)

        # legend strip for non-original panels
        if label != "original":
            lx = ox + MARGIN + 4
            ly = oy + LABEL_H + nh + 4
            leg_items = [
                ((255, 255, 255), "GT 2D"),
                (color, "Pred 2D"),
                (color, "3D cuboid"),
            ]
            for lc, lname in leg_items:
                draw.rectangle((lx, ly + 3, lx + 12, ly + 14), fill=lc)
                draw.text((lx + 16, ly), lname, fill=(200, 200, 200), font=font_sm)
                lx += int(draw.textlength(lname, font=font_sm)) + 32

    return canvas


# ── CLI helpers ───────────────────────────────────────────────────────────────

def _run_profile(
    profile: str,
    data_root: str,
    split: str,
    provider: str,
    timestamp_tag: str,
    extra_args: list[str],
) -> Path | None:
    output_parquet = Path(f"outputs/_compare_{timestamp_tag}/{profile}/preds.parquet")
    manifest_jsonl = Path(f"outputs/_compare_{timestamp_tag}/{profile}/preds_manifest.jsonl")
    run_dir = Path(f"outputs/_compare_{timestamp_tag}/{profile}")
    run_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "text2box_infer.cli",
        "--data-root", data_root,
        "--split", split,
        "--mode", "baseline-2d3d",
        "--provider", provider,
        "--prompt-profile", profile,
        "--limit-images", "1",
        "--debug",
        "--output-parquet", str(output_parquet),
        "--manifest-jsonl", str(manifest_jsonl),
    ] + extra_args

    env = {"PYTHONPATH": "src"}
    import os
    full_env = {**os.environ, **env}

    print(f"\n{'='*60}")
    print(f"Running profile: {profile}")
    print(f"  cmd: {' '.join(cmd)}")
    print(f"{'='*60}")

    result = subprocess.run(cmd, env=full_env)
    if result.returncode != 0:
        print(f"  [ERROR] profile {profile} exited with code {result.returncode}")
        return None

    return run_dir


def _get_first_image_id_and_bytes(run_dir: Path, data_root: str, split: str) -> tuple[int | None, bytes | None]:
    """Find the first debug JSON to get image_id, then load the image from dataset."""
    debug_dir = run_dir / "debug"
    if not debug_dir.exists():
        return None, None
    candidates = sorted(p for p in debug_dir.glob("*.json") if p.stem.isdigit())
    if not candidates:
        return None, None
    payload = _load_debug_payload(candidates[0])
    if payload is None:
        return None, None
    image_id = payload.get("image_id")
    if not isinstance(image_id, (int, float)):
        return None, None
    image_id = int(image_id)

    # Load the raw image from the dataset shard
    try:
        import pandas as pd
        from text2box_infer.data import ShardImageReader, build_image_lookup, load_split_tables
        _, images_df, _ = load_split_tables(data_root=Path(data_root), split=split)
        image_lookup = build_image_lookup(images_df)
        image_meta = image_lookup.get(image_id)
        if image_meta is None:
            return image_id, None
        reader = ShardImageReader(Path(data_root) / f"images_{split}")
        image_bytes = reader.read_image_bytes(image_id=image_id, shard_name=image_meta["shard"])
        reader.close()
        return image_id, image_bytes
    except Exception as exc:
        print(f"  [warn] could not load image bytes for image_id={image_id}: {exc}")
        return image_id, None


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all 3 prompt profiles on 1 image and visualize.")
    parser.add_argument("--data-root", default="Datasets/ycbv")
    parser.add_argument("--split", default="test")
    parser.add_argument("--provider", default="openai", choices=["openai", "ollama"])
    parser.add_argument("--no-pnp", action="store_true")
    parser.add_argument("--output", default="outputs/compare_all_profiles.png",
                        help="Output comparison PNG path.")
    args = parser.parse_args()

    from datetime import datetime, timezone
    timestamp_tag = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    extra = []
    if args.no_pnp:
        extra.append("--no-pnp")

    run_dirs: dict[str, Path | None] = {}
    for profile in PROFILES:
        run_dirs[profile] = _run_profile(
            profile=profile,
            data_root=args.data_root,
            split=args.split,
            provider=args.provider,
            timestamp_tag=timestamp_tag,
            extra_args=extra,
        )

    # Collect payloads and locate base image
    profile_payloads: dict[str, dict[str, Any]] = {}
    image_id: int | None = None
    base_image: Image.Image | None = None

    for profile in PROFILES:
        run_dir = run_dirs.get(profile)
        if run_dir is None:
            print(f"  [skip] {profile}: no run dir")
            continue
        json_path = _last_debug_json(run_dir)
        if json_path is None:
            print(f"  [skip] {profile}: no debug JSON found in {run_dir}/debug/")
            continue
        payload = _load_debug_payload(json_path)
        if payload is None:
            print(f"  [skip] {profile}: failed to load {json_path}")
            continue
        profile_payloads[profile] = payload
        print(f"  [ok] {profile}: loaded payload from {json_path}")

        if base_image is None:
            img_id = payload.get("image_id")
            if isinstance(img_id, (int, float)):
                image_id = int(img_id)
                # Try to load the image from the report PNG as fallback
                png_path = json_path.parent / f"{image_id:06d}_report.png"
                if png_path.exists():
                    report_img = Image.open(png_path).convert("RGB")
                    # The report PNG has the image embedded in it; extract the raw image
                    # instead load from dataset
                    pass

        # Load raw image from dataset (once)
        if base_image is None and image_id is not None:
            _, img_bytes = _get_first_image_id_and_bytes(run_dir, args.data_root, args.split)
            if img_bytes is not None:
                base_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    if base_image is None:
        print("[ERROR] Could not load base image. Cannot render comparison.")
        sys.exit(1)

    if not profile_payloads:
        print("[ERROR] No profile payloads loaded. Aborting.")
        sys.exit(1)

    print(f"\nRendering comparison for image_id={image_id} ...")
    comparison = render_comparison(base_image, profile_payloads)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.save(out_path)
    print(f"\n[done] Saved comparison to: {out_path}")

    # Also print per-profile debug report paths
    print("\nPer-profile debug reports:")
    for profile, run_dir in run_dirs.items():
        if run_dir is None:
            continue
        debug_dir = run_dir / "debug"
        if not debug_dir.exists():
            continue
        pngs = sorted(debug_dir.glob("*_report.png"))
        for p in pngs:
            print(f"  {profile}: {p}")


if __name__ == "__main__":
    main()
