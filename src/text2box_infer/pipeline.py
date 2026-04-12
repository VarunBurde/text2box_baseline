from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from .clients import create_provider
from .config import Settings
from .data import (
    ShardImageReader,
    build_image_lookup,
    load_inference_tables,
)
from .debug_artifacts import flush_image_debug_artifacts
from .geometry import (
    box_3d_cam_xyz_size_rpy_mm_deg_to_corners_cam_xyz_mm,
    denormalize_bbox_yxyx_to_xyxy,
    solve_pose_from_corners_norm_with_size,
)
from .output import append_manifest_record, init_manifest_jsonl, write_gts_like_parquet
from .parsing import parse_model_response
from .types import ModelRequest, PromptProfile, RunMode

LOGGER = logging.getLogger(__name__)


def _build_gt_lookup(
    data_root: Path,
    split: str,
) -> dict[int, dict[str, Any]]:
    """Load gts parquet and return query_id → gt dict (best-confidence instance)."""
    gts_path = data_root / f"gts_{split}.parquet"
    if not gts_path.exists():
        return {}
    try:
        gts_df = pd.read_parquet(gts_path)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Could not load GT file %s: %s", gts_path, exc)
        return {}

    lookup: dict[int, dict[str, Any]] = {}
    for row in gts_df.to_dict(orient="records"):
        query_id = int(row["query_id"])
        if query_id in lookup:
            continue  # keep first (lowest annotation_id) per query
        try:
            bbox_2d = [float(v) for v in row["bbox_2d"]]
            r_cam = [float(v) for v in row["R_cam_from_model"]]
            t_cam = [float(v) for v in row["t_cam_from_model"]]
            bbox_r = [float(v) for v in row["bbox_3d_R"]]
            bbox_t = [float(v) for v in row["bbox_3d_t"]]
            bbox_size = [float(v) for v in row["bbox_3d_size"]]
        except Exception:  # noqa: BLE001
            continue
        lookup[query_id] = {
            "bbox_xyxy": bbox_2d,
            "R_cam_from_model": r_cam,
            "t_cam_from_model": t_cam,
            "bbox_3d_R": bbox_r,
            "bbox_3d_t": bbox_t,
            "bbox_3d_size": bbox_size,
        }
    return lookup


def _gt_corners_cam_xyz_mm(gt: dict[str, Any]) -> list[list[float]] | None:
    """Return GT 3D bbox corners in camera frame (mm), without projection."""
    try:
        r_bbox = np.array(gt["bbox_3d_R"], dtype=np.float64).reshape(3, 3)
        t_bbox = np.array(gt["bbox_3d_t"], dtype=np.float64).reshape(3)
        size = np.array(gt["bbox_3d_size"], dtype=np.float64).reshape(3)
    except Exception:  # noqa: BLE001
        return None

    # Corners 0-3: front face (z = +half[2]), corners 4-7: back face.
    # Matches draw_cuboid_layered edge connectivity.
    half = size / 2.0
    signs = [
        [-1, -1, +1], [+1, -1, +1], [+1, +1, +1], [-1, +1, +1],
        [-1, -1, -1], [+1, -1, -1], [+1, +1, -1], [-1, +1, -1],
    ]
    corners_model = np.array([[s[0] * half[0], s[1] * half[1], s[2] * half[2]] for s in signs], dtype=np.float64)
    corners_cam = (r_bbox @ corners_model.T).T + t_bbox.reshape(1, 3)
    return corners_cam.tolist()


def _gt_corners_norm_1000(
    gt: dict[str, Any],
    intrinsics: list[float],
    image_width: int,
    image_height: int,
) -> list[list[float]] | None:
    """Project GT 3D bbox corners to normalized [0,1000] image coords."""
    if len(intrinsics) != 4 or image_width <= 0 or image_height <= 0:
        return None
    corners_cam = _gt_corners_cam_xyz_mm(gt)
    if corners_cam is None:
        return None
    return _project_cam_xyz_to_norm_1000(
        corners_cam_xyz_mm=corners_cam,
        intrinsics=intrinsics,
        image_width=image_width,
        image_height=image_height,
    )


def _infer_run_dir_from_manifest(manifest_jsonl: Path) -> Path:
    if manifest_jsonl.parent.name == "predictions":
        return manifest_jsonl.parent.parent
    return manifest_jsonl.parent


def _apply_query_limits(
    queries_df: pd.DataFrame,
    limit: int | None,
    limit_images: int | None,
) -> tuple[pd.DataFrame, dict[int, int]]:
    filtered = queries_df.sort_values("query_id").reset_index(drop=True)

    if limit is not None:
        filtered = filtered.head(limit)

    if limit_images is not None:
        selected_image_ids: list[int] = []
        seen: set[int] = set()
        for image_id in filtered["image_id"].tolist():
            image_id_int = int(image_id)
            if image_id_int in seen:
                continue
            seen.add(image_id_int)
            selected_image_ids.append(image_id_int)
            if len(selected_image_ids) >= int(limit_images):
                break
        filtered = filtered[filtered["image_id"].isin(selected_image_ids)].reset_index(drop=True)

    queries_per_image = {
        int(image_id): int(count)
        for image_id, count in filtered["image_id"].value_counts().to_dict().items()
    }
    return filtered, queries_per_image


def _project_cam_xyz_to_norm_1000(
    corners_cam_xyz_mm: list[list[float]],
    intrinsics: list[float],
    image_width: int,
    image_height: int,
) -> list[list[float]] | None:
    if len(intrinsics) != 4 or image_width <= 0 or image_height <= 0:
        return None
    if len(corners_cam_xyz_mm) != 8:
        return None

    fx, fy, cx, cy = [float(v) for v in intrinsics]
    out: list[list[float]] = []

    for corner in corners_cam_xyz_mm:
        if not isinstance(corner, list) or len(corner) != 3:
            return None
        x_mm, y_mm, z_mm = [float(v) for v in corner]
        if z_mm <= 1e-6:
            return None

        x_px = (fx * (x_mm / z_mm)) + cx
        y_px = (fy * (y_mm / z_mm)) + cy

        # Store unclamped coords so PIL can clip edges at image boundary naturally.
        # Clamping would snap off-screen corners to the image edge, making
        # the projected cuboid appear artificially large.
        x_norm = (x_px / float(image_width)) * 1000.0
        y_norm = (y_px / float(image_height)) * 1000.0
        out.append([y_norm, x_norm])

    return out


def run_inference(
    data_root: str | Path,
    split: str,
    provider_name: str,
    mode: RunMode,
    output_parquet: str | Path,
    manifest_jsonl: str | Path,
    settings: Settings,
    limit: int | None = None,
    limit_images: int | None = None,
    debug: bool = False,
    prompt_profile: PromptProfile = PromptProfile.SIMPLE,
) -> dict[str, Any]:
    wall_start = time.perf_counter()
    data_root = Path(data_root)
    output_parquet = Path(output_parquet)
    manifest_jsonl = Path(manifest_jsonl)
    provider_name_norm = str(provider_name).strip().lower()
    model_name = settings.ollama_model if provider_name_norm == "ollama" else settings.openai_model
    run_dir = _infer_run_dir_from_manifest(manifest_jsonl)
    debug_dir = run_dir / "debug"
    debug_enabled = bool(debug)
    if debug_enabled:
        debug_dir.mkdir(parents=True, exist_ok=True)

    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    if output_parquet.exists():
        output_parquet.unlink()
    init_manifest_jsonl(manifest_jsonl)

    images_df, queries_df = load_inference_tables(data_root=data_root, split=split)
    queries_df, queries_per_image = _apply_query_limits(
        queries_df=queries_df,
        limit=limit,
        limit_images=limit_images,
    )

    image_lookup = build_image_lookup(images_df)
    gt_lookup = _build_gt_lookup(data_root=data_root, split=split)

    provider = create_provider(provider_name, settings)
    image_reader = ShardImageReader(data_root / f"images_{split}")

    rows: list[dict[str, Any]] = []

    annotation_id = 0
    instance_counters: dict[int, int] = defaultdict(int)
    current_image_id: int | None = None
    current_image_records: list[dict[str, Any]] = []
    current_image_bytes: bytes | None = None
    current_image_query_done = 0
    completed_images = 0
    debug_images_written = 0
    manifest_records_written = 0

    processed_queries = 0
    skipped_queries = 0
    parsed_detections = 0
    written_detections = 0
    pnp_success = 0
    model_call_time_s = 0.0

    try:
        pbar = tqdm(queries_df.itertuples(index=False), total=len(queries_df))
        for query_row in pbar:
            processed_queries += 1
            query_id = int(query_row.query_id)
            image_id = int(query_row.image_id)
            query_text = str(query_row.query)
            image_meta = image_lookup.get(image_id)

            if current_image_id is None:
                current_image_query_done = 0
            elif image_id != current_image_id:
                current_image_query_done = 0
            current_image_query_done += 1

            shard_name = "unknown"
            if isinstance(image_meta, dict):
                shard_name = str(image_meta.get("shard", "unknown"))
            shard_tag = shard_name.replace("shard-", "").replace(".tar", "")
            pbar.set_postfix_str(
                (
                    f"img={image_id:06d} "
                    f"inst={current_image_query_done}/{queries_per_image.get(image_id, 0)} "
                    f"shard={shard_tag}"
                )
            )

            if current_image_id is None:
                current_image_id = image_id
            elif image_id != current_image_id:
                if debug_enabled:
                    flush_image_debug_artifacts(
                        debug_dir=debug_dir,
                        image_id=current_image_id,
                        image_records=current_image_records,
                        image_bytes=current_image_bytes,
                        model_name=model_name,
                    )
                    debug_images_written += 1
                current_image_records = []
                current_image_bytes = None

                completed_images += 1
                write_gts_like_parquet(rows=rows, output_path=output_parquet)
                LOGGER.info(
                    "Checkpoint saved after image_id=%s (completed_images=%d, rows=%d, manifest_records=%d, debug_images=%d)",
                    current_image_id,
                    completed_images,
                    len(rows),
                    manifest_records_written,
                    debug_images_written,
                )
                current_image_id = image_id
            if image_meta is None:
                skipped_queries += 1
                skipped_record = {
                    "query_id": query_id,
                    "image_id": image_id,
                    "query": query_text,
                    "provider": provider_name,
                    "status": "skipped",
                    "warning": "image_id not found in image metadata.",
                }
                append_manifest_record(
                    record=skipped_record,
                    manifest_path=manifest_jsonl,
                )
                manifest_records_written += 1
                if debug_enabled:
                    current_image_records.append(skipped_record)
                continue

            try:
                image_bytes = image_reader.read_image_bytes(
                    image_id=image_id,
                    shard_name=image_meta["shard"],
                )
            except Exception as exc:  # noqa: BLE001
                skipped_queries += 1
                skipped_record = {
                    "query_id": query_id,
                    "image_id": image_id,
                    "query": query_text,
                    "provider": provider_name,
                    "status": "skipped",
                    "warning": f"image read failed: {exc}",
                }
                append_manifest_record(
                    record=skipped_record,
                    manifest_path=manifest_jsonl,
                )
                manifest_records_written += 1
                if debug_enabled:
                    current_image_records.append(skipped_record)
                continue

            if debug_enabled and current_image_bytes is None:
                current_image_bytes = image_bytes

            request = ModelRequest(
                query=query_text,
                width=int(image_meta["width"]),
                height=int(image_meta["height"]),
                intrinsics=[float(v) for v in image_meta["intrinsics"]],
                mode=mode,
                prompt_profile=prompt_profile,
            )

            if len(request.intrinsics) != 4:
                skipped_queries += 1
                skipped_record = {
                    "query_id": query_id,
                    "image_id": image_id,
                    "query": query_text,
                    "provider": provider_name,
                    "status": "skipped",
                    "warning": "intrinsics must contain [fx, fy, cx, cy]",
                }
                append_manifest_record(
                    record=skipped_record,
                    manifest_path=manifest_jsonl,
                )
                manifest_records_written += 1
                if debug_enabled:
                    current_image_records.append(skipped_record)
                continue

            model_call_start = time.perf_counter()
            try:
                raw_response_text = provider.predict(image_bytes=image_bytes, request=request)
            except Exception as exc:  # noqa: BLE001
                skipped_queries += 1
                error_record = {
                    "query_id": query_id,
                    "image_id": image_id,
                    "query": query_text,
                    "provider": provider_name,
                    "status": "error",
                    "warning": f"provider call failed: {exc}",
                }
                append_manifest_record(
                    record=error_record,
                    manifest_path=manifest_jsonl,
                )
                manifest_records_written += 1
                if debug_enabled:
                    current_image_records.append(error_record)
                continue
            finally:
                model_call_time_s += time.perf_counter() - model_call_start

            parsed = parse_model_response(raw_response_text)
            parsed_detections += len(parsed.detections)

            gt_entry = gt_lookup.get(query_id)
            if gt_entry is not None:
                _intrinsics = [float(v) for v in image_meta["intrinsics"]]
                _w = int(image_meta["width"])
                _h = int(image_meta["height"])
                gt_corners_cam = _gt_corners_cam_xyz_mm(gt_entry)
                gt_corners_norm = _gt_corners_norm_1000(
                    gt=gt_entry,
                    intrinsics=_intrinsics,
                    image_width=_w,
                    image_height=_h,
                )
                gt_entry = {
                    **gt_entry,
                    "bbox_3d_corners_cam_xyz_mm": gt_corners_cam,
                    "bbox_3d_corners_norm_1000": gt_corners_norm,
                }

            query_manifest: dict[str, Any] = {
                "query_id": query_id,
                "image_id": image_id,
                "query": query_text,
                "provider": provider_name,
                "image_width": int(image_meta["width"]),
                "image_height": int(image_meta["height"]),
                "intrinsics": [float(v) for v in image_meta["intrinsics"]],
                "status": "ok",
                "parse_warning": parsed.parse_warning,
                "raw_response": raw_response_text,
                "parsed_detection_count": len(parsed.detections),
                "detections": [],
                "gt": gt_entry,
            }

            for detection in parsed.detections:
                if detection.bbox_2d_norm_1000 is None:
                    query_manifest["detections"].append(
                        {
                            "status": "skipped",
                            "warning": "missing bbox_2d_norm_1000",
                            "object_name": detection.object_name,
                        }
                    )
                    continue

                bbox_2d = denormalize_bbox_yxyx_to_xyxy(
                    bbox_norm_1000=detection.bbox_2d_norm_1000,
                    height=int(image_meta["height"]),
                    width=int(image_meta["width"]),
                )

                if bbox_2d[2] <= bbox_2d[0] or bbox_2d[3] <= bbox_2d[1]:
                    query_manifest["detections"].append(
                        {
                            "status": "skipped",
                            "warning": "invalid bbox extents after conversion",
                            "object_name": detection.object_name,
                        }
                    )
                    continue

                key = image_id
                instance_id = instance_counters[key]
                instance_counters[key] += 1

                row = {
                    "annotation_id": annotation_id,
                    "query_id": query_id,
                    "obj_id": -1,
                    "instance_id": instance_id,
                    "bbox_2d": bbox_2d,
                    "bbox_3d_R": None,
                    "bbox_3d_t": None,
                    "bbox_3d_size": None,
                    "R_cam_from_model": None,
                    "t_cam_from_model": None,
                    "visib_fract": None,
                }

                corners_norm_1000 = detection.bbox_3d_corners_norm_1000
                corners_source: str | None = None
                derived_from_box_3d = False

                if (
                    detection.bbox_3d_corners_cam_xyz_mm is None
                    and detection.box_3d_cam_xyz_size_rpy_mm_deg is not None
                ):
                    converted = box_3d_cam_xyz_size_rpy_mm_deg_to_corners_cam_xyz_mm(
                        detection.box_3d_cam_xyz_size_rpy_mm_deg
                    )
                    if converted is not None:
                        derived_corners_cam_xyz, derived_size_mm = converted
                        detection.bbox_3d_corners_cam_xyz_mm = derived_corners_cam_xyz
                        if detection.bbox_3d_size_mm is None:
                            detection.bbox_3d_size_mm = derived_size_mm
                        derived_from_box_3d = True

                if corners_norm_1000 is not None:
                    corners_source = "provided_norm_1000"
                elif detection.bbox_3d_corners_cam_xyz_mm is not None:
                    corners_norm_1000 = _project_cam_xyz_to_norm_1000(
                        corners_cam_xyz_mm=detection.bbox_3d_corners_cam_xyz_mm,
                        intrinsics=[float(v) for v in image_meta["intrinsics"]],
                        image_width=int(image_meta["width"]),
                        image_height=int(image_meta["height"]),
                    )
                    if corners_norm_1000 is not None:
                        if derived_from_box_3d:
                            corners_source = "projected_from_box_3d_rpy_mm_deg"
                        else:
                            corners_source = "projected_from_cam_xyz_mm"

                detection_manifest: dict[str, Any] = {
                    "status": "ok",
                    "object_name": detection.object_name,
                    "obj_id": -1,
                    "confidence": detection.confidence,
                    "bbox_2d_norm_1000": detection.bbox_2d_norm_1000,
                    "box_3d": detection.box_3d_cam_xyz_size_rpy_mm_deg,
                    "bbox_3d_corners_cam_xyz_mm": detection.bbox_3d_corners_cam_xyz_mm,
                    "bbox_3d_corners_norm_1000": corners_norm_1000,
                    "bbox_3d_size_mm": detection.bbox_3d_size_mm,
                    "corners_projection_source": corners_source,
                    "bbox_2d_xyxy": bbox_2d,
                    "pose_status": None,
                    "reprojection_error": None,
                    "permutation": None,
                    "pose_warning": None,
                }

                if mode == RunMode.BASELINE_2D3D:
                    if corners_norm_1000 is None:
                        detection_manifest["pose_status"] = "failed"
                        detection_manifest["pose_warning"] = (
                            "missing bbox_3d_corners_norm_1000 and could not project "
                            "bbox_3d_corners_cam_xyz_mm/box_3d"
                        )
                    elif detection.bbox_3d_size_mm is None:
                        detection_manifest["pose_status"] = "failed"
                        detection_manifest["pose_warning"] = "missing bbox_3d_size_mm"
                    else:
                        pose = solve_pose_from_corners_norm_with_size(
                            corners_norm_1000=corners_norm_1000,
                            intrinsics=[float(v) for v in image_meta["intrinsics"]],
                            bbox_3d_size_mm=detection.bbox_3d_size_mm,
                            image_height=int(image_meta["height"]),
                            image_width=int(image_meta["width"]),
                        )

                        if pose.success:
                            row["bbox_3d_R"] = pose.bbox_3d_R
                            row["bbox_3d_t"] = pose.bbox_3d_t
                            row["bbox_3d_size"] = pose.bbox_3d_size
                            row["R_cam_from_model"] = pose.r_cam_from_model
                            row["t_cam_from_model"] = pose.t_cam_from_model
                            pnp_success += 1
                            detection_manifest["pose_status"] = "ok"
                        else:
                            detection_manifest["pose_status"] = "failed"
                            detection_manifest["pose_warning"] = pose.message

                        detection_manifest["reprojection_error"] = pose.reprojection_error
                        detection_manifest["permutation"] = pose.permutation

                rows.append(row)
                annotation_id += 1
                written_detections += 1
                query_manifest["detections"].append(detection_manifest)

            append_manifest_record(record=query_manifest, manifest_path=manifest_jsonl)
            manifest_records_written += 1
            if debug_enabled:
                current_image_records.append(query_manifest)
    finally:
        image_reader.close()

    if current_image_id is not None and debug_enabled:
        flush_image_debug_artifacts(
            debug_dir=debug_dir,
            image_id=current_image_id,
            image_records=current_image_records,
            image_bytes=current_image_bytes,
            model_name=model_name,
        )
        debug_images_written += 1
        completed_images += 1
    elif current_image_id is not None:
        completed_images += 1

    write_gts_like_parquet(rows=rows, output_path=output_parquet)
    LOGGER.info(
        "Final checkpoint saved (completed_images=%d, rows=%d, manifest_records=%d, debug_images=%d)",
        completed_images,
        len(rows),
        manifest_records_written,
        debug_images_written,
    )
    inference_time_s = float(time.perf_counter() - wall_start)

    summary = {
        "queries_processed": processed_queries,
        "queries_skipped": skipped_queries,
        "detections_parsed": parsed_detections,
        "detections_written": written_detections,
        "pnp_success": pnp_success,
        "prompt_profile": prompt_profile.value,
        "images_completed": completed_images,
        "debug_images_written": debug_images_written,
        "manifest_records_written": manifest_records_written,
        "inference_time_s": inference_time_s,
        "model_call_time_s": float(model_call_time_s),
        "avg_model_call_time_s": float(model_call_time_s / max(1, processed_queries)),
        "output_parquet": str(output_parquet),
        "manifest_jsonl": str(manifest_jsonl),
        "debug_dir": str(debug_dir),
    }

    summary_json = manifest_jsonl.with_suffix(".summary.json")
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary_json"] = str(summary_json)

    LOGGER.info("Run summary: %s", summary)
    return summary
