from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from .clients import create_provider
from .config import Settings
from .data import (
    ShardImageReader,
    build_image_lookup,
    build_object_catalog,
    build_object_lookup,
    build_object_name_lookup,
    load_split_tables,
    resolve_obj_id,
)
from .debug_artifacts import flush_image_debug_artifacts
from .geometry import denormalize_bbox_yxyx_to_xyxy, solve_pose_from_corners_norm
from .output import append_manifest_record, init_manifest_jsonl, write_gts_like_parquet
from .parsing import parse_model_response
from .types import ModelRequest, PromptProfile, RunMode

LOGGER = logging.getLogger(__name__)


def _infer_run_dir_from_manifest(manifest_jsonl: Path) -> Path:
    if manifest_jsonl.parent.name == "predictions":
        return manifest_jsonl.parent.parent
    return manifest_jsonl.parent


def _load_gts_lookup(data_root: Path, split: str) -> dict[int, dict[str, Any]]:
    gts_lookup: dict[int, dict[str, Any]] = {}
    gts_path = data_root / f"gts_{split}.parquet"
    if not gts_path.exists():
        return gts_lookup

    gts_df = pd.read_parquet(
        gts_path,
        columns=[
            "query_id",
            "obj_id",
            "instance_id",
            "bbox_2d",
            "R_cam_from_model",
            "t_cam_from_model",
        ],
    )
    for row in gts_df.itertuples(index=False):
        gts_lookup[int(row.query_id)] = {
            "obj_id": int(row.obj_id),
            "instance_id": int(row.instance_id),
            "bbox_xyxy": [float(v) for v in list(row.bbox_2d)],
            "R_cam_from_model": [float(v) for v in list(row.R_cam_from_model)],
            "t_cam_from_model": [float(v) for v in list(row.t_cam_from_model)],
        }
    return gts_lookup


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

        x_norm = (x_px / float(image_width)) * 1000.0
        y_norm = (y_px / float(image_height)) * 1000.0
        x_norm = max(0.0, min(1000.0, x_norm))
        y_norm = max(0.0, min(1000.0, y_norm))
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
    prompt_profile: PromptProfile = PromptProfile.NORMALIZED_PNP,
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

    objects_df, images_df, queries_df = load_split_tables(data_root=data_root, split=split)

    gts_lookup = _load_gts_lookup(data_root=data_root, split=split)
    queries_df, queries_per_image = _apply_query_limits(
        queries_df=queries_df,
        limit=limit,
        limit_images=limit_images,
    )

    image_lookup = build_image_lookup(images_df)
    object_lookup = build_object_lookup(objects_df)
    object_name_lookup, searchable_names = build_object_name_lookup(objects_df)
    object_catalog = build_object_catalog(objects_df)

    provider = create_provider(provider_name, settings)
    image_reader = ShardImageReader(data_root / f"images_{split}")

    rows: list[dict[str, Any]] = []

    annotation_id = 0
    instance_counters: dict[tuple[int, int], int] = defaultdict(int)
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

            expected_obj_id: int | None = None
            gt_expected = gts_lookup.get(query_id)
            if isinstance(gt_expected, dict) and isinstance(gt_expected.get("obj_id"), int):
                expected_obj_id = int(gt_expected["obj_id"])

            shard_name = "unknown"
            if isinstance(image_meta, dict):
                shard_name = str(image_meta.get("shard", "unknown"))
            shard_tag = shard_name.replace("shard-", "").replace(".tar", "")
            pbar.set_postfix_str(
                (
                    f"img={image_id:06d} "
                    f"inst={current_image_query_done}/{queries_per_image.get(image_id, 0)} "
                    f"obj={expected_obj_id if expected_obj_id is not None else 'na'} "
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
                object_catalog=object_catalog,
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
            }
            if query_id in gts_lookup:
                gt_data = gts_lookup[query_id]
                query_manifest["expected_obj_id"] = int(gt_data["obj_id"])
                query_manifest["expected_instance_id"] = int(gt_data["instance_id"])
                query_manifest["gt"] = {
                    "bbox_xyxy": [float(v) for v in gt_data["bbox_xyxy"]],
                    "R_cam_from_model": [float(v) for v in gt_data["R_cam_from_model"]],
                    "t_cam_from_model": [float(v) for v in gt_data["t_cam_from_model"]],
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

                obj_id = resolve_obj_id(
                    object_name=detection.object_name,
                    query=query_text,
                    object_name_lookup=object_name_lookup,
                    searchable_names=searchable_names,
                )
                if obj_id is None:
                    query_manifest["detections"].append(
                        {
                            "status": "skipped",
                            "warning": "could not resolve obj_id",
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
                            "obj_id": obj_id,
                        }
                    )
                    continue

                key = (image_id, obj_id)
                instance_id = instance_counters[key]
                instance_counters[key] += 1

                row = {
                    "annotation_id": annotation_id,
                    "query_id": query_id,
                    "obj_id": obj_id,
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
                        corners_source = "projected_from_cam_xyz_mm"

                detection_manifest: dict[str, Any] = {
                    "status": "ok",
                    "object_name": detection.object_name,
                    "obj_id": obj_id,
                    "confidence": detection.confidence,
                    "bbox_2d_norm_1000": detection.bbox_2d_norm_1000,
                    "bbox_3d_corners_cam_xyz_mm": detection.bbox_3d_corners_cam_xyz_mm,
                    "bbox_3d_corners_norm_1000": corners_norm_1000,
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
                            "bbox_3d_corners_cam_xyz_mm"
                        )
                    else:
                        pose = solve_pose_from_corners_norm(
                            corners_norm_1000=corners_norm_1000,
                            intrinsics=[float(v) for v in image_meta["intrinsics"]],
                            object_meta=object_lookup[obj_id],
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
