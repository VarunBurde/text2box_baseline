"""Post-hoc visualization and replay for Text2Box debug reports."""
from __future__ import annotations

import argparse
import io
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from PIL import Image

from .data import ShardImageReader
from .evaluation.metrics import (
    compute_protocol_metrics_from_manifest,
    load_query_inputs_from_manifest,
)
from .rendering import (
    corner_list,
    denorm_bbox_yxyx_to_xyxy,
    float_list,
    format_metric,
    format_percent,
    render_columns_report,
)

SCHEMA_VERSION = "debug-columns-v1"


def parse_extra_config(raw: str | None) -> dict[str, str]:
    if not raw or not raw.strip():
        return {}
    result: dict[str, str] = {}
    for token in raw.split(","):
        item = token.strip()
        if not item:
            continue
        if "=" in item:
            key, value = item.split("=", 1)
            result[key.strip()] = value.strip()
        else:
            result[item] = "true"
    return result


def slugify(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", text.strip())
    return re.sub(r"-+", "-", cleaned).strip("-._") or "unknown"


def _fmt_config(value: Any) -> str:
    if value is None:
        return "none"
    if isinstance(value, float):
        return f"{value:.4g}".replace(".", "p")
    return str(value).replace(" ", "")


def infer_dataset_name(data_root: Path, override: str | None) -> str:
    if override and override.strip():
        return override.strip().lower()
    return data_root.name.strip().lower() or "dataset"


def infer_model_name_from_manifest(manifest_jsonl: Path, override: str) -> str:
    if override.strip().lower() != "auto":
        return override.strip()

    with manifest_jsonl.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            payload = json.loads(text)
            if not isinstance(payload, dict):
                continue
            for key in ("model", "model_name", "llm_model", "generator_model"):
                value = payload.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
            break
    return "unknown-model"


def prepare_run_output_paths(
    output_root: Path,
    dataset_name: str,
    model_name: str,
    timestamp_override: str | None,
    temperature: float | None,
    top_p: float | None,
    max_output_tokens: int | None,
    seed: int | None,
    config_tag: str | None,
    extra_config: dict[str, str],
) -> tuple[Path, Path, Path, dict[str, Any]]:
    timestamp = (
        timestamp_override.strip()
        if isinstance(timestamp_override, str) and timestamp_override.strip()
        else datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    )

    tokens: list[str] = []
    if temperature is not None:
        tokens.append(f"temp{_fmt_config(temperature)}")
    if top_p is not None:
        tokens.append(f"topP{_fmt_config(top_p)}")
    if max_output_tokens is not None:
        tokens.append(f"maxTok{_fmt_config(max_output_tokens)}")
    if seed is not None:
        tokens.append(f"seed{_fmt_config(seed)}")
    if config_tag and config_tag.strip():
        tokens.append(slugify(config_tag))
    for key in sorted(extra_config.keys()):
        tokens.append(f"{slugify(key)}{_fmt_config(extra_config[key])}")
    if not tokens:
        tokens = ["default"]

    run_slug = f"{timestamp}__{'_'.join(tokens)}"
    run_dir = output_root / slugify(dataset_name) / slugify(model_name) / run_slug
    debug_dir = run_dir / "debug"
    metrics_path = run_dir / "metrics" / "final_metrics.json"

    metadata: dict[str, Any] = {
        "dataset": slugify(dataset_name),
        "model_name": model_name,
        "model_slug": slugify(model_name),
        "timestamp": timestamp,
        "run_slug": run_slug,
        "temperature": temperature,
        "top_p": top_p,
        "max_output_tokens": max_output_tokens,
        "seed": seed,
        "config_tag": config_tag,
        "extra_config": extra_config,
        "debug_dir": str(debug_dir),
        "metrics_json": str(metrics_path),
    }
    return run_dir, debug_dir, metrics_path, metadata


class PostHocImageReader:
    def __init__(self, data_root: Path, split: str) -> None:
        self.data_root = data_root
        self.split = split
        self.images_split_dir = data_root / f"images_{split}"
        self.shard_lookup = self._load_shard_lookup()
        self.shard_reader: ShardImageReader | None = None
        if self.shard_lookup and self.images_split_dir.exists():
            self.shard_reader = ShardImageReader(images_split_dir=self.images_split_dir)

    def close(self) -> None:
        if self.shard_reader is not None:
            self.shard_reader.close()

    def read_image(self, image_id: int) -> Image.Image:
        if self.shard_reader is not None and image_id in self.shard_lookup:
            shard_name = self.shard_lookup[image_id]
            raw = self.shard_reader.read_image_bytes(image_id=image_id, shard_name=shard_name)
            return Image.open(io.BytesIO(raw)).convert("RGB")

        for name in (
            f"{image_id:08d}.jpg",
            f"{image_id:08d}.png",
            f"{image_id:06d}.jpg",
            f"{image_id:06d}.png",
        ):
            path = self.images_split_dir / name
            if path.exists():
                return Image.open(path).convert("RGB")

        raise FileNotFoundError(
            f"Could not load image_id={image_id} from {self.images_split_dir}"
        )

    def _load_shard_lookup(self) -> dict[int, str]:
        images_info_path = self.data_root / f"images_info_{self.split}.parquet"
        if not images_info_path.exists():
            return {}
        df = pd.read_parquet(images_info_path, columns=["image_id", "shard"])
        return {int(row.image_id): str(row.shard) for row in df.itertuples(index=False)}


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _pick_detection_from_instance(instance: dict[str, Any]) -> dict[str, Any] | None:
    parsed_raw = instance.get("parsed_detections")
    if not isinstance(parsed_raw, list):
        return None

    detections = [det for det in cast(list[Any], parsed_raw) if isinstance(det, dict)]
    if not detections:
        return None

    def _score(det: dict[str, Any]) -> tuple[int, float]:
        status = 1 if str(det.get("status") or "ok") == "ok" else 0
        conf = _safe_float(det.get("confidence"))
        return status, conf if conf is not None else -1.0

    return sorted(detections, key=_score, reverse=True)[0]


def _pred_bbox_from_det(det: dict[str, Any] | None, width: int, height: int) -> list[float] | None:
    if det is None:
        return None
    xyxy = float_list(det.get("bbox_2d_xyxy"), expected_len=4)
    if xyxy is not None:
        return xyxy
    norm = float_list(det.get("bbox_2d_norm_1000"), expected_len=4)
    if norm is None or width <= 0 or height <= 0:
        return None
    return denorm_bbox_yxyx_to_xyxy(norm, width=width, height=height)


def _gt_bbox_from_instance(instance: dict[str, Any]) -> list[float] | None:
    gt_raw = instance.get("gt")
    if not isinstance(gt_raw, dict):
        return None
    return float_list(gt_raw.get("bbox_xyxy"), expected_len=4)


def _load_bbox_object_lookup(data_root: Path) -> dict[int, dict[str, np.ndarray]]:
    objects_info_path = data_root / "objects_info.parquet"
    if not objects_info_path.exists():
        return {}

    df = pd.read_parquet(
        objects_info_path,
        columns=["obj_id", "bbox_3d_model_R", "bbox_3d_model_t", "bbox_3d_model_size"],
    )
    lookup: dict[int, dict[str, np.ndarray]] = {}
    for row in df.itertuples(index=False):
        try:
            obj_id = int(row.obj_id)
            lookup[obj_id] = {
                "bbox_3d_model_R": np.array(row.bbox_3d_model_R, dtype=np.float64).reshape(3, 3),
                "bbox_3d_model_t": np.array(row.bbox_3d_model_t, dtype=np.float64).reshape(3),
                "bbox_3d_model_size": np.array(row.bbox_3d_model_size, dtype=np.float64).reshape(3),
            }
        except Exception:
            continue
    return lookup


def _canonical_box_corners(size_xyz: np.ndarray) -> np.ndarray:
    sx, sy, sz = [float(v) for v in size_xyz.reshape(3).tolist()]
    return np.array(
        [
            [-sx / 2.0, -sy / 2.0, +sz / 2.0],
            [+sx / 2.0, -sy / 2.0, +sz / 2.0],
            [+sx / 2.0, +sy / 2.0, +sz / 2.0],
            [-sx / 2.0, +sy / 2.0, +sz / 2.0],
            [-sx / 2.0, -sy / 2.0, -sz / 2.0],
            [+sx / 2.0, -sy / 2.0, -sz / 2.0],
            [+sx / 2.0, +sy / 2.0, -sz / 2.0],
            [-sx / 2.0, +sy / 2.0, -sz / 2.0],
        ],
        dtype=np.float64,
    )


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
        # Store unclamped coords; PIL clips edges at image boundary naturally.
        x_norm = (x_px / float(image_width)) * 1000.0
        y_norm = (y_px / float(image_height)) * 1000.0
        out.append([y_norm, x_norm])

    return out


def _gt_corners_norm_from_instance(
    instance: dict[str, Any],
    width: int,
    height: int,
    object_lookup: dict[int, dict[str, np.ndarray]] | None,
) -> list[list[float]] | None:
    if object_lookup is None or width <= 0 or height <= 0:
        return None

    gt_raw = instance.get("gt")
    if not isinstance(gt_raw, dict):
        return None

    intrinsics = float_list(instance.get("intrinsics"), expected_len=4)
    if intrinsics is None:
        return None

    obj_id_raw = instance.get("obj_id")
    try:
        obj_id = int(obj_id_raw)
    except (TypeError, ValueError):
        return None

    object_meta = object_lookup.get(obj_id)
    if object_meta is None:
        return None

    try:
        r_cam_from_model = np.array(gt_raw.get("R_cam_from_model"), dtype=np.float64).reshape(3, 3)
        t_cam_from_model = np.array(gt_raw.get("t_cam_from_model"), dtype=np.float64).reshape(3)
    except Exception:
        return None

    r_bbox = r_cam_from_model @ object_meta["bbox_3d_model_R"]
    t_bbox = r_cam_from_model @ object_meta["bbox_3d_model_t"] + t_cam_from_model
    local = _canonical_box_corners(object_meta["bbox_3d_model_size"])
    corners_cam_xyz = (r_bbox @ local.T).T + t_bbox.reshape(1, 3)

    return _project_cam_xyz_to_norm_1000(
        corners_cam_xyz_mm=corners_cam_xyz.astype(float).tolist(),
        intrinsics=intrinsics,
        image_width=width,
        image_height=height,
    )


def _query_metrics_lookup(protocol_metrics: dict[str, Any]) -> dict[str, dict[str, Any]]:
    query_metrics = protocol_metrics.get("query_metrics")
    if not isinstance(query_metrics, dict):
        return {}
    return {
        str(key): cast(dict[str, Any], value)
        for key, value in query_metrics.items()
        if isinstance(value, dict)
    }


def _build_image_overview_rows(
    instances: list[dict[str, Any]],
    query_metrics: dict[str, dict[str, Any]] | None,
) -> list[dict[str, str]]:
    confs: list[float] = []
    pose_known = 0
    pose_ok = 0
    reproj_vals: list[float] = []
    iou2d_vals: list[float] = []
    iou3d_vals: list[float] = []
    acd_vals: list[float] = []
    hit2d50_vals: list[float] = []
    hit3d25_vals: list[float] = []

    total_parsed = 0
    for instance in instances:
        parsed = instance.get("parsed_detections")
        if isinstance(parsed, list):
            total_parsed += len(parsed)

        det = _pick_detection_from_instance(instance)
        if det is not None:
            conf = _safe_float(det.get("confidence"))
            if conf is not None:
                confs.append(conf)
            reproj = _safe_float(det.get("reprojection_error"))
            if reproj is not None:
                reproj_vals.append(reproj)
            pose_status = str(det.get("pose_status") or "").strip().lower()
            if pose_status in {"ok", "failed"}:
                pose_known += 1
                if pose_status == "ok":
                    pose_ok += 1

        if query_metrics is None:
            continue

        qid_raw = instance.get("query_id")
        qvals = query_metrics.get(str(qid_raw))
        if qvals is None:
            image_id = instance.get("image_id")
            inst_idx = instance.get("instance_idx")
            qvals = query_metrics.get(f"{image_id}_{inst_idx}")
        if qvals is None:
            continue

        for key, bucket in (
            ("best_iou2d", iou2d_vals),
            ("best_iou3d", iou3d_vals),
            ("best_acd3d", acd_vals),
            ("hit2d@50", hit2d50_vals),
            ("hit3d@25", hit3d25_vals),
        ):
            val = qvals.get(key)
            if isinstance(val, (int, float)):
                bucket.append(float(val))

    n_instances = len(instances)
    avg_conf = (sum(confs) / len(confs)) if confs else None
    avg_reproj = (sum(reproj_vals) / len(reproj_vals)) if reproj_vals else None
    pose_rate = (pose_ok / pose_known) if pose_known > 0 else None

    avg_iou2d = (sum(iou2d_vals) / len(iou2d_vals)) if iou2d_vals else None
    avg_iou3d = (sum(iou3d_vals) / len(iou3d_vals)) if iou3d_vals else None
    avg_acd = (sum(acd_vals) / len(acd_vals)) if acd_vals else None
    hit2d50 = (sum(hit2d50_vals) / len(hit2d50_vals)) if hit2d50_vals else None
    hit3d25 = (sum(hit3d25_vals) / len(hit3d25_vals)) if hit3d25_vals else None

    return [
        {"label": "queries", "value": str(n_instances)},
        {"label": "columns", "value": str(n_instances + 1)},
        {"label": "total detections", "value": str(total_parsed)},
        {"label": "avg confidence", "value": format_metric(avg_conf, 3)},
        {"label": "pose success", "value": format_percent(pose_rate, 1)},
        {"label": "avg reproj err", "value": format_metric(avg_reproj, 2)},
        {"label": "avg IoU2D", "value": format_metric(avg_iou2d, 3)},
        {"label": "avg IoU3D", "value": format_metric(avg_iou3d, 3)},
        {"label": "avg ACD3D", "value": format_metric(avg_acd, 2)},
        {"label": "hit2D@50", "value": format_percent(hit2d50, 1)},
        {"label": "hit3D@25", "value": format_percent(hit3d25, 1)},
    ]


def _instance_rows(
    instance: dict[str, Any],
    det: dict[str, Any] | None,
    qvals: dict[str, Any] | None,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = [
        {"label": "query_id", "value": str(instance.get("query_id", "n/a"))},
        {"label": "obj_id", "value": str(instance.get("obj_id", "n/a"))},
        {"label": "parse warning", "value": str(instance.get("parse_warning") or "none")},
    ]

    parsed = instance.get("parsed_detections")
    parsed_count = len(parsed) if isinstance(parsed, list) else 0
    rows.append({"label": "parsed detections", "value": str(parsed_count)})

    if det is not None:
        rows.extend(
            [
                {"label": "object", "value": str(det.get("object_name") or "n/a")},
                {"label": "confidence", "value": format_metric(det.get("confidence"), 3)},
                {"label": "pose", "value": str(det.get("pose_status") or "n/a")},
                {"label": "reproj err", "value": format_metric(det.get("reprojection_error"), 2)},
            ]
        )

    if qvals is not None:
        rows.extend(
            [
                {"label": "IoU2D", "value": format_metric(qvals.get("best_iou2d"), 3)},
                {"label": "IoU3D", "value": format_metric(qvals.get("best_iou3d"), 3)},
                {"label": "ACD3D", "value": format_metric(qvals.get("best_acd3d"), 2)},
                {"label": "hit2D@50", "value": str(int(bool(qvals.get("hit2d@50", 0))))},
                {"label": "hit3D@25", "value": str(int(bool(qvals.get("hit3d@25", 0))))},
            ]
        )

    return rows


def _payload_from_manifest_group(
    image_id: int,
    instances: list[dict[str, Any]],
    model_name: str,
    query_metrics: dict[str, dict[str, Any]] | None,
    object_lookup: dict[int, dict[str, np.ndarray]] | None,
) -> dict[str, Any]:
    width = int(instances[0].get("width", 0)) if instances else 0
    height = int(instances[0].get("height", 0)) if instances else 0

    cards: list[dict[str, Any]] = []
    for idx, instance in enumerate(instances):
        det = _pick_detection_from_instance(instance)
        pred_corners = corner_list(det.get("bbox_3d_corners_norm_1000")) if det else None
        gt_corners = _gt_corners_norm_from_instance(
            instance=instance,
            width=width,
            height=height,
            object_lookup=object_lookup,
        )
        qvals: dict[str, Any] | None = None
        if query_metrics is not None:
            qvals = query_metrics.get(str(instance.get("query_id")))
            if qvals is None:
                qvals = query_metrics.get(
                    f"{instance.get('image_id')}_{instance.get('instance_idx')}"
                )

        cards.append(
            {
                "title": f"Detection {idx + 1}",
                "query": str(instance.get("query") or ""),
                "rows": _instance_rows(instance=instance, det=det, qvals=qvals),
                "query_id": instance.get("query_id"),
                "gt_bbox_xyxy": _gt_bbox_from_instance(instance),
                "pred_bbox_xyxy": _pred_bbox_from_det(det, width=width, height=height),
                "gt_bbox_3d_corners_norm_1000": gt_corners,
                "pred_bbox_3d_corners_norm_1000": pred_corners,
                "metrics": qvals,
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "source": "posthoc-manifest",
        "image_id": int(image_id),
        "model_name": model_name,
        "image_width": width,
        "image_height": height,
        "overview_title": "RGB with GT and predicted boxes",
        "overview_rows": _build_image_overview_rows(instances, query_metrics),
        "instances": cards,
    }


def _group_instances_by_image(query_inputs: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for item in query_inputs:
        image_id = int(item.get("image_id", 0))
        if image_id <= 0:
            continue
        grouped.setdefault(image_id, []).append(item)

    for image_id in grouped:
        grouped[image_id].sort(
            key=lambda row: (int(row.get("instance_idx", 0)), int(row.get("query_id", 0)))
        )
    return grouped


def _load_debug_payload(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _iter_debug_json_paths(debug_json_dir: Path) -> list[Path]:
    return sorted(path for path in debug_json_dir.glob("*.json") if path.is_file())


def _average_from_debug_json(debug_dir: Path) -> dict[str, Any]:
    consumed = 0
    query_counts: list[float] = []

    for path in _iter_debug_json_paths(debug_dir):
        payload = _load_debug_payload(path)
        if payload is None:
            continue
        consumed += 1

        overview_rows = payload.get("overview_rows")
        if not isinstance(overview_rows, list):
            continue
        for row in overview_rows:
            if not isinstance(row, dict):
                continue
            if str(row.get("label")).strip().lower() == "queries":
                try:
                    query_counts.append(float(row.get("value")))
                except (TypeError, ValueError):
                    pass
                break

    return {
        "num_debug_json_files": int(consumed),
        "avg_instances_per_image": (
            float(sum(query_counts) / len(query_counts)) if query_counts else None
        ),
    }


def _write_payload_and_png(
    debug_dir: Path,
    payload: dict[str, Any],
    image: Image.Image,
) -> None:
    image_id = int(payload.get("image_id", 0))
    out_json = debug_dir / f"{image_id:06d}.json"
    out_png = debug_dir / f"{image_id:06d}_report.png"

    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report = render_columns_report(image=image, payload=payload)
    report.save(out_png)


def _run_replay_mode(
    debug_json_dir: Path,
    debug_dir_out: Path,
    image_reader: PostHocImageReader,
    image_ids: set[int] | None,
    limit: int | None,
) -> tuple[int, int]:
    processed = 0
    skipped = 0

    paths = _iter_debug_json_paths(debug_json_dir)
    if image_ids is not None:
        filtered_paths: list[Path] = []
        for path in paths:
            if not path.stem.isdigit():
                continue
            if int(path.stem) in image_ids:
                filtered_paths.append(path)
        paths = filtered_paths
    if limit is not None:
        paths = paths[:limit]

    for path in paths:
        payload = _load_debug_payload(path)
        if payload is None:
            skipped += 1
            continue

        image_id_raw = payload.get("image_id")
        if not isinstance(image_id_raw, (int, float)):
            skipped += 1
            continue
        image_id = int(image_id_raw)

        try:
            image = image_reader.read_image(image_id)
        except Exception as exc:
            skipped += 1
            print(f"[skip] image_id={image_id} reason={exc}")
            continue

        payload.setdefault("schema_version", SCHEMA_VERSION)
        _write_payload_and_png(debug_dir=debug_dir_out, payload=payload, image=image)
        processed += 1
        print(f"[ok] replay image_id={image_id}")

    return processed, skipped


def run_visualization(
    manifest_jsonl: Path | None,
    data_root: Path,
    split: str,
    output_root: Path,
    model_name: str = "auto",
    run_dir: Path | None = None,
    metrics_json_path: Path | None = None,
    timestamp: str | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    max_output_tokens: int | None = None,
    seed: int | None = None,
    config_tag: str | None = None,
    extra_config: dict[str, str] | None = None,
    image_ids: set[int] | None = None,
    limit: int | None = None,
    max_detections: int | None = None,
    dmax: int = 100,
    continuous_symmetry_steps: int = 36,
    dataset_name: str | None = None,
    debug_json_dir: Path | None = None,
) -> dict[str, Any]:
    """
    Generate simple debug columns.

    Preferred path: replay from per-image debug JSON files.
    Optional enriched path: build payloads from manifest + protocol metrics.
    """
    if extra_config is None:
        extra_config = {}

    if run_dir is not None:
        resolved_run_dir = run_dir
        debug_dir = resolved_run_dir / "debug"
        resolved_metrics_path = (
            metrics_json_path if metrics_json_path is not None else resolved_run_dir / "metrics" / "final_metrics.json"
        )
        resolved_model = model_name
        if manifest_jsonl is not None:
            resolved_model = infer_model_name_from_manifest(manifest_jsonl, model_name)

        metadata: dict[str, Any] = {
            "dataset": slugify(infer_dataset_name(data_root, dataset_name)),
            "model_name": resolved_model,
            "debug_dir": str(debug_dir),
            "metrics_json": str(resolved_metrics_path),
        }
    else:
        if manifest_jsonl is not None:
            resolved_model = infer_model_name_from_manifest(manifest_jsonl, model_name)
        else:
            resolved_model = model_name if model_name.strip().lower() != "auto" else "unknown-model"

        resolved_ds = infer_dataset_name(data_root, dataset_name)
        resolved_run_dir, debug_dir, resolved_metrics_path, metadata = prepare_run_output_paths(
            output_root=output_root,
            dataset_name=resolved_ds,
            model_name=resolved_model,
            timestamp_override=timestamp,
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_output_tokens,
            seed=seed,
            config_tag=config_tag,
            extra_config=extra_config,
        )

    debug_dir.mkdir(parents=True, exist_ok=True)
    resolved_metrics_path.parent.mkdir(parents=True, exist_ok=True)

    metadata.update(
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "split": split,
            "data_root": str(data_root),
            "manifest_jsonl": str(manifest_jsonl) if manifest_jsonl is not None else None,
            "debug_json_dir": str(debug_json_dir) if debug_json_dir is not None else None,
            "image_filter": sorted(image_ids) if image_ids is not None else None,
            "limit": limit,
            "max_detections": max_detections,
            "dmax": dmax,
            "continuous_symmetry_steps": continuous_symmetry_steps,
        }
    )

    metadata_path = (run_dir if run_dir is not None else resolved_run_dir) / "run_metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    image_reader = PostHocImageReader(data_root=data_root, split=split)

    final_metrics: dict[str, Any] = {
        "metrics": {},
        "counts": {},
    }

    processed = 0
    skipped = 0

    try:
        if debug_json_dir is not None:
            processed, skipped = _run_replay_mode(
                debug_json_dir=debug_json_dir,
                debug_dir_out=debug_dir,
                image_reader=image_reader,
                image_ids=image_ids,
                limit=limit,
            )
            final_metrics["mode"] = "replay"
        else:
            if manifest_jsonl is None:
                raise ValueError("manifest_jsonl is required when debug_json_dir is not provided")

            query_inputs = load_query_inputs_from_manifest(
                manifest_jsonl=manifest_jsonl,
                data_root=data_root,
                split=split,
            )
            grouped = _group_instances_by_image(query_inputs)
            selected_ids = sorted(grouped.keys())
            if image_ids is not None:
                selected_ids = [image_id for image_id in selected_ids if image_id in image_ids]
            if limit is not None:
                selected_ids = selected_ids[:limit]

            protocol_metrics = compute_protocol_metrics_from_manifest(
                manifest_jsonl=manifest_jsonl,
                data_root=data_root,
                split=split,
                dmax=dmax,
                continuous_symmetry_steps=continuous_symmetry_steps,
                include_details=False,
                include_query_metrics=True,
            )
            query_metrics = _query_metrics_lookup(protocol_metrics)
            object_lookup = _load_bbox_object_lookup(data_root)

            for image_id in selected_ids:
                instances = grouped.get(image_id, [])
                if max_detections is not None:
                    instances = instances[:max_detections]

                try:
                    image = image_reader.read_image(image_id)
                except Exception as exc:
                    skipped += 1
                    print(f"[skip] image_id={image_id} reason={exc}")
                    continue

                payload = _payload_from_manifest_group(
                    image_id=image_id,
                    instances=instances,
                    model_name=resolved_model,
                    query_metrics=query_metrics,
                    object_lookup=object_lookup,
                )
                _write_payload_and_png(debug_dir=debug_dir, payload=payload, image=image)
                processed += 1
                print(f"[ok] manifest image_id={image_id}")

            final_metrics["mode"] = "manifest"
            final_metrics["metrics"] = protocol_metrics.get("metrics", {})
            final_metrics["counts"] = dict(protocol_metrics.get("counts", {}) or {})

        final_metrics.setdefault("counts", {})
        final_metrics["counts"].update(
            {
                "num_reports_rendered": int(processed),
                "num_reports_skipped": int(skipped),
            }
        )
        final_metrics["averaged_from_debug_json"] = _average_from_debug_json(debug_dir)

    finally:
        image_reader.close()

    resolved_metrics_path.write_text(json.dumps(final_metrics, indent=2), encoding="utf-8")
    print(f"[metrics] saved {resolved_metrics_path}")
    print(f"Done. processed={processed} skipped={skipped}")
    return final_metrics


def parse_viz_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate simple Text2Box debug reports.")
    parser.add_argument("--manifest-jsonl", default=None)
    parser.add_argument("--debug-json-dir", default=None)
    parser.add_argument("--data-root", default="Datasets/ycbv")
    parser.add_argument("--split", default="test")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--model-name", default="auto")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--metrics-json", default=None)
    parser.add_argument("--timestamp", default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--max-output-tokens", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--config-tag", default=None)
    parser.add_argument("--extra-config", default=None)
    parser.add_argument("--dmax", type=int, default=100)
    parser.add_argument("--continuous-symmetry-steps", type=int, default=36)
    parser.add_argument("--image-ids", default=None, help="Comma-separated image IDs, e.g. 1,36,47")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-detections", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_viz_args()

    manifest_jsonl = Path(args.manifest_jsonl) if args.manifest_jsonl else None
    debug_json_dir = Path(args.debug_json_dir) if args.debug_json_dir else None

    if manifest_jsonl is None and debug_json_dir is None:
        raise ValueError("Provide --manifest-jsonl or --debug-json-dir")

    data_root = Path(args.data_root)
    output_root = Path(args.output_dir)

    if not data_root.exists():
        raise FileNotFoundError(f"Missing data root: {data_root}")
    if manifest_jsonl is not None and not manifest_jsonl.exists():
        raise FileNotFoundError(f"Missing manifest JSONL: {manifest_jsonl}")
    if debug_json_dir is not None and not debug_json_dir.exists():
        raise FileNotFoundError(f"Missing debug json dir: {debug_json_dir}")

    image_ids: set[int] | None = None
    if args.image_ids:
        image_ids = {int(val.strip()) for val in args.image_ids.split(",") if val.strip()}

    run_visualization(
        manifest_jsonl=manifest_jsonl,
        debug_json_dir=debug_json_dir,
        data_root=data_root,
        split=args.split,
        output_root=output_root,
        model_name=args.model_name,
        run_dir=Path(args.run_dir) if args.run_dir else None,
        metrics_json_path=Path(args.metrics_json) if args.metrics_json else None,
        timestamp=args.timestamp,
        temperature=args.temperature,
        top_p=args.top_p,
        max_output_tokens=args.max_output_tokens,
        seed=args.seed,
        config_tag=args.config_tag,
        extra_config=parse_extra_config(args.extra_config),
        image_ids=image_ids,
        limit=args.limit,
        max_detections=args.max_detections,
        dmax=args.dmax,
        continuous_symmetry_steps=args.continuous_symmetry_steps,
        dataset_name=args.dataset_name,
    )


if __name__ == "__main__":
    main()
