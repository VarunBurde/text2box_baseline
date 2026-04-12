"""Live debug artifact generation for inference-time visualization."""
from __future__ import annotations

import io
import json
import math
from pathlib import Path
from typing import Any, cast

from PIL import Image

from .rendering import (
    corner_list,
    denorm_bbox_yxyx_to_xyxy,
    float_list,
    format_metric,
    format_percent,
    render_columns_report,
)

SCHEMA_VERSION = "debug-columns-v1"


# ---------------------------------------------------------------------------
# Metric computation helpers
# ---------------------------------------------------------------------------

def _compute_iou_2d(gt: list[float], pred: list[float]) -> float | None:
    """Axis-aligned IoU from two xyxy boxes."""
    ix0 = max(gt[0], pred[0])
    iy0 = max(gt[1], pred[1])
    ix1 = min(gt[2], pred[2])
    iy1 = min(gt[3], pred[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    if inter == 0.0:
        return 0.0
    gt_area = max(0.0, gt[2] - gt[0]) * max(0.0, gt[3] - gt[1])
    pred_area = max(0.0, pred[2] - pred[0]) * max(0.0, pred[3] - pred[1])
    union = gt_area + pred_area - inter
    return inter / union if union > 0 else None


def _compute_acd_3d(
    pred_corners: list[list[float]],
    gt_corners: list[list[float]],
) -> float | None:
    """Mean Euclidean distance between corresponding corners (8×3, mm).

    Note: not symmetry-aware; use the evaluator for symmetry-corrected ACD3D.
    """
    if len(pred_corners) != 8 or len(gt_corners) != 8:
        return None
    try:
        total = sum(
            math.sqrt(sum((p - g) ** 2 for p, g in zip(pc, gc)))
            for pc, gc in zip(pred_corners, gt_corners)
        )
        return total / 8.0
    except Exception:  # noqa: BLE001
        return None


def _parse_corners_cam(raw: Any) -> list[list[float]] | None:
    """Parse a list of 8 × 3 cam-XYZ corner lists."""
    if not isinstance(raw, list) or len(raw) != 8:
        return None
    out: list[list[float]] = []
    for c in raw:
        if not isinstance(c, list) or len(c) != 3:
            return None
        try:
            out.append([float(v) for v in c])
        except (TypeError, ValueError):
            return None
    return out


def _build_instance_metrics(
    record: dict[str, Any],
    det: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Compute IoU2D, ACD3D, and hit flags for one query instance."""
    gt_raw = record.get("gt")
    if not isinstance(gt_raw, dict):
        return None

    metrics: dict[str, Any] = {}

    # ---- IoU2D ----
    gt_bbox = float_list(gt_raw.get("bbox_xyxy"), expected_len=4)
    pred_bbox_raw = det.get("bbox_2d_xyxy") if det else None
    pred_bbox = float_list(pred_bbox_raw, expected_len=4) if isinstance(pred_bbox_raw, list) else None

    if gt_bbox is not None and pred_bbox is not None:
        iou2d = _compute_iou_2d(gt_bbox, pred_bbox)
        if iou2d is not None:
            metrics["iou2d"] = round(iou2d, 4)
            metrics["hit2d@50"] = int(iou2d >= 0.50)
            metrics["hit2d@75"] = int(iou2d >= 0.75)

    # ---- ACD3D ----
    gt_corners_cam = _parse_corners_cam(gt_raw.get("bbox_3d_corners_cam_xyz_mm"))
    pred_corners_cam = _parse_corners_cam(det.get("bbox_3d_corners_cam_xyz_mm") if det else None)

    if gt_corners_cam is not None and pred_corners_cam is not None:
        acd = _compute_acd_3d(pred_corners_cam, gt_corners_cam)
        if acd is not None:
            metrics["acd3d_mm"] = round(acd, 1)

    return metrics if metrics else None


# ---------------------------------------------------------------------------


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _detection_list(record: dict[str, Any]) -> list[dict[str, Any]]:
    raw = record.get("detections")
    if not isinstance(raw, list):
        return []
    return [det for det in cast(list[Any], raw) if isinstance(det, dict)]


def _pick_detection(record: dict[str, Any]) -> dict[str, Any] | None:
    candidates = _detection_list(record)
    if not candidates:
        return None

    def _score(det: dict[str, Any]) -> tuple[int, float]:
        status = 1 if det.get("status") == "ok" else 0
        conf = _safe_float(det.get("confidence"))
        return status, conf if conf is not None else -1.0

    candidates_sorted = sorted(candidates, key=_score, reverse=True)
    return candidates_sorted[0] if candidates_sorted else None


def _resolve_pred_bbox(
    det: dict[str, Any] | None,
    width: int,
    height: int,
) -> list[float] | None:
    if det is None:
        return None
    bbox_xyxy = float_list(det.get("bbox_2d_xyxy"), expected_len=4)
    if bbox_xyxy is not None:
        return bbox_xyxy
    norm = float_list(det.get("bbox_2d_norm_1000"), expected_len=4)
    if norm is None or width <= 0 or height <= 0:
        return None
    return denorm_bbox_yxyx_to_xyxy(norm, width=width, height=height)


def _resolve_gt_bbox(record: dict[str, Any]) -> list[float] | None:
    gt_raw = record.get("gt")
    if not isinstance(gt_raw, dict):
        return None
    return float_list(gt_raw.get("bbox_xyxy"), expected_len=4)


def _resolve_gt_corners(record: dict[str, Any]) -> list[list[float]] | None:
    gt_raw = record.get("gt")
    if not isinstance(gt_raw, dict):
        return None
    return corner_list(gt_raw.get("bbox_3d_corners_norm_1000"))


def _build_overview_rows(
    records: list[dict[str, Any]],
    instance_metrics_list: list[dict[str, Any] | None] | None = None,
) -> list[dict[str, str]]:
    n_queries = len(records)
    parsed_counts: list[int] = []
    confidences: list[float] = []
    reproj_errors: list[float] = []
    pose_known = 0
    pose_ok = 0

    for record in records:
        dets = _detection_list(record)
        parsed_counts.append(len(dets))
        det = _pick_detection(record)
        if det is None:
            continue

        conf = _safe_float(det.get("confidence"))
        if conf is not None:
            confidences.append(conf)

        reproj = _safe_float(det.get("reprojection_error"))
        if reproj is not None:
            reproj_errors.append(reproj)

        pose_status = str(det.get("pose_status") or "").strip().lower()
        if pose_status in {"ok", "failed"}:
            pose_known += 1
            if pose_status == "ok":
                pose_ok += 1

    avg_parsed = (sum(parsed_counts) / len(parsed_counts)) if parsed_counts else None
    avg_conf = (sum(confidences) / len(confidences)) if confidences else None
    avg_reproj = (sum(reproj_errors) / len(reproj_errors)) if reproj_errors else None
    pose_success_rate = (pose_ok / pose_known) if pose_known > 0 else None

    rows: list[dict[str, str]] = [
        {"label": "queries", "value": str(n_queries)},
        {"label": "columns", "value": str(n_queries + 1)},
        {"label": "total detections", "value": str(sum(parsed_counts))},
        {"label": "avg parsed/query", "value": format_metric(avg_parsed, 2)},
        {"label": "avg confidence", "value": format_metric(avg_conf, 3)},
        {"label": "pose success", "value": format_percent(pose_success_rate, 1)},
        {"label": "avg reproj err", "value": format_metric(avg_reproj, 2)},
    ]

    # Aggregate per-instance metrics if available.
    if instance_metrics_list:
        iou2d_vals: list[float] = []
        hit2d50_vals: list[int] = []
        hit2d75_vals: list[int] = []
        acd3d_vals: list[float] = []
        for m in instance_metrics_list:
            if not isinstance(m, dict):
                continue
            if "iou2d" in m:
                iou2d_vals.append(float(m["iou2d"]))
                hit2d50_vals.append(int(m.get("hit2d@50", 0)))
                hit2d75_vals.append(int(m.get("hit2d@75", 0)))
            if "acd3d_mm" in m:
                acd3d_vals.append(float(m["acd3d_mm"]))

        if iou2d_vals:
            avg_iou2d = sum(iou2d_vals) / len(iou2d_vals)
            hit2d50_rate = sum(hit2d50_vals) / len(hit2d50_vals)
            hit2d75_rate = sum(hit2d75_vals) / len(hit2d75_vals)
            rows += [
                {"label": "avg IoU2D", "value": format_metric(avg_iou2d, 3)},
                {"label": "hit2d@50", "value": format_percent(hit2d50_rate, 1)},
                {"label": "hit2d@75", "value": format_percent(hit2d75_rate, 1)},
            ]
        if acd3d_vals:
            avg_acd3d = sum(acd3d_vals) / len(acd3d_vals)
            rows.append({"label": "avg ACD3D", "value": f"{avg_acd3d:.1f} mm"})

    return rows


def _pred_3d_status(pred_corners: list[list[float]] | None) -> str:
    if pred_corners is None:
        return "none"
    in_view = any(0.0 <= c[0] <= 1000.0 and 0.0 <= c[1] <= 1000.0 for c in pred_corners)
    return "visible" if in_view else "off-screen"


def _instance_rows(
    record: dict[str, Any],
    det: dict[str, Any] | None,
    pred_corners: list[list[float]] | None = None,
    metrics: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    rows.append({"label": "query_id", "value": str(record.get("query_id", "n/a"))})
    rows.append({"label": "status", "value": str(record.get("status", "n/a"))})

    parsed_count = len(_detection_list(record))
    rows.append({"label": "parsed detections", "value": str(parsed_count)})

    parse_warning = record.get("parse_warning")
    rows.append({"label": "parse warning", "value": str(parse_warning) if parse_warning else "none"})

    if det is None:
        rows.append({"label": "detection", "value": "not found"})
        return rows

    rows.extend(
        [
            {"label": "object", "value": str(det.get("object_name") or "n/a")},
            {"label": "confidence", "value": format_metric(det.get("confidence"), 3)},
            {"label": "det status", "value": str(det.get("status") or "n/a")},
            {"label": "pred 3D", "value": _pred_3d_status(pred_corners)},
            {"label": "pose", "value": str(det.get("pose_status") or "n/a")},
            {"label": "reproj err", "value": format_metric(det.get("reprojection_error"), 2)},
        ]
    )

    warning = det.get("pose_warning")
    rows.append({"label": "pose warning", "value": str(warning) if warning else "none"})

    # Per-instance metrics (IoU2D, hits, ACD3D).
    if isinstance(metrics, dict) and metrics:
        iou2d = metrics.get("iou2d")
        if iou2d is not None:
            rows.append({"label": "IoU2D", "value": format_metric(iou2d, 3)})
            rows.append({"label": "hit2d@50", "value": str(metrics.get("hit2d@50", "n/a"))})
            rows.append({"label": "hit2d@75", "value": str(metrics.get("hit2d@75", "n/a"))})
        acd3d = metrics.get("acd3d_mm")
        if acd3d is not None:
            rows.append({"label": "ACD3D", "value": f"{float(acd3d):.1f} mm"})

    return rows


def build_debug_payload(
    image_id: int,
    image_records: list[dict[str, Any]],
    model_name: str,
    image_size: tuple[int, int] | None,
) -> dict[str, Any]:
    width = int(image_size[0]) if image_size is not None else 0
    height = int(image_size[1]) if image_size is not None else 0

    instances: list[dict[str, Any]] = []
    instance_metrics_list: list[dict[str, Any] | None] = []
    for idx, record in enumerate(image_records):
        det = _pick_detection(record)
        pred_bbox = _resolve_pred_bbox(det, width=width, height=height)
        gt_bbox = _resolve_gt_bbox(record)
        gt_corners = _resolve_gt_corners(record)
        pred_corners = corner_list(det.get("bbox_3d_corners_norm_1000")) if det else None

        inst_metrics = _build_instance_metrics(record, det)
        instance_metrics_list.append(inst_metrics)

        instances.append(
            {
                "title": f"Detection {idx + 1}",
                "query": str(record.get("query") or ""),
                "rows": _instance_rows(record, det, pred_corners, metrics=inst_metrics),
                "query_id": record.get("query_id"),
                "gt_bbox_xyxy": gt_bbox,
                "pred_bbox_xyxy": pred_bbox,
                "gt_bbox_3d_corners_norm_1000": gt_corners,
                "pred_bbox_3d_corners_norm_1000": pred_corners,
                "metrics": inst_metrics,
            }
        )

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "source": "inference",
        "image_id": int(image_id),
        "model_name": model_name,
        "image_width": width,
        "image_height": height,
        "overview_title": "RGB with GT and predicted boxes",
        "overview_rows": _build_overview_rows(image_records, instance_metrics_list),
        "instances": instances,
    }
    return payload


def flush_image_debug_artifacts(
    debug_dir: Path,
    image_id: int,
    image_records: list[dict[str, Any]],
    image_bytes: bytes | None,
    model_name: str,
) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)

    image_size: tuple[int, int] | None = None
    base_rgb: Image.Image | None = None

    if image_bytes is not None:
        try:
            base_rgb = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            image_size = base_rgb.size
        except Exception:
            base_rgb = None
            image_size = None

    payload = build_debug_payload(
        image_id=int(image_id),
        image_records=image_records,
        model_name=model_name,
        image_size=image_size,
    )

    out_json = debug_dir / f"{int(image_id):06d}.json"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if base_rgb is not None:
        out_png = debug_dir / f"{int(image_id):06d}_report.png"
        report = render_columns_report(image=base_rgb, payload=payload)
        report.save(out_png)
