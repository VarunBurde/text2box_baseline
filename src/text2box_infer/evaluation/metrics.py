"""
Protocol metric computation for Text2Box-style outputs.

Provides AP2D/AP3D, AR2D/AR3D, and ACD3D metrics with symmetry-aware 3D IoU.
This module is the canonical library implementation; CLI entry is __main__.py.
"""
from __future__ import annotations

import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from scipy.spatial import QhullError

from text2box_infer.geometry.box_ops import denormalize_bbox_yxyx_to_xyxy, solve_pose_from_corners_norm


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class PredEntry:
    confidence: float
    iou2d: float | None
    iou3d: float | None
    acd3d: float | None


# ---------------------------------------------------------------------------
# Manifest / path utilities
# ---------------------------------------------------------------------------

def resolve_manifest_jsonl(
    manifest_jsonl_arg: str | None,
    split: str,
    predictions_root: Path,
) -> Path | None:
    if manifest_jsonl_arg is not None and str(manifest_jsonl_arg).strip():
        explicit = Path(str(manifest_jsonl_arg).strip())
        if not explicit.exists():
            raise FileNotFoundError(f"Manifest JSONL not found: {explicit}")
        return explicit

    candidates: list[Path] = []
    preferred = [
        predictions_root / f"preds_ollama_{split}_manifest.jsonl",
        predictions_root / f"preds_openai_{split}_manifest.jsonl",
    ]
    for path in preferred:
        if path.exists():
            candidates.append(path)

    if not candidates:
        discovered = [path for path in predictions_root.glob("**/*_manifest.jsonl") if path.is_file()]
        discovered.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        candidates = discovered

    if not candidates:
        return None

    if len(candidates) > 1:
        print(f"[manifest] multiple candidates found; using latest: {candidates[0]}")
    else:
        print(f"[manifest] auto-discovered: {candidates[0]}")
    return candidates[0]


def infer_run_dir_from_manifest(manifest_jsonl: Path) -> Path:
    if manifest_jsonl.parent.name == "predictions":
        return manifest_jsonl.parent.parent
    return manifest_jsonl.parent


# ---------------------------------------------------------------------------
# Array helpers
# ---------------------------------------------------------------------------

def _to_float_array(value: Any, expected_len: int | None = None) -> np.ndarray | None:
    if value is None:
        return None
    try:
        arr = np.array(value, dtype=np.float64).reshape(-1)
    except Exception:
        return None
    if expected_len is not None and arr.size != expected_len:
        return None
    return arr


def _float_list(value: Any, expected_len: int) -> list[float] | None:
    if not isinstance(value, list) or len(value) != expected_len:
        return None
    try:
        return [float(v) for v in value]
    except (TypeError, ValueError):
        return None


def _corner_list(value: Any) -> list[list[float]] | None:
    if not isinstance(value, list) or len(value) != 8:
        return None
    out: list[list[float]] = []
    for pt in value:
        if not isinstance(pt, list) or len(pt) != 2:
            return None
        try:
            out.append([float(pt[0]), float(pt[1])])
        except (TypeError, ValueError):
            return None
    return out


def _confidence_from_detection(det: dict[str, Any]) -> float:
    try:
        return float(det.get("confidence", 0.0))
    except (TypeError, ValueError):
        return 0.0


# ---------------------------------------------------------------------------
# Symmetry parsing
# ---------------------------------------------------------------------------

def _parse_symmetry_discrete(value: Any) -> list[tuple[np.ndarray, np.ndarray]]:
    out: list[tuple[np.ndarray, np.ndarray]] = []
    if value is None:
        return out

    arr = np.array(value, dtype=np.float64)
    if arr.size == 0:
        return out

    if arr.ndim == 1 and arr.size % 16 == 0:
        arr = arr.reshape(-1, 4, 4)
    elif arr.ndim == 2 and arr.shape[1] == 16:
        arr = arr.reshape(-1, 4, 4)
    elif arr.ndim == 3 and arr.shape[1:] == (4, 4):
        pass
    else:
        return out

    for mat in arr:
        r = mat[:3, :3].astype(np.float64)
        t = mat[:3, 3].astype(np.float64)
        out.append((r, t))
    return out


def _axis_angle_to_matrix(axis: np.ndarray, theta: float) -> np.ndarray:
    """Rodrigues rotation matrix for unit axis and angle theta."""
    ax = axis / max(1e-12, np.linalg.norm(axis))
    x, y, z = ax.tolist()
    ct = math.cos(theta)
    st = math.sin(theta)
    vt = 1.0 - ct
    return np.array(
        [
            [ct + x * x * vt, x * y * vt - z * st, x * z * vt + y * st],
            [y * x * vt + z * st, ct + y * y * vt, y * z * vt - x * st],
            [z * x * vt - y * st, z * y * vt + x * st, ct + z * z * vt],
        ],
        dtype=np.float64,
    )


def _parse_symmetry_continuous(
    value: Any,
    steps: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    out: list[tuple[np.ndarray, np.ndarray]] = []
    if value is None:
        return out

    items = value if isinstance(value, list) else [value]
    for item in items:
        if not isinstance(item, dict):
            continue
        axis = _to_float_array(item.get("axis"), expected_len=3)
        if axis is None:
            continue
        norm = float(np.linalg.norm(axis))
        if norm <= 1e-12:
            continue
        axis = axis / norm

        offset = _to_float_array(item.get("offset"), expected_len=3)
        if offset is None:
            offset = np.zeros(3, dtype=np.float64)

        for k in range(max(1, steps)):
            theta = (2.0 * math.pi * float(k)) / float(max(1, steps))
            r = _axis_angle_to_matrix(axis, theta)
            # t keeps the offset point fixed: offset = r @ offset + t  =>  t = offset - r @ offset
            t = offset - (r @ offset)
            out.append((r, t))

    return out


def build_symmetry_set(
    sym_discrete: Any,
    sym_continuous: Any,
    continuous_steps: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    transforms: list[tuple[np.ndarray, np.ndarray]] = [
        (np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64))
    ]
    transforms.extend(_parse_symmetry_discrete(sym_discrete))
    transforms.extend(_parse_symmetry_continuous(sym_continuous, steps=continuous_steps))

    deduped: list[tuple[np.ndarray, np.ndarray]] = []
    seen: set[tuple[float, ...]] = set()
    for r, t in transforms:
        key = tuple(np.round(np.concatenate([r.reshape(-1), t.reshape(-1)]), 6).tolist())
        if key in seen:
            continue
        seen.add(key)
        deduped.append((r, t))
    return deduped


# ---------------------------------------------------------------------------
# 3D geometry
# ---------------------------------------------------------------------------

def canonical_box_corners(size_xyz: np.ndarray) -> np.ndarray:
    """8 corners of an axis-aligned box centred at origin, ordered FTL/FTR/FBR/FBL/BTL/BTR/BBR/BBL."""
    sx, sy, sz = [float(v) for v in size_xyz]
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


def corners_from_bbox_pose(r_bbox: np.ndarray, t_bbox: np.ndarray, size_xyz: np.ndarray) -> np.ndarray:
    base = canonical_box_corners(size_xyz)
    return (r_bbox @ base.T).T + t_bbox.reshape(1, 3)


def apply_model_symmetry_to_pose(
    r_cam_from_model: np.ndarray,
    t_cam_from_model: np.ndarray,
    r_sym: np.ndarray,
    t_sym: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    r_new = r_cam_from_model @ r_sym
    t_new = (r_cam_from_model @ t_sym.reshape(3, 1)).reshape(3) + t_cam_from_model
    return r_new, t_new


def bbox_pose_from_model_pose(
    r_cam_from_model: np.ndarray,
    t_cam_from_model: np.ndarray,
    bbox_model_r: np.ndarray,
    bbox_model_t: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    r_bbox = r_cam_from_model @ bbox_model_r
    t_bbox = (r_cam_from_model @ bbox_model_t.reshape(3, 1)).reshape(3) + t_cam_from_model
    return r_bbox, t_bbox


def iou_xyxy(box_a: list[float], box_b: list[float]) -> float:
    ax0, ay0, ax1, ay1 = [float(v) for v in box_a]
    bx0, by0, bx1, by1 = [float(v) for v in box_b]

    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)

    iw = max(0.0, ix1 - ix0)
    ih = max(0.0, iy1 - iy0)
    inter = iw * ih

    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return float(inter / union)


def obb_planes(r_bbox: np.ndarray, t_bbox: np.ndarray, size_xyz: np.ndarray) -> list[tuple[np.ndarray, float]]:
    """Return 6 half-space planes (n, d) such that a point p is inside iff n @ p <= d for all planes."""
    axes = [r_bbox[:, 0], r_bbox[:, 1], r_bbox[:, 2]]
    center = t_bbox.reshape(3)
    extents = (size_xyz.reshape(3) * 0.5).astype(np.float64)

    planes: list[tuple[np.ndarray, float]] = []
    for i, axis in enumerate(axes):
        norm = float(np.linalg.norm(axis))
        if norm <= 1e-12:
            continue
        n = axis / norm
        d_pos = float(n @ center + extents[i])
        d_neg = float((-n) @ center + extents[i])
        planes.append((n, d_pos))
        planes.append((-n, d_neg))
    return planes


def intersection_vertices_from_planes(
    planes: list[tuple[np.ndarray, float]],
    det_eps: float = 1e-9,
    inside_eps: float = 1e-7,
    dedup_eps: float = 1e-6,
) -> np.ndarray:
    vertices: list[np.ndarray] = []
    for i, j, k in itertools.combinations(range(len(planes)), 3):
        a = np.vstack([planes[i][0], planes[j][0], planes[k][0]])
        b = np.array([planes[i][1], planes[j][1], planes[k][1]], dtype=np.float64)

        det = float(np.linalg.det(a))
        if abs(det) <= det_eps:
            continue

        try:
            x = np.linalg.solve(a, b)
        except np.linalg.LinAlgError:
            continue

        ok = True
        for n, d in planes:
            if float(n @ x) > d + inside_eps:
                ok = False
                break
        if not ok:
            continue

        duplicate = False
        for v in vertices:
            if float(np.linalg.norm(v - x)) <= dedup_eps:
                duplicate = True
                break
        if not duplicate:
            vertices.append(x)

    if not vertices:
        return np.zeros((0, 3), dtype=np.float64)
    return np.vstack(vertices)


def convex_hull_volume(points_xyz: np.ndarray) -> float:
    if points_xyz.shape[0] < 4:
        return 0.0
    try:
        hull = ConvexHull(points_xyz)
        return float(hull.volume)
    except (QhullError, ValueError):
        return 0.0


def iou_3d_oriented(
    r1: np.ndarray,
    t1: np.ndarray,
    size1: np.ndarray,
    r2: np.ndarray,
    t2: np.ndarray,
    size2: np.ndarray,
) -> float:
    vol1 = float(np.prod(np.maximum(size1.reshape(3), 0.0)))
    vol2 = float(np.prod(np.maximum(size2.reshape(3), 0.0)))
    if vol1 <= 0.0 or vol2 <= 0.0:
        return 0.0

    planes = obb_planes(r1, t1, size1) + obb_planes(r2, t2, size2)
    inter_vertices = intersection_vertices_from_planes(planes)
    inter_vol = convex_hull_volume(inter_vertices)
    union = vol1 + vol2 - inter_vol
    if union <= 0.0:
        return 0.0
    return float(inter_vol / union)


def corner_distance_mean(corners_a: np.ndarray, corners_b: np.ndarray) -> float:
    distances = np.linalg.norm(corners_a - corners_b, axis=1)
    return float(np.mean(distances))


# ---------------------------------------------------------------------------
# Core metric computation
# ---------------------------------------------------------------------------

def _compute_ap_ar(
    query_predictions: dict[str, list[PredEntry]],
    iou_attr: str,
    thresholds: list[float],
    dmax: int,
) -> tuple[dict[float, float], dict[float, float], float, float]:
    """
    COCO-style 101-point AP and AR at multiple IoU thresholds.

    One-to-one matching per query: the highest-confidence prediction that
    meets the IoU threshold is counted as TP; all others for that query are FP.
    Predictions are globally sorted by confidence to build the PR curve.
    """
    n_gt = len(query_predictions)
    if n_gt == 0:
        empty = {tau: 0.0 for tau in thresholds}
        return empty, empty, 0.0, 0.0

    ap_by_tau: dict[float, float] = {}
    ar_by_tau: dict[float, float] = {}

    recall_grid = np.linspace(0.0, 1.0, 101)

    for tau in thresholds:
        packed: list[tuple[float, int, int]] = []
        matched_total = 0

        for preds in query_predictions.values():
            ranked = sorted(preds, key=lambda p: p.confidence, reverse=True)[: max(1, dmax)]
            matched = False
            for pred in ranked:
                raw = getattr(pred, iou_attr)
                iou_value = float(raw) if raw is not None and math.isfinite(float(raw)) else -1.0
                is_tp = (not matched) and (iou_value >= tau)
                tp = 1 if is_tp else 0
                fp = 0 if is_tp else 1
                packed.append((pred.confidence, tp, fp))
                if is_tp:
                    matched = True
                    matched_total += 1

        if not packed:
            ap_by_tau[tau] = 0.0
            ar_by_tau[tau] = 0.0
            continue

        packed.sort(key=lambda item: item[0], reverse=True)
        tps = np.cumsum([item[1] for item in packed], dtype=np.float64)
        fps = np.cumsum([item[2] for item in packed], dtype=np.float64)

        recalls = tps / float(n_gt)
        precisions = tps / np.maximum(tps + fps, 1e-12)

        # Monotone precision envelope (right-to-left max)
        for idx in range(len(precisions) - 2, -1, -1):
            if precisions[idx] < precisions[idx + 1]:
                precisions[idx] = precisions[idx + 1]

        # 101-point interpolation
        interp_precisions: list[float] = []
        for r in recall_grid:
            valid = np.where(recalls >= r)[0]
            if valid.size == 0:
                interp_precisions.append(0.0)
            else:
                interp_precisions.append(float(np.max(precisions[valid])))

        ap_by_tau[tau] = float(np.mean(interp_precisions))
        ar_by_tau[tau] = float(matched_total / float(n_gt))

    map_value = float(np.mean(list(ap_by_tau.values()))) if ap_by_tau else 0.0
    mar_value = float(np.mean(list(ar_by_tau.values()))) if ar_by_tau else 0.0
    return ap_by_tau, ar_by_tau, map_value, mar_value


def _acd3d(
    query_predictions: dict[str, list[PredEntry]],
    dmax: int,
) -> tuple[float | None, int]:
    """
    Average Corner Distance in 3D (ACD3D).

    For each query, uses the ACD3D of the *highest-confidence* prediction that
    has a valid 3D estimate (not the oracle-best across all predictions).
    This is consistent with how AP is evaluated and reflects real detection quality.
    """
    top_conf_distances: list[float] = []

    for preds in query_predictions.values():
        ranked = sorted(preds, key=lambda p: p.confidence, reverse=True)[: max(1, dmax)]
        # Take the first valid ACD3D in confidence-ranked order (top-confidence prediction)
        for p in ranked:
            if p.acd3d is not None and math.isfinite(float(p.acd3d)):
                top_conf_distances.append(float(p.acd3d))
                break

    if not top_conf_distances:
        return None, 0
    return float(np.mean(top_conf_distances)), len(top_conf_distances)


def _metric_at(ap_by_tau: dict[float, float], tau: float) -> float:
    for key, value in ap_by_tau.items():
        if abs(float(key) - float(tau)) < 1e-9:
            return float(value)
    return 0.0


def _build_query_metrics(
    query_predictions: dict[str, list[PredEntry]],
    query_meta: dict[str, dict[str, Any]],
    dmax: int,
) -> dict[str, dict[str, float | int | str | None]]:
    result: dict[str, dict[str, float | int | str | None]] = {}
    for query_id, preds in query_predictions.items():
        ranked = sorted(preds, key=lambda p: p.confidence, reverse=True)[: max(1, dmax)]

        iou2d_vals = [float(p.iou2d) for p in ranked if p.iou2d is not None and math.isfinite(float(p.iou2d))]
        iou3d_vals = [float(p.iou3d) for p in ranked if p.iou3d is not None and math.isfinite(float(p.iou3d))]
        acd3d_vals = [float(p.acd3d) for p in ranked if p.acd3d is not None and math.isfinite(float(p.acd3d))]

        best_iou2d = max(iou2d_vals) if iou2d_vals else None
        best_iou3d = max(iou3d_vals) if iou3d_vals else None
        best_acd3d = min(acd3d_vals) if acd3d_vals else None
        top_conf = float(ranked[0].confidence) if ranked else None

        meta = query_meta.get(query_id, {})
        result[query_id] = {
            "query_id": int(meta.get("query_id", 0)),
            "image_id": int(meta.get("image_id", 0)),
            "instance_idx": int(meta.get("instance_idx", 0)),
            "obj_id": int(meta.get("obj_id", 0)),
            "query": str(meta.get("query", "")),
            "num_predictions": int(len(ranked)),
            "top_confidence": top_conf,
            "best_iou2d": best_iou2d,
            "best_iou3d": best_iou3d,
            "best_acd3d": best_acd3d,
            "hit2d@50": 1 if best_iou2d is not None and best_iou2d >= 0.50 else 0,
            "hit2d@75": 1 if best_iou2d is not None and best_iou2d >= 0.75 else 0,
            "hit3d@25": 1 if best_iou3d is not None and best_iou3d >= 0.25 else 0,
            "hit3d@50": 1 if best_iou3d is not None and best_iou3d >= 0.50 else 0,
        }
    return result


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            payload = json.loads(text)
            if isinstance(payload, dict):
                records.append(payload)
    return records


def load_query_inputs_from_manifest(
    manifest_jsonl: Path,
    data_root: Path,
    split: str,
) -> list[dict[str, Any]]:
    if not manifest_jsonl.exists():
        raise FileNotFoundError(f"Missing manifest JSONL: {manifest_jsonl}")

    queries_path = data_root / f"queries_{split}.parquet"
    gts_path = data_root / f"gts_{split}.parquet"
    images_info_path = data_root / f"images_info_{split}.parquet"

    if not queries_path.exists():
        raise FileNotFoundError(f"Missing queries parquet: {queries_path}")
    if not gts_path.exists():
        raise FileNotFoundError(f"Missing gts parquet: {gts_path}")
    if not images_info_path.exists():
        raise FileNotFoundError(f"Missing images info parquet: {images_info_path}")

    manifest_records = _load_jsonl_records(manifest_jsonl)
    manifest_by_query_id: dict[int, dict[str, Any]] = {}
    for record in manifest_records:
        query_id_val = record.get("query_id")
        try:
            query_id = int(query_id_val)
        except (TypeError, ValueError):
            continue
        manifest_by_query_id[query_id] = record

    queries_df = pd.read_parquet(queries_path, columns=["query_id", "image_id", "query"])
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
    images_df = pd.read_parquet(images_info_path, columns=["image_id", "width", "height", "intrinsics"])

    gt_lookup: dict[int, dict[str, Any]] = {}
    for row in gts_df.itertuples(index=False):
        gt_lookup[int(row.query_id)] = {
            "obj_id": int(row.obj_id),
            "instance_id": int(row.instance_id),
            "bbox_xyxy": [float(v) for v in np.array(row.bbox_2d, dtype=np.float64).reshape(-1).tolist()],
            "R_cam_from_model": [
                float(v) for v in np.array(row.R_cam_from_model, dtype=np.float64).reshape(-1).tolist()
            ],
            "t_cam_from_model": [
                float(v) for v in np.array(row.t_cam_from_model, dtype=np.float64).reshape(-1).tolist()
            ],
        }

    image_lookup: dict[int, dict[str, Any]] = {}
    for row in images_df.itertuples(index=False):
        image_lookup[int(row.image_id)] = {
            "width": int(row.width),
            "height": int(row.height),
            "intrinsics": [float(v) for v in np.array(row.intrinsics, dtype=np.float64).reshape(-1).tolist()],
        }

    query_inputs: list[dict[str, Any]] = []
    for row in queries_df.itertuples(index=False):
        query_id = int(row.query_id)
        image_id = int(row.image_id)
        query_text = str(row.query)

        gt = gt_lookup.get(query_id)
        image_meta = image_lookup.get(image_id)
        if gt is None or image_meta is None:
            continue

        manifest_rec = manifest_by_query_id.get(query_id, {})
        detections_raw = (
            manifest_rec.get("detections")
            if isinstance(manifest_rec.get("detections"), list)
            else []
        )

        parsed_detections: list[dict[str, Any]] = []
        for det in detections_raw:
            if not isinstance(det, dict):
                continue
            if det.get("status") not in (None, "ok"):
                continue

            bbox_norm = det.get("bbox_2d_norm_1000")
            bbox_xyxy = det.get("bbox_2d_xyxy")
            if bbox_norm is None and bbox_xyxy is None:
                continue

            parsed_detections.append(
                {
                    "object_name": det.get("object_name"),
                    "obj_id": det.get("obj_id"),
                    "confidence": det.get("confidence"),
                    "bbox_2d_norm_1000": det.get("bbox_2d_norm_1000"),
                    "bbox_2d_xyxy": det.get("bbox_2d_xyxy"),
                    "bbox_3d_corners_norm_1000": det.get("bbox_3d_corners_norm_1000"),
                    "pose_status": det.get("pose_status"),
                    "pose_warning": det.get("pose_warning"),
                    "reprojection_error": det.get("reprojection_error"),
                }
            )

        query_inputs.append(
            {
                "query_id": query_id,
                "image_id": image_id,
                "instance_idx": int(gt["instance_id"]),
                "obj_id": int(gt["obj_id"]),
                "query": query_text,
                "width": int(image_meta["width"]),
                "height": int(image_meta["height"]),
                "intrinsics": [float(v) for v in image_meta["intrinsics"]],
                "raw_response": manifest_rec.get("raw_response"),
                "parse_warning": manifest_rec.get("parse_warning"),
                "parsed_detections": parsed_detections,
                "gt": {
                    "bbox_xyxy": [float(v) for v in gt["bbox_xyxy"]],
                    "R_cam_from_model": [float(v) for v in gt["R_cam_from_model"]],
                    "t_cam_from_model": [float(v) for v in gt["t_cam_from_model"]],
                },
            }
        )

    return query_inputs


def _load_object_lookup(data_root: Path, continuous_symmetry_steps: int) -> dict[int, dict[str, Any]]:
    objects_info_path = data_root / "objects_info.parquet"
    if not objects_info_path.exists():
        raise FileNotFoundError(f"Missing objects info parquet: {objects_info_path}")

    object_df = pd.read_parquet(
        objects_info_path,
        columns=[
            "obj_id",
            "bbox_3d_model_R",
            "bbox_3d_model_t",
            "bbox_3d_model_size",
            "symmetries_discrete",
            "symmetries_continuous",
        ],
    )
    object_lookup: dict[int, dict[str, Any]] = {}
    for row in object_df.itertuples(index=False):
        object_lookup[int(row.obj_id)] = {
            "bbox_3d_model_R": np.array(row.bbox_3d_model_R, dtype=np.float64).reshape(3, 3),
            "bbox_3d_model_t": np.array(row.bbox_3d_model_t, dtype=np.float64).reshape(3),
            "bbox_3d_model_size": np.array(row.bbox_3d_model_size, dtype=np.float64).reshape(3),
            "symmetry_set": build_symmetry_set(
                sym_discrete=row.symmetries_discrete,
                sym_continuous=row.symmetries_continuous,
                continuous_steps=continuous_symmetry_steps,
            ),
        }
    return object_lookup


def _eval_detections_for_query(
    item: dict[str, Any],
    object_lookup: dict[int, dict[str, Any]],
) -> list[PredEntry]:
    gt = item.get("gt") if isinstance(item.get("gt"), dict) else {}
    gt_bbox = _float_list(gt.get("bbox_xyxy"), expected_len=4)
    gt_r = _to_float_array(gt.get("R_cam_from_model"), expected_len=9)
    gt_t = _to_float_array(gt.get("t_cam_from_model"), expected_len=3)

    intrinsics = _float_list(item.get("intrinsics"), expected_len=4)
    width = int(item.get("width", 0))
    height = int(item.get("height", 0))
    image_size = (width, height) if width > 0 and height > 0 else None
    obj_id = int(item.get("obj_id", 0))
    object_meta = object_lookup.get(obj_id)

    detections = item.get("parsed_detections") if isinstance(item.get("parsed_detections"), list) else []
    records: list[PredEntry] = []

    for det in detections:
        if not isinstance(det, dict):
            continue

        confidence = _confidence_from_detection(det)
        iou2d_val: float | None = None
        iou3d_star: float | None = None
        acd3d_star: float | None = None

        pred_bbox_xyxy = _float_list(det.get("bbox_2d_xyxy"), expected_len=4)
        bbox_norm = _float_list(det.get("bbox_2d_norm_1000"), expected_len=4)
        if pred_bbox_xyxy is None and bbox_norm is not None and image_size is not None:
            w, h = image_size
            pred_bbox_xyxy = denormalize_bbox_yxyx_to_xyxy(bbox_norm, height=h, width=w)

        if pred_bbox_xyxy is not None and gt_bbox is not None:
            iou2d_val = iou_xyxy(pred_bbox_xyxy, gt_bbox)

        corners_norm = _corner_list(det.get("bbox_3d_corners_norm_1000"))
        can_eval_3d = (
            corners_norm is not None
            and intrinsics is not None
            and image_size is not None
            and object_meta is not None
            and gt_r is not None
            and gt_t is not None
        )

        if can_eval_3d:
            w, h = image_size
            pose = solve_pose_from_corners_norm(
                corners_norm_1000=corners_norm,
                intrinsics=intrinsics,
                object_meta={
                    "bbox_3d_model_R": object_meta["bbox_3d_model_R"].reshape(-1).tolist(),
                    "bbox_3d_model_t": object_meta["bbox_3d_model_t"].reshape(-1).tolist(),
                    "bbox_3d_model_size": object_meta["bbox_3d_model_size"].reshape(-1).tolist(),
                },
                image_height=h,
                image_width=w,
            )

            if (
                pose.success
                and pose.bbox_3d_R is not None
                and pose.bbox_3d_t is not None
                and pose.bbox_3d_size is not None
            ):
                pred_r_bbox = np.array(pose.bbox_3d_R, dtype=np.float64).reshape(3, 3)
                pred_t_bbox = np.array(pose.bbox_3d_t, dtype=np.float64).reshape(3)
                pred_size = np.array(pose.bbox_3d_size, dtype=np.float64).reshape(3)
                pred_corners = corners_from_bbox_pose(pred_r_bbox, pred_t_bbox, pred_size)

                gt_r_cam = gt_r.reshape(3, 3)
                gt_t_cam = gt_t.reshape(3)

                best_iou = -1.0
                best_acd = float("inf")

                # Note: best_iou and best_acd are tracked independently across symmetry variants.
                # Each is the oracle-best over the symmetry set, which is standard BOP-style evaluation.
                for sym_r, sym_t in object_meta["symmetry_set"]:
                    gt_r_sym, gt_t_sym = apply_model_symmetry_to_pose(
                        r_cam_from_model=gt_r_cam,
                        t_cam_from_model=gt_t_cam,
                        r_sym=sym_r,
                        t_sym=sym_t,
                    )
                    gt_bbox_r, gt_bbox_t = bbox_pose_from_model_pose(
                        r_cam_from_model=gt_r_sym,
                        t_cam_from_model=gt_t_sym,
                        bbox_model_r=object_meta["bbox_3d_model_R"],
                        bbox_model_t=object_meta["bbox_3d_model_t"],
                    )
                    gt_corners = corners_from_bbox_pose(
                        r_bbox=gt_bbox_r,
                        t_bbox=gt_bbox_t,
                        size_xyz=object_meta["bbox_3d_model_size"],
                    )

                    iou3d = iou_3d_oriented(
                        r1=pred_r_bbox, t1=pred_t_bbox, size1=pred_size,
                        r2=gt_bbox_r, t2=gt_bbox_t, size2=object_meta["bbox_3d_model_size"],
                    )
                    acd = corner_distance_mean(pred_corners, gt_corners)

                    if iou3d > best_iou:
                        best_iou = iou3d
                    if acd < best_acd:
                        best_acd = acd

                if best_iou >= 0.0:
                    iou3d_star = float(best_iou)
                if math.isfinite(best_acd):
                    acd3d_star = float(best_acd)

        records.append(PredEntry(confidence=confidence, iou2d=iou2d_val, iou3d=iou3d_star, acd3d=acd3d_star))

    return records


def _compute_metrics_from_predictions(
    query_predictions: dict[str, list[PredEntry]],
    query_meta: dict[str, dict[str, Any]],
    query_count: int,
    dmax: int,
    include_details: bool,
    include_query_metrics: bool,
    thresholds_2d: list[float],
    thresholds_3d: list[float],
) -> dict[str, Any]:
    ap2d_by_tau, ar2d_by_tau, ap2d, ar2d = _compute_ap_ar(
        query_predictions=query_predictions,
        iou_attr="iou2d",
        thresholds=thresholds_2d,
        dmax=dmax,
    )
    ap3d_by_tau, ar3d_by_tau, ap3d, ar3d = _compute_ap_ar(
        query_predictions=query_predictions,
        iou_attr="iou3d",
        thresholds=thresholds_3d,
        dmax=dmax,
    )
    acd3d, acd_matched = _acd3d(query_predictions=query_predictions, dmax=dmax)

    metrics = {
        "AP2D": float(ap2d),
        "AP2D@50": _metric_at(ap2d_by_tau, 0.50),
        "AP2D@75": _metric_at(ap2d_by_tau, 0.75),
        "AR2D": float(ar2d),
        "AP3D": float(ap3d),
        "AP3D@25": _metric_at(ap3d_by_tau, 0.25),
        "AP3D@50": _metric_at(ap3d_by_tau, 0.50),
        "AR3D": float(ar3d),
        "ACD3D": acd3d,
    }

    summary: dict[str, Any] = {
        "metrics": metrics,
        "counts": {
            "num_queries": int(query_count),
            "num_predictions_evaluated": int(len(query_predictions)),
        },
    }

    if include_query_metrics:
        summary["query_metrics"] = _build_query_metrics(query_predictions, query_meta, dmax)

    if include_details:
        summary["protocol"] = {
            "D_max": int(dmax),
            "num_queries": int(query_count),
            "matching": "Greedy one-to-one per query; global confidence sort; 101-point AP interpolation.",
        }
        summary["details"] = {
            "thresholds_2d": thresholds_2d,
            "thresholds_3d": thresholds_3d,
            "ap2d_per_threshold": {f"{k:.2f}": float(v) for k, v in ap2d_by_tau.items()},
            "ar2d_per_threshold": {f"{k:.2f}": float(v) for k, v in ar2d_by_tau.items()},
            "ap3d_per_threshold": {f"{k:.2f}": float(v) for k, v in ap3d_by_tau.items()},
            "ar3d_per_threshold": {f"{k:.2f}": float(v) for k, v in ar3d_by_tau.items()},
            "acd3d_matched_queries": int(acd_matched),
        }

    return summary


# ---------------------------------------------------------------------------
# Public API: compute metrics
# ---------------------------------------------------------------------------

def compute_protocol_metrics_from_manifest(
    manifest_jsonl: Path,
    data_root: Path,
    split: str,
    dmax: int = 100,
    continuous_symmetry_steps: int = 36,
    include_details: bool = False,
    include_query_metrics: bool = False,
) -> dict[str, Any]:
    """Compute AP2D/AP3D/AR/ACD3D from a manifest JSONL + dataset parquet tables."""
    query_inputs = load_query_inputs_from_manifest(
        manifest_jsonl=manifest_jsonl,
        data_root=data_root,
        split=split,
    )
    object_lookup = _load_object_lookup(data_root, continuous_symmetry_steps)

    thresholds_2d = [round(0.50 + 0.05 * i, 2) for i in range(10)]
    thresholds_3d = [round(0.05 + 0.05 * i, 2) for i in range(10)]

    query_predictions: dict[str, list[PredEntry]] = {}
    query_meta: dict[str, dict[str, Any]] = {}

    for item in query_inputs:
        query_id = str(int(item.get("query_id", 0)))
        query_meta[query_id] = {
            "query_id": int(item.get("query_id", 0)),
            "image_id": int(item.get("image_id", 0)),
            "instance_idx": int(item.get("instance_idx", 0)),
            "obj_id": int(item.get("obj_id", 0)),
            "query": str(item.get("query", "")),
        }
        query_predictions[query_id] = _eval_detections_for_query(item, object_lookup)

    summary = _compute_metrics_from_predictions(
        query_predictions=query_predictions,
        query_meta=query_meta,
        query_count=len(query_inputs),
        dmax=dmax,
        include_details=include_details,
        include_query_metrics=include_query_metrics,
        thresholds_2d=thresholds_2d,
        thresholds_3d=thresholds_3d,
    )

    # Patch counts with manifest-level info
    summary["counts"]["num_manifest_records"] = int(len(query_inputs))
    return summary


def compute_protocol_metrics(
    per_instance_dir: Path,
    data_root: Path,
    split: str,
    dmax: int = 100,
    continuous_symmetry_steps: int = 36,
    include_details: bool = False,
    include_query_metrics: bool = False,
) -> dict[str, Any]:
    """Legacy: compute metrics from per-instance JSON files (use manifest mode when possible)."""
    images_info_path = data_root / f"images_info_{split}.parquet"
    objects_info_path = data_root / "objects_info.parquet"

    if not per_instance_dir.exists():
        raise FileNotFoundError(f"Missing per-instance directory: {per_instance_dir}")
    if not images_info_path.exists():
        raise FileNotFoundError(f"Missing images info parquet: {images_info_path}")
    if not objects_info_path.exists():
        raise FileNotFoundError(f"Missing objects info parquet: {objects_info_path}")

    image_df = pd.read_parquet(images_info_path, columns=["image_id", "width", "height"])
    image_lookup: dict[int, tuple[int, int]] = {}
    for row in image_df.itertuples(index=False):
        image_lookup[int(row.image_id)] = (int(row.width), int(row.height))

    object_lookup = _load_object_lookup(data_root, continuous_symmetry_steps)

    thresholds_2d = [round(0.50 + 0.05 * i, 2) for i in range(10)]
    thresholds_3d = [round(0.05 + 0.05 * i, 2) for i in range(10)]

    query_predictions: dict[str, list[PredEntry]] = {}
    query_meta: dict[str, dict[str, Any]] = {}
    query_count = 0

    for path in sorted(per_instance_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))

        image_id = int(payload.get("image_id"))
        instance_idx = int(payload.get("instance_idx"))
        obj_id = int(payload.get("obj_id"))
        query_id = f"{image_id}_{instance_idx}"
        query_count += 1

        image_size = image_lookup.get(image_id)
        width, height = image_size if image_size else (0, 0)

        gt = payload.get("gt") if isinstance(payload.get("gt"), dict) else {}

        # Build a unified item dict for _eval_detections_for_query
        item: dict[str, Any] = {
            "query_id": query_id,
            "image_id": image_id,
            "instance_idx": instance_idx,
            "obj_id": obj_id,
            "width": width,
            "height": height,
            "intrinsics": payload.get("intrinsics"),
            "gt": {
                "bbox_xyxy": gt.get("bbox_xyxy"),
                "R_cam_from_model": gt.get("R_cam_from_model") or gt.get("R"),
                "t_cam_from_model": gt.get("t_cam_from_model") or gt.get("t"),
            },
            "parsed_detections": payload.get("parsed_detections") if isinstance(payload.get("parsed_detections"), list) else [],
        }

        query_meta[query_id] = {
            "query_id": query_id,
            "image_id": image_id,
            "instance_idx": instance_idx,
            "obj_id": obj_id,
            "query": str(payload.get("query", "")),
        }
        query_predictions[query_id] = _eval_detections_for_query(item, object_lookup)

    return _compute_metrics_from_predictions(
        query_predictions=query_predictions,
        query_meta=query_meta,
        query_count=query_count,
        dmax=dmax,
        include_details=include_details,
        include_query_metrics=include_query_metrics,
        thresholds_2d=thresholds_2d,
        thresholds_3d=thresholds_3d,
    )
