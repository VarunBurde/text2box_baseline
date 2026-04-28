"""Per-detection processing extracted from run_inference's main loop."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..geometry import (
    box_3d_cam_xyz_size_rpy_mm_deg_to_corners_cam_xyz_mm,
    denormalize_bbox_yxyx_to_xyxy,
    project_cam_xyz_to_norm_1000,
    solve_pose_from_corners_norm_with_size,
)
from ..types import IntermediateDetection, ParsedResponse, RunMode


@dataclass
class DetectionOutcome:
    """Result of processing one parsed detection.

    ``row`` is None when the detection was rejected before reaching the parquet
    output (missing bbox or invalid extents). ``pose_succeeded`` is True only
    when BASELINE_2D3D mode ran PnP successfully.
    """

    manifest: dict[str, Any]
    row: dict[str, Any] | None
    pose_succeeded: bool


def build_query_manifest(
    *,
    query_id: int,
    image_id: int,
    query: str,
    provider: str,
    image_meta: dict[str, Any],
    parsed: ParsedResponse,
    raw_response: str,
    gt_entry: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "query_id": query_id,
        "image_id": image_id,
        "query": query,
        "provider": provider,
        "image_width": int(image_meta["width"]),
        "image_height": int(image_meta["height"]),
        "intrinsics": [float(v) for v in image_meta["intrinsics"]],
        "status": "ok",
        "parse_warning": parsed.parse_warning,
        "raw_response": raw_response,
        "parsed_detection_count": len(parsed.detections),
        "detections": [],
        "gt": gt_entry,
    }


def _empty_row(annotation_id: int, query_id: int, instance_id: int, bbox_2d: list[float]) -> dict[str, Any]:
    return {
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


def _resolve_corners_norm(
    detection: IntermediateDetection,
    *,
    intrinsics: list[float],
    width: int,
    height: int,
) -> tuple[list[list[float]] | None, str | None]:
    """Return (corners_norm_1000, source_label) using cached or projected corners.

    Mutates ``detection`` to backfill ``bbox_3d_corners_cam_xyz_mm`` and
    ``bbox_3d_size_mm`` when they can be derived from ``box_3d_cam_xyz_size_rpy_mm_deg``.
    """
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

    corners_norm_1000 = detection.bbox_3d_corners_norm_1000
    if corners_norm_1000 is not None:
        return corners_norm_1000, "provided_norm_1000"
    if detection.bbox_3d_corners_cam_xyz_mm is None:
        return None, None
    projected = project_cam_xyz_to_norm_1000(
        corners_cam_xyz_mm=detection.bbox_3d_corners_cam_xyz_mm,
        intrinsics=intrinsics,
        image_width=width,
        image_height=height,
    )
    if projected is None:
        return None, None
    source = "projected_from_box_3d_rpy_mm_deg" if derived_from_box_3d else "projected_from_cam_xyz_mm"
    return projected, source


def _solve_and_fill_pose(
    *,
    row: dict[str, Any],
    detection_manifest: dict[str, Any],
    detection: IntermediateDetection,
    corners_norm_1000: list[list[float]] | None,
    intrinsics: list[float],
    width: int,
    height: int,
) -> bool:
    """Run PnP for BASELINE_2D3D mode; populate row + manifest. Return success."""
    if corners_norm_1000 is None:
        detection_manifest["pose_status"] = "failed"
        detection_manifest["pose_warning"] = (
            "missing bbox_3d_corners_norm_1000 and could not project "
            "bbox_3d_corners_cam_xyz_mm/box_3d"
        )
        return False
    if detection.bbox_3d_size_mm is None:
        detection_manifest["pose_status"] = "failed"
        detection_manifest["pose_warning"] = "missing bbox_3d_size_mm"
        return False

    pose = solve_pose_from_corners_norm_with_size(
        corners_norm_1000=corners_norm_1000,
        intrinsics=intrinsics,
        bbox_3d_size_mm=detection.bbox_3d_size_mm,
        image_height=height,
        image_width=width,
    )
    detection_manifest["reprojection_error"] = pose.reprojection_error
    detection_manifest["permutation"] = pose.permutation

    if pose.success:
        row["bbox_3d_R"] = pose.bbox_3d_R
        row["bbox_3d_t"] = pose.bbox_3d_t
        row["bbox_3d_size"] = pose.bbox_3d_size
        row["R_cam_from_model"] = pose.r_cam_from_model
        row["t_cam_from_model"] = pose.t_cam_from_model
        detection_manifest["pose_status"] = "ok"
        return True

    detection_manifest["pose_status"] = "failed"
    detection_manifest["pose_warning"] = pose.message
    return False


def process_detection(
    detection: IntermediateDetection,
    *,
    image_meta: dict[str, Any],
    mode: RunMode,
    query_id: int,
    instance_id: int,
    annotation_id: int,
) -> DetectionOutcome:
    """Validate, project corners, and (in BASELINE_2D3D) solve pose for one detection."""
    if detection.bbox_2d_norm_1000 is None:
        return DetectionOutcome(
            manifest={
                "status": "skipped",
                "warning": "missing bbox_2d_norm_1000",
                "object_name": detection.object_name,
            },
            row=None,
            pose_succeeded=False,
        )

    width = int(image_meta["width"])
    height = int(image_meta["height"])
    intrinsics = [float(v) for v in image_meta["intrinsics"]]

    bbox_2d = denormalize_bbox_yxyx_to_xyxy(
        bbox_norm_1000=detection.bbox_2d_norm_1000,
        height=height,
        width=width,
    )
    if bbox_2d[2] <= bbox_2d[0] or bbox_2d[3] <= bbox_2d[1]:
        return DetectionOutcome(
            manifest={
                "status": "skipped",
                "warning": "invalid bbox extents after conversion",
                "object_name": detection.object_name,
            },
            row=None,
            pose_succeeded=False,
        )

    corners_norm_1000, corners_source = _resolve_corners_norm(
        detection, intrinsics=intrinsics, width=width, height=height,
    )
    row = _empty_row(annotation_id, query_id, instance_id, bbox_2d)
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

    pose_succeeded = False
    if mode == RunMode.BASELINE_2D3D:
        pose_succeeded = _solve_and_fill_pose(
            row=row,
            detection_manifest=detection_manifest,
            detection=detection,
            corners_norm_1000=corners_norm_1000,
            intrinsics=intrinsics,
            width=width,
            height=height,
        )

    return DetectionOutcome(manifest=detection_manifest, row=row, pose_succeeded=pose_succeeded)
