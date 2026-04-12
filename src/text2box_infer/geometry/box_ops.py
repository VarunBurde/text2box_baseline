from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

from ..types import PoseResult


def denormalize_bbox_yxyx_to_xyxy(
    bbox_norm_1000: list[float], height: int, width: int
) -> list[float]:
    ymin, xmin, ymax, xmax = [float(v) for v in bbox_norm_1000]

    ymin = _clamp(ymin, 0.0, 1000.0)
    xmin = _clamp(xmin, 0.0, 1000.0)
    ymax = _clamp(ymax, 0.0, 1000.0)
    xmax = _clamp(xmax, 0.0, 1000.0)

    x0 = (xmin / 1000.0) * float(width)
    y0 = (ymin / 1000.0) * float(height)
    x1 = (xmax / 1000.0) * float(width)
    y1 = (ymax / 1000.0) * float(height)

    x_min, x_max = sorted([x0, x1])
    y_min, y_max = sorted([y0, y1])

    x_min = _clamp(x_min, 0.0, float(width))
    x_max = _clamp(x_max, 0.0, float(width))
    y_min = _clamp(y_min, 0.0, float(height))
    y_max = _clamp(y_max, 0.0, float(height))

    return [x_min, y_min, x_max, y_max]


def box_3d_cam_xyz_size_rpy_mm_deg_to_corners_cam_xyz_mm(
    box_3d: list[float],
) -> tuple[list[list[float]], list[float]] | None:
    if len(box_3d) != 9:
        return None

    (
        x_center_mm,
        y_center_mm,
        z_center_mm,
        x_size_mm,
        y_size_mm,
        z_size_mm,
        roll_deg,
        pitch_deg,
        yaw_deg,
    ) = [float(v) for v in box_3d]

    center = np.array([x_center_mm, y_center_mm, z_center_mm], dtype=np.float64)
    size_mm = np.array([x_size_mm, y_size_mm, z_size_mm], dtype=np.float64)

    if not np.all(np.isfinite(center)):
        return None
    if not np.all(np.isfinite(size_mm)):
        return None
    if np.any(size_mm <= 0.0):
        return None

    roll = math.radians(float(roll_deg))
    pitch = math.radians(float(pitch_deg))
    yaw = math.radians(float(yaw_deg))

    sr = math.sin(roll / 2.0)
    sp = math.sin(pitch / 2.0)
    sy = math.sin(yaw / 2.0)
    cr = math.cos(roll / 2.0)
    cp = math.cos(pitch / 2.0)
    cz = math.cos(yaw / 2.0)

    qx = sr * cp * cz - cr * sp * sy
    qy = cr * sp * cz + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cz
    qw = cr * cp * cz + sr * sp * sy

    rotation_matrix = np.array(
        [
            [1.0 - 2.0 * qy**2 - 2.0 * qz**2, 2.0 * qx * qy - 2.0 * qw * qz, 2.0 * qx * qz + 2.0 * qw * qy],
            [2.0 * qx * qy + 2.0 * qw * qz, 1.0 - 2.0 * qx**2 - 2.0 * qz**2, 2.0 * qy * qz - 2.0 * qw * qx],
            [2.0 * qx * qz - 2.0 * qw * qy, 2.0 * qy * qz + 2.0 * qw * qx, 1.0 - 2.0 * qx**2 - 2.0 * qy**2],
        ],
        dtype=np.float64,
    )

    local_corners = _build_centered_cuboid_corners(size_mm).astype(np.float64)
    corners_cam_xyz = (rotation_matrix @ local_corners.T).T + center.reshape(1, 3)

    return (
        corners_cam_xyz.astype(float).tolist(),
        size_mm.astype(float).tolist(),
    )


def solve_pose_from_corners_norm(
    corners_norm_1000: list[list[float]],
    intrinsics: list[float],
    object_meta: dict[str, Any],
    image_height: int,
    image_width: int,
    max_reprojection_error: float = 120.0,
) -> PoseResult:
    if len(corners_norm_1000) != 8:
        return PoseResult(
            success=False,
            r_cam_from_model=None,
            t_cam_from_model=None,
            bbox_3d_R=None,
            bbox_3d_t=None,
            bbox_3d_size=None,
            reprojection_error=None,
            message="Expected exactly 8 projected corners.",
        )

    if len(intrinsics) != 4:
        return PoseResult(
            success=False,
            r_cam_from_model=None,
            t_cam_from_model=None,
            bbox_3d_R=None,
            bbox_3d_t=None,
            bbox_3d_size=None,
            reprojection_error=None,
            message="Intrinsics must have 4 values: [fx, fy, cx, cy].",
        )

    image_points = _denormalize_corners_yx_to_xy(
        corners_norm_1000,
        height=int(image_height),
        width=int(image_width),
    )

    fx, fy, cx, cy = [float(v) for v in intrinsics]
    camera_matrix = np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    object_points = _build_object_corners_model(object_meta).astype(np.float64)

    best: dict[str, Any] | None = None

    for permutation in _candidate_permutations():
        permuted_img = image_points[np.array(permutation, dtype=np.int64)]

        ok, rvec, tvec, _ = cv2.solvePnPRansac(
            object_points,
            permuted_img,
            camera_matrix,
            distCoeffs=None,
            iterationsCount=200,
            reprojectionError=8.0,
            confidence=0.99,
            flags=cv2.SOLVEPNP_EPNP,
        )
        if not ok:
            continue

        refine_ok, rvec_refined, tvec_refined = cv2.solvePnP(
            object_points,
            permuted_img,
            camera_matrix,
            distCoeffs=None,
            rvec=rvec,
            tvec=tvec,
            useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if refine_ok:
            rvec = rvec_refined
            tvec = tvec_refined

        reprojection_error = _mean_reprojection_error(
            object_points,
            permuted_img,
            camera_matrix,
            rvec,
            tvec,
        )

        if (best is None) or (reprojection_error < best["reprojection_error"]):
            best = {
                "rvec": rvec,
                "tvec": tvec,
                "permutation": permutation,
                "reprojection_error": reprojection_error,
            }

    if best is None:
        return PoseResult(
            success=False,
            r_cam_from_model=None,
            t_cam_from_model=None,
            bbox_3d_R=None,
            bbox_3d_t=None,
            bbox_3d_size=None,
            reprojection_error=None,
            message="solvePnP failed for all corner permutations.",
        )

    if not math.isfinite(best["reprojection_error"]):
        return PoseResult(
            success=False,
            r_cam_from_model=None,
            t_cam_from_model=None,
            bbox_3d_R=None,
            bbox_3d_t=None,
            bbox_3d_size=None,
            reprojection_error=best["reprojection_error"],
            permutation=list(best["permutation"]),
            message="Reprojection error is not finite.",
        )

    if best["reprojection_error"] > max_reprojection_error:
        return PoseResult(
            success=False,
            r_cam_from_model=None,
            t_cam_from_model=None,
            bbox_3d_R=None,
            bbox_3d_t=None,
            bbox_3d_size=None,
            reprojection_error=float(best["reprojection_error"]),
            permutation=list(best["permutation"]),
            message=(
                "Reprojection error is above threshold: "
                f"{best['reprojection_error']:.3f} > {max_reprojection_error:.3f}"
            ),
        )

    r_cam_from_model, _ = cv2.Rodrigues(best["rvec"])
    t_cam_from_model = best["tvec"].reshape(3)

    bbox_3d_model_R = np.array(object_meta["bbox_3d_model_R"], dtype=np.float64).reshape(3, 3)
    bbox_3d_model_t = np.array(object_meta["bbox_3d_model_t"], dtype=np.float64).reshape(3)
    bbox_3d_model_size = np.array(object_meta["bbox_3d_model_size"], dtype=np.float64).reshape(3)

    bbox_3d_R = r_cam_from_model @ bbox_3d_model_R
    bbox_3d_t = r_cam_from_model @ bbox_3d_model_t + t_cam_from_model

    return PoseResult(
        success=True,
        r_cam_from_model=r_cam_from_model.reshape(-1).astype(float).tolist(),
        t_cam_from_model=t_cam_from_model.astype(float).tolist(),
        bbox_3d_R=bbox_3d_R.reshape(-1).astype(float).tolist(),
        bbox_3d_t=bbox_3d_t.reshape(-1).astype(float).tolist(),
        bbox_3d_size=bbox_3d_model_size.astype(float).tolist(),
        reprojection_error=float(best["reprojection_error"]),
        permutation=list(best["permutation"]),
        message=None,
    )


def solve_pose_from_corners_norm_with_size(
    corners_norm_1000: list[list[float]],
    intrinsics: list[float],
    bbox_3d_size_mm: list[float],
    image_height: int,
    image_width: int,
    max_reprojection_error: float = 120.0,
) -> PoseResult:
    if len(corners_norm_1000) != 8:
        return PoseResult(
            success=False,
            r_cam_from_model=None,
            t_cam_from_model=None,
            bbox_3d_R=None,
            bbox_3d_t=None,
            bbox_3d_size=None,
            reprojection_error=None,
            message="Expected exactly 8 projected corners.",
        )

    if len(intrinsics) != 4:
        return PoseResult(
            success=False,
            r_cam_from_model=None,
            t_cam_from_model=None,
            bbox_3d_R=None,
            bbox_3d_t=None,
            bbox_3d_size=None,
            reprojection_error=None,
            message="Intrinsics must have 4 values: [fx, fy, cx, cy].",
        )

    if len(bbox_3d_size_mm) != 3:
        return PoseResult(
            success=False,
            r_cam_from_model=None,
            t_cam_from_model=None,
            bbox_3d_R=None,
            bbox_3d_t=None,
            bbox_3d_size=None,
            reprojection_error=None,
            message="bbox_3d_size_mm must have 3 values: [length_mm, width_mm, height_mm].",
        )

    size_mm = np.array([float(v) for v in bbox_3d_size_mm], dtype=np.float64).reshape(3)
    if np.any(size_mm <= 0.0):
        return PoseResult(
            success=False,
            r_cam_from_model=None,
            t_cam_from_model=None,
            bbox_3d_R=None,
            bbox_3d_t=None,
            bbox_3d_size=None,
            reprojection_error=None,
            message="bbox_3d_size_mm values must be strictly positive.",
        )

    image_points = _denormalize_corners_yx_to_xy(
        corners_norm_1000,
        height=int(image_height),
        width=int(image_width),
    )

    fx, fy, cx, cy = [float(v) for v in intrinsics]
    camera_matrix = np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    object_points = _build_centered_cuboid_corners(size_mm).astype(np.float64)

    best: dict[str, Any] | None = None

    for permutation in _candidate_permutations():
        permuted_img = image_points[np.array(permutation, dtype=np.int64)]

        ok, rvec, tvec, _ = cv2.solvePnPRansac(
            object_points,
            permuted_img,
            camera_matrix,
            distCoeffs=None,
            iterationsCount=200,
            reprojectionError=8.0,
            confidence=0.99,
            flags=cv2.SOLVEPNP_EPNP,
        )
        if not ok:
            continue

        refine_ok, rvec_refined, tvec_refined = cv2.solvePnP(
            object_points,
            permuted_img,
            camera_matrix,
            distCoeffs=None,
            rvec=rvec,
            tvec=tvec,
            useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if refine_ok:
            rvec = rvec_refined
            tvec = tvec_refined

        reprojection_error = _mean_reprojection_error(
            object_points,
            permuted_img,
            camera_matrix,
            rvec,
            tvec,
        )

        if (best is None) or (reprojection_error < best["reprojection_error"]):
            best = {
                "rvec": rvec,
                "tvec": tvec,
                "permutation": permutation,
                "reprojection_error": reprojection_error,
            }

    if best is None:
        return PoseResult(
            success=False,
            r_cam_from_model=None,
            t_cam_from_model=None,
            bbox_3d_R=None,
            bbox_3d_t=None,
            bbox_3d_size=None,
            reprojection_error=None,
            message="solvePnP failed for all corner permutations.",
        )

    if not math.isfinite(best["reprojection_error"]):
        return PoseResult(
            success=False,
            r_cam_from_model=None,
            t_cam_from_model=None,
            bbox_3d_R=None,
            bbox_3d_t=None,
            bbox_3d_size=None,
            reprojection_error=best["reprojection_error"],
            permutation=list(best["permutation"]),
            message="Reprojection error is not finite.",
        )

    if best["reprojection_error"] > max_reprojection_error:
        return PoseResult(
            success=False,
            r_cam_from_model=None,
            t_cam_from_model=None,
            bbox_3d_R=None,
            bbox_3d_t=None,
            bbox_3d_size=None,
            reprojection_error=float(best["reprojection_error"]),
            permutation=list(best["permutation"]),
            message=(
                "Reprojection error is above threshold: "
                f"{best['reprojection_error']:.3f} > {max_reprojection_error:.3f}"
            ),
        )

    r_cam_from_model, _ = cv2.Rodrigues(best["rvec"])
    t_cam_from_model = best["tvec"].reshape(3)

    return PoseResult(
        success=True,
        r_cam_from_model=r_cam_from_model.reshape(-1).astype(float).tolist(),
        t_cam_from_model=t_cam_from_model.astype(float).tolist(),
        bbox_3d_R=r_cam_from_model.reshape(-1).astype(float).tolist(),
        bbox_3d_t=t_cam_from_model.astype(float).tolist(),
        bbox_3d_size=size_mm.astype(float).tolist(),
        reprojection_error=float(best["reprojection_error"]),
        permutation=list(best["permutation"]),
        message=None,
    )


def _denormalize_corners_yx_to_xy(
    corners_norm_1000: list[list[float]], height: int, width: int
) -> np.ndarray:
    image_points: list[list[float]] = []
    for point in corners_norm_1000:
        y, x = float(point[0]), float(point[1])
        y = _clamp(y, 0.0, 1000.0)
        x = _clamp(x, 0.0, 1000.0)

        px = (x / 1000.0) * float(width)
        py = (y / 1000.0) * float(height)
        image_points.append([px, py])

    return np.array(image_points, dtype=np.float64)


def _build_object_corners_model(object_meta: dict[str, Any]) -> np.ndarray:
    size = np.array(object_meta["bbox_3d_model_size"], dtype=np.float64)
    bbox_3d_model_R = np.array(object_meta["bbox_3d_model_R"], dtype=np.float64).reshape(3, 3)
    bbox_3d_model_t = np.array(object_meta["bbox_3d_model_t"], dtype=np.float64).reshape(3, 1)

    sx, sy, sz = size.tolist()
    corners_box = np.array(
        [
            [-sx / 2.0, -sy / 2.0, +sz / 2.0],  # Front-Top-Left
            [+sx / 2.0, -sy / 2.0, +sz / 2.0],  # Front-Top-Right
            [+sx / 2.0, +sy / 2.0, +sz / 2.0],  # Front-Bottom-Right
            [-sx / 2.0, +sy / 2.0, +sz / 2.0],  # Front-Bottom-Left
            [-sx / 2.0, -sy / 2.0, -sz / 2.0],  # Back-Top-Left
            [+sx / 2.0, -sy / 2.0, -sz / 2.0],  # Back-Top-Right
            [+sx / 2.0, +sy / 2.0, -sz / 2.0],  # Back-Bottom-Right
            [-sx / 2.0, +sy / 2.0, -sz / 2.0],  # Back-Bottom-Left
        ],
        dtype=np.float64,
    )

    corners_model = (bbox_3d_model_R @ corners_box.T) + bbox_3d_model_t
    return corners_model.T


def _build_centered_cuboid_corners(size_xyz: np.ndarray) -> np.ndarray:
    sx, sy, sz = [float(v) for v in size_xyz.reshape(3).tolist()]
    return np.array(
        [
            [-sx / 2.0, -sy / 2.0, +sz / 2.0],  # Front-Top-Left
            [+sx / 2.0, -sy / 2.0, +sz / 2.0],  # Front-Top-Right
            [+sx / 2.0, +sy / 2.0, +sz / 2.0],  # Front-Bottom-Right
            [-sx / 2.0, +sy / 2.0, +sz / 2.0],  # Front-Bottom-Left
            [-sx / 2.0, -sy / 2.0, -sz / 2.0],  # Back-Top-Left
            [+sx / 2.0, -sy / 2.0, -sz / 2.0],  # Back-Top-Right
            [+sx / 2.0, +sy / 2.0, -sz / 2.0],  # Back-Bottom-Right
            [-sx / 2.0, +sy / 2.0, -sz / 2.0],  # Back-Bottom-Left
        ],
        dtype=np.float64,
    )


def _mean_reprojection_error(
    object_points: np.ndarray,
    image_points: np.ndarray,
    camera_matrix: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
) -> float:
    projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, None)
    projected_xy = projected.reshape(-1, 2)
    distances = np.linalg.norm(projected_xy - image_points, axis=1)
    return float(np.mean(distances))


def _candidate_permutations() -> list[list[int]]:
    base = [
        [0, 1, 2, 3, 4, 5, 6, 7],
        [1, 2, 3, 0, 5, 6, 7, 4],
        [2, 3, 0, 1, 6, 7, 4, 5],
        [3, 0, 1, 2, 7, 4, 5, 6],
        [0, 3, 2, 1, 4, 7, 6, 5],
        [3, 2, 1, 0, 7, 6, 5, 4],
        [2, 1, 0, 3, 6, 5, 4, 7],
        [1, 0, 3, 2, 5, 4, 7, 6],
    ]

    extended: list[list[int]] = []
    for permutation in base:
        extended.append(permutation)
        swapped = [idx + 4 if idx < 4 else idx - 4 for idx in permutation]
        extended.append(swapped)

    deduped: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    for permutation in extended:
        key = tuple(permutation)
        if key not in seen:
            seen.add(key)
            deduped.append(permutation)
    return deduped


def _clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))
