"""PnP-based pose estimation from projected 3D corners."""
from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

from ..types import PoseResult
from .corners import canonical_box_corners, object_corners_in_model_frame

DEFAULT_MAX_REPROJECTION_ERROR = 120.0


def _failure(message: str, **extra: Any) -> PoseResult:
    return PoseResult(
        success=False,
        r_cam_from_model=None,
        t_cam_from_model=None,
        bbox_3d_R=None,
        bbox_3d_t=None,
        bbox_3d_size=None,
        reprojection_error=extra.get("reprojection_error"),
        permutation=extra.get("permutation"),
        message=message,
    )


def _denormalize_corners_yx_to_xy(
    corners_norm_1000: list[list[float]], height: int, width: int
) -> np.ndarray:
    out: list[list[float]] = []
    for point in corners_norm_1000:
        y = max(0.0, min(1000.0, float(point[0])))
        x = max(0.0, min(1000.0, float(point[1])))
        out.append([(x / 1000.0) * float(width), (y / 1000.0) * float(height)])
    return np.array(out, dtype=np.float64)


def _camera_matrix(intrinsics: list[float]) -> np.ndarray:
    fx, fy, cx, cy = [float(v) for v in intrinsics]
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def _candidate_permutations() -> list[list[int]]:
    """All corner-orderings to try when matching predicted 2D corners to model 3D corners.

    The VLM's corner ordering is not guaranteed to match the canonical model
    indexing, so we try 8 cyclic rotations of the top face × 2 mirror flips ×
    optional top↔bottom swap. PnP is run for each and the lowest reprojection
    error wins.
    """
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
        extended.append([idx + 4 if idx < 4 else idx - 4 for idx in permutation])
    deduped: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    for permutation in extended:
        key = tuple(permutation)
        if key not in seen:
            seen.add(key)
            deduped.append(permutation)
    return deduped


def _mean_reprojection_error(
    object_points: np.ndarray,
    image_points: np.ndarray,
    camera_matrix: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
) -> float:
    projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, None)
    distances = np.linalg.norm(projected.reshape(-1, 2) - image_points, axis=1)
    return float(np.mean(distances))


def _best_pnp_fit(
    object_points: np.ndarray,
    image_points: np.ndarray,
    camera_matrix: np.ndarray,
) -> dict[str, Any] | None:
    """Run PnP for every corner permutation and return the one with lowest reprojection error.

    Each candidate is solved with EPNP+RANSAC then refined with iterative PnP.
    """
    best: dict[str, Any] | None = None
    for permutation in _candidate_permutations():
        permuted = image_points[np.array(permutation, dtype=np.int64)]

        ok, rvec, tvec, _ = cv2.solvePnPRansac(
            object_points,
            permuted,
            camera_matrix,
            distCoeffs=None,
            iterationsCount=200,
            reprojectionError=8.0,
            confidence=0.99,
            flags=cv2.SOLVEPNP_EPNP,
        )
        if not ok:
            continue

        refine_ok, rvec_r, tvec_r = cv2.solvePnP(
            object_points,
            permuted,
            camera_matrix,
            distCoeffs=None,
            rvec=rvec,
            tvec=tvec,
            useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if refine_ok:
            rvec, tvec = rvec_r, tvec_r

        err = _mean_reprojection_error(object_points, permuted, camera_matrix, rvec, tvec)
        if best is None or err < best["reprojection_error"]:
            best = {"rvec": rvec, "tvec": tvec, "permutation": permutation, "reprojection_error": err}

    return best


def _validate_pnp_inputs(
    corners_norm_1000: list[list[float]],
    intrinsics: list[float],
) -> str | None:
    if len(corners_norm_1000) != 8:
        return "Expected exactly 8 projected corners."
    if len(intrinsics) != 4:
        return "Intrinsics must have 4 values: [fx, fy, cx, cy]."
    return None


def solve_pose_from_corners_norm(
    corners_norm_1000: list[list[float]],
    intrinsics: list[float],
    object_meta: dict[str, Any],
    image_height: int,
    image_width: int,
    max_reprojection_error: float = DEFAULT_MAX_REPROJECTION_ERROR,
) -> PoseResult:
    """Solve full model→camera pose using known object 3D model corners."""
    err = _validate_pnp_inputs(corners_norm_1000, intrinsics)
    if err is not None:
        return _failure(err)

    image_points = _denormalize_corners_yx_to_xy(corners_norm_1000, image_height, image_width)
    camera_matrix = _camera_matrix(intrinsics)
    object_points = object_corners_in_model_frame(object_meta)

    best = _best_pnp_fit(object_points, image_points, camera_matrix)
    if best is None:
        return _failure("solvePnP failed for all corner permutations.")
    if not math.isfinite(best["reprojection_error"]):
        return _failure(
            "Reprojection error is not finite.",
            reprojection_error=best["reprojection_error"],
            permutation=list(best["permutation"]),
        )
    if best["reprojection_error"] > max_reprojection_error:
        return _failure(
            f"Reprojection error is above threshold: "
            f"{best['reprojection_error']:.3f} > {max_reprojection_error:.3f}",
            reprojection_error=float(best["reprojection_error"]),
            permutation=list(best["permutation"]),
        )

    r_cam_from_model, _ = cv2.Rodrigues(best["rvec"])
    t_cam_from_model = best["tvec"].reshape(3)

    bbox_R = np.asarray(object_meta["bbox_3d_model_R"], dtype=np.float64).reshape(3, 3)
    bbox_t = np.asarray(object_meta["bbox_3d_model_t"], dtype=np.float64).reshape(3)
    bbox_size = np.asarray(object_meta["bbox_3d_model_size"], dtype=np.float64).reshape(3)

    bbox_3d_R = r_cam_from_model @ bbox_R
    bbox_3d_t = r_cam_from_model @ bbox_t + t_cam_from_model

    return PoseResult(
        success=True,
        r_cam_from_model=r_cam_from_model.reshape(-1).astype(float).tolist(),
        t_cam_from_model=t_cam_from_model.astype(float).tolist(),
        bbox_3d_R=bbox_3d_R.reshape(-1).astype(float).tolist(),
        bbox_3d_t=bbox_3d_t.reshape(-1).astype(float).tolist(),
        bbox_3d_size=bbox_size.astype(float).tolist(),
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
    max_reprojection_error: float = DEFAULT_MAX_REPROJECTION_ERROR,
) -> PoseResult:
    """Solve bbox→camera pose using a known bbox size; bbox frame == model frame."""
    err = _validate_pnp_inputs(corners_norm_1000, intrinsics)
    if err is not None:
        return _failure(err)
    if len(bbox_3d_size_mm) != 3:
        return _failure("bbox_3d_size_mm must have 3 values: [length_mm, width_mm, height_mm].")

    size_mm = np.array([float(v) for v in bbox_3d_size_mm], dtype=np.float64).reshape(3)
    if np.any(size_mm <= 0.0):
        return _failure("bbox_3d_size_mm values must be strictly positive.")

    image_points = _denormalize_corners_yx_to_xy(corners_norm_1000, image_height, image_width)
    camera_matrix = _camera_matrix(intrinsics)
    object_points = canonical_box_corners(size_mm)

    best = _best_pnp_fit(object_points, image_points, camera_matrix)
    if best is None:
        return _failure("solvePnP failed for all corner permutations.")
    if not math.isfinite(best["reprojection_error"]):
        return _failure(
            "Reprojection error is not finite.",
            reprojection_error=best["reprojection_error"],
            permutation=list(best["permutation"]),
        )
    if best["reprojection_error"] > max_reprojection_error:
        return _failure(
            f"Reprojection error is above threshold: "
            f"{best['reprojection_error']:.3f} > {max_reprojection_error:.3f}",
            reprojection_error=float(best["reprojection_error"]),
            permutation=list(best["permutation"]),
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
