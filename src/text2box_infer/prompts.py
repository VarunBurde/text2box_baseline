from __future__ import annotations

from textwrap import dedent

from .types import ModelRequest, PromptProfile


def build_prompt(request: ModelRequest) -> str:
    catalog_preview = ", ".join(request.object_catalog[:120])

    if request.prompt_profile == PromptProfile.DIRECT_JSON:
        return _build_prompt_direct_json(request=request, catalog_preview=catalog_preview)

    if request.prompt_profile == PromptProfile.NORMALIZED:
        return _build_prompt_normalized(request=request, catalog_preview=catalog_preview)

    return _build_prompt_normalized_pnp(request=request, catalog_preview=catalog_preview)


def _build_prompt_direct_json(request: ModelRequest, catalog_preview: str) -> str:
    return dedent(
        f"""
        You are a spatial grounding engine.

        Task:
        - Query: {request.query}
        - Image width: {request.width}
        - Image height: {request.height}
        - Camera intrinsics [fx, fy, cx, cy]: {request.intrinsics}

        Predict final coordinates directly and return ONLY valid JSON with this exact schema:
        {{
          "detections": [
            {{
              "object_name": "string or null",
              "bbox_2d_norm_1000": [ymin, xmin, ymax, xmax],
              "bbox_3d_corners_cam_xyz_mm": [
                [x, y, z],
                [x, y, z],
                [x, y, z],
                [x, y, z],
                [x, y, z],
                [x, y, z],
                [x, y, z],
                [x, y, z]
              ],
              "confidence": 0.0
            }}
          ]
        }}

        Output contract (STRICT):
        - Return exactly one JSON object and nothing else.
        - Return one of these forms only:
          {{"detections": []}}
          or
          {{"detections": [{{single_detection}}]}}
        - detections must contain at most one detection for the query object only.
        - Use keys exactly as shown in the schema.
        - Do not repeat keys.
        - Do not add trailing commas.
        - Do not include comments, markdown, or explanations.

        Corner order for bbox_3d_corners_cam_xyz_mm:
        1 Front-Top-Left
        2 Front-Top-Right
        3 Front-Bottom-Right
        4 Front-Bottom-Left
        5 Back-Top-Left
        6 Back-Top-Right
        7 Back-Bottom-Right
        8 Back-Bottom-Left

        Rules:
        - Output bbox_3d_corners_cam_xyz_mm for every detection.
        - bbox_3d_corners_cam_xyz_mm must be camera-frame metric coordinates in millimeters.
        - Each cam corner must be exactly 3 numbers: [x_mm, y_mm, z_mm].
        - z_mm must be positive for all corners.
        - Use normalized coordinates in range 0..1000 for bbox_2d_norm_1000.
        - bbox format is [ymin, xmin, ymax, xmax].
        - 3D corners must be exactly 8 points in the listed order.
        - Infer amodal geometry (include occluded parts).
        - If object extends outside the image, clip to 0..1000.
        - confidence must be a number in [0, 1].
        - No markdown, no code block, no explanation.
        - If object is not found, return {{"detections": []}}.
        - Prefer object_name values from this catalog when possible:
          {catalog_preview}
        """
    ).strip()


def _build_prompt_normalized(request: ModelRequest, catalog_preview: str) -> str:
    return dedent(
        f"""
        You are a spatial grounding engine.

        Task:
        - Query: {request.query}
        - Image width: {request.width}
        - Image height: {request.height}
        - Camera intrinsics [fx, fy, cx, cy]: {request.intrinsics}

        Use a normalized coordinate system from 0 to 1000 for both axes.
        Return ONLY valid JSON with this exact schema:
        {{
          "detections": [
            {{
              "object_name": "string or null",
              "bbox_2d_norm_1000": [ymin, xmin, ymax, xmax],
              "bbox_3d_corners_norm_1000": [
                [y, x],
                [y, x],
                [y, x],
                [y, x],
                [y, x],
                [y, x],
                [y, x],
                [y, x]
              ],
              "confidence": 0.0
            }}
          ]
        }}

        Corner order for bbox_3d_corners_norm_1000 (exactly this order):
        1 Front-Top-Left
        2 Front-Top-Right
        3 Front-Bottom-Right
        4 Front-Bottom-Left
        5 Back-Top-Left
        6 Back-Top-Right
        7 Back-Bottom-Right
        8 Back-Bottom-Left

        Rules:
        - bbox format is [ymin, xmin, ymax, xmax].
        - bbox_2d_norm_1000 must be AMODAL: include full object extent, including occluded parts.
        - Do not return only the visible silhouette; estimate hidden extent from object geometry.
        - Return exactly 8 corners as [y, x] numeric pairs in the required order.
        - 3D corners should represent a full amodal cuboid; infer occluded corners.
        - If object extends outside the image, clip coordinates to 0..1000.
        - All coordinates must be numbers between 0 and 1000.
        - No markdown, no code block, no explanation.
        - If object is not found, return {{"detections": []}}.
        - Prefer object_name values from this catalog when possible:
          {catalog_preview}
        """
    ).strip()


def _build_prompt_normalized_pnp(request: ModelRequest, catalog_preview: str) -> str:
    return dedent(
        f"""
        You are a spatial grounding engine.

        Task:
        - Query: {request.query}
        - Image width: {request.width}
        - Image height: {request.height}
        - Camera intrinsics [fx, fy, cx, cy]: {request.intrinsics}

        Use a normalized coordinate system from 0 to 1000 for both axes.
        Return ONLY valid JSON with this exact schema:
        {{
          "detections": [
            {{
              "object_name": "string or null",
              "bbox_2d_norm_1000": [ymin, xmin, ymax, xmax],
              "bbox_3d_corners_norm_1000": [
                [y, x],
                [y, x],
                [y, x],
                [y, x],
                [y, x],
                [y, x],
                [y, x],
                [y, x]
              ],
              "confidence": 0.0
            }}
          ]
        }}

        Corner order for bbox_3d_corners_norm_1000 (exactly this order):
        1 Front-Top-Left
        2 Front-Top-Right
        3 Front-Bottom-Right
        4 Front-Bottom-Left
        5 Back-Top-Left
        6 Back-Top-Right
        7 Back-Bottom-Right
        8 Back-Bottom-Left

        Rules:
        - bbox format is [ymin, xmin, ymax, xmax].
        - bbox_2d_norm_1000 must be AMODAL: include full object extent, including occluded parts.
        - Do not return only the visible silhouette; estimate hidden extent from object geometry.
        - Use bbox_2d_norm_1000 as one cuboid face: the front face must be exactly
          [[ymin, xmin], [ymin, xmax], [ymax, xmax], [ymax, xmin]].
        - This means corners 1..4 must match bbox_2d_norm_1000 exactly in [y, x] order.
        - Return exactly 8 corners as [y, x] numeric pairs in the required order.
        - Front face and back face must both be present and not identical.
        - The 4 front-face corners must form a non-zero-area quadrilateral.
        - The 4 back-face corners must form a non-zero-area quadrilateral.
        - Corresponding front/back corners should be separated to indicate depth.
        - Do not collapse all 8 corners to one plane, one line, or one point.
        - Infer occluded corners to form a full amodal cuboid.
        - If geometry is too uncertain to output a valid 8-corner cuboid, output {{"detections": []}}.
        - If object extends outside the image, clip coordinates to 0..1000.
        - All coordinates must be numbers between 0 and 1000.
        - No markdown, no code block, no explanation.
        - Prefer object_name values from this catalog when possible:
          {catalog_preview}
        """
    ).strip()
