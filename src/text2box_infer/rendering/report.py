"""Top-level multi-column report orchestrator: header, summary, detections, footer."""
from __future__ import annotations

from typing import Any

from PIL import Image, ImageDraw, ImageFont

from .detection_card import render_detection_cards
from .layout import draw_header, draw_legend_footer, row_pairs, wrap_text_lines
from .primitives import (
    BG,
    DET_3D_ROW_H,
    DET_COL_W,
    DET_IMG_H,
    FOOTER_H,
    GAP,
    HEADER_H,
    IMG_PAD,
    MARGIN,
    ROW_H,
    SCALE,
    SUMMARY_2D_IMG_H,
    SUMMARY_3D_IMG_H,
    SUMMARY_COL_W,
    load_font,
)
from .summary_card import render_summary_card


def _estimate_column_heights(
    instances: list[dict[str, Any]],
    overview_rows: list[tuple[str, str]],
    small_font: ImageFont.ImageFont,
) -> tuple[int, list[int]]:
    summary_h = (
        20 * SCALE + SUMMARY_2D_IMG_H + 3 * SCALE + SUMMARY_3D_IMG_H + 3 * SCALE
        + max(1, len(overview_rows)) * ROW_H + 4 * SCALE
    )
    dummy = Image.new("RGB", (10, 10), color=BG)
    probe = ImageDraw.Draw(dummy)
    inst_heights: list[int] = []
    for inst in instances:
        query = str(inst.get("query") or "")
        q_lines = wrap_text_lines(probe, query, small_font, DET_COL_W - 2 * IMG_PAD - 8, max_lines=4)
        rows = row_pairs(inst.get("rows"))
        inst_h = (
            20 * SCALE + DET_IMG_H + 3 * SCALE + DET_3D_ROW_H + 3 * SCALE
            + (1 + len(q_lines)) * ROW_H + 2 * SCALE
            + max(1, len(rows)) * ROW_H + 4 * SCALE
        )
        inst_heights.append(inst_h)
    return summary_h, inst_heights


def render_columns_report(image: Image.Image, payload: dict[str, Any]) -> Image.Image:
    """Render N+1 debug columns (image summary + one card per detection instance)."""
    image_id_raw = payload.get("image_id")
    image_id = int(image_id_raw) if isinstance(image_id_raw, (int, float)) else None
    model_name = str(payload.get("model_name") or "unknown-model")

    overview_rows = row_pairs(payload.get("overview_rows"))
    overview_title = str(payload.get("overview_title") or "Image summary")

    raw_instances = payload.get("instances")
    instances: list[dict[str, Any]] = (
        [inst for inst in raw_instances if isinstance(inst, dict)]
        if isinstance(raw_instances, list) else []
    )

    fonts = {
        "title": load_font(24 * SCALE),
        "body": load_font(15 * SCALE),
        "small": load_font(13 * SCALE),
        "metric": load_font(12 * SCALE),
        "badge": load_font(11 * SCALE),
    }

    summary_h, inst_heights = _estimate_column_heights(instances, overview_rows, fonts["small"])
    body_h = max([summary_h] + inst_heights) if inst_heights else summary_h

    n_cols = 1 + len(instances)
    canvas_w = 2 * MARGIN + SUMMARY_COL_W
    if n_cols > 1:
        canvas_w += len(instances) * (DET_COL_W + GAP)
    canvas_h = MARGIN + HEADER_H + GAP + body_h + GAP + FOOTER_H + MARGIN

    canvas = Image.new("RGB", (canvas_w, canvas_h), color=BG)
    draw = ImageDraw.Draw(canvas)

    draw_header(
        draw=draw, canvas_w=canvas_w, image_id=image_id, model_name=model_name,
        title_font=fonts["title"], body_font=fonts["body"],
    )

    top_y = MARGIN + HEADER_H + GAP
    render_summary_card(
        canvas=canvas, draw=draw, image=image,
        instances=instances, overview_rows=overview_rows, overview_title=overview_title,
        sx=MARGIN, top_y=top_y, body_h=body_h, fonts=fonts,
    )
    render_detection_cards(
        canvas=canvas, draw=draw, image=image, instances=instances,
        start_x=MARGIN + SUMMARY_COL_W + GAP, top_y=top_y, body_h=body_h, fonts=fonts,
    )

    footer_y = top_y + body_h + GAP
    draw_legend_footer(draw=draw, canvas_w=canvas_w, y=footer_y, font=fonts["small"])
    return canvas
