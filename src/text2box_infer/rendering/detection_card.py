"""Per-detection card rendering: thumbnails, badges, query text, metric rows."""
from __future__ import annotations

from typing import Any

from PIL import Image, ImageDraw, ImageFont

from ..evaluation.iou import iou_xyxy
from ..utils import corner_list, float_list
from .layout import (
    badge_color_iou,
    badge_color_reproj,
    draw_rows,
    row_pairs,
    wrap_text_lines,
)
from .overlays import draw_2d_overlay, draw_3d_gt_pred_overlay_preview
from .primitives import (
    ACCENT,
    BADGE_RED,
    DET_3D_ROW_H,
    DET_COL_W,
    DET_IMG_H,
    GAP,
    IMG_PAD,
    MUTED,
    PANEL,
    PANEL_BORDER,
    ROW_H,
    SCALE,
    TEXT,
    draw_badge,
    draw_card,
    fit_to_box,
)


def _draw_detection_2d_badges(
    detection_img: Image.Image,
    *,
    iou_val: float | None,
    confidence_str: str | None,
    badge_font: ImageFont.ImageFont,
) -> None:
    draw = ImageDraw.Draw(detection_img)
    bx, by = 4, detection_img.height - 15 * SCALE - 6
    if iou_val is not None:
        bx = draw_badge(
            draw, bx, by, f"IoU {iou_val:.2f}",
            bg=badge_color_iou(iou_val), font=badge_font,
        ) + 4
    if confidence_str and confidence_str != "n/a":
        try:
            draw_badge(
                draw, bx, by, f"conf {float(confidence_str):.2f}",
                bg=ACCENT, font=badge_font,
            )
        except ValueError:
            pass


def _draw_detection_3d_badges(
    mini_img: Image.Image,
    *,
    pose_status: str | None,
    reproj_str: str | None,
    badge_font: ImageFont.ImageFont,
) -> None:
    draw = ImageDraw.Draw(mini_img)
    bx, by = 4, mini_img.height - 15 * SCALE - 6
    if pose_status == "ok" and reproj_str and reproj_str != "n/a":
        try:
            reproj_f = float(reproj_str)
            if reproj_f > 0.001:
                draw_badge(
                    draw, bx, by, f"reproj {reproj_f:.1f}px",
                    bg=badge_color_reproj(reproj_f), font=badge_font,
                )
        except ValueError:
            pass
    elif pose_status == "failed":
        draw_badge(draw, bx, by, "pose failed", bg=BADGE_RED, font=badge_font)


def render_detection_cards(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    image: Image.Image,
    instances: list[dict[str, Any]],
    start_x: int,
    top_y: int,
    body_h: int,
    fonts: dict[str, ImageFont.ImageFont],
) -> None:
    x = start_x
    for idx, inst in enumerate(instances):
        draw_card(draw, x, top_y, DET_COL_W, body_h, fill=PANEL, outline=PANEL_BORDER, radius=10)
        title = str(inst.get("title") or f"Detection {idx + 1}")
        draw.text((x + IMG_PAD, top_y + 3 * SCALE), title, fill=ACCENT, font=fonts["body"])

        gt_bbox = float_list(inst.get("gt_bbox_xyxy"), expected_len=4)
        pred_bbox = float_list(inst.get("pred_bbox_xyxy"), expected_len=4)
        pred_corners = corner_list(inst.get("pred_bbox_3d_corners_norm_1000"))
        gt_corners = corner_list(inst.get("gt_bbox_3d_corners_norm_1000"))

        inst_row_dict = dict(row_pairs(inst.get("rows")))
        metrics_raw = inst.get("metrics")
        inst_metrics: dict[str, Any] = metrics_raw if isinstance(metrics_raw, dict) else {}

        det_2d = draw_2d_overlay(image, gt_bbox, pred_bbox, label=f"D{idx + 1}")
        det_img = fit_to_box(det_2d, DET_COL_W - 2 * IMG_PAD, DET_IMG_H)
        iou_val: float | None = inst_metrics.get("iou2d") if inst_metrics else None
        if iou_val is None and gt_bbox is not None and pred_bbox is not None:
            iou_val = iou_xyxy(gt_bbox, pred_bbox)
        _draw_detection_2d_badges(
            det_img, iou_val=iou_val,
            confidence_str=inst_row_dict.get("confidence"), badge_font=fonts["badge"],
        )
        canvas.paste(det_img, (x + IMG_PAD, top_y + 20 * SCALE))

        three_d_y = top_y + 20 * SCALE + DET_IMG_H + 3 * SCALE
        draw.text((x + IMG_PAD, three_d_y + 2), "3D GT vs Pred", fill=MUTED, font=fonts["small"])
        mini_y = three_d_y + ROW_H
        mini_h = DET_3D_ROW_H - ROW_H
        mini_w = DET_COL_W - 2 * IMG_PAD
        combined_3d = draw_3d_gt_pred_overlay_preview(image, gt_corners, pred_corners)
        mini_img = fit_to_box(combined_3d, mini_w, mini_h)
        _draw_detection_3d_badges(
            mini_img,
            pose_status=inst_row_dict.get("pose"),
            reproj_str=inst_row_dict.get("reproj err"),
            badge_font=fonts["badge"],
        )
        canvas.paste(mini_img, (x + IMG_PAD, mini_y))

        text_y = three_d_y + DET_3D_ROW_H + 3 * SCALE
        draw.text((x + IMG_PAD, text_y + 2), "Query:", fill=MUTED, font=fonts["small"])
        q_lines = wrap_text_lines(
            draw=draw, text=str(inst.get("query") or ""),
            font=fonts["small"],
            max_width=DET_COL_W - 2 * IMG_PAD - 8, max_lines=4,
        )
        text_y += ROW_H
        for line in q_lines:
            draw.text((x + IMG_PAD, text_y + 2), line, fill=TEXT, font=fonts["small"])
            text_y += ROW_H

        text_y += 2 * SCALE
        rows = row_pairs(inst.get("rows")) or [("info", "no rows")]
        draw_rows(
            draw=draw, rows=rows,
            x=x + IMG_PAD, y=text_y,
            width=DET_COL_W - 2 * IMG_PAD, row_h=ROW_H,
            label_font=fonts["metric"], value_font=fonts["metric"],
        )

        x += DET_COL_W + GAP
