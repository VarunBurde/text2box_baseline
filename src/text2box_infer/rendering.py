"""Shared simple report renderer for Text2Box debug and visualization flows."""
from __future__ import annotations

import math
from typing import Any

from PIL import Image, ImageDraw, ImageFont

# Colors
GT_COLOR = (22, 163, 74)
PRED_COLOR = (220, 38, 38)
CUBE_FRONT = (185, 28, 28)
CUBE_BACK = (127, 29, 29)
CUBE_GT_FRONT = (21, 128, 61)
CUBE_GT_BACK = (22, 101, 52)
BG = (241, 245, 249)
PANEL = (255, 255, 255)
PANEL_BORDER = (203, 213, 225)
HEADER_BG = (15, 23, 42)
HEADER_TEXT = (248, 250, 252)
ACCENT = (59, 130, 246)
TEXT = (15, 23, 42)
MUTED = (100, 116, 139)

# Layout
MARGIN = 20
GAP = 12
HEADER_H = 56
FOOTER_H = 42
IMG_PAD = 10
ROW_H = 18
SUMMARY_COL_W = 440
DET_COL_W = 336
DET_IMG_H = 220
# Summary and detection columns use the same image height so all rows align.
SUMMARY_2D_IMG_H = DET_IMG_H
SUMMARY_3D_IMG_H = DET_IMG_H
# 3D row reserves one text-label row above the image.
DET_3D_ROW_H = ROW_H + DET_IMG_H


def float_list(value: Any, expected_len: int) -> list[float] | None:
    if not isinstance(value, list) or len(value) != expected_len:
        return None
    try:
        return [float(v) for v in value]
    except (TypeError, ValueError):
        return None


def corner_list(value: Any) -> list[list[float]] | None:
    if not isinstance(value, list) or len(value) != 8:
        return None
    out: list[list[float]] = []
    for point in value:
        if not isinstance(point, list) or len(point) != 2:
            return None
        try:
            out.append([float(point[0]), float(point[1])])
        except (TypeError, ValueError):
            return None
    return out


def format_metric(value: Any, precision: int = 3) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{precision}f}"
    except (TypeError, ValueError):
        return str(value)


def format_percent(value: Any, precision: int = 1) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value) * 100.0:.{precision}f}%"
    except (TypeError, ValueError):
        return str(value)


def _iou_2d(gt: list[float], pred: list[float]) -> float | None:
    """2D IoU between two xyxy boxes."""
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


def _draw_badge(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    text: str,
    fg: tuple[int, int, int] = (255, 255, 255),
    bg: tuple[int, int, int] = (30, 30, 30),
    font: ImageFont.ImageFont | None = None,
) -> int:
    """Draw a pill badge and return x position after the badge."""
    if font is None:
        font = load_font(11)
    tw = int(draw.textlength(text, font=font)) + 8
    th = 15
    draw.rounded_rectangle((x, y, x + tw, y + th), radius=3, fill=bg)
    draw.text((x + 4, y + 1), text, fill=fg, font=font)
    return x + tw


def denorm_bbox_yxyx_to_xyxy(norm_bbox: list[float], width: int, height: int) -> list[float]:
    ymin, xmin, ymax, xmax = norm_bbox
    return [
        float(xmin) / 1000.0 * width,
        float(ymin) / 1000.0 * height,
        float(xmax) / 1000.0 * width,
        float(ymax) / 1000.0 * height,
    ]


def load_font(size: int) -> ImageFont.ImageFont:
    for name in ("DejaVuSans.ttf", "Arial.ttf", "Helvetica.ttc"):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def fit_to_box(image: Image.Image, max_w: int, max_h: int) -> Image.Image:
    src_w, src_h = image.size
    if src_w <= 0 or src_h <= 0:
        return Image.new("RGB", (max_w, max_h), color=(220, 220, 220))
    scale = min(float(max_w) / float(src_w), float(max_h) / float(src_h))
    out_w = max(1, int(round(src_w * scale)))
    out_h = max(1, int(round(src_h * scale)))
    resized = image.resize((out_w, out_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (max_w, max_h), color=(245, 245, 245))
    canvas.paste(resized, ((max_w - out_w) // 2, (max_h - out_h) // 2))
    return canvas


def draw_card(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    width: int,
    height: int,
    fill: tuple[int, int, int] = PANEL,
    outline: tuple[int, int, int] = PANEL_BORDER,
    radius: int = 10,
) -> None:
    draw.rounded_rectangle(
        (x, y, x + width, y + height),
        radius=radius,
        fill=fill,
        outline=outline,
        width=1,
    )


def draw_dashed_segment(
    draw: ImageDraw.ImageDraw,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    color: tuple[int, int, int],
    width: int = 2,
    dash: int = 8,
    gap: int = 5,
) -> None:
    length = math.hypot(x1 - x0, y1 - y0)
    if length < 1:
        return
    dx = (x1 - x0) / length
    dy = (y1 - y0) / length
    pos = 0.0
    while pos < length:
        end_pos = min(pos + dash, length)
        draw.line(
            [
                (x0 + dx * pos, y0 + dy * pos),
                (x0 + dx * end_pos, y0 + dy * end_pos),
            ],
            fill=color,
            width=width,
        )
        pos = end_pos + gap


def draw_dashed_rect(
    draw: ImageDraw.ImageDraw,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    color: tuple[int, int, int],
    width: int = 2,
    dash: int = 10,
    gap: int = 5,
) -> None:
    for ax, ay, bx, by in [
        (x0, y0, x1, y0),
        (x1, y0, x1, y1),
        (x1, y1, x0, y1),
        (x0, y1, x0, y0),
    ]:
        draw_dashed_segment(draw, ax, ay, bx, by, color, width=width, dash=dash, gap=gap)


def draw_cuboid_layered(
    draw: ImageDraw.ImageDraw,
    corners_norm: list[list[float]],
    img_w: int,
    img_h: int,
    front_color: tuple[int, int, int] = CUBE_FRONT,
    back_color: tuple[int, int, int] = CUBE_BACK,
    front_w: int = 3,
    back_w: int = 1,
) -> None:
    points = [
        (float(corner[1]) * img_w / 1000.0, float(corner[0]) * img_h / 1000.0)
        for corner in corners_norm
    ]
    back_edges = [(4, 5), (5, 6), (6, 7), (7, 4)]
    conn_edges = [(0, 4), (1, 5), (2, 6), (3, 7)]
    front_edges = [(0, 1), (1, 2), (2, 3), (3, 0)]

    for i, j in back_edges:
        draw_dashed_segment(draw, *points[i], *points[j], back_color, width=back_w, dash=6, gap=4)
    for i, j in conn_edges:
        draw.line([points[i], points[j]], fill=back_color, width=back_w)
    for i, j in front_edges:
        draw.line([points[i], points[j]], fill=front_color, width=front_w)


def draw_2d_overlay(
    image: Image.Image,
    gt_bbox: list[float] | None,
    pred_bbox: list[float] | None,
    label: str | None = None,
) -> Image.Image:
    """Draw only 2-D GT (dashed green) and predicted (solid red) bounding boxes."""
    out = image.copy()
    draw = ImageDraw.Draw(out)

    if gt_bbox is not None:
        draw_dashed_rect(draw, gt_bbox[0], gt_bbox[1], gt_bbox[2], gt_bbox[3], GT_COLOR, width=3)

    if pred_bbox is not None:
        # White underlay makes the predicted box visible on bright backgrounds.
        draw.rectangle((pred_bbox[0], pred_bbox[1], pred_bbox[2], pred_bbox[3]), outline=(255, 255, 255), width=5)
        draw.rectangle((pred_bbox[0], pred_bbox[1], pred_bbox[2], pred_bbox[3]), outline=PRED_COLOR, width=3)

    if label is not None and pred_bbox is not None:
        lbl_font = load_font(12)
        lbl_w = int(draw.textlength(label, font=lbl_font)) + 8
        lbl_h = 16
        lbl_x = max(0, int(pred_bbox[0]))
        lbl_y = max(0, int(pred_bbox[1]) - lbl_h - 2)
        draw.rounded_rectangle(
            (lbl_x, lbl_y, lbl_x + lbl_w, lbl_y + lbl_h),
            radius=3,
            fill=(255, 230, 60),
            outline=(40, 40, 40),
        )
        draw.text((lbl_x + 4, lbl_y + 1), label, fill=(20, 20, 20), font=lbl_font)

    return out


def _corners_in_view(corners: list[list[float]]) -> bool:
    """Return True if at least one corner projects within [0, 1000] in both axes."""
    return any(0.0 <= c[0] <= 1000.0 and 0.0 <= c[1] <= 1000.0 for c in corners)


def draw_3d_gt_pred_overlay_preview(
    image: Image.Image,
    gt_corners_norm: list[list[float]] | None,
    pred_corners_norm: list[list[float]] | None,
) -> Image.Image:
    out = image.copy()
    draw = ImageDraw.Draw(out)
    img_w, img_h = out.size
    lbl_font = load_font(12)

    if gt_corners_norm is not None and img_w > 0 and img_h > 0:
        draw_cuboid_layered(
            draw,
            gt_corners_norm,
            img_w,
            img_h,
            front_color=CUBE_GT_FRONT,
            back_color=CUBE_GT_BACK,
            front_w=3,
            back_w=2,
        )

    if pred_corners_norm is not None and img_w > 0 and img_h > 0:
        if _corners_in_view(pred_corners_norm):
            draw_cuboid_layered(
                draw,
                pred_corners_norm,
                img_w,
                img_h,
                front_color=CUBE_FRONT,
                back_color=CUBE_BACK,
                front_w=3,
                back_w=2,
            )
        else:
            # All pred corners project outside the image — likely a bad depth estimate.
            msg = "pred: off-screen"
            msg_w = int(draw.textlength(msg, font=lbl_font))
            draw.text(((img_w - msg_w) // 2, img_h - 20), msg, fill=PRED_COLOR, font=lbl_font)

    if gt_corners_norm is None and pred_corners_norm is None:
        na = "n/a"
        na_w = int(draw.textlength(na, font=lbl_font))
        draw.text(((img_w - na_w) // 2, max(0, img_h // 2 - 6)), na, fill=MUTED, font=lbl_font)

    return out


def _row_pairs(rows: Any) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    if not isinstance(rows, list):
        return out
    for entry in rows:
        if isinstance(entry, tuple) and len(entry) == 2:
            out.append((str(entry[0]), str(entry[1])))
            continue
        if isinstance(entry, list) and len(entry) == 2:
            out.append((str(entry[0]), str(entry[1])))
            continue
        if isinstance(entry, dict):
            label = entry.get("label")
            value = entry.get("value")
            if label is None:
                continue
            out.append((str(label), str(value if value is not None else "n/a")))
    return out


def _wrap_text_lines(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    max_width: int,
    max_lines: int,
) -> list[str]:
    clean = " ".join(str(text).split())
    if not clean:
        return ["n/a"]

    words = clean.split(" ")
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if draw.textlength(candidate, font=font) <= max_width:
            current = candidate
            continue
        if current:
            lines.append(current)
        current = word
        if len(lines) >= max_lines:
            break

    if len(lines) < max_lines and current:
        lines.append(current)

    if len(lines) > max_lines:
        lines = lines[:max_lines]
    if len(lines) == max_lines and " ".join(lines) != clean:
        last = lines[-1]
        while last and draw.textlength(f"{last}...", font=font) > max_width:
            last = last[:-1]
        lines[-1] = f"{last}..." if last else "..."

    return lines


def _draw_rows(
    draw: ImageDraw.ImageDraw,
    rows: list[tuple[str, str]],
    x: int,
    y: int,
    width: int,
    row_h: int,
    label_font: ImageFont.ImageFont,
    value_font: ImageFont.ImageFont,
) -> int:
    for idx, (label, value) in enumerate(rows):
        fill = (248, 250, 252) if idx % 2 == 0 else (241, 245, 249)
        draw.rectangle((x, y, x + width, y + row_h - 1), fill=fill)
        draw.text((x + 6, y + 2), label, fill=MUTED, font=label_font)
        val_text = str(value)
        val_w = int(draw.textlength(val_text, font=value_font))
        draw.text((x + width - val_w - 6, y + 2), val_text, fill=TEXT, font=value_font)
        y += row_h
    return y


def draw_header(
    draw: ImageDraw.ImageDraw,
    canvas_w: int,
    image_id: int | None,
    model_name: str,
    title_font: ImageFont.ImageFont,
    body_font: ImageFont.ImageFont,
) -> None:
    draw_card(
        draw,
        MARGIN,
        MARGIN,
        canvas_w - 2 * MARGIN,
        HEADER_H,
        fill=HEADER_BG,
        outline=HEADER_BG,
        radius=12,
    )
    title = f"Image {int(image_id):06d}" if image_id is not None else "Image report"
    draw.text((MARGIN + 16, MARGIN + 14), title, fill=HEADER_TEXT, font=title_font)
    chip = f"  {model_name}  "
    chip_w = int(draw.textlength(chip, font=body_font)) + 6
    chip_h = 28
    chip_x = canvas_w - MARGIN - chip_w - 12
    chip_y = MARGIN + (HEADER_H - chip_h) // 2
    draw.rounded_rectangle(
        (chip_x, chip_y, chip_x + chip_w, chip_y + chip_h),
        radius=10,
        fill=(59, 130, 246),
        outline=(147, 197, 253),
    )
    draw.text((chip_x + 4, chip_y + 6), chip, fill=HEADER_TEXT, font=body_font)


def draw_legend_footer(
    draw: ImageDraw.ImageDraw,
    canvas_w: int,
    y: int,
    font: ImageFont.ImageFont,
) -> None:
    draw_card(draw, MARGIN, y, canvas_w - 2 * MARGIN, FOOTER_H, fill=PANEL, outline=PANEL_BORDER, radius=8)
    lx = MARGIN + 12
    ly = y + 12
    draw.text((lx, ly), "Legend:", fill=TEXT, font=font)
    lx += int(draw.textlength("Legend:", font=font)) + 12

    items: list[tuple[tuple[int, int, int], str]] = [
        (GT_COLOR, "GT box"),
        (PRED_COLOR, "Pred box"),
        (CUBE_GT_FRONT, "GT 3D"),
        (CUBE_FRONT, "Pred 3D"),
    ]
    for color, label in items:
        draw.rectangle((lx, ly + 1, lx + 12, ly + 13), fill=color, outline=(40, 40, 40))
        lx += 16
        draw.text((lx, ly), label, fill=TEXT, font=font)
        lx += int(draw.textlength(label, font=font)) + 14


def render_columns_report(image: Image.Image, payload: dict[str, Any]) -> Image.Image:
    """
    Render N+1 debug columns.

    First column is image summary, followed by one full-image column per instance.
    """
    image_id_raw = payload.get("image_id")
    image_id = int(image_id_raw) if isinstance(image_id_raw, (int, float)) else None
    model_name = str(payload.get("model_name") or "unknown-model")

    overview_rows = _row_pairs(payload.get("overview_rows"))
    overview_title = str(payload.get("overview_title") or "Image summary")

    raw_instances = payload.get("instances")
    instances: list[dict[str, Any]] = []
    if isinstance(raw_instances, list):
        instances = [inst for inst in raw_instances if isinstance(inst, dict)]

    title_font = load_font(24)
    body_font = load_font(15)
    small_font = load_font(13)
    metric_font = load_font(12)
    badge_font = load_font(11)

    # Estimate heights for each card to size a common row.
    summary_h = 38 + SUMMARY_2D_IMG_H + 8 + SUMMARY_3D_IMG_H + 8 + max(1, len(overview_rows)) * ROW_H + 12

    dummy = Image.new("RGB", (10, 10), color=BG)
    probe = ImageDraw.Draw(dummy)
    inst_heights: list[int] = []
    for inst in instances:
        query = str(inst.get("query") or "")
        q_lines = _wrap_text_lines(probe, query, small_font, DET_COL_W - 2 * IMG_PAD - 8, max_lines=4)
        rows = _row_pairs(inst.get("rows"))
        inst_h = (
            38
            + DET_IMG_H
            + 8
            + DET_3D_ROW_H
            + 8
            + (1 + len(q_lines)) * ROW_H
            + 8
            + max(1, len(rows)) * ROW_H
            + 12
        )
        inst_heights.append(inst_h)

    body_h = max([summary_h] + inst_heights) if inst_heights else summary_h

    n_cols = 1 + len(instances)
    canvas_w = 2 * MARGIN + SUMMARY_COL_W
    if n_cols > 1:
        canvas_w += len(instances) * (DET_COL_W + GAP)
    canvas_h = MARGIN + HEADER_H + GAP + body_h + GAP + FOOTER_H + MARGIN

    canvas = Image.new("RGB", (canvas_w, canvas_h), color=BG)
    draw = ImageDraw.Draw(canvas)

    draw_header(
        draw=draw,
        canvas_w=canvas_w,
        image_id=image_id,
        model_name=model_name,
        title_font=title_font,
        body_font=body_font,
    )

    top_y = MARGIN + HEADER_H + GAP

    # Summary card with separate 2D and 3D overlays.
    sx = MARGIN
    draw_card(draw, sx, top_y, SUMMARY_COL_W, body_h, fill=PANEL, outline=PANEL_BORDER, radius=10)
    draw.text((sx + IMG_PAD, top_y + 8), overview_title, fill=ACCENT, font=body_font)

    all_2d_overlay = image.copy()
    all_3d_overlay = image.copy()

    # Accumulate per-image stats for summary badges.
    summary_n_detected = 0
    summary_ious: list[float] = []
    summary_reprojections: list[float] = []
    summary_pose_ok = 0
    summary_pose_total = 0

    for idx, inst in enumerate(instances):
        gt_bbox = float_list(inst.get("gt_bbox_xyxy"), expected_len=4)
        pred_bbox = float_list(inst.get("pred_bbox_xyxy"), expected_len=4)
        pred_corners = corner_list(inst.get("pred_bbox_3d_corners_norm_1000"))
        gt_corners = corner_list(inst.get("gt_bbox_3d_corners_norm_1000"))

        if pred_bbox is not None:
            summary_n_detected += 1
        if gt_bbox is not None and pred_bbox is not None:
            iou = _iou_2d(gt_bbox, pred_bbox)
            if iou is not None:
                summary_ious.append(iou)

        inst_row_dict = {lbl: val for lbl, val in _row_pairs(inst.get("rows"))}
        pose_s = inst_row_dict.get("pose")
        reproj_s = inst_row_dict.get("reproj err")
        if pose_s in ("ok", "failed"):
            summary_pose_total += 1
            if pose_s == "ok":
                summary_pose_ok += 1
        if reproj_s and reproj_s != "n/a":
            try:
                summary_reprojections.append(float(reproj_s))
            except ValueError:
                pass

        all_2d_overlay = draw_2d_overlay(
            image=all_2d_overlay,
            gt_bbox=gt_bbox,
            pred_bbox=pred_bbox,
            label=f"D{idx + 1}",
        )
        all_3d_overlay = draw_3d_gt_pred_overlay_preview(
            image=all_3d_overlay,
            gt_corners_norm=gt_corners,
            pred_corners_norm=pred_corners,
        )

    summary_avg_iou = sum(summary_ious) / len(summary_ious) if summary_ious else None
    summary_avg_reproj = sum(summary_reprojections) / len(summary_reprojections) if summary_reprojections else None

    summary_2d_img = fit_to_box(all_2d_overlay, SUMMARY_COL_W - 2 * IMG_PAD, SUMMARY_2D_IMG_H)
    # Badges on summary 2D thumbnail.
    s2_draw = ImageDraw.Draw(summary_2d_img)
    bx, by = 4, 4
    bx = _draw_badge(s2_draw, bx, by, f"det {summary_n_detected}/{len(instances)}", fg=(255, 255, 255), bg=(30, 30, 30), font=badge_font) + 4
    if summary_avg_iou is not None:
        iou_bg = (34, 197, 94) if summary_avg_iou >= 0.5 else (251, 146, 60) if summary_avg_iou >= 0.25 else (239, 68, 68)
        _draw_badge(s2_draw, bx, by, f"avg IoU {summary_avg_iou:.2f}", fg=(255, 255, 255), bg=iou_bg, font=badge_font)

    summary_3d_img = fit_to_box(all_3d_overlay, SUMMARY_COL_W - 2 * IMG_PAD, SUMMARY_3D_IMG_H)
    # Badges on summary 3D thumbnail.
    s3_draw = ImageDraw.Draw(summary_3d_img)
    bx, by = 4, 4
    if summary_pose_total > 0:
        if summary_pose_ok == summary_pose_total:
            pose_bg: tuple[int, int, int] = (34, 197, 94)
        elif summary_pose_ok == 0:
            pose_bg = (239, 68, 68)
        else:
            pose_bg = (251, 146, 60)
        bx = _draw_badge(s3_draw, bx, by, f"pose {summary_pose_ok}/{summary_pose_total}", fg=(255, 255, 255), bg=pose_bg, font=badge_font) + 4
    if summary_avg_reproj is not None:
        reproj_bg = (34, 197, 94) if summary_avg_reproj < 10 else (251, 146, 60) if summary_avg_reproj < 30 else (239, 68, 68)
        _draw_badge(s3_draw, bx, by, f"avg reproj {summary_avg_reproj:.1f}px", fg=(255, 255, 255), bg=reproj_bg, font=badge_font)

    summary_2d_y = top_y + 30
    summary_3d_y = summary_2d_y + SUMMARY_2D_IMG_H + 8
    canvas.paste(summary_2d_img, (sx + IMG_PAD, summary_2d_y))
    canvas.paste(summary_3d_img, (sx + IMG_PAD, summary_3d_y))

    sy = summary_3d_y + SUMMARY_3D_IMG_H + 8
    rows_to_draw = overview_rows if overview_rows else [("info", "no summary rows")]
    _draw_rows(
        draw=draw,
        rows=rows_to_draw,
        x=sx + IMG_PAD,
        y=sy,
        width=SUMMARY_COL_W - 2 * IMG_PAD,
        row_h=ROW_H,
        label_font=metric_font,
        value_font=metric_font,
    )

    # Instance cards.
    x = sx + SUMMARY_COL_W + GAP
    for idx, inst in enumerate(instances):
        draw_card(draw, x, top_y, DET_COL_W, body_h, fill=PANEL, outline=PANEL_BORDER, radius=10)
        title = str(inst.get("title") or f"Detection {idx + 1}")
        draw.text((x + IMG_PAD, top_y + 8), title, fill=ACCENT, font=body_font)

        gt_bbox = float_list(inst.get("gt_bbox_xyxy"), expected_len=4)
        pred_bbox = float_list(inst.get("pred_bbox_xyxy"), expected_len=4)
        pred_corners = corner_list(inst.get("pred_bbox_3d_corners_norm_1000"))
        gt_corners = corner_list(inst.get("gt_bbox_3d_corners_norm_1000"))

        # Extract per-instance metrics from rows for badge overlays.
        inst_row_dict = {lbl: val for lbl, val in _row_pairs(inst.get("rows"))}
        inst_pose_s = inst_row_dict.get("pose")
        inst_reproj_s = inst_row_dict.get("reproj err")
        inst_conf_s = inst_row_dict.get("confidence")

        # Top row: 2D bboxes only (no cuboids).
        det_2d_img = draw_2d_overlay(
            image=image,
            gt_bbox=gt_bbox,
            pred_bbox=pred_bbox,
            label=f"D{idx + 1}",
        )
        det_img = fit_to_box(det_2d_img, DET_COL_W - 2 * IMG_PAD, DET_IMG_H)
        # IoU and confidence badges on 2D thumbnail.
        # Prefer pre-computed iou2d from metrics dict; fall back to inline computation.
        inst_metrics = inst.get("metrics") if isinstance(inst.get("metrics"), dict) else {}
        det_draw = ImageDraw.Draw(det_img)
        bx_d, by_d = 4, det_img.height - 18
        iou_val = inst_metrics.get("iou2d") if inst_metrics else None
        if iou_val is None and gt_bbox is not None and pred_bbox is not None:
            iou_val = _iou_2d(gt_bbox, pred_bbox)
        if iou_val is not None:
            iou_bg = (34, 197, 94) if iou_val >= 0.5 else (251, 146, 60) if iou_val >= 0.25 else (239, 68, 68)
            bx_d = _draw_badge(det_draw, bx_d, by_d, f"IoU {iou_val:.2f}", fg=(255, 255, 255), bg=iou_bg, font=badge_font) + 4
        if inst_conf_s and inst_conf_s != "n/a":
            try:
                _draw_badge(det_draw, bx_d, by_d, f"conf {float(inst_conf_s):.2f}", fg=(255, 255, 255), bg=ACCENT, font=badge_font)
            except ValueError:
                pass
        canvas.paste(det_img, (x + IMG_PAD, top_y + 30))

        three_d_y = top_y + 30 + DET_IMG_H + 8
        draw.text((x + IMG_PAD, three_d_y + 2), "3D GT vs Pred", fill=MUTED, font=small_font)

        mini_y = three_d_y + ROW_H
        mini_h = DET_3D_ROW_H - ROW_H
        mini_w = DET_COL_W - 2 * IMG_PAD

        combined_3d = draw_3d_gt_pred_overlay_preview(
            image=image,
            gt_corners_norm=gt_corners,
            pred_corners_norm=pred_corners,
        )

        mini_img = fit_to_box(combined_3d, mini_w, mini_h)
        # Reproj / pose badge on 3D thumbnail.
        mini_draw = ImageDraw.Draw(mini_img)
        bx_m, by_m = 4, mini_img.height - 18
        if inst_pose_s == "ok" and inst_reproj_s and inst_reproj_s != "n/a":
            try:
                reproj_f = float(inst_reproj_s)
                reproj_bg = (34, 197, 94) if reproj_f < 10 else (251, 146, 60) if reproj_f < 30 else (239, 68, 68)
                _draw_badge(mini_draw, bx_m, by_m, f"reproj {reproj_f:.1f}px", fg=(255, 255, 255), bg=reproj_bg, font=badge_font)
            except ValueError:
                pass
        elif inst_pose_s == "failed":
            _draw_badge(mini_draw, bx_m, by_m, "pose failed", fg=(255, 255, 255), bg=(239, 68, 68), font=badge_font)
        canvas.paste(mini_img, (x + IMG_PAD, mini_y))

        text_y = three_d_y + DET_3D_ROW_H + 8
        draw.text((x + IMG_PAD, text_y + 2), "Query:", fill=MUTED, font=small_font)
        query = str(inst.get("query") or "")
        q_lines = _wrap_text_lines(
            draw=draw,
            text=query,
            font=small_font,
            max_width=DET_COL_W - 2 * IMG_PAD - 8,
            max_lines=4,
        )
        text_y += ROW_H
        for line in q_lines:
            draw.text((x + IMG_PAD, text_y + 2), line, fill=TEXT, font=small_font)
            text_y += ROW_H

        text_y += 6
        rows = _row_pairs(inst.get("rows"))
        rows = rows if rows else [("info", "no rows")]
        _draw_rows(
            draw=draw,
            rows=rows,
            x=x + IMG_PAD,
            y=text_y,
            width=DET_COL_W - 2 * IMG_PAD,
            row_h=ROW_H,
            label_font=metric_font,
            value_font=metric_font,
        )

        x += DET_COL_W + GAP

    footer_y = top_y + body_h + GAP
    draw_legend_footer(draw=draw, canvas_w=canvas_w, y=footer_y, font=small_font)
    return canvas
