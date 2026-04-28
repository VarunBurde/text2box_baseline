from __future__ import annotations

import argparse
import io
import json
import re
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image


GTS_COLUMNS = [
    "annotation_id",
    "query_id",
    "obj_id",
    "instance_id",
    "bbox_2d",
    "bbox_3d_R",
    "bbox_3d_t",
    "bbox_3d_size",
    "R_cam_from_model",
    "t_cam_from_model",
    "visib_fract",
]

OBJ_ID_PATTERN = re.compile(r"(\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert handal_sample.json (in unstructured BOP tree) into the "
            "BOP-Text2Box parquet/tar format used by this repository."
        )
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("unstructed_dataset/handal"),
        help="Root containing handal_sample.json and val/<scene>/... folders.",
    )
    parser.add_argument(
        "--sample-json",
        type=Path,
        default=None,
        help="Path to sample JSON. Defaults to <source-root>/handal_sample.json.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("Datasets/handal"),
        help="Destination BOP-Text2Box dataset root.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        help="Split name to write (default: val).",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=1000,
        help="Number of images per tar shard.",
    )
    return parser.parse_args()


def parse_obj_id(global_object_id: str) -> int:
    match = OBJ_ID_PATTERN.search(global_object_id)
    if not match:
        raise ValueError(f"Could not parse obj_id from global_object_id={global_object_id!r}")
    return int(match.group(1))


def xyxy_to_xywh(bbox_xyxy: list[float]) -> list[float]:
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]


def iou_xywh(a: list[float], b: list[float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b

    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0

    union = aw * ah + bw * bh - inter
    return float(inter / union) if union > 0.0 else 0.0


def flatten_3x3(matrix: Any) -> list[float]:
    if not isinstance(matrix, list) or len(matrix) != 3:
        raise ValueError(f"Expected 3x3 list matrix, got: {matrix!r}")
    out: list[float] = []
    for row in matrix:
        if not isinstance(row, list) or len(row) != 3:
            raise ValueError(f"Expected 3x3 list matrix, got: {matrix!r}")
        out.extend(float(v) for v in row)
    return out


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_tar_shards(
    output_root: Path,
    split: str,
    image_records: list[dict[str, Any]],
) -> None:
    images_dir = output_root / f"images_{split}"
    ensure_dir(images_dir)

    by_shard: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in image_records:
        by_shard[str(record["shard"])].append(record)

    for shard_name in sorted(by_shard.keys()):
        shard_path = images_dir / shard_name
        with tarfile.open(shard_path, mode="w") as tar_handle:
            for record in sorted(by_shard[shard_name], key=lambda r: int(r["image_id"])):
                image_id = int(record["image_id"])
                image_path = Path(str(record["_src_image_path"]))
                data = image_path.read_bytes()

                member_name = f"{image_id:08d}.jpg"
                info = tarfile.TarInfo(name=member_name)
                info.size = len(data)
                info.mtime = 0
                tar_handle.addfile(info, io.BytesIO(data))


def convert_dataset(
    source_root: Path,
    sample_json_path: Path,
    output_root: Path,
    split: str,
    shard_size: int,
) -> None:
    if shard_size <= 0:
        raise ValueError("shard_size must be > 0")

    entries: list[dict[str, Any]] = json.loads(sample_json_path.read_text(encoding="utf-8"))
    entries = [entry for entry in entries if str(entry.get("split", "")).strip() == split]
    if not entries:
        raise ValueError(f"No entries found for split={split!r} in {sample_json_path}")

    frame_to_image: dict[tuple[str, int], dict[str, Any]] = {}
    image_records: list[dict[str, Any]] = []

    for entry in entries:
        scene_id = str(entry["scene_id"])
        frame_id = int(entry["frame_id"])
        key = (scene_id, frame_id)

        if key in frame_to_image:
            continue

        rgb_path = source_root / split / scene_id / "rgb" / f"{frame_id:06d}.jpg"
        if not rgb_path.exists():
            raise FileNotFoundError(f"Missing RGB image: {rgb_path}")

        with Image.open(rgb_path) as image_obj:
            width, height = image_obj.size

        image_id = len(image_records)
        shard_idx = image_id // shard_size
        shard_name = f"shard-{shard_idx:06d}.tar"

        intrinsics_raw = entry.get("cam_intrinsics")
        if isinstance(intrinsics_raw, dict):
            keys = ("fx", "fy", "cx", "cy")
            if not all(key in intrinsics_raw for key in keys):
                raise ValueError(
                    f"Entry scene={scene_id} frame={frame_id} has invalid cam_intrinsics: {intrinsics_raw!r}"
                )
            intrinsics = [float(intrinsics_raw[key]) for key in keys]
        elif isinstance(intrinsics_raw, list) and len(intrinsics_raw) == 4:
            intrinsics = [float(v) for v in intrinsics_raw]
        else:
            raise ValueError(
                f"Entry scene={scene_id} frame={frame_id} has invalid cam_intrinsics: {intrinsics_raw!r}"
            )

        record = {
            "image_id": image_id,
            "shard": shard_name,
            "width": int(width),
            "height": int(height),
            "intrinsics": intrinsics,
            "_src_image_path": str(rgb_path),
            "scene_id": scene_id,
            "frame_id": frame_id,
        }
        frame_to_image[key] = record
        image_records.append(record)

    scene_gt_cache: dict[str, dict[str, Any]] = {}
    scene_info_cache: dict[str, dict[str, Any]] = {}

    def load_scene(scene_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
        if scene_id not in scene_gt_cache:
            scene_dir = source_root / split / scene_id
            gt_path = scene_dir / "scene_gt.json"
            info_path = scene_dir / "scene_gt_info.json"
            if not gt_path.exists() or not info_path.exists():
                raise FileNotFoundError(
                    f"Missing scene GT files in {scene_dir} (scene_gt.json / scene_gt_info.json)"
                )
            scene_gt_cache[scene_id] = json.loads(gt_path.read_text(encoding="utf-8"))
            scene_info_cache[scene_id] = json.loads(info_path.read_text(encoding="utf-8"))
        return scene_gt_cache[scene_id], scene_info_cache[scene_id]

    query_rows: list[dict[str, Any]] = []
    gt_rows: list[dict[str, Any]] = []

    object_names: dict[int, set[str]] = defaultdict(set)
    object_sizes: dict[int, list[list[float]]] = defaultdict(list)

    annotation_id = 0
    query_id = 0

    low_iou_matches: list[tuple[str, int, str, float]] = []

    for entry in entries:
        scene_id = str(entry["scene_id"])
        frame_id = int(entry["frame_id"])
        frame_key = str(frame_id)
        image_meta = frame_to_image[(scene_id, frame_id)]
        image_id = int(image_meta["image_id"])

        scene_gt, scene_info = load_scene(scene_id)
        if frame_key not in scene_gt or frame_key not in scene_info:
            raise KeyError(f"Frame {frame_key} missing in scene GT for scene={scene_id}")

        gt_frame: list[dict[str, Any]] = scene_gt[frame_key]
        info_frame: list[dict[str, Any]] = scene_info[frame_key]

        if len(gt_frame) != len(info_frame):
            raise ValueError(
                f"scene={scene_id} frame={frame_key}: scene_gt / scene_gt_info length mismatch"
            )

        for spec in entry.get("target_specs", []):
            queries = spec.get("queries")
            targets = spec.get("target_objects")

            if not isinstance(queries, list) or not isinstance(targets, list):
                continue

            query_ids_for_spec: list[int] = []
            for query_item in queries:
                if isinstance(query_item, str):
                    query_text = query_item
                elif isinstance(query_item, dict) and isinstance(query_item.get("query"), str):
                    query_text = query_item["query"]
                else:
                    continue

                query_rows.append(
                    {
                        "query_id": query_id,
                        "image_id": image_id,
                        "query": query_text.strip(),
                    }
                )
                query_ids_for_spec.append(query_id)
                query_id += 1

            for target in targets:
                if not isinstance(target, dict):
                    continue

                global_object_id = str(target["global_object_id"])
                obj_id = parse_obj_id(global_object_id)
                object_names[obj_id].add(global_object_id)

                bbox_xyxy = [float(v) for v in target["bbox_2d"]]
                bbox_xywh = xyxy_to_xywh(bbox_xyxy)

                candidates: list[tuple[float, int]] = []
                for idx, (gt_item, info_item) in enumerate(zip(gt_frame, info_frame)):
                    if int(gt_item["obj_id"]) != obj_id:
                        continue
                    iou = iou_xywh(bbox_xywh, [float(v) for v in info_item["bbox_obj"]])
                    candidates.append((iou, idx))

                if not candidates:
                    raise ValueError(
                        f"No scene GT candidate for obj_id={obj_id} in scene={scene_id} frame={frame_id}"
                    )

                candidates.sort(key=lambda pair: pair[0], reverse=True)
                best_iou, best_idx = candidates[0]
                if best_iou < 0.9:
                    low_iou_matches.append((scene_id, frame_id, global_object_id, best_iou))

                gt_item = gt_frame[best_idx]
                info_item = info_frame[best_idx]

                bbox_3d_r = flatten_3x3(target["bbox_3d_R"])
                bbox_3d_t = [float(v) for v in target["bbox_3d_t"]]
                bbox_3d_size = [float(v) for v in target["bbox_3d_size"]]
                object_sizes[obj_id].append(bbox_3d_size)

                r_cam_from_model = [float(v) for v in gt_item["cam_R_m2c"]]
                t_cam_from_model = [float(v) for v in gt_item["cam_t_m2c"]]
                visib_fract = target.get("visib_fract", info_item.get("visib_fract"))
                visib_value = float(visib_fract) if visib_fract is not None else None

                for qid in query_ids_for_spec:
                    gt_rows.append(
                        {
                            "annotation_id": annotation_id,
                            "query_id": qid,
                            "obj_id": obj_id,
                            "instance_id": int(best_idx),
                            "bbox_2d": bbox_xyxy,
                            "bbox_3d_R": bbox_3d_r,
                            "bbox_3d_t": bbox_3d_t,
                            "bbox_3d_size": bbox_3d_size,
                            "R_cam_from_model": r_cam_from_model,
                            "t_cam_from_model": t_cam_from_model,
                            "visib_fract": visib_value,
                        }
                    )
                    annotation_id += 1

    object_rows: list[dict[str, Any]] = []
    identity_r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    zero_t = [0.0, 0.0, 0.0]

    for obj_id in sorted(object_names.keys()):
        names = sorted(object_names[obj_id])
        size_samples = object_sizes[obj_id]
        if not size_samples:
            raise ValueError(f"No size samples found for obj_id={obj_id}")

        # Size is stable in handal_sample, but we average to be robust.
        sx = sum(sample[0] for sample in size_samples) / len(size_samples)
        sy = sum(sample[1] for sample in size_samples) / len(size_samples)
        sz = sum(sample[2] for sample in size_samples) / len(size_samples)

        object_rows.append(
            {
                "obj_id": int(obj_id),
                "bop_dataset": source_root.name,
                "bop_obj_id": int(obj_id),
                "name": names[0],
                "symmetries_discrete": None,
                "symmetries_continuous": None,
                "bbox_3d_model_R": identity_r,
                "bbox_3d_model_t": zero_t,
                "bbox_3d_model_size": [float(sx), float(sy), float(sz)],
            }
        )

    ensure_dir(output_root)

    images_table = [
        {
            "image_id": rec["image_id"],
            "shard": rec["shard"],
            "width": rec["width"],
            "height": rec["height"],
            "intrinsics": rec["intrinsics"],
        }
        for rec in image_records
    ]

    pd.DataFrame(object_rows).to_parquet(
        output_root / "objects_info.parquet", index=False, compression="zstd"
    )
    pd.DataFrame(images_table).to_parquet(
        output_root / f"images_info_{split}.parquet", index=False, compression="zstd"
    )
    pd.DataFrame(query_rows).to_parquet(
        output_root / f"queries_{split}.parquet", index=False, compression="zstd"
    )

    gt_df = pd.DataFrame(gt_rows)
    for column in GTS_COLUMNS:
        if column not in gt_df.columns:
            gt_df[column] = None
    gt_df = gt_df[GTS_COLUMNS]
    gt_df.to_parquet(output_root / f"gts_{split}.parquet", index=False, compression="zstd")

    write_tar_shards(output_root=output_root, split=split, image_records=image_records)

    print(f"Converted entries: {len(entries)}")
    print(f"Unique images: {len(image_records)}")
    print(f"Queries: {len(query_rows)}")
    print(f"GT rows: {len(gt_rows)}")
    print(f"Objects: {len(object_rows)}")
    if low_iou_matches:
        print(f"Low-IoU matches (<0.9): {len(low_iou_matches)}")
    else:
        print("All target-to-instance matches IoU >= 0.9")
    print(f"Output root: {output_root}")


def main() -> None:
    args = parse_args()

    source_root = args.source_root.resolve()
    sample_json_path = args.sample_json.resolve() if args.sample_json else (source_root / "handal_sample.json").resolve()
    output_root = args.output_root.resolve()

    convert_dataset(
        source_root=source_root,
        sample_json_path=sample_json_path,
        output_root=output_root,
        split=str(args.split),
        shard_size=int(args.shard_size),
    )


if __name__ == "__main__":
    main()
