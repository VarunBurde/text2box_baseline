from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

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


def write_gts_like_parquet(rows: list[dict[str, Any]], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataframe = pd.DataFrame(rows)
    for column in GTS_COLUMNS:
        if column not in dataframe.columns:
            dataframe[column] = None

    dataframe = dataframe[GTS_COLUMNS]
    dataframe.to_parquet(output_path, index=False, compression="zstd")


def write_manifest_jsonl(records: list[dict[str, Any]], manifest_path: str | Path) -> None:
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    with manifest_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def init_manifest_jsonl(manifest_path: str | Path) -> None:
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("", encoding="utf-8")


def append_manifest_record(record: dict[str, Any], manifest_path: str | Path) -> None:
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True) + "\n")
