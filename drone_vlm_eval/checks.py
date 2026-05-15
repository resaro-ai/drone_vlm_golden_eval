"""Dataset quality utilities for the drone curation pipeline."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd

from drone_vlm_eval.image_io import blur_score, brightness_score


def _has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    if str(value).lower() in {"", "none", "nan"}:
        return False
    return True


def compute_image_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with per-row blur_score, brightness, and bbox_ratio.

    bbox_ratio is the largest bounding-box area relative to image area.
    It is only computed for drone-present rows with valid boxes; NaN otherwise.
    """
    rows = []
    for _, row in df.iterrows():
        image_path = Path(str(row["image_path"])) if _has_value(row.get("image_path")) else None

        blur = None
        brightness = None
        if image_path and image_path.exists():
            blur = blur_score(image_path)
            brightness = brightness_score(image_path)

        bbox_ratio = None
        if bool(row.get("drone_present", False)):
            w = float(row.get("width", 0) or 0)
            h = float(row.get("height", 0) or 0)
            area = w * h
            if area > 0:
                max_frac = 0.0
                for box in row.get("bbox") or []:
                    try:
                        xmin, ymin, xmax, ymax = [float(v) for v in box]
                        frac = max(0.0, xmax - xmin) * max(0.0, ymax - ymin) / area
                        max_frac = max(max_frac, frac)
                    except (TypeError, ValueError):
                        pass
                bbox_ratio = max_frac if max_frac > 0 else None

        rows.append({
            "image_id": str(row["image_id"]),
            "blur_score": blur,
            "brightness": brightness,
            "bbox_ratio": bbox_ratio,
        })

    return pd.DataFrame(rows)


def assign_curation_status(row: pd.Series) -> str:
    confusers = row.get("possible_confusers") or []
    if confusers and confusers != ["none"]:
        return "needs_review"
    return "keep"
