"""Shared enums and dataclasses for the drone VLM eval demo."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Optional


class Background(StrEnum):
    SKY = "sky"
    TREES = "trees"
    BUILDING = "building"
    CLUTTERED = "cluttered"
    UNKNOWN = "unknown"


class Lighting(StrEnum):
    NORMAL = "normal"
    DARK = "dark"
    BRIGHT = "bright"
    BACKLIT = "backlit"
    UNKNOWN = "unknown"


class DroneVisibility(StrEnum):
    LARGE = "large"
    MEDIUM = "medium"
    SMALL = "small"


class BlurBucket(StrEnum):
    SHARP = "sharp"
    MILD = "mild"
    BLURRY = "blurry"


class CameraAngle(StrEnum):
    TOP_DOWN = "top_down"
    HIGH_ANGLE = "high_angle"
    EYE_LEVEL = "eye_level"
    LOW_ANGLE = "low_angle"
    WORMS_EYE = "worms_eye"
    UNKNOWN = "unknown"


class DepthRange(StrEnum):
    CLOSE_UP = "close_up"
    MID_RANGE = "mid_range"
    LANDSCAPE = "landscape"
    UNKNOWN = "unknown"


class CurationStatus(StrEnum):
    KEEP = "keep"
    NEEDS_REVIEW = "needs_review"
    REJECT = "reject"


@dataclass(frozen=True)
class BoundingBox:
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    @property
    def width(self) -> float:
        return self.xmax - self.xmin

    @property
    def height(self) -> float:
        return self.ymax - self.ymin

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    def to_list(self) -> list[float]:
        return [self.xmin, self.ymin, self.xmax, self.ymax]


@dataclass
class CurationAnnotation:
    background: Background = Background.UNKNOWN
    lighting: Lighting = Lighting.UNKNOWN
    drone_visibility: Optional[DroneVisibility] = None
    blur_bucket: BlurBucket = BlurBucket.SHARP
    possible_confusers: list[str] = field(default_factory=list)
    camera_angle: CameraAngle = CameraAngle.UNKNOWN
    depth_range: DepthRange = DepthRange.UNKNOWN
    annotator: str = "unset"
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "background": self.background.value,
            "lighting": self.lighting.value,
            "drone_visibility": self.drone_visibility.value if self.drone_visibility is not None else None,
            "blur_bucket": self.blur_bucket.value,
            "possible_confusers": self.possible_confusers,
            "camera_angle": self.camera_angle.value,
            "depth_range": self.depth_range.value,
            "annotator": self.annotator,
            "raw": self.raw,
        }


@dataclass
class DroneRecord:
    image_id: str
    image_path: Path
    width: int
    height: int
    drone_present: bool
    bbox: list[BoundingBox]
    source_split: str
    xml_path: Path | None = None
    augmentation_type: str = "real"
    source_image_id: str | None = None
    is_synthetic: bool = False
    curation: CurationAnnotation | None = None

    def to_dict(self) -> dict[str, Any]:
        curation = self.curation.to_dict() if self.curation else {}
        return {
            "image_id": self.image_id,
            "image_path": str(self.image_path),
            "width": self.width,
            "height": self.height,
            "drone_present": self.drone_present,
            "bbox": [box.to_list() for box in self.bbox],
            "box_count": len(self.bbox),
            "source_split": self.source_split,
            "xml_path": str(self.xml_path) if self.xml_path else None,
            "augmentation_type": self.augmentation_type,
            "source_image_id": self.source_image_id,
            "is_synthetic": self.is_synthetic,
            **curation,
        }
