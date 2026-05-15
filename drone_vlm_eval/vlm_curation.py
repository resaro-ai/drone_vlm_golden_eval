"""AI-assisted curation adapters."""

from __future__ import annotations

import json
import os
import re
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from drone_vlm_eval.image_io import image_to_base64
from drone_vlm_eval.schemas import (
    Background,
    BlurBucket,
    CameraAngle,
    CurationAnnotation,
    DepthRange,
    DroneVisibility,
    Lighting,
)


class CurationAnnotator(ABC):
    name: str

    @abstractmethod
    def annotate(self, row: dict[str, Any]) -> CurationAnnotation:
        """Return structured curation metadata for one dataset row."""


class SceneCurationSchema(BaseModel):
    background: Literal["sky", "trees", "building", "cluttered"] = Field(
        description="Dominant visual background."
    )
    lighting: Literal["normal", "dark", "bright", "backlit"] = Field(
        description="Dominant lighting condition."
    )
    blur_bucket: Literal["sharp", "mild", "blurry"] = Field(
        description="Overall image blur level."
    )
    possible_confusers: list[Literal["bird", "aircraft", "helicopter", "none"]] = Field(
        description="Objects that could be confused with drones; use ['none'] if no confuser is visible."
    )
    camera_angle: Literal["top_down", "high_angle", "eye_level", "low_angle", "worms_eye"] = Field(
        description=(
            "Camera perspective relative to the scene: "
            "top_down=looking straight down, high_angle=above subject looking down, "
            "eye_level=roughly at horizon, low_angle=below subject level, worms_eye=looking up."
        )
    )
    depth_range: Literal["close_up", "mid_range", "landscape"] = Field(
        description=(
            "Approximate depth/distance of the primary subject: "
            "close_up=subject fills >25% of frame, mid_range=5–25%, landscape=<5% or a wide outdoor scene."
        )
    )
    caption: str = Field(description="One concise sentence describing the scene and visual evidence.")


def _coerce_enum(enum_cls: type[Any], value: Any, default: Any) -> Any:
    try:
        return enum_cls(str(value))
    except (TypeError, ValueError):
        return default


def _coerce_confusers(value: Any) -> list[str]:
    if not isinstance(value, list):
        return ["none"]
    allowed = {"bird", "aircraft", "helicopter", "none"}
    aliases = {"airplane": "aircraft", "plane": "aircraft", "kite": "bird"}
    confusers: set[str] = set()
    for item in value:
        key = str(item).strip().lower().rstrip(".,;:!?")
        key = aliases.get(key, key)
        if key in allowed:
            confusers.add(key)
    if not confusers or confusers == {"none"}:
        return ["none"]
    confusers.discard("none")
    return sorted(confusers)


def _json_from_text(text: str) -> dict[str, Any]:
    """Parse a JSON object from a model response.

    Tolerates: fenced code blocks, <think>...</think> reasoning blocks (Qwen3 etc.),
    array-wrapped objects, and multiple JSON fragments (returns the last complete object).
    """
    raw = text.strip()
    # Strip think blocks; fall back to raw if nothing remains (answer may be inside)
    stripped = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    working = stripped if stripped else raw
    if working.startswith("```"):
        working = re.sub(r"^```(?:json)?\s*", "", working, flags=re.IGNORECASE)
        working = re.sub(r"\s*```$", "", working).strip()

    # Fast path: direct parse
    try:
        parsed = json.loads(working)
        if isinstance(parsed, dict):
            return parsed
        # Unwrap single-element array
        if isinstance(parsed, list) and len(parsed) == 1 and isinstance(parsed[0], dict):
            return parsed[0]
    except json.JSONDecodeError:
        pass

    # Slow path: scan for the last complete JSON object in the text.
    # Using raw_decode so we don't require the whole string to be valid JSON,
    # and taking the *last* match because reasoning models often emit JSON
    # fragments while thinking before producing the final answer.
    decoder = json.JSONDecoder()
    last_obj: dict[str, Any] | None = None
    i = 0
    while i < len(working):
        idx = working.find("{", i)
        if idx == -1:
            break
        try:
            obj, end = decoder.raw_decode(working, idx)
            if isinstance(obj, dict):
                last_obj = obj
            i = end
        except json.JSONDecodeError:
            i = idx + 1

    if last_obj is not None:
        return last_obj
    raise ValueError(f"Curation response contained no JSON object. Raw response:\n{raw!r}")


def _visibility_from_ground_truth(row: dict[str, Any]) -> DroneVisibility | None:
    """Bucket drone size from annotation boxes, not from the VLM."""
    if not bool(row.get("drone_present", False)):
        return None

    width = float(row.get("width", 0) or 0)
    height = float(row.get("height", 0) or 0)
    image_area = width * height
    if image_area <= 0:
        return DroneVisibility.SMALL

    max_frac = 0.0
    for box in row.get("bbox") or []:
        try:
            xmin, ymin, xmax, ymax = [float(v) for v in box]
        except (TypeError, ValueError):
            continue
        box_area = max(0.0, xmax - xmin) * max(0.0, ymax - ymin)
        max_frac = max(max_frac, box_area / image_area)

    if max_frac >= 0.02:
        return DroneVisibility.LARGE
    if max_frac >= 0.005:
        return DroneVisibility.MEDIUM
    return DroneVisibility.SMALL


_CURATION_PROMPT = """\
You are curating drone-detection evaluation images for operating-domain coverage.

Use only the controlled ODD values defined in the response JSON schema.
Do not classify whether a drone is present; the dataset annotations provide that ground truth.
Do not estimate drone size or visibility; those are derived from bounding boxes.
If no drone-like confuser is visible, set possible_confusers to ["none"].
Do not add any fields outside this JSON object.
"""


class GeminiCurationAnnotator(CurationAnnotator):
    """Gemini Flash adapter for non-label ODD metadata."""

    name = "gemini"

    def __init__(
        self,
        model_id: str = "gemini-3.1-flash-lite-preview",
        cache_dir: Path | None = None,
        force_refresh: bool = False,
        api_key: str | None = None,
        requests_per_minute: int = 15,
    ) -> None:
        self.model_id = model_id
        self.cache_dir = cache_dir
        self.force_refresh = force_refresh
        self.api_key = (
            api_key
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
        )
        self._client: Any = None
        self._rpm = requests_per_minute
        self._request_times: list[float] = []

    def _rate_limit(self) -> None:
        """Block until within the configured requests-per-minute budget."""
        if self._rpm <= 0:
            return
        now = time.monotonic()
        self._request_times = [t for t in self._request_times if now - t < 60.0]
        if len(self._request_times) >= self._rpm:
            sleep_for = 60.0 - (now - self._request_times[0])
            if sleep_for > 0:
                time.sleep(sleep_for)
        self._request_times.append(time.monotonic())

    def _ensure_client(self) -> None:
        if self._client is not None:
            return
        if not self.api_key:
            raise RuntimeError(
                "Gemini curation requires GEMINI_API_KEY in your environment or .env file."
            )
        try:
            from google import genai
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Gemini curation requires the `google-genai` package. Run `uv sync` first."
            ) from exc
        self._client = genai.Client(api_key=self.api_key)

    def _cache_path(self, image_id: str) -> Path | None:
        return self.cache_dir / f"{image_id}.json" if self.cache_dir else None

    def _read_cache(self, image_id: str, row: dict[str, Any]) -> CurationAnnotation | None:
        cache_path = self._cache_path(image_id)
        if not cache_path or self.force_refresh or not cache_path.exists():
            return None
        try:
            cached = json.loads(cache_path.read_text())
            if "blur_bucket" not in cached:
                return None
            raw = cached.get("raw", {})
            return CurationAnnotation(
                background=_coerce_enum(Background, cached.get("background"), Background.UNKNOWN),
                lighting=_coerce_enum(Lighting, cached.get("lighting"), Lighting.UNKNOWN),
                drone_visibility=_visibility_from_ground_truth(row),
                blur_bucket=_coerce_enum(BlurBucket, cached.get("blur_bucket"), BlurBucket.SHARP),
                possible_confusers=_coerce_confusers(cached.get("possible_confusers")),
                camera_angle=_coerce_enum(CameraAngle, cached.get("camera_angle"), CameraAngle.UNKNOWN),
                depth_range=_coerce_enum(DepthRange, cached.get("depth_range"), DepthRange.UNKNOWN),
                annotator=self.name,
                raw=raw if isinstance(raw, dict) else {},
            )
        except (json.JSONDecodeError, ValueError, KeyError):
            return None

    def _write_cache(self, image_id: str, annotation: CurationAnnotation) -> None:
        cache_path = self._cache_path(image_id)
        if not cache_path:
            return
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(annotation.to_dict(), default=str))

    def _request_json(self, row: dict[str, Any]) -> dict[str, Any]:
        from PIL import Image

        self._ensure_client()

        image_path = Path(str(row["image_path"]))
        image = Image.open(image_path).convert("RGB")
        dataset_context = {
            "image_id": str(row["image_id"]),
            "ground_truth_drone_present": "yes" if bool(row.get("drone_present", False)) else "no",
            "ground_truth_box_count": int(row.get("box_count", 0) or 0),
        }
        prompt = _CURATION_PROMPT + "\nDataset annotation context:\n" + json.dumps(dataset_context)

        self._rate_limit()
        response = self._client.models.generate_content(
            model=self.model_id,
            contents=[prompt, image],
            config={
                "response_mime_type": "application/json",
                "response_json_schema": SceneCurationSchema.model_json_schema(),
            },
        )
        return _json_from_text(response.text or "")

    def annotate(self, row: dict[str, Any]) -> CurationAnnotation:
        """Run Gemini over one image and return ODD curation metadata."""
        image_id = str(row["image_id"])

        cached = self._read_cache(image_id, row)
        if cached:
            return cached

        raw_response = self._request_json(row)

        annotation = CurationAnnotation(
            background=_coerce_enum(Background, raw_response.get("background"), Background.UNKNOWN),
            lighting=_coerce_enum(Lighting, raw_response.get("lighting"), Lighting.UNKNOWN),
            drone_visibility=_visibility_from_ground_truth(row),
            blur_bucket=_coerce_enum(BlurBucket, raw_response.get("blur_bucket"), BlurBucket.SHARP),
            possible_confusers=_coerce_confusers(raw_response.get("possible_confusers")),
            camera_angle=_coerce_enum(CameraAngle, raw_response.get("camera_angle"), CameraAngle.UNKNOWN),
            depth_range=_coerce_enum(DepthRange, raw_response.get("depth_range"), DepthRange.UNKNOWN),
            annotator=self.name,
            raw={
                "model": self.model_id,
                "caption": str(raw_response.get("caption", "")),
                "gemini_response": raw_response,
                "visibility_source": "ground_truth_bbox",
                "schema_version": "odd_v3",
            },
        )

        self._write_cache(image_id, annotation)
        return annotation


class OpenAICurationAnnotator(CurationAnnotator):
    """OpenAI curation annotator."""

    name = "openai"

    def __init__(
        self,
        model_id: str = "gpt-5.4-nano",
        base_url: str | None = None,
        cache_dir: Path | None = None,
        force_refresh: bool = False,
        api_key: str | None = None,
    ) -> None:
        self.model_id = model_id
        self.base_url = base_url
        self.cache_dir = cache_dir
        self.force_refresh = force_refresh
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._client: Any = None

    def _ensure_client(self) -> None:
        if self._client is not None:
            return
        try:
            from openai import OpenAI
        except ModuleNotFoundError as exc:
            raise RuntimeError("OpenAI curation requires the `openai` package.") from exc
        kwargs: dict[str, Any] = {"api_key": self.api_key}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        self._client = OpenAI(**kwargs)

    def _cache_path(self, image_id: str) -> Path | None:
        return self.cache_dir / f"{image_id}.json" if self.cache_dir else None

    def _read_cache(self, image_id: str, row: dict[str, Any]) -> CurationAnnotation | None:
        cache_path = self._cache_path(image_id)
        if not cache_path or self.force_refresh or not cache_path.exists():
            return None
        try:
            cached = json.loads(cache_path.read_text())
            if "blur_bucket" not in cached:
                return None
            raw = cached.get("raw", {})
            return CurationAnnotation(
                background=_coerce_enum(Background, cached.get("background"), Background.UNKNOWN),
                lighting=_coerce_enum(Lighting, cached.get("lighting"), Lighting.UNKNOWN),
                drone_visibility=_visibility_from_ground_truth(row),
                blur_bucket=_coerce_enum(BlurBucket, cached.get("blur_bucket"), BlurBucket.SHARP),
                possible_confusers=_coerce_confusers(cached.get("possible_confusers")),
                camera_angle=_coerce_enum(CameraAngle, cached.get("camera_angle"), CameraAngle.UNKNOWN),
                depth_range=_coerce_enum(DepthRange, cached.get("depth_range"), DepthRange.UNKNOWN),
                annotator=self.name,
                raw=raw if isinstance(raw, dict) else {},
            )
        except (json.JSONDecodeError, ValueError, KeyError):
            return None

    def _write_cache(self, image_id: str, annotation: CurationAnnotation) -> None:
        cache_path = self._cache_path(image_id)
        if not cache_path:
            return
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(annotation.to_dict(), default=str))

    def _request_json(self, row: dict[str, Any]) -> dict[str, Any]:
        self._ensure_client()

        image_path = Path(str(row["image_path"]))
        image_b64 = image_to_base64(image_path)
        dataset_context = {
            "image_id": str(row["image_id"]),
            "ground_truth_drone_present": "yes" if bool(row.get("drone_present", False)) else "no",
            "ground_truth_box_count": int(row.get("box_count", 0) or 0),
        }
        prompt = (
            _CURATION_PROMPT
            + "\nDataset annotation context:\n"
            + json.dumps(dataset_context)
        )

        response = self._client.responses.parse(
            model=self.model_id,
            input=[
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                "You are a precise image curation assistant. "
                                "Return only valid JSON with no markdown."
                            ),
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {
                            "type": "input_image",
                            "image_url": f"data:image/jpeg;base64,{image_b64}",
                        },
                    ],
                },
            ],
            text_format=SceneCurationSchema,
            temperature=0.0,
            max_output_tokens=2048,
        )
        return response.output_parsed.model_dump()

    def annotate(self, row: dict[str, Any]) -> CurationAnnotation:
        """Run OpenAI curation over one image."""
        image_id = str(row["image_id"])

        cached = self._read_cache(image_id, row)
        if cached:
            return cached

        raw_response = self._request_json(row)

        annotation = CurationAnnotation(
            background=_coerce_enum(Background, raw_response.get("background"), Background.UNKNOWN),
            lighting=_coerce_enum(Lighting, raw_response.get("lighting"), Lighting.UNKNOWN),
            drone_visibility=_visibility_from_ground_truth(row),
            blur_bucket=_coerce_enum(BlurBucket, raw_response.get("blur_bucket"), BlurBucket.SHARP),
            possible_confusers=_coerce_confusers(raw_response.get("possible_confusers")),
            camera_angle=_coerce_enum(CameraAngle, raw_response.get("camera_angle"), CameraAngle.UNKNOWN),
            depth_range=_coerce_enum(DepthRange, raw_response.get("depth_range"), DepthRange.UNKNOWN),
            annotator=self.name,
            raw={
                "model": self.model_id,
                "base_url": self.base_url,
                "caption": str(raw_response.get("caption", "")),
                "openai_response": raw_response,
                "visibility_source": "ground_truth_bbox",
                "schema_version": "odd_v3",
            },
        )

        self._write_cache(image_id, annotation)
        return annotation
