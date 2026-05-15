from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ModuleNotFoundError:
        return
    load_dotenv(_PROJECT_ROOT / ".env")


def _annotator_from_env() -> str:
    annotator = os.environ.get("DRONE_DEMO_ANNOTATOR", "gemini").strip().lower()
    supported = {"gemini", "openai"}
    if annotator not in supported:
        supported_list = ", ".join(sorted(supported))
        raise ValueError(f"Unsupported curation annotator. Supported values: {supported_list}.")
    return annotator


@dataclass
class DemoConfig:
    """Runtime configuration for the drone VLM golden eval demo.

    All expensive steps reference max_*_rows to cap processing during live
    demos. Set to 0 for no limit (process all rows).
    """

    # Directories
    data_dir: Path = field(default_factory=lambda: _PROJECT_ROOT / "data")
    artifact_dir: Path = field(default_factory=lambda: _PROJECT_ROOT / "artifacts")

    # Row limits (0 = no limit, process all rows)
    max_curation_rows: int = 0
    max_augmentation_rows: int = 0
    max_eval_rows: int = 0

    # Feature flags
    force_refresh: bool = False

    # Backend selection
    annotator: str = "gemini"
    vlm_model: str = "gpt-5.5"

    # Model IDs
    gemini_model_id: str = "gemini-3.1-flash-lite-preview"
    gemini_rpm: int = 15  # max Gemini requests per minute (0 = unlimited)
    openai_curation_model: str = "gpt-5.4-nano"
    openai_curation_base_url: str | None = None

    # Coverage
    coverage_min_count: int = 5

    # Caching
    curation_cache_dir: Path | None = None
    openai_image_cache_dir: Path | None = None

    def __post_init__(self) -> None:
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

        if self.curation_cache_dir is None:
            self.curation_cache_dir = self.artifact_dir / f"{self.annotator}_cache"
        if self.openai_image_cache_dir is None:
            self.openai_image_cache_dir = self.artifact_dir / "openai_image_cache"

        self.curation_cache_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_env(cls) -> "DemoConfig":
        """Build config from environment variables with sensible defaults.

        Environment variables:
            DRONE_DEMO_ANNOTATOR: curation annotator (default: ``"gemini"``)
            VLM_MODEL: OpenAI model name for VLM eval (default: gpt-5.5)
            GEMINI_MODEL_ID: Gemini curation model (default: gemini-3.1-flash-lite-preview)
            GEMINI_RPM: Gemini requests per minute cap (default: 15, 0 = unlimited)
            GEMINI_API_KEY: Google AI Studio API key
            OPENAI_MODEL: OpenAI curation model (default: gpt-5.4-nano)
            OPENAI_BASE_URL: OpenAI-compatible curation endpoint
            OPENAI_API_KEY: API key for OpenAI
            DRONE_DEMO_MAX_CURATION: max curation rows
            DRONE_DEMO_MAX_AUGMENTATION: max augmentation rows
            DRONE_DEMO_MAX_EVAL: max evaluation rows
            DRONE_DEMO_FORCE_REFRESH: ``"1"`` to bypass caches
        """
        _load_dotenv()
        return cls(
            annotator=_annotator_from_env(),
            vlm_model=os.environ.get("VLM_MODEL", "gpt-5.5"),
            max_curation_rows=int(os.environ.get("DRONE_DEMO_MAX_CURATION", "0")),
            max_augmentation_rows=int(os.environ.get("DRONE_DEMO_MAX_AUGMENTATION", "0")),
            max_eval_rows=int(os.environ.get("DRONE_DEMO_MAX_EVAL", "0")),
            force_refresh=os.environ.get("DRONE_DEMO_FORCE_REFRESH", "") == "1",
            gemini_model_id=os.environ.get("GEMINI_MODEL_ID", "gemini-3.1-flash-lite-preview"),
            gemini_rpm=int(os.environ.get("GEMINI_RPM", "15")),
            openai_curation_model=os.environ.get("OPENAI_MODEL", "gpt-5.4-nano"),
            openai_curation_base_url=os.environ.get("OPENAI_BASE_URL") or None,
        )
