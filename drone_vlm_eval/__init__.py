"""Utilities for the drone VLM golden-evaluation mini demo."""

from drone_vlm_eval.augmentation import BlurAugmenter, OpenAIImageAugmenter
from drone_vlm_eval.checks import assign_curation_status, compute_image_stats
from drone_vlm_eval.config import DemoConfig
from drone_vlm_eval.coverage import analyze_coverage, compute_key_crosstabs, crosstab, export_coverage_artifacts
from drone_vlm_eval.dataset import discover_existing_snippets, load_snippets
from drone_vlm_eval.download import download_snippets
from drone_vlm_eval.vlm_connector import VLMConnector
from drone_vlm_eval.vlm_curation import GeminiCurationAnnotator, OpenAICurationAnnotator
from drone_vlm_eval.vlm_eval import QuerySetBuilder, run_retrieval_evaluation

__all__ = [
    # Config
    "DemoConfig",
    # Dataset
    "discover_existing_snippets",
    "download_snippets",
    "load_snippets",
    # Curation
    "GeminiCurationAnnotator",
    "OpenAICurationAnnotator",
    # Quality
    "assign_curation_status",
    "compute_image_stats",
    # Coverage
    "analyze_coverage",
    "crosstab",
    "compute_key_crosstabs",
    "export_coverage_artifacts",
    # Augmentation
    "BlurAugmenter",
    "OpenAIImageAugmenter",
    # VLM retrieval eval
    "VLMConnector",
    "QuerySetBuilder",
    "run_retrieval_evaluation",
]
