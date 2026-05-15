# %% [markdown]
# # AI Engineer SG Drone VLM Golden Evaluation Demo
#
# **60-minute live workshop walkthrough.**
#
# Our job: A perimeter security team would like to efficiently review
# video footage to extract scenarios of interest using natural language.
# This workshop builds a VLM-powered image retrieval pipeline that lets
# operators query a collection of captured frames — for example,
# "show me low-light footage with a drone next to a building" — and receive
# a shortlist to review rather than watching hours of footage manually
#
# ## What we will do:
# We start with a raw drone dataset, use AI to accelerate curation, apply quality
# gates, identify test cases of interest, inspect coverage gaps, generate targeted
# augmentation candidates, and preview/run VLM triage evaluation.
#
# ---
#
# ## Workshop Narrative
#
# | Time | Phase |
# |------|-------|
# | 5 min | Load and normalize data |
# | 10 min | AI-assisted curation with Gemini Flash |
# | 10 min | Quality checks and filtering |
# | 10 min | Test cases of interest and coverage gaps |
# | 10 min | Targeted augmentation |
# | 10 min | VLM triage evaluation |
# | 5 min | Wrap-up |

# %% [markdown]
# ## Runtime Configuration
#
# All expensive steps respect row limits set here. Set to `0` for "process all
# available rows." Override with environment variables if desired (see
# `DemoConfig.from_env()`).

# %%
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from rich import box as rich_box
from rich.console import Console
from rich.table import Table

try:
    from IPython.display import display
except ImportError:
    display = print  # type: ignore[assignment]

from drone_vlm_eval.augmentation import BlurAugmenter, OpenAIImageAugmenter
from drone_vlm_eval.checks import assign_curation_status, compute_image_stats
from drone_vlm_eval.config import DemoConfig
from drone_vlm_eval.coverage import (
    analyze_coverage,
    compute_key_crosstabs,
    crosstab,
    export_coverage_artifacts,
)
from drone_vlm_eval.dataset import discover_existing_snippets, load_snippets
from drone_vlm_eval.download import download_snippets
from drone_vlm_eval.vlm_connector import VLMConnector
from drone_vlm_eval.vlm_curation import GeminiCurationAnnotator, OpenAICurationAnnotator
from drone_vlm_eval.vlm_eval import (
    QuerySetBuilder,
    run_retrieval_evaluation,
)

console = Console()


def stratified_sample_by_presence(df: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    """Sample positives and negatives, assigning odd remainders to positives."""
    if max_rows == 0 or max_rows >= len(df):
        return df.copy()

    positives = df[df["drone_present"] == True]  # noqa: E712
    negatives = df[df["drone_present"] == False]  # noqa: E712

    target_pos = (max_rows + 1) // 2 if len(positives) else 0
    pos_n = min(target_pos, len(positives))
    neg_n = min(max_rows - pos_n, len(negatives))

    remaining = max_rows - pos_n - neg_n
    if remaining > 0 and len(positives) > pos_n:
        extra = min(remaining, len(positives) - pos_n)
        pos_n += extra
        remaining -= extra
    if remaining > 0 and len(negatives) > neg_n:
        neg_n += min(remaining, len(negatives) - neg_n)

    samples = []
    if pos_n:
        samples.append(positives.sample(n=pos_n, random_state=42))
    if neg_n:
        samples.append(negatives.sample(n=neg_n, random_state=42))
    return pd.concat(samples, ignore_index=True) if samples else df.head(0).copy()


# %%
cfg = DemoConfig.from_env()

config_table = Table(title="Demo Configuration", show_header=False, box=rich_box.SIMPLE)
config_table.add_column("Setting", style="cyan bold", no_wrap=True)
config_table.add_column("Value")
config_table.add_row("annotator", cfg.annotator)
config_table.add_row("weather_augmenter", "openai_weather")
config_table.add_row("vlm_model", cfg.vlm_model)
config_table.add_row("max_curation_rows", str(cfg.max_curation_rows or "all"))
config_table.add_row("max_eval_rows", str(cfg.max_eval_rows or "all"))
config_table.add_row("force_refresh", str(cfg.force_refresh))
config_table.add_row("artifact_dir", str(cfg.artifact_dir))
console.print(config_table)

# %% [markdown]
# ## 0-5 min - Load and Normalize Data
#
# Two snippets, 100 images each:
# - **Test**: 100 images, 51 with XML bounding boxes (mixed positive/negative)
# - **Train**: 100 images, 100 with XML bounding boxes (all positive)
#
# Expected: **200 rows**, **151 positives** (have XML), **49 negatives**.

# %%
paths = discover_existing_snippets(cfg.data_dir)
if "test" not in paths or "train" not in paths:
    try:
        downloaded = download_snippets(cfg.data_dir)
        paths = {**downloaded, **paths}
    except Exception as exc:
        console.print(f"[yellow]Download skipped or failed:[/yellow] {exc}")
        console.print("Continuing with locally discovered snippets.")

sources_table = Table(title="Data Sources", box=rich_box.SIMPLE)
sources_table.add_column("Split", style="cyan bold")
sources_table.add_column("Path")
for split, p in sorted(paths.items()):
    sources_table.add_row(split, str(p))
console.print(sources_table)

# %%
df = load_snippets(paths)

stats_table = Table(title=f"Dataset ({len(df)} rows)", box=rich_box.SIMPLE_HEAD)
stats_table.add_column("Split", style="cyan")
stats_table.add_column("Drone Present", justify="center")
stats_table.add_column("Count", justify="right")
for (split, present), count in (
    df.groupby(["source_split", "drone_present"]).size().items()
):
    stats_table.add_row(split, "yes" if present else "no", str(count))
console.print(stats_table)

console.print(
    f"Positives: [green]{df['drone_present'].sum()}[/green]  "
    f"Negatives: [red]{(~df['drone_present']).sum()}[/red]  "
    f"With XML: {df['xml_path'].notna().sum()}"
)

# Show sample rows
display_cols = [
    "image_id",
    "source_split",
    "drone_present",
    "box_count",
    "width",
    "height",
]
console.print("\n[bold]Sample positives[/bold]")
display(df[df["drone_present"]][display_cols].head(3))

console.print("\n[bold]Sample negatives[/bold]")
display(df[~df["drone_present"]][display_cols].head(3))

# %% [markdown]
# ## 5-15 min - AI-Assisted Curation
#
# We run a VLM over configurable rows to produce structured scene metadata.
# This accelerates review; it is NOT the source of truth. The dataset
# XML labels remain our anchor.
#
# **Ground-truth fields** (from XML/bbox):
# - `drone_present`: yes / no
# - `drone_visibility`: large / medium / small (derived from bbox size)
#
# **VLM-annotated fields**:
# - `background`: sky / trees / building / cluttered
# - `lighting`: normal / dark / bright / backlit
# - `blur_bucket`: sharp / mild / blurry
# - `possible_confusers`: bird / aircraft / helicopter / none
# - `camera_angle`: top_down / high_angle / eye_level / low_angle / worms_eye
# - `depth_range`: close_up / mid_range / landscape

# %%
console.print(f"Loading {cfg.annotator} annotator...")
if cfg.annotator == "gemini":
    annotator = GeminiCurationAnnotator(
        model_id=cfg.gemini_model_id,
        cache_dir=cfg.curation_cache_dir,
        force_refresh=cfg.force_refresh,
        requests_per_minute=cfg.gemini_rpm,
    )
elif cfg.annotator == "openai":
    annotator = OpenAICurationAnnotator(
        model_id=cfg.openai_curation_model,
        base_url=cfg.openai_curation_base_url,
        cache_dir=cfg.curation_cache_dir,
        force_refresh=cfg.force_refresh,
    )
else:
    raise ValueError(f"Unsupported curation annotator: {cfg.annotator}")
console.print(f"Annotator: [cyan bold]{annotator.name}[/cyan bold]")

# %%
# Select rows for curation (respecting max_curation_rows with stratified sampling)
if cfg.max_curation_rows > 0 and cfg.max_curation_rows < len(df):
    curation_source = stratified_sample_by_presence(df, cfg.max_curation_rows)
    console.print(
        f"Curation limited to {len(curation_source)} rows (stratified sample of {len(df)} total)"
    )
else:
    curation_source = df

rows_to_annotate = curation_source.to_dict(orient="records")

annotations = []
for i, row in enumerate(rows_to_annotate):
    annotation = annotator.annotate(row)
    annotations.append(annotation.to_dict())
    if (i + 1) % 20 == 0:
        console.print(f"  ... {i + 1}/{len(rows_to_annotate)} annotated")

curation_df = pd.DataFrame(annotations)
df_curated = pd.concat(
    [curation_source.reset_index(drop=True), curation_df.reset_index(drop=True)],
    axis=1,
)

# Assign curation status
df_curated["curation_status"] = df_curated.apply(assign_curation_status, axis=1)

console.print(
    f"\nCuration complete: [bold]{len(df_curated)}[/bold] rows  "
    f"annotator: [cyan]{df_curated['annotator'].iloc[0]}[/cyan]"
)
display(df_curated["curation_status"].value_counts().rename("count").reset_index())

# Show a few examples
sample_cols = [
    "image_id",
    "drone_present",
    "drone_visibility",
    "background",
    "lighting",
    "blur_bucket",
    "possible_confusers",
    "camera_angle",
    "depth_range",
    "curation_status",
]
console.print("\n[bold]Curation samples[/bold]")
display(df_curated[sample_cols].head(8))

# %%
# Write curated dataset
df_curated.to_json(
    cfg.artifact_dir / "curated_dataset.jsonl", orient="records", lines=True
)
console.print(
    f"[green]✓[/green] curated_dataset.jsonl → {cfg.artifact_dir / 'curated_dataset.jsonl'}"
)

# %% [markdown]
# ## 10 min - Quality Checks and Filtering
#
# Rather than binary pass/fail checks we compute per-image statistics and
# visualise their distributions. Inspect the histograms, then adjust the
# filtering thresholds in the cell below.
#
# | Statistic | What it measures |
# |-----------|-----------------|
# | `blur_score` | Laplacian variance proxy — higher = sharper |
# | `brightness` | Mean pixel luminance [0, 255] |
# | `bbox_ratio` | Largest drone bbox / image area (drone-present rows only) |

# %%
console.print("Computing image statistics...")
stats_df = compute_image_stats(df_curated)
df_curated = df_curated.merge(stats_df, on="image_id", how="left")

display(df_curated[["blur_score", "brightness", "bbox_ratio"]].describe().round(3))

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("Image Quality Distributions", fontweight="bold")

axes[0].hist(
    df_curated["blur_score"].dropna(), bins=30, color="steelblue", edgecolor="white"
)
axes[0].set_title("Blur Score (higher = sharper)")
axes[0].set_xlabel("Laplacian variance")
axes[0].axvline(20, color="red", linestyle="--", label="default min")
axes[0].legend()

axes[1].hist(
    df_curated["brightness"].dropna(), bins=30, color="goldenrod", edgecolor="white"
)
axes[1].set_title("Brightness (mean luminance)")
axes[1].set_xlabel("Mean pixel value [0–255]")
axes[1].axvline(25, color="red", linestyle="--", label="dark min")
axes[1].axvline(235, color="orange", linestyle="--", label="bright max")
axes[1].legend()

bbox_vals = df_curated.loc[df_curated["drone_present"] == True, "bbox_ratio"].dropna()  # noqa: E712
axes[2].hist(bbox_vals, bins=30, color="mediumseagreen", edgecolor="white")
axes[2].set_title("Subject-to-Frame Ratio\n(drone-present rows)")
axes[2].set_xlabel("Largest bbox area / image area")

plt.tight_layout()
plt.show()

# %%
# --- Filtering thresholds ---
# Adjust after inspecting the distributions above. Set to None to skip a check.
MIN_BLUR: float | None = 20.0
MIN_BRIGHTNESS: float | None = 25.0
MAX_BRIGHTNESS: float | None = 235.0

blur_ok = df_curated["blur_score"].isna() | (
    MIN_BLUR is None or df_curated["blur_score"] >= MIN_BLUR
)
brightness_ok = df_curated["brightness"].isna() | (
    (MIN_BRIGHTNESS is None or df_curated["brightness"] >= MIN_BRIGHTNESS)
    & (MAX_BRIGHTNESS is None or df_curated["brightness"] <= MAX_BRIGHTNESS)
)
keep_mask = blur_ok & brightness_ok
df_golden = df_curated[keep_mask].copy()
dropped_n = (~keep_mask).sum()

df_golden.to_json(
    cfg.artifact_dir / "golden_dataset.jsonl", orient="records", lines=True
)
console.print(
    f"Stats filtering: [green]{len(df_golden)}[/green] kept, [red]{dropped_n}[/red] dropped\n"
    f"[green]✓[/green] golden_dataset.jsonl → {cfg.artifact_dir / 'golden_dataset.jsonl'}"
)
display(df_golden[sample_cols].head(5))

# %% [markdown]
# ## 10 min - Test Cases of Interest and Coverage Gaps
#
# We analyze coverage across these compact dimensions:
# - `drone_present`, `drone_visibility`, `background`, `lighting`,
#   `blur_bucket`, `possible_confusers`
#
# Gaps (buckets with < 5 examples) drive augmentation or future data collection.

# %%
coverage_dimensions = [
    "drone_present",
    "drone_visibility",
    "background",
    "lighting",
    "blur_bucket",
    "possible_confusers",
]

coverage_tables, gaps = analyze_coverage(df_golden, coverage_dimensions, min_count=10)

# Compute cross-tabs
crosstabs = compute_key_crosstabs(df_golden)

# Export all coverage artifacts
export_coverage_artifacts(coverage_tables, gaps, crosstabs, cfg.artifact_dir)

# Per-dimension coverage tables
for name, table in coverage_tables.items():
    console.print(f"\n[bold]{name}[/bold]")
    display(table)

# Coverage gaps
if gaps:
    gap_table = Table(
        title=f"Coverage Gaps  (min_count={cfg.coverage_min_count})",
        box=rich_box.SIMPLE_HEAD,
    )
    gap_table.add_column("Dimension", style="cyan")
    gap_table.add_column("Value")
    gap_table.add_column("Count", justify="right", style="red bold")
    gap_table.add_column("Required", justify="right")
    for gap in gaps:
        gap_table.add_row(gap.dimension, gap.value, str(gap.count), str(gap.min_count))
    console.print(gap_table)
else:
    console.print("[green]No coverage gaps found![/green]")

# Key cross-tabs
for title, key in [
    ("drone_present × drone_visibility", "drone_present_x_drone_visibility"),
    ("drone_present × possible_confusers", "drone_present_x_possible_confusers"),
]:
    console.print(f"\n[bold]{title}[/bold]")
    display(crosstabs.get(key, pd.DataFrame()))

# %% [markdown]
# ## 35-45 min - Targeted Augmentation
#
# Two augmentation paths:
# 1. **Blur** (conventional, deterministic): Gaussian blur on positive images.
# 2. **Weather** (OpenAI gpt-image-2 image editing): weather variants.
#
# Generated rows are **candidates**: they preserve source lineage and
# provenance. They are NOT silently merged into the golden candidate.

# %%
positive_targets = df_golden[df_golden["drone_present"]].copy()
console.print(f"Positive target rows available: [bold]{len(positive_targets)}[/bold]")

# --- Blur augmentation ---
console.print("\n[bold]Blur augmentation[/bold]")
max_blur = cfg.max_augmentation_rows if cfg.max_augmentation_rows > 0 else 2
blur_rows = BlurAugmenter().augment(
    positive_targets, cfg.artifact_dir / "augmented" / "blur", max_rows=max_blur
)
console.print(f"Blur candidates: [green]{len(blur_rows)}[/green]")
blur_rows["blur_bucket"] = "blurry"

# --- Weather augmentation ---
# Edit these prompts to try different augmentation styles.
WEATHER_AUGMENT_PROMPTS = {
    "fog": "same scene with dense fog, low visibility, foggy atmosphere, drone partially obscured",
    "rain": "same scene with heavy rain, rain streaks on lens, dark overcast sky, wet surfaces",
    "overcast": "same scene with overcast cloudy sky, grey muted lighting, flat shadows, no direct sunlight",
    "night": "same scene at night, dark sky, navigation lights visible, soft moonlight",
}

console.print("\n[bold]Weather augmentation[/bold]")
# Number of source rows to augment. Each row gets all prompts applied,
# so total generated = max_source_rows × len(WEATHER_AUGMENT_PROMPTS).
max_source_rows = cfg.max_augmentation_rows if cfg.max_augmentation_rows > 0 else 2
weather_rows = OpenAIImageAugmenter(
    prompts=WEATHER_AUGMENT_PROMPTS,
    cache_dir=cfg.openai_image_cache_dir,
).augment(
    positive_targets,
    cfg.artifact_dir / "augmented" / "weather",
    max_rows=max_source_rows,
)
console.print(f"Weather candidates: [green]{len(weather_rows)}[/green] (openai_image)")
# Patch lighting to reflect augmented conditions; fog also degrades sharpness
_lighting_patch = {"night": "dark", "rain": "dark", "fog": "dark"}
weather_rows["lighting"] = (
    weather_rows["augment_label"].map(_lighting_patch).fillna(weather_rows["lighting"])
)
weather_rows.loc[weather_rows["augment_label"] == "fog", "blur_bucket"] = "blurry"

# Combine and export
candidate_augmented = pd.concat([blur_rows, weather_rows], ignore_index=True)
candidate_augmented.to_json(
    cfg.artifact_dir / "augmented_candidates.jsonl", orient="records", lines=True
)

# Show provenance fields
aug_display = [
    "image_id",
    "source_image_id",
    "augmentation_type",
    "augment_label",
    "is_synthetic",
    "openai_augment_status",
]
available = [c for c in aug_display if c in candidate_augmented.columns]
console.print(f"\n[bold]Augmented candidates ({len(candidate_augmented)} total)[/bold]")
display(candidate_augmented[available])
console.print(
    f"[green]✓[/green] augmented_candidates.jsonl → {cfg.artifact_dir / 'augmented_candidates.jsonl'}"
)

# Extended evaluation set: golden + successfully generated augmented images
aug_ok = candidate_augmented[candidate_augmented["is_synthetic"]].copy()
df_eval_extended = pd.concat([df_golden, aug_ok], ignore_index=True)
console.print(
    f"Eval set: [bold]{len(df_golden)}[/bold] golden + [bold]{len(aug_ok)}[/bold] augmented "
    f"= [bold]{len(df_eval_extended)}[/bold] total"
)

# %% [markdown]
# ## 10 min - VLM Retrieval Evaluation
#
# We evaluate candidate VLMs as zero-shot image retrievers. For each query
# the model is shown an image and asked whether it matches — responding with
# a graded verbal answer:
#
#   "definitely yes" / "probably yes" / "uncertain" / "probably no" / "definitely no"
#
# Queries are auto-selected from the curated metadata so that ground truth is
# always coherent with the query text. Each query gets 5 positive frames
# (meeting the metadata condition) and 5 negative frames (not meeting it).
# The evaluation pool combines golden and successfully generated augmented images
# so that augmentation conditions (blur, fog, rain, night) are exercised.
#
# **Metrics** (threshold: probably yes or above)
# - Precision: of retrieved frames, what fraction are actually relevant?
# - Recall: of all relevant frames, what fraction were retrieved?
# - F1: harmonic mean

# %%
# --- Systems under test ---
# Edit this list to change which models are benchmarked.
SYSTEMS_UNDER_TEST = [
    "gpt-5.4-nano",
    "gpt-5.4-mini",
]

# Build and display the query set once — shared across all models
query_specs = QuerySetBuilder(n_queries=5, n_samples=5).build(df_eval_extended)

qs_table = Table(title="Evaluation Query Set", box=rich_box.SIMPLE_HEAD)
qs_table.add_column("ID", style="cyan", no_wrap=True)
qs_table.add_column("Query")
qs_table.add_column("Pos", justify="right")
qs_table.add_column("Neg", justify="right")
for qs in query_specs:
    qs_table.add_row(
        qs.query_id, qs.query_text, str(len(qs.positive_ids)), str(len(qs.negative_ids))
    )
console.print(qs_table)
console.print(
    f"[dim]{len(SYSTEMS_UNDER_TEST)} model(s) × {len(query_specs)} queries × 10 frames "
    f"= {len(SYSTEMS_UNDER_TEST) * len(query_specs) * 10} total VLM calls[/dim]"
)

# Check API key once before entering the loop
_connector_check = VLMConnector()
if not _connector_check.api_key:
    console.print("[yellow]OPENAI_API_KEY not set. Skipping VLM evaluation.[/yellow]")
    console.print(
        "Set [cyan]OPENAI_API_KEY[/cyan] (and optionally [cyan]OPENAI_BASE_URL[/cyan]) "
        "in your .env file or environment to run this step."
    )
else:
    _drop = {"image_path", "xml_path", "bbox", "raw"}
    meta_cols = ["image_id"] + [
        c for c in df_eval_extended.columns if c not in _drop and c != "image_id"
    ]
    all_eval_dfs: list[pd.DataFrame] = []

    for model_name in SYSTEMS_UNDER_TEST:
        console.print(f"\n[bold cyan]Model: {model_name}[/bold cyan]")
        connector = VLMConnector(model=model_name)

        frame_results, query_metrics, summary = run_retrieval_evaluation(
            df=df_eval_extended,
            connector=connector,
            cfg=cfg,
            query_specs=query_specs,
            run_label=model_name,
        )

        # Build enriched per-frame DataFrame for this model
        id_to_query_text = {qs.query_id: qs.query_text for qs in query_specs}
        eval_rows = [
            {
                "model": model_name,
                "query_id": r.query_id,
                "query_text": id_to_query_text.get(r.query_id, r.query_id),
                "image_id": r.image_id,
                "is_positive": r.is_positive,
                "graded_response": r.graded_response,
                "score": r.score,
                "retrieved": r.retrieved,
                "correct": r.retrieved == r.is_positive,
            }
            for r in frame_results
        ]
        model_df = pd.DataFrame(eval_rows)
        model_df = model_df.merge(
            df_eval_extended[meta_cols], on="image_id", how="left"
        )
        all_eval_dfs.append(model_df)

    # Combined DataFrame across all models
    eval_df = pd.concat(all_eval_dfs, ignore_index=True)
    detailed_path = cfg.artifact_dir / "retrieval_eval_detailed.jsonl"
    eval_df.to_json(detailed_path, orient="records", lines=True)
    console.print(
        f"\n[green]✓[/green] retrieval_eval_detailed.jsonl → {detailed_path} ({len(eval_df)} rows)"
    )

# %%
# Aggregation — rerun this cell freely without re-calling the VLM.
# Loads from disk so it works after a kernel restart too.
eval_df = pd.read_json(cfg.artifact_dir / "retrieval_eval_detailed.jsonl", lines=True)
eval_df["is_positive"] = eval_df["is_positive"].astype(bool)
eval_df["retrieved"] = eval_df["retrieved"].astype(bool)


def _query_metrics(g: pd.DataFrame) -> pd.Series:
    tp = (g["is_positive"] & g["retrieved"]).sum()
    fp = (~g["is_positive"] & g["retrieved"]).sum()
    fn = (g["is_positive"] & ~g["retrieved"]).sum()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return pd.Series(
        {
            "precision": round(precision, 2),
            "recall": round(recall, 2),
            "f1": round(f1, 2),
        }
    )


metrics_df = (
    eval_df.groupby(["model", "query_text"], sort=False)
    .apply(_query_metrics, include_groups=False)
    .reset_index()
)
matrix = metrics_df.pivot(
    index="query_text", columns="model", values=["precision", "recall", "f1"]
)
matrix.columns = [f"{model} {metric}" for metric, model in matrix.columns]
matrix.index.name = None
matrix.loc["mean"] = matrix.mean()

console.print("\n[bold]Evaluation Matrix (Precision / Recall / F1 per query)[/bold]")
display(matrix)

# --- Breakdown by augmentation condition ---
_MIN_CONDITION_SAMPLES = 5

_aug_label = (
    eval_df["augment_label"]
    if "augment_label" in eval_df.columns
    else pd.Series(pd.NA, index=eval_df.index, dtype=object)
)
_aug_type = (
    eval_df["augmentation_type"]
    if "augmentation_type" in eval_df.columns
    else pd.Series(pd.NA, index=eval_df.index, dtype=object)
)
eval_df["condition"] = (
    _aug_label
    .fillna(_aug_type.map({"gaussian_blur": "blur"}))
    .fillna("original")
)


def _condition_metrics(g: pd.DataFrame) -> pd.Series:
    if len(g) < _MIN_CONDITION_SAMPLES:
        return pd.Series({"precision": float("nan"), "recall": float("nan"), "f1": float("nan")})
    return _query_metrics(g)


condition_metrics = (
    eval_df.groupby(["model", "condition"], sort=False)
    .apply(_condition_metrics, include_groups=False)
    .reset_index()
)
condition_matrix = condition_metrics.pivot(
    index="condition", columns="model", values=["precision", "recall", "f1"]
)
condition_matrix.columns = [
    f"{model} {metric}" for metric, model in condition_matrix.columns
]
condition_matrix.index.name = None
condition_matrix.loc["mean"] = condition_matrix.mean()

console.print("\n[bold]Breakdown by Augmentation Condition[/bold]")
display(condition_matrix)

# %% [markdown]
# ## Wrap-Up
#
# ### What we built today:
#
# | Artifact | Purpose |
# |----------|---------|
# | `curated_dataset.jsonl` | Full dataset with AI-assisted curation metadata |
# | `golden_dataset.jsonl` | Quality-filtered trusted evaluation set |
# | `coverage_*.csv` | Per-dimension coverage tables |
# | `coverage_gaps.csv` | Buckets below minimum count |
# | `coverage_crosstab_*.csv` | Key cross-tabulations |
# | `augmented_candidates.jsonl` | Blur + weather augmentation candidates |
# | `retrieval_results_{model}.jsonl` | Per-frame graded relevance scores per model |
# | `retrieval_eval_detailed.jsonl` | Per-frame scores joined with full image metadata |
# | `retrieval_eval_summary_{model}.json` | Per-query and aggregate P/R/F1 per model |

# %%
artifacts = [
    {
        "file": str(f.relative_to(cfg.artifact_dir)),
        "size_kb": round(f.stat().st_size / 1024, 1),
    }
    for f in sorted(cfg.artifact_dir.rglob("*"))
    if f.is_file()
]
display(pd.DataFrame(artifacts))
