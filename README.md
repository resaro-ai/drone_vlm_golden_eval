# Drone VLM Golden Evaluation Demo

Demo for building a golden evaluation pipeline
around a VLM drone-alert triage use case for AI Engineer Singapore Workshop on 15 May 2026.

The runnable walkthrough is [`demo.py`](./demo.py), a plain Python file with
`# %%` cell markers. Step through it in VS Code, Cursor, or any
Jupyter-compatible editor.

## Quick Start

```bash
cd ~/Documents/drone_vlm_golden_eval
uv run python demo.py
```

Create a Gemini API key in Google AI Studio, then put it in `.env`:

```bash
cp .env.example .env
# edit .env and set GEMINI_API_KEY
```

### Dataset Viewer

A Gradio-based viewer (`viewer.py`) runs alongside the demo so you can
inspect artifacts as the pipeline produces them. Launch it in a **separate
terminal** before or while stepping through `demo.py`:

```bash
uv run python viewer.py
# → http://127.0.0.1:7860
```

Hit **↻ Refresh** after each demo phase to pick up new artifacts. The gallery
lets you browse curated / golden / augmented / eval images with bounding-box
overlays and filter by split, drone presence, visibility, and background. Click
any image to view its full metadata and edit labels inline.

Use `--port` to change the port and `--host 0.0.0.0` to share on your LAN.

## Workshop Flow (~60 min)

| Time | Phase | Key Output |
|------|-------|------------|
| 0-5 min | Load and normalize | 200 rows (151 pos, 49 neg) |
| 5-15 min | AI-assisted curation | `curated_dataset.jsonl` |
| 15-25 min | Quality checks | `golden_dataset.jsonl` |
| 25-35 min | Coverage gaps | `coverage_*.csv`, `coverage_crosstab_*.csv` |
| 35-45 min | Augmentation | `augmented_candidates.jsonl` |
| 45-55 min | VLM retrieval eval | `retrieval_eval_detailed.jsonl`, `retrieval_eval_summary_{model}.json` |
| 55-60 min | Wrap-up | Artifact inventory |

## Configuration

All run-time settings live in `DemoConfig` (see
`drone_vlm_eval/config.py`). Overrides via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `DRONE_DEMO_ANNOTATOR` | `gemini` | `gemini` or `openai` curation annotator |
| `GEMINI_API_KEY` | - | Required for Gemini Flash curation |
| `GEMINI_MODEL_ID` | `gemini-3.1-flash-lite-preview` | Gemini curation model |
| `GEMINI_RPM` | `15` | Max Gemini requests per minute (0 = unlimited) |
| `OPENAI_MODEL` | `gpt-5.4-nano` | OpenAI-compatible curation model |
| `OPENAI_BASE_URL` | - | Optional OpenAI-compatible curation endpoint |
| `VLM_MODEL` | `gpt-5.5` | OpenAI-compatible model for VLM retrieval eval |
| `OPENAI_API_KEY` | - | Required for OpenAI curation and VLM eval |
| `DRONE_DEMO_MAX_CURATION` | `0` (all) | Max curation rows |
| `DRONE_DEMO_MAX_AUGMENTATION` | `0` (all) | Max augmented rows |
| `DRONE_DEMO_MAX_EVAL` | `0` (all) | Max eval rows |
| `DRONE_DEMO_FORCE_REFRESH` | - | Set to `1` to bypass caches |

Example with limits for a quick demo:

```bash
DRONE_DEMO_MAX_CURATION=20 DRONE_DEMO_MAX_EVAL=10 \
  uv run python demo.py
```

## Artifacts

All generated artifacts land in `artifacts/`:

```text
artifacts/
├── curated_dataset.jsonl          # Full dataset plus curation metadata
├── golden_dataset.jsonl           # Quality-filtered trusted evaluation set
├── coverage_{dim}.csv             # Per-dimension coverage tables
├── coverage_gaps.csv              # Buckets below min_count
├── coverage_crosstab_*.csv        # Key cross-tabulations
├── augmented/
│   ├── blur/                      # Gaussian blur candidates
│   └── weather/                   # OpenAI weather-edit candidates
├── augmented_candidates.jsonl     # All augmented rows plus provenance
├── {annotator}_cache/             # Per-image curation cache
├── openai_image_cache/            # Per-image OpenAI image-edit cache
├── retrieval_results_{model}.jsonl        # Per-frame graded responses per model
├── retrieval_eval_detailed.jsonl          # Per-frame scores joined with metadata
└── retrieval_eval_summary_{model}.json    # Per-query and aggregate P/R/F1 per model
```

## Optional Integrations

### Curation Annotators

Set `GEMINI_API_KEY` in `.env`, then:

```bash
uv run python demo.py
```

To use OpenAI instead:

```bash
DRONE_DEMO_ANNOTATOR=openai \
OPENAI_API_KEY=sk-... \
  uv run python demo.py
```

Curation annotators return the controlled non-label ODD fields used by the rest
of the demo: `background`, `lighting`, `blur_bucket`, `possible_confusers`,
`camera_angle`, and `depth_range`. The dataset XML/bbox annotations remain the
source for `drone_present`, while `drone_visibility` is derived deterministically
from bounding-box size. The free-form caption and raw JSON response are kept
under `raw`.

| ODD Dimension | Possible Values | Source |
|---------------|-----------------|--------|
| `drone_present` | yes / no | XML/bbox ground truth |
| `drone_visibility` | large / medium / small | Derived from bbox size |
| `background` | sky / trees / building / cluttered | Curation annotator |
| `lighting` | normal / dark / bright / backlit | Curation annotator |
| `blur_bucket` | sharp / mild / blurry | Curation annotator |
| `possible_confusers` | bird / aircraft / helicopter / none | Curation annotator |
| `camera_angle` | top_down / high_angle / eye_level / low_angle / worms_eye | Curation annotator |
| `depth_range` | close_up / mid_range / landscape | Curation annotator |

Annotations are cached per `image_id` in `artifacts/{annotator}_cache/`. Use
`DRONE_DEMO_FORCE_REFRESH=1` to bypass cache.

### VLM Retrieval Evaluation

Set `OPENAI_API_KEY` and optionally `OPENAI_BASE_URL` and `VLM_MODEL`:

```bash
OPENAI_API_KEY=sk-... VLM_MODEL=gpt-5.5 \
  uv run python demo.py
```

For each query the model is shown an image and responds with one of:
`definitely yes` / `probably yes` / `uncertain` / `probably no` / `definitely no`.
Scores ≥ 4 ("probably yes" or above) count as retrieved. Queries are
auto-selected from the curated metadata so ground truth is always coherent with
the query text. Metrics (precision, recall, F1) are reported per query and in
aggregate, and can be recomputed from `retrieval_eval_detailed.jsonl` without
re-calling the VLM.

## Data

Two 100-image snippets (test + train) are expected or downloaded:

- Test snippet (100 images, 51 XMLs)
- Train snippet (100 images, 100 XMLs)

Total: 200 rows (151 positives, 49 negatives).

If downloads are unavailable, put extracted folders under
`data/raw/`.
