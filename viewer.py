#!/usr/bin/env python3
"""Drone VLM Golden Eval — Dataset Viewer.

Launch alongside demo.py to preview artifacts as the pipeline runs.
Hit Refresh to pick up new/updated files after each phase.

Usage:
    python viewer.py                          # defaults: artifacts/ port 7860
    python viewer.py --port 8787              # custom port
    python viewer.py --host 0.0.0.0           # share on LAN
"""

from __future__ import annotations

import argparse
import io
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageDraw

# ---------------------------------------------------------------------------
# Arg parsing (must come before gr import so Gradio doesn't swallow flags)
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Drone VLM dataset viewer")
parser.add_argument("--artifacts-dir", default="artifacts", type=Path)
parser.add_argument("--port", default=7860, type=int)
parser.add_argument("--host", default="127.0.0.1")
parser.add_argument("--share", action="store_true")
args = parser.parse_args()

import gradio as gr

ARTIFACTS = Path(args.artifacts_dir).resolve()
LAST_REFRESH = datetime.now()
APP_CSS = """
footer {visibility: hidden}
.refresh-btn { background: #3b82f6 !important; color: white !important; }
"""

# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------


def _load_jsonl(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        return None
    df = pd.DataFrame(rows)
    # Normalise booleans stored as strings
    for col in ("drone_present", "is_synthetic"):
        if col in df.columns and df[col].dtype == object:
            df[col] = df[col].map(
                lambda v: v if isinstance(v, bool) else str(v).lower() == "true"
            )
    return df


def load_curated() -> pd.DataFrame | None:
    return _load_jsonl(ARTIFACTS / "curated_dataset.jsonl")


def load_golden() -> pd.DataFrame | None:
    return _load_jsonl(ARTIFACTS / "golden_dataset.jsonl")


def load_augmented() -> pd.DataFrame | None:
    return _load_jsonl(ARTIFACTS / "augmented_candidates.jsonl")


def load_eval() -> pd.DataFrame | None:
    return _load_jsonl(ARTIFACTS / "retrieval_eval_detailed.jsonl")


def load_raw() -> pd.DataFrame | None:
    """Load raw snippets from data/raw/ (before any curation)."""
    try:
        from drone_vlm_eval.dataset import discover_existing_snippets, load_snippets
        paths = discover_existing_snippets(Path("data"))
        if not paths:
            return None
        return load_snippets(paths)
    except Exception:
        return None


def load_dataset(name: str) -> tuple[pd.DataFrame | None, str]:
    """Return (DataFrame, status_message)."""
    loaders = {
        "raw": load_raw,
        "curated": load_curated,
        "golden": load_golden,
        "augmented": load_augmented,
        "eval": load_eval,
    }
    df = loaders[name]()
    if df is None:
        return None, f"⚠️  {name}_dataset.jsonl not found"
    return df, f"✅ {name} — {len(df)} rows"


# ---------------------------------------------------------------------------
# Image + bbox helpers
# ---------------------------------------------------------------------------


def _draw_boxes_on_image(
    image_path: str,
    bboxes: list[list[float]] | None,
    figsize: tuple[int, int] = (8, 6),
) -> Image.Image | None:
    """Draw bounding boxes on an image, return a PIL Image."""
    path = Path(image_path)
    if not path.exists():
        return None
    try:
        img = Image.open(path).convert("RGB")
    except Exception:
        return None

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img)
    ax.axis("off")

    if bboxes:
        for box in bboxes:
            if not box or len(box) < 4:
                continue
            xmin, ymin, xmax, ymax = float(box[0]), float(box[1]), float(box[2]), float(box[3])
            w, h = xmax - xmin, ymax - ymin
            rect = mpatches.Rectangle(
                (xmin, ymin), w, h,
                linewidth=2, edgecolor="lime", facecolor="none",
            )
            ax.add_patch(rect)

    plt.tight_layout(pad=0)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _side_by_side(
    source_path: str | None,
    aug_path: str | None,
    source_boxes: list | None = None,
    aug_boxes: list | None = None,
) -> Image.Image | None:
    """Produce a side-by-side comparison image."""
    imgs = []
    for path, boxes in ((source_path, source_boxes), (aug_path, aug_boxes)):
        if path and Path(path).exists():
            try:
                raw = Image.open(path).convert("RGB")
                if boxes:
                    raw = _draw_boxes_on_image(str(path), boxes, figsize=(6, 4.5))
                imgs.append(raw.resize((480, 360)))
            except Exception:
                imgs.append(Image.new("RGB", (480, 360), color=(30, 30, 30)))
        else:
            imgs.append(Image.new("RGB", (480, 360), color=(30, 30, 30)))

    combined = Image.new("RGB", (960, 360))
    combined.paste(imgs[0], (0, 0))
    combined.paste(imgs[1], (480, 0))
    return combined


def _format_metadata(row: dict) -> str:
    """Build a markdown metadata block for a row."""
    lines: list[str] = []
    fields = [
        ("image_id", "**ID**"),
        ("source_split", "Split"),
        ("drone_present", "Drone Present"),
        ("drone_visibility", "Visibility"),
        ("background", "Background"),
        ("lighting", "Lighting"),
        ("blur_bucket", "Blur"),
        ("possible_confusers", "Confusers"),
        ("camera_angle", "Camera Angle"),
        ("depth_range", "Depth"),
        ("curation_status", "Curation"),
        ("augmentation_type", "Aug Type"),
        ("augment_label", "Weather Effect"),
        ("openai_augment_status", "Aug Status"),
        ("source_image_id", "Source ID"),
        ("is_synthetic", "Synthetic"),
        ("blur_score", "Blur Score"),
        ("brightness", "Brightness"),
        ("bbox_ratio", "Bbox Ratio"),
        ("model", "Model"),
        ("query_id", "Query"),
        ("graded_response", "VLM Verdict"),
        ("score", "Score"),
        ("correct", "Correct"),
        ("retrieved", "Retrieved"),
    ]
    for key, label in fields:
        val = row.get(key)
        if val is None or val == "" or val == "None" or val == "unknown":
            continue
        if isinstance(val, float) and (val != val):  # NaN
            continue
        if isinstance(val, list):
            val = ", ".join(str(v) for v in val if str(v) != "none")
            if not val:
                continue
        elif isinstance(val, float):
            val = f"{val:.2f}"
        else:
            val = str(val)
        lines.append(f"- {label}: `{val}`")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tab 1: Dataset Gallery
# ---------------------------------------------------------------------------

THUMB_SIZE = (200, 150)

# Cache for resolving image_id → image_path across datasets
_image_path_cache: dict[str, Path] = {}


def _resolve_image_path(image_id: str) -> Path | None:
    """Find the image path for an image_id by searching all datasets."""
    if image_id in _image_path_cache:
        return _image_path_cache[image_id]
    for loader in (load_curated, load_golden, load_augmented):
        df = loader()
        if df is not None:
            match = df[df["image_id"] == image_id]
            if len(match):
                p = match.iloc[0].get("image_path")
                if p:
                    path = Path(str(p))
                    if path.exists():
                        _image_path_cache[image_id] = path
                        return path
    return None


def _make_thumbnail(row: dict) -> Image.Image | None:
    """Render a thumbnail with bounding boxes for gallery display."""
    path = Path(str(row.get("image_path", "")))
    if not path.is_file():
        # Try resolving from other datasets
        resolved = _resolve_image_path(str(row.get("image_id", "")))
        if resolved:
            path = resolved
        else:
            return None
    try:
        img = Image.open(path).convert("RGB")
    except Exception:
        return None
    img = img.resize(THUMB_SIZE)

    bboxes = row.get("bbox")
    if bboxes and isinstance(bboxes, list) and len(bboxes) > 0:
        # Scale boxes to thumbnail size
        orig_w = float(row.get("width", img.size[0]) or img.size[0])
        orig_h = float(row.get("height", img.size[1]) or img.size[1])
        sx = THUMB_SIZE[0] / max(orig_w, 1)
        sy = THUMB_SIZE[1] / max(orig_h, 1)
        draw_img = img.copy()
        draw = ImageDraw.Draw(draw_img)
        for box in bboxes:
            if not box or len(box) < 4:
                continue
            xmin, ymin, xmax, ymax = float(box[0]) * sx, float(box[1]) * sy, float(box[2]) * sx, float(box[3]) * sy
            draw.rectangle([xmin, ymin, xmax, ymax], outline="lime", width=2)
        return draw_img
    return img


def _build_gallery():
    with gr.Blocks() as demo:
        gr.Markdown("## Dataset Gallery")

        with gr.Row():
            dataset_radio = gr.Radio(
                choices=["curated", "golden", "augmented", "eval"],
                value="curated",
                label="Dataset",
                interactive=True,
            )
            refresh_btn = gr.Button("↻  Refresh", variant="primary", scale=0)
        status_md = gr.Markdown("")

        with gr.Row():
            split_filter = gr.Dropdown(
                choices=[], value=None, label="Split", scale=1, allow_custom_value=True,
            )
            present_filter = gr.Dropdown(
                choices=[], value=None, label="Drone Present", scale=1, allow_custom_value=True,
            )
            vis_filter = gr.Dropdown(
                choices=[], value=None, label="Visibility", scale=1, allow_custom_value=True,
            )
            bg_filter = gr.Dropdown(
                choices=[], value=None, label="Background", scale=1, allow_custom_value=True,
            )

        with gr.Row():
            model_filter = gr.Dropdown(
                choices=[], value=None, label="Model", scale=1, allow_custom_value=True,
            )
            query_filter = gr.Dropdown(
                choices=[], value=None, label="Query", scale=1, allow_custom_value=True,
            )

        gallery = gr.Gallery(
            label="Images", columns=5, rows=3, height=520,
            object_fit="contain", allow_preview=True,
        )

        # --- Editable detail panel ---
        with gr.Group():
            gr.Markdown("### ✏️ Edit Labels")
            with gr.Row():
                edit_image_id = gr.Textbox(label="Image ID", interactive=False, scale=1)
                edit_split = gr.Textbox(label="Split", interactive=False, scale=1)
                edit_aug_type = gr.Textbox(label="Aug Type", interactive=False, scale=1)
                edit_aug_label = gr.Textbox(label="Weather Effect", scale=1)
            with gr.Row():
                edit_drone_present = gr.Checkbox(label="Drone Present", scale=1)
                edit_visibility = gr.Dropdown(
                    choices=["large", "medium", "small", ""], label="Visibility", scale=1, allow_custom_value=True,
                )
                edit_curation = gr.Dropdown(
                    choices=["keep", "needs_review", "reject"], label="Curation", scale=1, allow_custom_value=True,
                )
                edit_synthetic = gr.Checkbox(label="Synthetic", interactive=False, scale=1)
            with gr.Row():
                edit_background = gr.Dropdown(
                    choices=["sky", "trees", "building", "cluttered", "unknown"],
                    label="Background", scale=1, allow_custom_value=True,
                )
                edit_lighting = gr.Dropdown(
                    choices=["normal", "dark", "bright", "backlit", "unknown"],
                    label="Lighting", scale=1, allow_custom_value=True,
                )
                edit_blur = gr.Dropdown(
                    choices=["sharp", "mild", "blurry"], label="Blur", scale=1, allow_custom_value=True,
                )
            with gr.Row():
                edit_camera = gr.Dropdown(
                    choices=["top_down", "high_angle", "eye_level", "low_angle", "worms_eye", "unknown"],
                    label="Camera Angle", scale=1, allow_custom_value=True,
                )
                edit_depth = gr.Dropdown(
                    choices=["close_up", "mid_range", "landscape", "unknown"],
                    label="Depth", scale=1, allow_custom_value=True,
                )
                edit_confusers = gr.Textbox(label="Confusers (comma-separated)", scale=1)
            with gr.Row():
                edit_model = gr.Textbox(label="Model", interactive=False, scale=1)
                edit_query = gr.Textbox(label="Query", interactive=False, scale=1)
                edit_verdict = gr.Dropdown(
                    choices=["definitely yes", "probably yes", "uncertain", "probably no", "definitely no"],
                    label="VLM Verdict", scale=1, allow_custom_value=True,
                )
                edit_score = gr.Dropdown(
                    choices=["1", "2", "3", "4", "5"], label="Score", scale=1, allow_custom_value=True,
                )
            with gr.Row():
                save_btn = gr.Button("💾 Save Changes", variant="primary", scale=1)
                save_status = gr.Markdown("")

        edit_outputs = [
            edit_image_id, edit_split, edit_aug_type, edit_aug_label,
            edit_drone_present, edit_visibility, edit_curation, edit_synthetic,
            edit_background, edit_lighting, edit_blur,
            edit_camera, edit_depth, edit_confusers,
            edit_model, edit_query, edit_verdict, edit_score,
            save_status,
        ]

        # --- State ---
        df_state = gr.State()
        filtered_state = gr.State()
        edit_idx = gr.State(None)
        dataset_name_state = gr.State("curated")

        # --- Helpers ---
        def _apply_filters(df: pd.DataFrame, split_val, present_val, vis_val, bg_val, model_val, query_val):
            mask = pd.Series(True, index=df.index)
            if split_val and split_val != "All" and "source_split" in df.columns:
                mask &= df["source_split"].astype(str) == split_val
            if present_val and present_val != "All" and "drone_present" in df.columns:
                mask &= df["drone_present"].astype(str) == present_val
            if vis_val and vis_val != "All" and "drone_visibility" in df.columns:
                mask &= df["drone_visibility"].astype(str) == vis_val
            if bg_val and bg_val != "All" and "background" in df.columns:
                mask &= df["background"].astype(str) == bg_val
            if model_val and model_val != "All" and "model" in df.columns:
                mask &= df["model"].astype(str) == model_val
            if query_val and query_val != "All" and "query_id" in df.columns:
                mask &= df["query_id"].astype(str) == query_val
            return df[mask].reset_index(drop=True)

        def _build_filter_choices(df: pd.DataFrame):
            def _col_choices(col: str) -> list[str]:
                if col not in df.columns:
                    return []
                return [str(c) for c in sorted(df[col].dropna().unique())]

            split_choices = _col_choices("source_split")
            present_choices = _col_choices("drone_present")
            vis_choices = [
                v for v in _col_choices("drone_visibility")
                if v and v not in ("None", "none", "")
            ]
            bg_choices = [
                v for v in _col_choices("background")
                if v and v not in ("None", "none", "unknown", "")
            ]
            model_choices = _col_choices("model")
            query_choices = _col_choices("query_id")
            return split_choices, present_choices, vis_choices, bg_choices, model_choices, query_choices

        # --- Load / refresh ---
        def _load_dataset(dataset_name):
            global LAST_REFRESH, _image_path_cache
            LAST_REFRESH = datetime.now()
            _image_path_cache.clear()  # refresh resolution cache
            df, msg = load_dataset(dataset_name)
            empty_edits = [gr.update(value="") if i != 18 else "" for i in range(19)]  # 18 fields + 1 markdown
            if df is None:
                return (
                    df, df, [], msg,
                    gr.update(choices=[]), gr.update(choices=[]),
                    gr.update(choices=[]), gr.update(choices=[]),
                    gr.update(choices=[]), gr.update(choices=[]),
                    dataset_name, *empty_edits,
                )
            sc, pc, vc, bc, mc, qc = _build_filter_choices(df)
            filtered = _apply_filters(df, None, None, None, None, None, None)
            thumbs = [_make_thumbnail(r) for _, r in filtered.iterrows()]
            thumbs = [t for t in thumbs if t is not None]
            status = f"✅ {dataset_name} — {len(filtered)} shown of {len(df)} rows  ·  {LAST_REFRESH.strftime('%H:%M:%S')}"
            return (
                df, filtered, thumbs, status,
                gr.update(choices=sc, value=None),
                gr.update(choices=pc, value=None),
                gr.update(choices=vc, value=None),
                gr.update(choices=bc, value=None),
                gr.update(choices=mc, value=None),
                gr.update(choices=qc, value=None),
                dataset_name, *empty_edits,
            )

        load_outputs = [df_state, filtered_state, gallery, status_md,
                        split_filter, present_filter, vis_filter, bg_filter,
                        model_filter, query_filter,
                        dataset_name_state, *edit_outputs]

        refresh_btn.click(_load_dataset, inputs=[dataset_radio], outputs=load_outputs)
        dataset_radio.change(_load_dataset, inputs=[dataset_radio], outputs=load_outputs)

        # --- Filter changes → rebuild gallery ---
        def _refilter(df_dict, split_val, present_val, vis_val, bg_val, model_val, query_val):
            empty_edits = [gr.update(value="") if i != 18 else "" for i in range(19)]
            if df_dict is None:
                return [df_dict, [], "*No data loaded.*", *empty_edits]
            df = df_dict if isinstance(df_dict, pd.DataFrame) else pd.DataFrame(df_dict)
            filtered = _apply_filters(df, split_val, present_val, vis_val, bg_val, model_val, query_val)
            thumbs = [_make_thumbnail(r) for _, r in filtered.iterrows()]
            thumbs = [t for t in thumbs if t is not None]
            return [filtered, thumbs, f"{len(filtered)} of {len(df)} rows", *empty_edits]

        filter_inputs = [df_state, split_filter, present_filter, vis_filter, bg_filter, model_filter, query_filter]
        filter_outputs = [filtered_state, gallery, status_md, *edit_outputs]

        for filt in (split_filter, present_filter, vis_filter, bg_filter, model_filter, query_filter):
            filt.change(_refilter, inputs=filter_inputs, outputs=filter_outputs)

        # --- Gallery select → populate editable form ---
        def _on_select(filtered_dict, evt: gr.SelectData):
            empty = [gr.update(value="") if i != 18 else "*No data.*" for i in range(19)]
            if filtered_dict is None:
                return [evt.index, *empty]
            filtered = (filtered_dict if isinstance(filtered_dict, pd.DataFrame)
                        else pd.DataFrame(filtered_dict))
            idx = evt.index
            if idx < 0 or idx >= len(filtered):
                return [None, *empty]
            row = filtered.iloc[idx].to_dict()

            def _str(v):
                if v is None or v == "" or v == "None" or v == "unknown":
                    return ""
                if isinstance(v, float) and (v != v):  # NaN
                    return ""
                if isinstance(v, list):
                    return ", ".join(str(x) for x in v if str(x) != "none")
                if isinstance(v, bool):
                    return v
                return str(v)

            return [
                idx,
                gr.update(value=_str(row.get("image_id"))),
                gr.update(value=_str(row.get("source_split"))),
                gr.update(value=_str(row.get("augmentation_type"))),
                gr.update(value=_str(row.get("augment_label", ""))),
                gr.update(value=bool(row.get("drone_present", False))),
                gr.update(value=_str(row.get("drone_visibility"))),
                gr.update(value=_str(row.get("curation_status"))),
                gr.update(value=bool(row.get("is_synthetic", False))),
                gr.update(value=_str(row.get("background"))),
                gr.update(value=_str(row.get("lighting"))),
                gr.update(value=_str(row.get("blur_bucket"))),
                gr.update(value=_str(row.get("camera_angle"))),
                gr.update(value=_str(row.get("depth_range"))),
                gr.update(value=_str(row.get("possible_confusers"))),
                gr.update(value=_str(row.get("model"))),
                gr.update(value=_str(row.get("query_id"))),
                gr.update(value=_str(row.get("graded_response"))),
                gr.update(value=_str(row.get("score"))),
                f"Editing: **{row.get('image_id')}**  |  Box count: **{len(row.get('bbox') or [])}**",
            ]

        gallery.select(_on_select, inputs=[filtered_state], outputs=[edit_idx, *edit_outputs])

        # --- Save edits back to JSONL ---
        def _save_edits(
            dataset_name, edit_idx_val, filtered_dict,
            image_id, split, aug_type, aug_label,
            drone_present, visibility, curation, synthetic,
            background, lighting, blur, camera, depth, confusers_str,
            model, query, verdict, score,
        ):
            if dataset_name == "raw":
                return "⚠️  Raw dataset is read-only (loaded from data/raw/)."
            if edit_idx_val is None or filtered_dict is None:
                return "⚠️  No row selected."
            filtered = (filtered_dict if isinstance(filtered_dict, pd.DataFrame)
                        else pd.DataFrame(filtered_dict))
            idx = int(edit_idx_val)
            if idx < 0 or idx >= len(filtered):
                return "⚠️  Invalid row index."

            row = filtered.iloc[idx]
            actual_image_id = str(row.get("image_id", ""))

            # Build updated field dict
            confusers = [c.strip() for c in (confusers_str or "").split(",") if c.strip()]
            if not confusers:
                confusers = ["none"]

            updates = {
                "augment_label": aug_label if aug_label else None,
                "drone_present": drone_present if isinstance(drone_present, bool) else str(drone_present).lower() == "true",
                "drone_visibility": visibility if visibility else None,
                "curation_status": curation if curation else "keep",
                "background": background if background else "unknown",
                "lighting": lighting if lighting else "unknown",
                "blur_bucket": blur if blur else "sharp",
                "camera_angle": camera if camera else "unknown",
                "depth_range": depth if depth else "unknown",
                "possible_confusers": confusers,
                "graded_response": verdict if verdict else None,
                "score": int(score) if score and str(score).isdigit() else None,
            }

            # Determine which JSONL to write
            if dataset_name == "eval":
                jsonl_path = ARTIFACTS / "retrieval_eval_detailed.jsonl"
            else:
                jsonl_path = ARTIFACTS / f"{dataset_name}_dataset.jsonl"
            if not jsonl_path.exists():
                return f"⚠️  File not found: {jsonl_path.name}"

            # Read all rows, find & update
            rows = []
            found = False
            with open(jsonl_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    r = json.loads(line)
                    if str(r.get("image_id", "")) == actual_image_id:
                        r.update(updates)
                        found = True
                    rows.append(r)

            if not found:
                return f"⚠️  image_id {actual_image_id} not found in {jsonl_path.name}"

            with open(jsonl_path, "w") as f:
                for r in rows:
                    f.write(json.dumps(r, default=str) + "\n")

            return f"✅ Saved **{actual_image_id}** → {jsonl_path.name}"

        save_btn.click(
            _save_edits,
            inputs=[
                dataset_name_state, edit_idx, filtered_state,
                edit_image_id, edit_split, edit_aug_type, edit_aug_label,
                edit_drone_present, edit_visibility, edit_curation, edit_synthetic,
                edit_background, edit_lighting, edit_blur,
                edit_camera, edit_depth, edit_confusers,
                edit_model, edit_query, edit_verdict, edit_score,
            ],
            outputs=[save_status],
        )

    return demo


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Drone VLM Dataset Viewer") as app:
        gr.Markdown(
            "# 🚁 Drone VLM Golden Eval — Dataset Viewer\n"
            "Preview artifacts as the demo pipeline runs. Hit **Refresh** after each phase."
        )
        _build_gallery()
    return app


if __name__ == "__main__":
    app = build_app()
    app.queue(default_concurrency_limit=4)
    print(f"Viewer starting at http://{args.host}:{args.port}")
    print(f"Artifacts dir: {ARTIFACTS}")
    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        css=APP_CSS,
        theme=gr.themes.Soft(),
    )
