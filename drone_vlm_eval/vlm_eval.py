"""VLM retrieval evaluation: query construction, graded scoring, and precision/recall metrics."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from drone_vlm_eval.config import DemoConfig
from drone_vlm_eval.image_io import image_to_base64 as _image_to_base64


# ---------------------------------------------------------------------------
# Graded response constants
# ---------------------------------------------------------------------------

GRADED_RESPONSES = [
    "definitely yes",
    "probably yes",
    "uncertain",
    "probably no",
    "definitely no",
]

SCORE_MAP: dict[str, int] = {
    "definitely yes": 5,
    "probably yes": 4,
    "uncertain": 3,
    "probably no": 2,
    "definitely no": 1,
}

RETRIEVAL_THRESHOLD = 4  # score >= 4 (probably yes or above) counts as retrieved

RETRIEVAL_PROMPT = (
    "You are reviewing drone surveillance footage for a perimeter security team.\n"
    "\n"
    'Query: "{query}"\n'
    "\n"
    "Does this image match the query? Respond with exactly one of the following options:\n"
    "- definitely yes\n"
    "- probably yes\n"
    "- uncertain\n"
    "- probably no\n"
    "- definitely no\n"
    "\n"
    "Respond with only your chosen option, nothing else."
)


# ---------------------------------------------------------------------------
# Query templates
# ---------------------------------------------------------------------------
# Each entry: (query_id, natural language query, positive filter function)
# Filters are applied to the curated DataFrame. Templates are tried in order;
# the first n_queries with >= n_samples positives AND >= n_samples negatives are used.

_CANDIDATE_TEMPLATES: list[tuple[str, str, Any]] = [
    (
        "q_drone_present",
        "footage containing a drone",
        lambda df: df["drone_present"] == True,  # noqa: E712
    ),
    (
        "q_dark_lighting",
        "drone footage in dark or low-light conditions",
        lambda df: (df["drone_present"] == True) & (df["lighting"] == "dark"),  # noqa: E712
    ),
    (
        "q_sky_background",
        "a drone flying against a clear sky background",
        lambda df: (df["drone_present"] == True) & (df["background"] == "sky"),  # noqa: E712
    ),
    (
        "q_small_drone",
        "footage containing a small or distant drone",
        lambda df: (df["drone_present"] == True) & (df["drone_visibility"] == "small"),  # noqa: E712
    ),
    (
        "q_building_background",
        "a drone filmed near or against a building",
        lambda df: (df["drone_present"] == True) & (df["background"] == "building"),  # noqa: E712
    ),
    (
        "q_blurry",
        "blurry or motion-affected drone footage",
        lambda df: (df["drone_present"] == True) & (df["blur_bucket"] == "blurry"),  # noqa: E712
    ),
    (
        "q_no_drone",
        "footage with no drone present",
        lambda df: df["drone_present"] == False,  # noqa: E712
    ),
    (
        "q_dark_sky",
        "a drone at night or in dark conditions against a sky background",
        lambda df: (df["drone_present"] == True) & (df["lighting"] == "dark") & (df["background"] == "sky"),  # noqa: E712
    ),
    (
        "q_bright_lighting",
        "drone footage in bright or overexposed lighting conditions",
        lambda df: (df["drone_present"] == True) & (df["lighting"] == "bright"),  # noqa: E712
    ),
    (
        "q_small_dark",
        "a small or distant drone in dark or low-light conditions",
        lambda df: (df["drone_present"] == True) & (df["drone_visibility"] == "small") & (df["lighting"] == "dark"),  # noqa: E712
    ),
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class QuerySpec:
    """A single evaluation query with its sampled positive and negative frames."""
    query_id: str
    query_text: str
    positive_ids: list[str]
    negative_ids: list[str]

    @property
    def all_ids(self) -> list[str]:
        return self.positive_ids + self.negative_ids


@dataclass
class FrameResult:
    """Graded relevance result for a single (query, frame) pair."""
    query_id: str
    image_id: str
    is_positive: bool
    raw_response: str
    graded_response: str
    score: int
    retrieved: bool


@dataclass
class QueryMetrics:
    """Precision, recall, and F1 for a single query."""
    query_id: str
    query_text: str
    n_positive: int
    n_negative: int
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int
    precision: float
    recall: float
    f1: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "query_text": self.query_text,
            "n_positive": self.n_positive,
            "n_negative": self.n_negative,
            "tp": self.true_positives,
            "fp": self.false_positives,
            "fn": self.false_negatives,
            "tn": self.true_negatives,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
        }


# ---------------------------------------------------------------------------
# Query set builder
# ---------------------------------------------------------------------------

class QuerySetBuilder:
    """Selects evaluation queries from candidate templates based on dataset coverage."""

    def __init__(
        self,
        n_queries: int = 5,
        n_samples: int = 5,
        random_state: int = 42,
    ) -> None:
        self.n_queries = n_queries
        self.n_samples = n_samples
        self.random_state = random_state

    def build(self, df: pd.DataFrame) -> list[QuerySpec]:
        """Return up to n_queries QuerySpecs, each with n_samples positives and negatives.

        Templates are tried in order. A template is skipped if the dataset has
        fewer than n_samples positives or negatives for it.
        """
        specs: list[QuerySpec] = []

        for query_id, query_text, pos_filter in _CANDIDATE_TEMPLATES:
            if len(specs) >= self.n_queries:
                break

            try:
                pos_mask = pos_filter(df)
            except KeyError:
                continue

            positives = df[pos_mask]
            negatives = df[~pos_mask]

            if len(positives) < self.n_samples or len(negatives) < self.n_samples:
                continue

            pos_sample = positives.sample(n=self.n_samples, random_state=self.random_state)
            neg_sample = negatives.sample(n=self.n_samples, random_state=self.random_state)

            specs.append(QuerySpec(
                query_id=query_id,
                query_text=query_text,
                positive_ids=pos_sample["image_id"].tolist(),
                negative_ids=neg_sample["image_id"].tolist(),
            ))

        return specs


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def _parse_graded_response(raw: str) -> tuple[str, int]:
    """Extract the graded response token and its score from raw VLM output.

    Matches longest option first to avoid "yes" matching inside "definitely yes".
    Falls back to "uncertain" if no token is found.
    """
    normalised = raw.strip().lower()
    for response in sorted(GRADED_RESPONSES, key=len, reverse=True):
        if response in normalised:
            return response, SCORE_MAP[response]
    return "uncertain", SCORE_MAP["uncertain"]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _compute_metrics(spec: QuerySpec, results: list[FrameResult]) -> QueryMetrics:
    tp = sum(1 for r in results if r.is_positive and r.retrieved)
    fp = sum(1 for r in results if not r.is_positive and r.retrieved)
    fn = sum(1 for r in results if r.is_positive and not r.retrieved)
    tn = sum(1 for r in results if not r.is_positive and not r.retrieved)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return QueryMetrics(
        query_id=spec.query_id,
        query_text=spec.query_text,
        n_positive=len(spec.positive_ids),
        n_negative=len(spec.negative_ids),
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        true_negatives=tn,
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_retrieval_evaluation(
    df: pd.DataFrame,
    connector: Any,
    cfg: DemoConfig,
    query_specs: list[QuerySpec] | None = None,
    run_label: str | None = None,
) -> tuple[list[FrameResult], list[QueryMetrics], dict[str, Any]]:
    """Run VLM retrieval evaluation over a query set.

    Args:
        df: Golden dataset DataFrame with image_path and metadata columns.
        connector: VLMConnector instance.
        cfg: DemoConfig with artifact_dir.
        query_specs: Pre-built query specs. Built automatically from df if None.
        run_label: Optional string (e.g. model name) used to suffix artifact filenames
            so multiple runs don't overwrite each other.

    Returns:
        (all_frame_results, per_query_metrics, summary_dict)
    """
    if query_specs is None:
        query_specs = QuerySetBuilder().build(df)

    if not query_specs:
        print("No queries could be built — check that curation metadata columns are present.")
        return [], [], {}

    id_to_row: dict[str, Any] = {str(row["image_id"]): row for _, row in df.iterrows()}
    all_results: list[FrameResult] = []
    all_metrics: list[QueryMetrics] = []

    for qs in query_specs:
        print(f'\n[Query] {qs.query_id}: "{qs.query_text}"')
        print(f"  positives={len(qs.positive_ids)}  negatives={len(qs.negative_ids)}")

        prompt = RETRIEVAL_PROMPT.format(query=qs.query_text)

        frame_list = (
            [(iid, True) for iid in qs.positive_ids]
            + [(iid, False) for iid in qs.negative_ids]
        )

        def _score_frame(item: tuple[str, bool]) -> FrameResult | None:
            image_id, is_positive = item
            row = id_to_row.get(image_id)
            if row is None:
                print(f"  ! {image_id}: not in dataset, skipping")
                return None

            image_path = Path(str(row["image_path"]))
            if not image_path.exists():
                print(f"  ! {image_id}: image file missing, skipping")
                return None

            raw_response = ""
            try:
                image_b64 = _image_to_base64(image_path)
                raw_response = connector.call(image_b64, prompt)
            except Exception as exc:
                print(f"  x {image_id}: API error — {exc}")

            graded, score = _parse_graded_response(raw_response)
            retrieved = score >= RETRIEVAL_THRESHOLD
            return FrameResult(
                query_id=qs.query_id,
                image_id=image_id,
                is_positive=is_positive,
                raw_response=raw_response,
                graded_response=graded,
                score=score,
                retrieved=retrieved,
            )

        max_workers = getattr(connector, "max_workers", 1)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            scored = list(executor.map(_score_frame, frame_list))

        query_results = [r for r in scored if r is not None]

        for r in query_results:
            label = "pos" if r.is_positive else "neg"
            icon = "✓" if r.retrieved == r.is_positive else "✗"
            print(f"  {icon} [{label}] {r.image_id}: {r.graded_response!r} → {'retrieved' if r.retrieved else 'not retrieved'}")

        metrics = _compute_metrics(qs, query_results)
        all_results.extend(query_results)
        all_metrics.append(metrics)

        print(
            f"  → Precision: {metrics.precision:.0%}  "
            f"Recall: {metrics.recall:.0%}  "
            f"F1: {metrics.f1:.0%}  "
            f"(TP={metrics.true_positives} FP={metrics.false_positives} "
            f"FN={metrics.false_negatives} TN={metrics.true_negatives})"
        )

    # --- Artifacts ---
    suffix = f"_{run_label}" if run_label else ""
    results_path = cfg.artifact_dir / f"retrieval_results{suffix}.jsonl"
    with results_path.open("w") as f:
        for r in all_results:
            f.write(json.dumps({
                "query_id": r.query_id,
                "image_id": r.image_id,
                "is_positive": r.is_positive,
                "graded_response": r.graded_response,
                "score": r.score,
                "retrieved": r.retrieved,
            }) + "\n")

    n = len(all_metrics)
    summary: dict[str, Any] = {
        "model": run_label,
        "retrieval_threshold": RETRIEVAL_THRESHOLD,
        "n_queries": n,
        "mean_precision": round(sum(m.precision for m in all_metrics) / n, 4),
        "mean_recall": round(sum(m.recall for m in all_metrics) / n, 4),
        "mean_f1": round(sum(m.f1 for m in all_metrics) / n, 4),
        "per_query": [m.to_dict() for m in all_metrics],
    }

    sum_path = cfg.artifact_dir / f"retrieval_eval_summary{suffix}.json"
    sum_path.write_text(json.dumps(summary, indent=2))

    # --- Summary table ---
    title = f"VLM Retrieval Evaluation — {run_label}" if run_label else "VLM Retrieval Evaluation"
    print("\n" + "=" * 68)
    print(title)
    print("=" * 68)
    header = f"{'Query':<38}  {'P':>6}  {'R':>6}  {'F1':>6}"
    print(header)
    print("-" * 68)
    for m in all_metrics:
        label = (m.query_text[:36] + "..") if len(m.query_text) > 38 else m.query_text
        print(f"{label:<38}  {m.precision:>6.0%}  {m.recall:>6.0%}  {m.f1:>6.0%}")
    print("-" * 68)
    print(
        f"{'Mean':<38}  {summary['mean_precision']:>6.0%}  "
        f"{summary['mean_recall']:>6.0%}  {summary['mean_f1']:>6.0%}"
    )
    print(f"\nArtifacts → {results_path.name}, {sum_path.name}")

    return all_results, all_metrics, summary
