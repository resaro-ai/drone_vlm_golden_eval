"""Coverage and gap analysis for curated drone test cases."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class CoverageGap:
    dimension: str
    value: str
    count: int
    min_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "dimension": self.dimension,
            "value": self.value,
            "count": self.count,
            "min_count": self.min_count,
        }


def analyze_coverage(
    df: pd.DataFrame,
    dimensions: list[str],
    min_count: int = 5,
) -> tuple[dict[str, pd.DataFrame], list[CoverageGap]]:
    tables: dict[str, pd.DataFrame] = {}
    gaps: list[CoverageGap] = []

    if df.empty:
        return {}, []

    for dimension in dimensions:
        if dimension not in df.columns:
            continue

        # For list columns (possible_confusers), explode then count.
        # For empty lists, replace with ["none"] before exploding so those
        # rows aren't silently dropped.
        col = df[dimension]
        if isinstance(col.iloc[0], list):
            col = col.apply(lambda v: v if v else ["none"])
            counts = col.explode().fillna("unknown").astype(str).value_counts()
        else:
            if dimension == "drone_present":
                col = col.map({True: "yes", False: "no"}).fillna("uncertain")
            counts = col.fillna("unknown").astype(str).value_counts()

        counts_df = (
            counts.rename_axis(dimension)
            .reset_index(name="count")
        )
        counts_df["has_gap"] = counts_df["count"] < min_count
        tables[dimension] = counts_df

        for _, row in counts_df[counts_df["has_gap"]].iterrows():
            gaps.append(
                CoverageGap(
                    dimension=dimension,
                    value=str(row[dimension]),
                    count=int(row["count"]),
                    min_count=min_count,
                )
            )

    return tables, gaps


def crosstab(df: pd.DataFrame, left: str, right: str) -> pd.DataFrame:
    """Compute a crosstab between two columns, handling list columns."""
    if left not in df.columns or right not in df.columns or df.empty:
        return pd.DataFrame()

    left_col = df[left]
    right_col = df[right]
    if left == "drone_present":
        left_col = left_col.map({True: "yes", False: "no"}).fillna("uncertain")
    if right == "drone_present":
        right_col = right_col.map({True: "yes", False: "no"}).fillna("uncertain")

    # Handle list columns by joining
    if isinstance(left_col.iloc[0], list):
        left_col = left_col.apply(lambda v: ", ".join(v) if v else "none")
    if isinstance(right_col.iloc[0], list):
        right_col = right_col.apply(lambda v: ", ".join(v) if v else "none")

    return pd.crosstab(left_col.astype(str), right_col.astype(str), margins=True)


def compute_key_crosstabs(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Compute the key cross-tabs needed for the workshop.

    Returns:
        Dict mapping crosstab name to DataFrame.
    """
    tabs: dict[str, pd.DataFrame] = {}

    df_with_drone = df[df["drone_visibility"].notna()] if "drone_visibility" in df.columns else df
    tabs["drone_present_x_drone_visibility"] = crosstab(df_with_drone, "drone_present", "drone_visibility")
    tabs["drone_present_x_possible_confusers"] = crosstab(df, "drone_present", "possible_confusers")
    tabs["drone_present_x_camera_angle"] = crosstab(df, "drone_present", "camera_angle")
    tabs["drone_present_x_depth_range"] = crosstab(df, "drone_present", "depth_range")
    tabs["drone_visibility_x_lighting"] = crosstab(df_with_drone, "drone_visibility", "lighting")
    tabs["drone_visibility_x_blur_bucket"] = crosstab(df_with_drone, "drone_visibility", "blur_bucket")
    tabs["camera_angle_x_depth_range"] = crosstab(df, "camera_angle", "depth_range")

    return tabs


def export_coverage_artifacts(
    tables: dict[str, pd.DataFrame],
    gaps: list[CoverageGap],
    crosstabs: dict[str, pd.DataFrame],
    artifact_dir: Path,
) -> None:
    """Write all coverage artifacts to *artifact_dir*."""
    # Per-dimension CSVs
    for name, table in tables.items():
        table.to_csv(artifact_dir / f"coverage_{name}.csv", index=False)

    # Gaps CSV
    pd.DataFrame([gap.to_dict() for gap in gaps]).to_csv(
        artifact_dir / "coverage_gaps.csv", index=False
    )

    # Cross-tab CSVs
    for name, table in crosstabs.items():
        table.to_csv(artifact_dir / f"coverage_crosstab_{name}.csv", index=True)
