"""Lightweight, scriptable EDA helpers for Task 1.

Run from CLI:
    python -m scripts.eda --data data/insurance.csv --out reports/eda
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd

from scripts.data_processing import load_and_clean_data


def _ensure_out_dir(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def summarize_dataframe(df: pd.DataFrame) -> Dict[str, Path]:
    """Generate summary CSVs for basic profiling."""
    outputs: Dict[str, Path] = {}
    out_dir = _ensure_out_dir(Path("reports/eda"))

    summary_path = out_dir / "summary_stats.csv"
    df.describe(include="all").to_csv(summary_path)
    outputs["summary"] = summary_path

    missing_path = out_dir / "missing_values.csv"
    df.isna().sum().to_csv(missing_path)
    outputs["missing"] = missing_path

    corr_path = out_dir / "correlation.csv"
    df.select_dtypes(include="number").corr().to_csv(corr_path)
    outputs["correlation"] = corr_path

    return outputs


def plot_distributions(df: pd.DataFrame, out_dir: Path) -> Path:
    out_dir = _ensure_out_dir(out_dir)
    numeric_cols = list(df.select_dtypes(include="number").columns)
    if not numeric_cols:
        raise ValueError("No numeric columns available for distribution plots")

    n_cols = min(3, len(numeric_cols))
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for ax, col in zip(axes, numeric_cols):
        df[col].hist(ax=ax, bins=20, color="#4c72b0", edgecolor="white")
        ax.set_title(f"Distribution: {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Count")

    for ax in axes[len(numeric_cols) :]:
        ax.set_visible(False)

    plt.tight_layout()
    path = out_dir / "numeric_distributions.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_charges_relationships(df: pd.DataFrame, out_dir: Path) -> Path:
    out_dir = _ensure_out_dir(out_dir)
    if "charges" not in df.columns:
        raise ValueError("Column 'charges' not found for relationship plots")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    df.plot.scatter(x="bmi", y="charges", ax=axes[0], color="#dd8452", alpha=0.7)
    axes[0].set_title("Charges vs BMI")

    df.plot.scatter(x="age", y="charges", ax=axes[1], color="#55a868", alpha=0.7)
    axes[1].set_title("Charges vs Age")

    plt.tight_layout()
    path = out_dir / "charges_relationships.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def run_basic_eda(data_path: str, out_dir: str = "reports/eda") -> Dict[str, Path]:
    """Execute minimal, repeatable EDA and persist outputs.

    Returns a mapping of artifact labels to file paths.
    """
    df = load_and_clean_data(data_path)
    out_dir_path = Path(out_dir)

    artifacts: Dict[str, Path] = {}
    artifacts.update(summarize_dataframe(df))
    artifacts["distributions"] = plot_distributions(df, out_dir_path)
    artifacts["charges_relationships"] = plot_charges_relationships(df, out_dir_path)
    return artifacts


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run minimal EDA and save artifacts")
    parser.add_argument("--data", type=str, default="data/insurance.csv", help="Path to CSV data")
    parser.add_argument("--out", type=str, default="reports/eda", help="Directory to store EDA outputs")
    args = parser.parse_args()

    artifacts = run_basic_eda(args.data, args.out)
    for label, path in artifacts.items():
        print(f"{label}: {path}")
