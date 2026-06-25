"""Reusable plotting helpers for the Customer Intelligence Platform.

These thin wrappers keep notebooks and the Streamlit dashboard consistent and
DRY. They set a shared visual style and centralize figure persistence so the
EDA notebook (and later phases) can do ``save_fig(fig, "name")`` without
repeating boilerplate.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import get_config


# --------------------------------------------------------------------------- #
# Style
# --------------------------------------------------------------------------- #
def set_plot_style() -> None:
    """Apply a consistent seaborn/matplotlib style across the platform.

    Safe to call multiple times. Configures the default figure size, the
    ``whitegrid`` seaborn theme, and slightly larger fonts for readability in
    notebooks and exported PNGs.
    """
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "figure.figsize": (10, 5),
            "figure.dpi": 100,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
        }
    )


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def save_fig(
    fig: plt.Figure,
    name: str,
    subdir: str = "eda",
    figures_root: Optional[Path] = None,
) -> Path:
    """Persist ``fig`` to ``reports/figures/<subdir>/<name>.png`` and return path.

    The parent directories are created on demand. Calling this does *not* show
    the figure — the caller decides whether to ``plt.show()``.

    Args:
        fig: The matplotlib figure to save.
        name: File stem (no extension); e.g. ``"monthly_revenue"``.
        subdir: Sub-folder under ``reports/figures/`` (e.g. ``"eda"``).
        figures_root: Override the default ``reports/figures`` root. Defaults
            to ``<project_root>/reports/figures``.

    Returns:
        The absolute path the PNG was written to.
    """
    cfg = get_config()
    if figures_root is None:
        figures_root = cfg["paths"]["project_root"] / "reports" / "figures"
    out_dir = Path(figures_root) / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}.png"
    fig.savefig(out_path)
    return out_path


# --------------------------------------------------------------------------- #
# Reusable analytical plots
# --------------------------------------------------------------------------- #
def plot_distribution(
    df: pd.DataFrame,
    col: str,
    log: bool = False,
    title: Optional[str] = None,
) -> plt.Figure:
    """Plot a histogram + boxplot of a numeric column side by side.

    Args:
        df: Source DataFrame.
        col: Numeric column to visualize.
        log: When ``True``, apply a log1p transform (useful for heavily
            right-skewed columns like ``TotalPrice``).
        title: Optional plot title; defaults to the column name.

    Returns:
        The created :class:`matplotlib.figure.Figure`.
    """
    series = pd.to_numeric(df[col], errors="coerce").dropna()
    label = col + (" (log1p)" if log else "")
    if log:
        series = pd.Series(np.log1p(series.values), index=series.index)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.histplot(series, bins=50, kde=True, ax=axes[0], color="#4C72B0")
    axes[0].set_title(f"Distribution of {label}")
    axes[0].set_xlabel(label)

    sns.boxplot(x=series, ax=axes[1], color="#55A868")
    axes[1].set_title(f"Boxplot of {label}")
    axes[1].set_xlabel(label)

    fig.suptitle(title or col, y=1.02, fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_top_n(
    df: pd.DataFrame,
    by: str,
    n: int = 15,
    value_col: str = "TotalPrice",
    agg: str = "sum",
    title: Optional[str] = None,
    palette: str = "viridis",
) -> plt.Figure:
    """Horizontal bar chart of the top ``n`` groups by an aggregate.

    Args:
        df: Source DataFrame.
        by: Column to group by (e.g. ``"StockCode"``, ``"Country"``).
        n: Number of top groups to show.
        value_col: Column to aggregate (default ``"TotalPrice"``).
        agg: Aggregation function name (``"sum"``, ``"count"``, ``"mean"``).
        title: Optional chart title.
        palette: seaborn palette name.

    Returns:
        The created :class:`matplotlib.figure.Figure`.
    """
    grouped = df.groupby(by)[value_col].agg(agg).sort_values(ascending=False).head(n)
    fig, ax = plt.subplots(figsize=(10, max(4, n * 0.35)))
    # Map each bar to a gradient color from the named palette. Using a manual
    # color list avoids the seaborn>=0.13 "palette without hue" FutureWarning.
    cmap = sns.color_palette(palette, n_colors=len(grouped))
    sns.barplot(x=grouped.values, y=grouped.index.astype(str), ax=ax, hue=grouped.index.astype(str), palette=cmap, legend=False)
    ax.set_title(title or f"Top {n} by {agg}({value_col})")
    ax.set_xlabel(f"{agg.title()} of {value_col}")
    ax.set_ylabel(by)
    fig.tight_layout()
    return fig
