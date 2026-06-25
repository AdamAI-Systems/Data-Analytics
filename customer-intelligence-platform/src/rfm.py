"""RFM (Recency–Frequency–Monetary) scoring and K-Means segmentation.

Transforms a cleaned transactional DataFrame into per-customer RFM features,
quantile-scores each dimension (1–5), and clusters customers into five actionable
segments using K-Means on the scaled RFM matrix.

Typical usage::

    from src.preprocessing import preprocess
    from src.rfm import build_rfm_table, score_rfm, segment_customers

    df = preprocess()
    rfm = build_rfm_table(df)
    rfm = score_rfm(rfm)
    segments = segment_customers(rfm)   # adds 'Segment' column

Segment labels (high → low value):
    Champions, Loyal, Potential, At-Risk, Lost
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from .config import get_config

# --------------------------------------------------------------------------- #
# Segment labels — ordered from highest to lowest customer value.
# --------------------------------------------------------------------------- #
SEGMENT_LABELS: list[str] = ["Champions", "Loyal", "Potential", "At-Risk", "Lost"]

# Mapping used by ``assign_segment_labels``: cluster index → label name.
# Populated dynamically after K-Means runs (clusters are sorted by their
# mean RFM score so index 0 is always the best cluster).
_LABEL_MAP: dict[int, str] = {}


# --------------------------------------------------------------------------- #
# 1. Build the raw RFM table
# --------------------------------------------------------------------------- #
def build_rfm_table(
    df: pd.DataFrame,
    snapshot_date: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """Aggregate transactions to one row per customer with R, F, M values.

    Args:
        df: A *cleaned* transactional DataFrame (cancellations already
            removed). Must contain ``CustomerID``, ``InvoiceDate``, and
            ``TotalPrice``.
        snapshot_date: The reference date for computing **Recency** (days
            since the customer's last purchase). When ``None``, defaults to
            one day after the latest ``InvoiceDate`` in the dataset.

    Returns:
        A DataFrame indexed by ``CustomerID`` with three columns:
        ``Recency`` (days), ``Frequency`` (count of unique invoices),
        ``Monetary`` (sum of ``TotalPrice``).

    Example:
        >>> rfm = build_rfm_table(clean_df, snapshot_date=pd.Timestamp("2011-12-10"))
        >>> rfm.describe()
    """
    df = df.copy()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df = df.dropna(subset=["InvoiceDate", "CustomerID"])
    df["CustomerID"] = df["CustomerID"].astype(int)

    if snapshot_date is None:
        snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    rfm = (
        df.groupby("CustomerID")
        .agg(
            Recency=("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
            Frequency=("InvoiceNo", "nunique"),
            Monetary=("TotalPrice", "sum"),
        )
        .astype(float)
    )

    print(f"✅ RFM table built — {len(rfm):,} customers, snapshot: {snapshot_date.date()}")
    return rfm


# --------------------------------------------------------------------------- #
# 2. Quantile scoring (1–5)
# --------------------------------------------------------------------------- #
def score_rfm(
    rfm: pd.DataFrame,
    n_bins: int = 5,
) -> pd.DataFrame:
    """Assign quantile-based scores (1–5) to each RFM dimension.

    - **Recency**: lower is better → score 5 for the most recent customers.
    - **Frequency / Monetary**: higher is better → score 5 for the top tier.

    ``pd.qcut`` is used with ``duplicates='drop'`` so the function does not
    break when many customers share the same value.

    Args:
        rfm: Output of :func:`build_rfm_table`.
        n_bins: Number of quantile bins (default 5 → scores 1–5).

    Returns:
        The input DataFrame with three extra columns: ``R_Score``,
        ``F_Score``, ``M_Score``, and a composite ``RFM_Score``
        (concatenated string e.g. ``"555"``).
    """
    rfm = rfm.copy()

    def _qcut_series(series: pd.Series, ascending: bool = True) -> pd.Series:
        """Quantile-bin a series and return integer labels 1–n_bins."""
        labels = list(range(1, n_bins + 1))
        if ascending:
            labels = labels[::-1]  # Reverse so recent / high = 5
        try:
            return pd.qcut(series, q=n_bins, labels=labels, duplicates="drop").astype(int)
        except ValueError:
            # Fallback for very small datasets where qcut can't split.
            ranks = series.rank(pct=True)
            bins = pd.cut(ranks, bins=n_bins, labels=labels, include_lowest=True)
            return bins.astype(int)

    rfm["R_Score"] = _qcut_series(rfm["Recency"], ascending=True)
    rfm["F_Score"] = _qcut_series(rfm["Frequency"], ascending=False)
    rfm["M_Score"] = _qcut_series(rfm["Monetary"], ascending=False)
    rfm["RFM_Score"] = (
        rfm["R_Score"].astype(str)
        + rfm["F_Score"].astype(str)
        + rfm["M_Score"].astype(str)
    )

    print("✅ RFM quantile scores assigned (1–5 per dimension)")
    return rfm


# --------------------------------------------------------------------------- #
# 3. K-Means clustering → 5 segments
# --------------------------------------------------------------------------- #
def segment_customers(
    rfm: pd.DataFrame,
    n_clusters: int = 5,
    random_state: Optional[int] = None,
    n_init: int = 10,
) -> pd.DataFrame:
    """Run K-Means on scaled RFM features and label customers into segments.

    Steps:

    1. Scale ``Recency``, ``Frequency``, ``Monetary`` with
       :class:`~sklearn.preprocessing.StandardScaler`.
    2. Fit :class:`~sklearn.cluster.KMeans` with ``n_clusters``.
    3. Sort clusters by their mean composite RFM score (best → worst).
    4. Map sorted cluster indices to :data:`SEGMENT_LABELS`.

    Args:
        rfm: Output of :func:`score_rfm` (must include ``R_Score``,
            ``F_Score``, ``M_Score``).
        n_clusters: Number of K-Means clusters (default 5).
        random_state: RNG seed. Defaults to ``config.sampling.random_seed``.
        n_init: Number of K-Means initialisations (default 10).

    Returns:
        The input DataFrame with an additional ``Segment`` column
        containing one of :data:`SEGMENT_LABELS`.
    """
    if random_state is None:
        random_state = get_config()["sampling"]["random_seed"]

    features = rfm[["Recency", "Frequency", "Monetary"]].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init)
    rfm = rfm.copy()
    rfm["Cluster"] = km.fit_predict(X_scaled)

    # Sort clusters by mean composite score so label 0 = best.
    cluster_scores = (
        rfm.groupby("Cluster")[["R_Score", "F_Score", "M_Score"]]
        .mean()
        .sum(axis=1)
        .sort_values(ascending=False)
    )

    sorted_labels = SEGMENT_LABELS[: len(cluster_scores)]
    _LABEL_MAP.clear()
    _LABEL_MAP.update(
        {old: new for old, new in zip(cluster_scores.index, sorted_labels)}
    )

    rfm["Segment"] = rfm["Cluster"].map(_LABEL_MAP)

    print(f"✅ K-Means segmentation done — {n_clusters} clusters:")
    for label in sorted_labels:
        count = (rfm["Segment"] == label).sum()
        print(f"   {label:12s} : {count:,} customers")

    return rfm


# --------------------------------------------------------------------------- #
# 4. Convenience: full pipeline
# --------------------------------------------------------------------------- #
def rfm_pipeline(
    df: Optional[pd.DataFrame] = None,
    snapshot_date: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """End-to-end RFM: build → score → segment.

    Args:
        df: Cleaned transactions. When ``None``, calls
            :func:`src.preprocessing.preprocess` to load + clean.
        snapshot_date: Forwarded to :func:`build_rfm_table`.

    Returns:
        The segmented RFM DataFrame (``CustomerID`` index with R, F, M,
        scores, and ``Segment``).
    """
    if df is None:
        from .preprocessing import preprocess  # Lazy import to avoid cycles.

        df = preprocess(keep_returns=False, save=False)

    rfm = build_rfm_table(df, snapshot_date=snapshot_date)
    rfm = score_rfm(rfm)
    rfm = segment_customers(rfm)
    return rfm


if __name__ == "__main__":
    result = rfm_pipeline()
    print("\nPreview:")
    print(result.head(10))
