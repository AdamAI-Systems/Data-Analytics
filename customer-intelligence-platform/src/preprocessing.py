"""Cleaning and feature-engineering for transactional data.

The pipeline is split into small, independently testable steps so each can be
reused (e.g. the dashboard may call :func:`clean_transactions` to mirror exactly
what the training pipeline did):

    load_transactions  →  clean_transactions  →  engineer_features  →  save_processed

``preprocess`` is the convenience entry point that chains them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from .config import get_config
from .data_loader import load_transactions


def clean_transactions(
    df: pd.DataFrame,
    keep_returns: bool = False,
) -> pd.DataFrame:
    """Remove invalid / non-analytical rows from a raw transactional DataFrame.

    By default the following rows are dropped (typical for RFM / LTV / churn
    modeling):

    - Rows with a missing ``CustomerID`` (anonymous purchases).
    - Cancellations — invoices whose ``InvoiceNo`` starts with ``"C"``.
    - Returns / non-positive movements — ``Quantity <= 0`` or ``UnitPrice <= 0``.
    - Exact duplicate rows.

    Args:
        df: A DataFrame following the canonical schema (see
            :mod:`src.data_loader`).
        keep_returns: When ``True``, cancellations and non-positive
            quantities/prices are *kept*. Anonymous customers and duplicates
            are still removed. Useful for analyses that care about return
            behaviour.

    Returns:
        A cleaned copy of ``df``. Prints how many rows were removed at each
        step.
    """
    before = len(df)
    df = df.copy()

    # 1) Drop anonymous customers (no RFM/LTV/Churn modeling is possible).
    df = df.dropna(subset=["CustomerID"])
    print(f"   • dropped {before - len(df):,} rows with missing CustomerID")

    # 2) Optionally drop returns & cancellations.
    if not keep_returns:
        pre = len(df)

        is_cancelled = df["InvoiceNo"].astype(str).str.startswith(
            get_config()["schema"]["cancelled_invoice_prefix"]
        )
        non_positive = (df["Quantity"] <= 0) | (df["UnitPrice"] <= 0)
        df = df[~(is_cancelled | non_positive)]

        print(f"   • dropped {pre - len(df):,} returns / cancellations")

    # 3) Drop exact duplicates.
    pre = len(df)
    df = df.drop_duplicates()
    print(f"   • dropped {pre - len(df):,} duplicate rows")

    print(f"   ✓ clean: {before:,} → {len(df):,} rows")
    return df.reset_index(drop=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cast columns to correct dtypes and (re)compute ``TotalPrice``.

    Ensures:

    - ``InvoiceDate`` is ``datetime64``.
    - ``Quantity`` and ``UnitPrice`` are ``float``.
    - ``CustomerID`` is an integer-as-string (it is an identifier, not a
      measure).
    - ``TotalPrice = Quantity * UnitPrice``.

    Args:
        df: A (optionally cleaned) transactional DataFrame.

    Returns:
        A copy of ``df`` with corrected dtypes and a recomputed ``TotalPrice``.
    """
    df = df.copy()

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce")

    # CustomerID is an identifier — keep its digits but as a nullable Int so
    # non-integer sentinel values don't corrupt the column.
    df["CustomerID"] = pd.to_numeric(df["CustomerID"], errors="coerce").astype("Int64")

    # Re-engineer TotalPrice from the cleaned values.
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

    # Tidy text columns.
    for col in ("InvoiceNo", "StockCode", "Description", "Country"):
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip()

    return df


def save_processed(
    df: pd.DataFrame,
    path: Optional[Path | str] = None,
) -> Path:
    """Persist a cleaned DataFrame to a parquet file for fast reuse.

    Args:
        df: DataFrame to save.
        path: Destination path. Defaults to
            ``config.paths.processed_parquet``. The parent directory is
            created if needed.

    Returns:
        The absolute path the file was written to.
    """
    cfg = get_config()
    if path is None:
        path = cfg["paths"]["processed_parquet"]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"✅ Saved {len(df):,} rows → {path}")
    return path


def preprocess(
    path: Optional[Path | str] = None,
    keep_returns: bool = False,
    save: bool = True,
) -> pd.DataFrame:
    """Full pipeline: load → clean → feature-engineer → (optionally) save.

    Args:
        path: Path to raw data; forwarded to :func:`load_transactions`.
        keep_returns: Forwarded to :func:`clean_transactions`.
        save: When ``True`` (default) the result is also written to parquet
            via :func:`save_processed`.

    Returns:
        The cleaned, feature-engineered DataFrame.
    """
    print("📥 Loading transactions ...")
    df = load_transactions(path)

    print("🧹 Cleaning ...")
    df = clean_transactions(df, keep_returns=keep_returns)

    print("🔧 Engineering features ...")
    df = engineer_features(df)

    if save:
        save_processed(df)

    print(f"✓ Preprocessing done — {len(df):,} rows ready for modeling.")
    return df


if __name__ == "__main__":  # Run the full pipeline when executed directly.
    preprocess()
