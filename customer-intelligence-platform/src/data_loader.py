"""Raw-data loading and schema normalization for the Customer Intelligence Platform.

This module is the *only* place that knows how to read the original UCI
"Online Retail II" file (two Excel sheets, non-standard column names) and turn
it into a DataFrame that follows the canonical schema defined in
:mod:`src.config`. Every downstream module (preprocessing, RFM, LTV, churn)
should consume the output of :func:`load_transactions`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from .config import CANONICAL_COLUMNS, get_config


def normalize_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Rename UCI columns to the canonical schema and ensure ``TotalPrice``.

    The raw file exposes ``Invoice`` / ``Price`` / ``Customer ID`` (note the
    space). This function maps them to ``InvoiceNo`` / ``UnitPrice`` /
    ``CustomerID`` and adds the engineered ``TotalPrice`` column so the result
    matches :data:`src.config.CANONICAL_COLUMNS`.

    Args:
        df: A raw DataFrame, possibly with UCI-style column names.

    Returns:
        A copy of ``df`` with canonical column names and a ``TotalPrice``
        column. Columns already in canonical form are left untouched; any
        canonical column absent from the input is created and filled with
        ``NaN``.
    """
    cfg = get_config()
    rename_map = {k: v for k, v in cfg["schema"]["rename_map"].items() if k in df.columns}
    df = df.rename(columns=rename_map).copy()

    # Engineer TotalPrice if it is not already present.
    if "TotalPrice" not in df.columns:
        qty = pd.to_numeric(df.get("Quantity"), errors="coerce")
        price = pd.to_numeric(df.get("UnitPrice"), errors="coerce")
        df["TotalPrice"] = qty * price

    # Guarantee every canonical column exists (fill missing ones with NaN).
    for col in CANONICAL_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA

    return df[CANONICAL_COLUMNS].copy()


def load_raw(path: Optional[Path | str] = None) -> pd.DataFrame:
    """Read the raw Online Retail II file (``.xlsx`` or ``.csv``).

    For the ``.xlsx`` distribution the data is split across two sheets
    ("Year 2009-2010" and "Year 2010-2011"); both are concatenated vertically.
    No cleaning is performed — this returns the data exactly as stored on disk.

    Args:
        path: Path to ``online_retail_II.xlsx`` or a ``.csv`` export. When
            ``None``, falls back to ``config.paths.raw_xlsx`` then
            ``config.paths.sample_csv``.

    Returns:
        The raw, unnormalized DataFrame (column names as found on disk).

    Raises:
        FileNotFoundError: If neither ``path`` nor the default locations exist.
    """
    cfg = get_config()
    if path is None:
        raw_xlsx = cfg["paths"]["raw_xlsx"]
        sample_csv = cfg["paths"]["sample_csv"]
        path = raw_xlsx if raw_xlsx.exists() else sample_csv

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No raw data found at '{path}'. Run "
            f"'python scripts/download_data.py' or "
            f"'python scripts/make_sample_data.py' first."
        )

    suffix = path.suffix.lower()
    if suffix == ".xlsx":
        sheets = cfg["uci"]["sheets"]
        # Read every known sheet; ignore sheets that do not exist so the loader
        # still works on single-sheet exports.
        frames = [
            pd.read_excel(path, sheet_name=sheet)
            for sheet in sheets
            if sheet in pd.ExcelFile(path).sheet_names
        ]
        if not frames:  # Fallback: read all sheets available.
            frames = list(pd.read_excel(path, sheet_name=None).values())
        df = pd.concat(frames, ignore_index=True)
    elif suffix in {".csv"}:
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file type '{suffix}'. Use .xlsx or .csv.")

    print(f"✅ Loaded raw data: {len(df):,} rows × {len(df.columns)} cols from '{path.name}'")
    return df


def load_transactions(path: Optional[Path | str] = None) -> pd.DataFrame:
    """Load raw data and normalize it to the canonical schema.

    This is the main entry point for the rest of the platform. It chains
    :func:`load_raw` and :func:`normalize_schema` and prints a short summary
    (row count, date range, number of customers) so callers get quick feedback.

    Args:
        path: Optional explicit path; see :func:`load_raw`.

    Returns:
        A DataFrame with exactly the columns in
        :data:`src.config.CANONICAL_COLUMNS` and a ``TotalPrice`` column.
    """
    raw = load_raw(path)
    df = normalize_schema(raw)

    print("— Normalized to canonical schema —")
    print(f"   rows      : {len(df):,}")
    if "CustomerID" in df.columns and df["CustomerID"].notna().any():
        n_customers = df["CustomerID"].nunique()
        print(f"   customers : {n_customers:,}")
    if "InvoiceDate" in df.columns:
        dates = pd.to_datetime(df["InvoiceDate"], errors="coerce").dropna()
        if not dates.empty:
            print(f"   date range: {dates.min().date()} → {dates.max().date()}")
    return df


if __name__ == "__main__":  # Quick smoke test when run directly.
    df = load_transactions()
    print(df.head())
