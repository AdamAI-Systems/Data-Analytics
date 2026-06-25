"""Central configuration for the Customer Intelligence Platform.

All tunable constants (paths, schema mappings, seeds) live here so that the
data-loader, preprocessing, scripts and dashboard read from a single source of
truth — mirroring the ``get_config()`` pattern used across the sibling projects
in this repository (e.g. ``Archaeological_Landmark_Classifier``).
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Project layout
# --------------------------------------------------------------------------- #
# Resolve paths relative to the project root (the folder that contains src/).
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# --------------------------------------------------------------------------- #
# Canonical transaction schema
# --------------------------------------------------------------------------- #
# The cleaned, in-memory DataFrame always exposes these nine columns. The raw
# UCI "Online Retail II" file uses slightly different names (see RENAME_MAP
# below); ``data_loader.normalize_schema`` converts them to this canonical form
# so every downstream module (RFM / LTV / Churn) can rely on stable names.
CANONICAL_COLUMNS: list[str] = [
    "InvoiceNo",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "UnitPrice",
    "CustomerID",
    "Country",
    "TotalPrice",
]

# Map raw UCI column names -> canonical names. Keys are the names found in the
# original .xlsx file; values are the canonical names every module expects.
RENAME_MAP: dict[str, str] = {
    "Invoice": "InvoiceNo",
    "Price": "UnitPrice",
    "Customer ID": "CustomerID",
}

# A cancellation is signalled by an Invoice number that starts with "C".
CANCELLED_INVOICE_PREFIX: str = "C"


def get_config() -> dict:
    """Return the platform's central configuration dictionary.

    Returns:
        A nested dictionary with the following top-level keys:

        - ``paths``: absolute :class:`~pathlib.Path` objects for every
          important file/folder (raw data, processed data, sample data,
          models, project root).
        - ``schema``: canonical column list and the raw->canonical rename map.
        - ``uci``: download URL + Excel sheet names for the source dataset.
        - ``preprocessing``: tunables shared by ``preprocessing.clean_*``.
        - ``sampling``: random seed and sample size for ``make_sample_data``.

    Example:
        >>> cfg = get_config()
        >>> cfg["sampling"]["random_seed"]
        42
    """
    return {
        "paths": {
            "project_root": PROJECT_ROOT,
            "data_dir": PROJECT_ROOT / "data",
            "raw_dir": PROJECT_ROOT / "data" / "raw",
            "processed_dir": PROJECT_ROOT / "data" / "processed",
            "raw_xlsx": PROJECT_ROOT / "data" / "raw" / "online_retail_II.xlsx",
            "processed_parquet": PROJECT_ROOT
            / "data"
            / "processed"
            / "transactions.parquet",
            "sample_csv": PROJECT_ROOT / "data" / "sample_data.csv",
            "models_dir": PROJECT_ROOT / "models",
        },
        "schema": {
            "canonical_columns": CANONICAL_COLUMNS,
            "rename_map": RENAME_MAP,
            "cancelled_invoice_prefix": CANCELLED_INVOICE_PREFIX,
        },
        "uci": {
            # Official UCI ML Repository dataset page (id 502). The .xlsx is
            # served from this domain; download_data.py resolves the file URL.
            "dataset_url": "https://archive.ics.uci.edu/dataset/502/online+retail+ii",
            "file_url": "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip",
            "sheets": ["Year 2009-2010", "Year 2010-2011"],
        },
        "preprocessing": {
            "keep_returns": False,
        },
        "sampling": {
            "random_seed": 42,
            "sample_size": 5000,
        },
    }
