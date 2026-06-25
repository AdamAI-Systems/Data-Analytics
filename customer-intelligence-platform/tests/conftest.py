"""Shared pytest fixtures and path setup for the Customer Intelligence tests.

Importing ``src`` and ``scripts`` from the tests requires the project root on
``sys.path``; doing it here once keeps every test module clean. Fixtures build
small, deterministic DataFrames from the canonical schema so tests run fast
and offline (no UCI download, no model training on the full dataset).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

# Make ``src`` importable whether pytest is launched from the project root or
# from inside ``tests/``.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Also expose ``scripts`` so the sample generator can be imported.
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def sample_df() -> pd.DataFrame:
    """A deterministic ~500-row synthetic transactional DataFrame.

    Built from the same generator the dashboard uses (``make_sample_data``)
    but smaller so the test suite stays fast. Follows the canonical schema and
    includes ~5% cancellations/returns so :func:`clean_transactions` has rows
    to drop.
    """
    from make_sample_data import generate_sample

    return generate_sample(sample_size=500, seed=42, out_path=PROJECT_ROOT / ".tmp_test_sample.csv")


@pytest.fixture(scope="session")
def clean_df(sample_df) -> pd.DataFrame:
    """Cleaned + feature-engineered transactions (the modelling-ready form)."""
    from src.preprocessing import clean_transactions, engineer_features

    return engineer_features(clean_transactions(sample_df, keep_returns=False))


@pytest.fixture
def tiny_raw_df() -> pd.DataFrame:
    """A hand-built 4-row frame exercising every cleaning rule.

    Rows:
      1. valid purchase
      2. cancellation (InvoiceNo starts with "C")
      3. negative quantity (a return)
      4. exact duplicate of row 1
    """
    return pd.DataFrame(
        [
            {"InvoiceNo": "570001", "StockCode": "A", "Description": "x", "Quantity": 4,
             "InvoiceDate": "2011-01-02 09:00", "UnitPrice": 2.5, "CustomerID": 12346, "Country": "United Kingdom"},
            {"InvoiceNo": "C570002", "StockCode": "A", "Description": "x", "Quantity": -1,
             "InvoiceDate": "2011-01-03 09:00", "UnitPrice": 2.5, "CustomerID": 12346, "Country": "United Kingdom"},
            {"InvoiceNo": "570003", "StockCode": "B", "Description": "y", "Quantity": -2,
             "InvoiceDate": "2011-01-04 09:00", "UnitPrice": 3.0, "CustomerID": 12347, "Country": "Germany"},
            {"InvoiceNo": "570001", "StockCode": "A", "Description": "x", "Quantity": 4,
             "InvoiceDate": "2011-01-02 09:00", "UnitPrice": 2.5, "CustomerID": 12346, "Country": "United Kingdom"},
        ]
    )
