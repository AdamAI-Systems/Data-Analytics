"""Tests for schema normalization (``src.data_loader``).

These focus on ``normalize_schema`` — the function that turns UCI-style raw
columns into the canonical form every downstream module relies on. ``load_raw``
/ ``load_transactions`` are integration tests (they touch the filesystem /
network) and are covered by the dashboard's runtime instead.
"""

import pandas as pd

from src.config import CANONICAL_COLUMNS
from src.data_loader import normalize_schema


def test_rename_uci_columns_to_canonical():
    raw = pd.DataFrame(
        [
            {"Invoice": "1", "Price": 2.0, "Customer ID": 5,
             "StockCode": "A", "Description": "x", "Quantity": 3,
             "InvoiceDate": "2011-01-01", "Country": "UK"},
        ]
    )
    out = normalize_schema(raw)
    assert set(out.columns) == set(CANONICAL_COLUMNS)
    assert "InvoiceNo" in out.columns and "Invoice" not in out.columns
    assert "UnitPrice" in out.columns and "Price" not in out.columns
    assert "CustomerID" in out.columns and "Customer ID" not in out.columns


def test_engineers_total_price_when_missing():
    raw = pd.DataFrame(
        [{"InvoiceNo": "1", "Quantity": 4, "UnitPrice": 2.5}]
    )
    out = normalize_schema(raw)
    assert "TotalPrice" in out.columns
    assert out["TotalPrice"].iloc[0] == 10.0


def test_keeps_existing_total_price():
    raw = pd.DataFrame(
        [{"InvoiceNo": "1", "Quantity": 2, "UnitPrice": 3.0, "TotalPrice": 999.0}]
    )
    out = normalize_schema(raw)
    # An explicit TotalPrice is preserved (not recomputed/overwritten).
    assert out["TotalPrice"].iloc[0] == 999.0


def test_missing_canonical_columns_filled_with_na():
    raw = pd.DataFrame([{"InvoiceNo": "1", "Quantity": 1}])
    out = normalize_schema(raw)
    for col in CANONICAL_COLUMNS:
        assert col in out.columns
    # Columns absent from the input should be NaN, not raise.
    assert pd.isna(out["CustomerID"].iloc[0])


def test_does_not_mutate_input():
    raw = pd.DataFrame([{"InvoiceNo": "1", "Quantity": 1, "UnitPrice": 1.0}])
    original_cols = list(raw.columns)
    normalize_schema(raw)
    assert list(raw.columns) == original_cols  # input untouched
