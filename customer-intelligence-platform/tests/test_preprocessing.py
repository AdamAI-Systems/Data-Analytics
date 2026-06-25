"""Tests for cleaning + feature engineering (``src.preprocessing``).

These guard the exact cleaning rules the modelling pipeline depends on:
anonymous rows, cancellations, returns, and duplicates must all be removed by
default. The ``tiny_raw_df`` fixture is purpose-built to exercise each rule.
"""

import pandas as pd

from src.preprocessing import clean_transactions, engineer_features


# --------------------------------------------------------------------------- #
# clean_transactions
# --------------------------------------------------------------------------- #
def test_clean_drops_cancellations_and_returns_and_dupes(tiny_raw_df):
    clean = clean_transactions(tiny_raw_df, keep_returns=False)

    # 4 input rows: 1 valid + 1 cancellation + 1 negative-qty + 1 duplicate.
    # All but the first valid row should be removed.
    assert len(clean) == 1
    assert (clean["InvoiceNo"] == "570001").all()
    assert (clean["Quantity"] > 0).all()
    assert (clean["UnitPrice"] > 0).all()


def test_clean_keep_returns_keeps_non_positive_but_still_drops_anon_and_dupes(tiny_raw_df):
    # With keep_returns=True the cancellation/return rows survive; only the
    # exact duplicate (and anonymous rows — none here) go.
    clean = clean_transactions(tiny_raw_df, keep_returns=True)
    assert len(clean) == 3
    # No exact duplicates remain.
    assert clean.duplicated().sum() == 0


def test_clean_drops_missing_customer_id():
    df = pd.DataFrame(
        [
            {"InvoiceNo": "1", "Quantity": 1, "UnitPrice": 1.0, "CustomerID": 1.0},
            {"InvoiceNo": "2", "Quantity": 1, "UnitPrice": 1.0, "CustomerID": None},
        ]
    )
    clean = clean_transactions(df, keep_returns=False)
    assert len(clean) == 1
    assert clean["CustomerID"].iloc[0] == 1.0


def test_clean_resets_index(tiny_raw_df):
    clean = clean_transactions(tiny_raw_df, keep_returns=False)
    assert clean.index.equals(pd.RangeIndex(len(clean)))


def test_clean_does_not_mutate_input(tiny_raw_df):
    before = len(tiny_raw_df)
    clean_transactions(tiny_raw_df, keep_returns=False)
    assert len(tiny_raw_df) == before  # input untouched


# --------------------------------------------------------------------------- #
# engineer_features
# --------------------------------------------------------------------------- #
def test_engineer_features_dtypes_and_total_price(sample_df):
    feat = engineer_features(sample_df)

    assert pd.api.types.is_datetime64_any_dtype(feat["InvoiceDate"])
    assert pd.api.types.is_numeric_dtype(feat["Quantity"])
    assert pd.api.types.is_numeric_dtype(feat["UnitPrice"])
    # CustomerID kept as a nullable integer identifier.
    assert str(feat["CustomerID"].dtype) in {"Int64", "float64"}

    # TotalPrice must equal Quantity * UnitPrice for every row.
    assert ((feat["Quantity"] * feat["UnitPrice"]).round(4)
            == feat["TotalPrice"].round(4)).all()


def test_engineer_features_does_not_mutate_input(sample_df):
    before = sample_df["Quantity"].iloc[0]
    engineer_features(sample_df)
    assert sample_df["Quantity"].iloc[0] == before
