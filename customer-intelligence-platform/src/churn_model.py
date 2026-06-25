"""Churn prediction with XGBoost and SHAP explainability.

A customer is labeled as **churned** if they have not purchased within a
defined inactivity window (default 90 days) before the end of the observation
period. Per-customer features are engineered from the transaction history (RFM
metrics + behavioural aggregates) and fed into an XGBoost binary classifier.

SHAP values are computed post-hoc so every prediction is interpretable.

Typical usage::

    from src.churn_model import build_churn_dataset, train_churn_model

    X, y = build_churn_dataset(clean_df)
    model, metrics = train_churn_model(X, y)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import train_test_split

from .config import get_config

# --------------------------------------------------------------------------- #
# 1. Feature engineering from transactions
# --------------------------------------------------------------------------- #
def build_churn_features(
    df: pd.DataFrame,
    snapshot_date: Optional[pd.Timestamp] = None,
    churn_days: int = 90,
) -> pd.DataFrame:
    """Engineer per-customer features for churn prediction.

    Computes RFM metrics plus behavioural features (standard deviation of
    inter-purchase days, basket diversity, etc.).

    Args:
        df: Cleaned transactions (no returns, no anonymous customers).
        snapshot_date: Observation cut-off. Defaults to ``df.InvoiceDate.max()``.
        churn_days: A customer is **churned** if their last purchase is more
            than ``churn_days`` before ``snapshot_date``.

    Returns:
        A DataFrame indexed by ``CustomerID`` with feature columns and a
        boolean ``churn`` target column.
    """
    df = df.copy()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df = df.dropna(subset=["InvoiceDate", "CustomerID"])
    df["CustomerID"] = df["CustomerID"].astype(int)

    if snapshot_date is None:
        snapshot_date = df["InvoiceDate"].max()

    # --- Core RFM --------------------------------------------------------- #
    agg = df.groupby("CustomerID").agg(
        recency=("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
        frequency=("InvoiceNo", "nunique"),
        monetary=("TotalPrice", "sum"),
        avg_basket_size=("Quantity", "mean"),
        avg_basket_value=("TotalPrice", "mean"),
        std_basket_value=("TotalPrice", "std"),
        n_products=("StockCode", "nunique"),
        n_invoices=("InvoiceNo", "count"),  # total line items
        first_purchase=("InvoiceDate", "min"),
        last_purchase=("InvoiceDate", "max"),
    )

    # --- Derived features ------------------------------------------------- #
    agg["tenure_days"] = (snapshot_date - agg["first_purchase"]).dt.days
    agg["recency_ratio"] = agg["recency"] / agg["tenure_days"].clip(lower=1)
    agg["products_per_invoice"] = agg["n_products"] / agg["frequency"].clip(lower=1)
    agg["avg_days_between"] = agg["tenure_days"] / agg["frequency"].clip(lower=1)
    agg["monetary_per_product"] = agg["monetary"] / agg["n_products"].clip(lower=1)

    # Fill NaNs (e.g. std_basket_value for single-purchase customers).
    agg = agg.fillna(0)

    # --- Churn label ------------------------------------------------------ #
    agg["churn"] = (agg["recency"] > churn_days).astype(int)

    # Drop datetime columns (not useful for XGBoost).
    agg = agg.drop(columns=["first_purchase", "last_purchase"], errors="ignore")

    churn_pct = agg["churn"].mean() * 100
    print(f"✅ Churn features built — {len(agg):,} customers, churn rate: {churn_pct:.1f}%")
    return agg


# --------------------------------------------------------------------------- #
# 2. Train / evaluate
# --------------------------------------------------------------------------- #
def train_churn_model(
    features: Optional[pd.DataFrame] = None,
    df: Optional[pd.DataFrame] = None,
    test_size: float = 0.2,
    random_state: Optional[int] = None,
    churn_days: int = 90,
) -> tuple[xgb.XGBClassifier, dict, pd.DataFrame]:
    """Train an XGBoost churn classifier and return metrics + SHAP values.

    Args:
        features: Output of :func:`build_churn_features`. When ``None``,
            built from ``df``.
        df: Cleaned transactions (used only if ``features`` is ``None``).
        test_size: Hold-out fraction for evaluation.
        random_state: RNG seed. Defaults to ``config.sampling.random_seed``.
        churn_days: Forwarded to :func:`build_churn_features`.

    Returns:
        A tuple of (model, metrics_dict, shap_df) where ``metrics_dict``
        contains ``auc_roc``, ``auc_pr``, ``classification_report`` (str), and
        ``shap_df`` is a DataFrame of SHAP values (customers × features).
    """
    if random_state is None:
        random_state = get_config()["sampling"]["random_seed"]

    if features is None:
        if df is None:
            from .preprocessing import preprocess

            df = preprocess(keep_returns=False, save=False)
        features = build_churn_features(df, churn_days=churn_days)

    y = features.pop("churn")
    X = features

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Compute scale_pos_weight for class imbalance.
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos_weight = neg / max(pos, 1)

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="auc",
        random_state=random_state,
        verbosity=0,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    # --- Metrics ---------------------------------------------------------- #
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    metrics = {
        "auc_roc": roc_auc_score(y_test, y_prob),
        "auc_pr": average_precision_score(y_test, y_prob),
        "classification_report": classification_report(y_test, y_pred, target_names=["Active", "Churned"]),
    }

    print(f"✅ XGBoost trained — AUC-ROC: {metrics['auc_roc']:.4f} | AUC-PR: {metrics['auc_pr']:.4f}")
    print("\n" + metrics["classification_report"])

    # --- SHAP ------------------------------------------------------------- #
    try:
        import shap

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        shap_df = pd.DataFrame(shap_values, columns=X.columns, index=X_test.index)
    except ImportError:
        print("⚠️  shap not installed — skipping SHAP values")
        shap_df = pd.DataFrame()

    return model, metrics, shap_df


# --------------------------------------------------------------------------- #
# 3. Persist / load
# --------------------------------------------------------------------------- #
def save_model(
    model: xgb.XGBClassifier,
    path: Optional[Path | str] = None,
) -> Path:
    """Save the trained XGBoost model to disk.

    Args:
        model: Fitted XGBClassifier.
        path: Destination path. Defaults to ``models/churn_model.json``.

    Returns:
        The path the model was written to.
    """
    cfg = get_config()
    if path is None:
        path = cfg["paths"]["models_dir"] / "churn_model.json"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(path))
    print(f"✅ Churn model saved → {path}")
    return path


if __name__ == "__main__":
    features = build_churn_features()
    model, metrics, shap_df = train_churn_model(features)
    save_model(model)
