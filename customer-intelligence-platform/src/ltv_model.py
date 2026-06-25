"""Probabilistic Customer Lifetime Value (CLV) using BG/NBD + Gamma-Gamma.

Instead of the more common but less rigorous regression-based CLV, this module
uses two well-established probabilistic models from the ``lifetimes`` library:

1. **BG/NBD** (Beta-Geometric / Negative-Binomial Distribution) — models
   the *probability* that a customer is still "alive" (will purchase again)
   and predicts the expected number of future transactions.
2. **Gamma-Gamma** — models the *monetary value* per transaction as
   independent of purchase frequency, yielding an estimated average transaction
   value.

CLV = predicted transactions × predicted avg. value, projected over a
user-chosen horizon (3 / 6 / 12 months).

Typical usage::

    from src.ltv_model import fit_ltv_models, predict_clv

    summary, bgf, ggf = fit_ltv_models(clean_df)
    clv_12m = predict_clv(summary, ggf, horizon_months=12)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
from lifetimes import BetaGeoFitter, GammaGammaFitter

from .config import get_config


# --------------------------------------------------------------------------- #
# 1. Prepare the summary-from-transactions data
# --------------------------------------------------------------------------- #
def build_summary_data(
    df: pd.DataFrame,
    observation_period_end: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """Convert a cleaned transactional DataFrame to ``lifetimes`` summary format.

    The output is the exact format expected by ``BetaGeoFitter.fit()`` with
    columns: ``customer_id``, ``frequency``, ``recency``, ``T`` (age).

    Args:
        df: Cleaned transactions (no returns, no anonymous customers).
        observation_period_end: Cut-off date. Defaults to ``df.InvoiceDate.max()``.

    Returns:
        A ``lifetimes``-compatible summary DataFrame.
    """
    from lifetimes.utils import summary_from_transaction_data

    df = df.copy()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df = df.dropna(subset=["InvoiceDate", "CustomerID"])
    df["CustomerID"] = df["CustomerID"].astype(str)

    if observation_period_end is None:
        observation_period_end = df["InvoiceDate"].max()

    summary = summary_from_transaction_data(
        transactions=df,
        customer_id_col="CustomerID",
        datetime_col="InvoiceDate",
        monetary_value_col="TotalPrice",
        observation_period_end=observation_period_end,
    )
    print(f"✅ Lifetimes summary built — {len(summary):,} customers")
    return summary


# --------------------------------------------------------------------------- #
# 2. Fit both models
# --------------------------------------------------------------------------- #
def fit_ltv_models(
    df: Optional[pd.DataFrame] = None,
    summary: Optional[pd.DataFrame] = None,
    penalizer_coef: float = 0.01,
) -> tuple[pd.DataFrame, BetaGeoFitter, GammaGammaFitter]:
    """Fit BG/NBD and Gamma-Gamma models and return enriched summaries.

    Args:
        df: Cleaned transactions. When provided (and ``summary`` is ``None``),
            the summary is built via :func:`build_summary_data`.
        summary: Pre-built lifetimes summary. Takes precedence over ``df``.
        penalizer_coef: L2 regularisation strength for both models.

    Returns:
        A tuple of (summary_with_predictions, bgf, ggf) where the summary
        includes columns: ``predicted_purchases_3m``, ``predicted_purchases_6m``,
        ``predicted_purchases_12m``, ``expected_avg_profit``, and
        ``clv_3m``, ``clv_6m``, ``clv_12m``.
    """
    if summary is None:
        if df is None:
            from .preprocessing import preprocess

            df = preprocess(keep_returns=False, save=False)
        summary = build_summary_data(df)

    # --- BG/NBD: transaction frequency & alive probability ---------------- #
    bgf = BetaGeoFitter(penalizer_coef=penalizer_coef)
    bgf.fit(summary["frequency"], summary["recency"], summary["T"])
    print(f"✅ BG/NBD fitted (penalizer={penalizer_coef})")

    # --- Gamma-Gamma: monetary value -------------------------------------- #
    # Gamma-Gamma requires customers with at least one repeat purchase.
    returning = summary.query("frequency > 0").copy()
    ggf = GammaGammaFitter(penalizer_coef=penalizer_coef)
    ggf.fit(returning["frequency"], returning["monetary_value"])
    print(f"✅ Gamma-Gamma fitted on {len(returning):,} repeat customers")

    # --- Merge predictions back into summary ------------------------------ #
    # Predicted number of purchases over horizons (in days).
    days_in_month = 30.44  # average
    for months, col in [(3, "predicted_purchases_3m"), (6, "predicted_purchases_6m"), (12, "predicted_purchases_12m")]:
        summary[col] = bgf.conditional_expected_number_of_purchases_up_to_time(
            months * days_in_month,
            summary["frequency"],
            summary["recency"],
            summary["T"],
        )

    # Expected average profit per transaction (Gamma-Gamma).
    summary["expected_avg_profit"] = ggf.conditional_expected_average_profit(
        summary["frequency"], summary["monetary_value"]
    )

    # CLV = predicted purchases × expected avg profit.
    for months, pcol, clv_col in [
        (3, "predicted_purchases_3m", "clv_3m"),
        (6, "predicted_purchases_6m", "clv_6m"),
        (12, "predicted_purchases_12m", "clv_12m"),
    ]:
        summary[clv_col] = ggf.customer_lifetime_value(
            bgf,
            summary["frequency"],
            summary["recency"],
            summary["T"],
            summary["monetary_value"],
            time=months * days_in_month,  # months in "time" parameter = months
            discount_rate=0.0,
        )

    print(f"✅ CLV predictions added (3m / 6m / 12m horizons)")
    return summary, bgf, ggf


# --------------------------------------------------------------------------- #
# 3. Convenience wrappers
# --------------------------------------------------------------------------- #
def predict_clv(
    summary: pd.DataFrame,
    ggf: GammaGammaFitter,
    horizon_months: int = 12,
) -> pd.Series:
    """Return the CLV series for a specific horizon.

    Args:
        summary: Output of :func:`fit_ltv_models`.
        ggf: Fitted Gamma-Gamma model.
        horizon_months: Forecast window (3, 6, or 12).

    Returns:
        A :class:`~pandas.Series` indexed by customer id.
    """
    col = f"clv_{horizon_months}m"
    if col not in summary.columns:
        raise KeyError(
            f"Column '{col}' not found. Call fit_ltv_models() first."
        )
    return summary[col]


def save_models(
    bgf: BetaGeoFitter,
    ggf: GammaGammaFitter,
    bgf_path: Optional[Path | str] = None,
    ggf_path: Optional[Path | str] = None,
) -> None:
    """Persist fitted models to disk for the dashboard to reload.

    Args:
        bgf: Fitted BetaGeoFitter.
        ggf: Fitted GammaGammaFitter.
        bgf_path: Save location for BG/NBD. Defaults to ``models/bgf.pkl``.
        ggf_path: Save location for Gamma-Gamma. Defaults to ``models/ggf.pkl``.
    """
    cfg = get_config()
    if bgf_path is None:
        bgf_path = cfg["paths"]["models_dir"] / "bgf.pkl"
    if ggf_path is None:
        ggf_path = cfg["paths"]["models_dir"] / "ggf.pkl"

    Path(bgf_path).parent.mkdir(parents=True, exist_ok=True)
    bgf.save_model(str(bgf_path))
    ggf.save_model(str(ggf_path))
    print(f"✅ Models saved → {bgf_path}, {ggf_path}")


if __name__ == "__main__":
    from .preprocessing import preprocess

    clean = preprocess(keep_returns=False, save=False)
    summary, bgf, ggf = fit_ltv_models(clean)
    save_models(bgf, ggf)
    print("\nTop 10 customers by 12-month CLV:")
    print(summary.nlargest(10, "clv_12m")[["frequency", "recency", "monetary_value", "clv_12m"]])
