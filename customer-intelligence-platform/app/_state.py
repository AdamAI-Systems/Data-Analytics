"""Shared, cached loaders for the Customer Intelligence dashboard.

Why this module exists
----------------------
Streamlit's native multi-page apps discover ``app/pages/*.py`` and run each
page in its own script execution. Each page therefore needs access to the
*same* cleaned transactions and fitted models the home page built.

Importing the entry module (``streamlit_dashboard``) from a page to reuse its
``@st.cache_data`` helpers is fragile: importing the entry module by name
re-executes it, which re-calls :func:`st.set_page_config` and raises a
``StreamlitAPIException``. To avoid that, every cached loader lives here in a
plain module that is safe to import from any page without side effects.

Pages should retrieve shared objects via :func:`get_df` /
:func:`get_churn_days`, which read the values the home page stashes in
``st.session_state`` (falling back to recomputation if a page is opened
directly).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Ensure the project root is on sys.path so ``import src...`` works regardless
# of the working directory Streamlit was launched from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# --------------------------------------------------------------------------- #
# 1. Ensure sample data exists (so the dashboard works on first launch)
# --------------------------------------------------------------------------- #
def ensure_sample_data() -> Path:
    """Return the path to ``data/sample_data.csv``, generating it if absent.

    The sample is deliberately not committed (it is regenerable from a fixed
    seed). Without this guard the dashboard would ``FileNotFoundError`` on a
    fresh clone, contradicting the README's "launch immediately" promise.
    """
    from src.config import get_config

    sample_path: Path = get_config()["paths"]["sample_csv"]
    if not sample_path.exists():
        # Build the synthetic sample in-process. ``make_sample_data`` lives
        # under ``scripts/`` (not a package), so load it by file path. Its
        # ``if __name__ == "__main__"`` guard means importing does not run
        # generation — we call generate_sample() explicitly, which writes
        # the CSV and normalises to the canonical schema itself.
        import importlib.util

        gen_path = PROJECT_ROOT / "scripts" / "make_sample_data.py"
        spec = importlib.util.spec_from_file_location("_make_sample_data", gen_path)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[arg-type]
        mod.generate_sample()
    return sample_path


# --------------------------------------------------------------------------- #
# 2. Cached data + model loaders
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="⏳ Loading and cleaning transactions...")
def load_data(
    uploaded_file=None,
    use_sample: bool = True,
) -> pd.DataFrame:
    """Load and clean transactional data.

    Priority:
    1. Uploaded CSV (if provided).
    2. Synthetic sample (if ``use_sample`` is True) — generated on demand.
    3. Full UCI dataset from ``data/raw/`` (fallback).
    """
    from src.preprocessing import clean_transactions, engineer_features

    if uploaded_file is not None:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file)
        st.sidebar.success(f"✅ Loaded {len(df):,} rows from upload")
    elif use_sample:
        sample_path = ensure_sample_data()
        df = pd.read_csv(sample_path)
        st.sidebar.success(f"✅ Sample data ({len(df):,} rows)")
    else:
        from src.data_loader import load_transactions

        df = load_transactions()
        st.sidebar.success(f"✅ Loaded {len(df):,} rows (full dataset)")

    df = clean_transactions(df, keep_returns=False)
    df = engineer_features(df)
    return df


@st.cache_data(show_spinner="⏳ Building RFM table...")
def get_rfm(df: pd.DataFrame):
    """Return the scored + segmented RFM table (cached on ``df``)."""
    from src.rfm import build_rfm_table, score_rfm, segment_customers

    rfm = build_rfm_table(df)
    rfm = score_rfm(rfm)
    rfm = segment_customers(rfm)
    return rfm


@st.cache_resource(show_spinner="⏳ Fitting LTV models...")
def get_ltv(df: pd.DataFrame):
    """Fit BG/NBD + Gamma-Gamma and return ``(summary, bgf, ggf)``."""
    from src.ltv_model import fit_ltv_models

    summary, bgf, ggf = fit_ltv_models(df)
    return summary, bgf, ggf


@st.cache_resource(show_spinner="⏳ Training churn model...")
def get_churn(df: pd.DataFrame, churn_days: int = 90):
    """Build churn features and train XGBoost. Cached on (``df``, ``churn_days``)."""
    from src.churn_model import build_churn_features, train_churn_model

    features = build_churn_features(df, churn_days=churn_days)
    model, metrics, shap_df = train_churn_model(features, churn_days=churn_days)
    return features, model, metrics, shap_df


# --------------------------------------------------------------------------- #
# 3. Page-facing accessors — read session_state, fall back to recomputation
# --------------------------------------------------------------------------- #
def get_df() -> pd.DataFrame:
    """Return the active transactions DataFrame for the current page.

    Prefer the value the home page stashed in ``st.session_state`` (so an
    uploaded CSV propagates to every page). If a page is opened directly
    (session_state empty), recompute via :func:`load_data` — which is cached,
    so this is cheap and consistent.
    """
    df = st.session_state.get("df")
    if df is None:
        df = load_data()
        st.session_state["df"] = df
    return df


def get_churn_days() -> int:
    """Return the sidebar's churn threshold, defaulting to 90."""
    return int(st.session_state.get("churn_days", 90))
