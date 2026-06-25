"""Customer Intelligence Platform — Multi-page Streamlit Dashboard.

This is the entry point (``streamlit run app/streamlit_dashboard.py``).

Pages are discovered automatically from ``app/pages/*.py`` (Streamlit's
native multi-page convention). The main file configures the sidebar, data
loading, and shared state.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
import pandas as pd

# Ensure the project root is on sys.path so ``src`` is importable.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# --------------------------------------------------------------------------- #
# Page config (must be the very first Streamlit call)
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Customer Intelligence Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

from _state import load_data, get_rfm, get_ltv, get_churn  # noqa: E402

# --------------------------------------------------------------------------- #
# Sidebar — data loading + shared state
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.title("🧠 Customer Intelligence")

    st.markdown("---")
    st.header("📁 Data Source")

    uploaded_file = st.file_uploader(
        "Upload a CSV (schema: InvoiceNo, StockCode, Description, "
        "Quantity, InvoiceDate, UnitPrice, CustomerID, Country)",
        type=["csv"],
        key="csv_upload",
    )

    use_sample = st.checkbox(
        "Use synthetic sample (5,000 rows)",
        value=(uploaded_file is None),
        disabled=(uploaded_file is not None),
    )

    st.markdown("---")
    st.header("⚙️ Settings")
    # Bind the slider to session_state so every page reads the live value
    # (the Churn page trains its model against this threshold). Using a key
    # instead of a local variable is what makes the control actually work.
    st.slider(
        "Churn threshold (days)",
        30, 180, st.session_state.get("churn_days", 90),
        step=10,
        key="churn_days",
    )

    st.markdown("---")
    st.caption("© 2026 AdamAI-Systems")


# --------------------------------------------------------------------------- #
# Trigger loading (so cached results are available for sub-pages) and expose
# the active DataFrame to every page via session_state.
# --------------------------------------------------------------------------- #
df = load_data(uploaded_file, use_sample)
st.session_state["df"] = df

# --------------------------------------------------------------------------- #
# Home page content
# --------------------------------------------------------------------------- #
st.title("🧠 Customer Intelligence Platform")
st.markdown(
    "End-to-end customer analytics: **segmentation**, **lifetime-value "
    "forecasting**, and **churn prediction** — all in one dashboard."
)

st.markdown("---")

col1, col2, col3 = st.columns(3)
col1.metric("Transactions", f"{len(df):,}")
col2.metric("Customers", f"{df['CustomerID'].nunique():,}")
col3.metric("Products", f"{df['StockCode'].nunique():,}")

st.markdown("---")

st.subheader("📊 Data Preview")
st.dataframe(df.head(10), use_container_width=True)

st.markdown("---")

st.subheader("🚀 Navigate")
st.markdown(
    "Use the sidebar **pages** (▸ arrows at top-left or the dropdown) "
    "to explore:\n"
    "\n"
    "1. **Segmentation** — RFM scoring + K-Means (5 segments)\n"
    "2. **Lifetime Value** — BG/NBD + Gamma-Gamma CLV (3/6/12 months)\n"
    "3. **Churn** — XGBoost + SHAP explainability"
)
