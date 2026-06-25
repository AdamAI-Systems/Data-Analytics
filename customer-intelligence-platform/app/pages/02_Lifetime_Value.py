"""Lifetime Value page — BG/NBD + Gamma-Gamma CLV predictions."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.visualizations import set_plot_style

set_plot_style()

# ---------------------------------------------------------------------------
# Access cached data via the shared _state module.
# ---------------------------------------------------------------------------
import _state  # noqa: E402

df = _state.get_df()
summary, bgf, ggf = _state.get_ltv(df)

# NOTE: ``lifetimes`` builds the summary with ``CustomerID`` cast to **str**
# (see src.ltv_model.build_summary_data), so the index is a string. Keep
# lookups string-keyed to avoid ``KeyError``.

st.title("💰 Customer Lifetime Value")
st.markdown("Predicted CLV using BG/NBD (transaction frequency) + Gamma-Gamma (monetary value).")

# --- Key metrics ----------------------------------------------------------- #
col1, col2, col3 = st.columns(3)
col1.metric("Total 12m CLV", f"£{summary['clv_12m'].sum():,.0f}")
col2.metric("Avg CLV (12m)", f"£{summary['clv_12m'].mean():,.0f}")
col3.metric("Median CLV (12m)", f"£{summary['clv_12m'].median():,.0f}")

# --- CLV distribution ------------------------------------------------------ #
st.subheader("CLV Distribution by Horizon")
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, col in zip(axes, ["clv_3m", "clv_6m", "clv_12m"]):
    sns.histplot(summary[col].clip(lower=0), bins=50, kde=True, ax=ax)
    ax.set_title(col.replace("_", " ").upper())
    ax.set_xlabel("Predicted CLV (£)")
fig.suptitle("CLV distributions", fontsize=14, fontweight="bold", y=1.02)
fig.tight_layout()
st.pyplot(fig)

# --- Alive probability ----------------------------------------------------- #
# Compute into a *standalone Series* rather than mutating the cached summary
# DataFrame (``summary`` comes from ``@st.cache_resource`` and must stay pure).
st.subheader("🔴 Customer Alive Probability")
alive_prob = bgf.conditional_probability_alive(
    summary["frequency"], summary["recency"], summary["T"]
)
# Normalise the index to str so it matches the summary index dtype and is
# comparable across the dashboard.
alive_prob.index = alive_prob.index.astype(str)

fig2, ax2 = plt.subplots(figsize=(10, 4))
sns.histplot(alive_prob, bins=30, kde=True, ax=ax2, color="#4C72B0")
ax2.set_title("P(customer is alive) — BG/NBD")
ax2.set_xlabel("Probability")
st.pyplot(fig2)

col1, col2 = st.columns(2)
col1.metric("At-risk (P < 0.3)", f"{(alive_prob < 0.3).sum():,}")
col2.metric("Very alive (P > 0.9)", f"{(alive_prob > 0.9).sum():,}")

# --- Customer drill-down --------------------------------------------------- #
st.subheader("🔍 Customer Lookup")
# ``summary.index`` is string-typed (see note above).
selected = st.selectbox("Select a customer", summary.index.astype(str))
if selected:
    row = summary.loc[selected]
    cols = st.columns(4)
    cols[0].metric("3-month CLV", f"£{row['clv_3m']:,.0f}")
    cols[1].metric("6-month CLV", f"£{row['clv_6m']:,.0f}")
    cols[2].metric("12-month CLV", f"£{row['clv_12m']:,.0f}")
    cols[3].metric("Alive Probability", f"{alive_prob.loc[selected]:.2f}")
