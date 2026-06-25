"""Segmentation page — RFM scoring + K-Means customer clusters."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.rfm import SEGMENT_LABELS
from src.visualizations import set_plot_style

set_plot_style()

# ---------------------------------------------------------------------------
# Access cached data via the shared _state module (safe to import — unlike the
# entry module, importing it does NOT re-run st.set_page_config).
# ---------------------------------------------------------------------------
import _state  # noqa: E402

df = _state.get_df()
rfm = _state.get_rfm(df)

st.title("📊 Customer Segmentation")
st.markdown("RFM scoring (1–5) + K-Means clustering into 5 actionable segments.")

# --- Segment overview ----------------------------------------------------- #
# reindex(SEGMENT_LABELS) can introduce NaN rows when a label is absent from
# the data (e.g. tiny uploads); drop them so the counts/percentages stay sane.
segment_summary = (
    rfm.groupby("Segment")
    .agg(
        customers=("Recency", "count"),
        avg_recency=("Recency", "mean"),
        avg_frequency=("Frequency", "mean"),
        avg_monetary=("Monetary", "mean"),
    )
    .reindex(SEGMENT_LABELS)
    .dropna(how="all")
)
total_customers = segment_summary["customers"].sum()
segment_summary["%"] = (segment_summary["customers"] / total_customers * 100).round(1)

st.subheader("Segment Overview")
st.dataframe(segment_summary, use_container_width=True)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Segment Sizes")
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    ax1.pie(
        segment_summary["customers"],
        labels=segment_summary.index,
        autopct="%1.1f%%",
        colors=sns.color_palette("Set2", len(segment_summary)),
        startangle=140,
    )
    ax1.set_title("Customer distribution")
    st.pyplot(fig1)

with col2:
    st.subheader("Avg RFM Heatmap")
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    metrics = segment_summary[["avg_recency", "avg_frequency", "avg_monetary"]]
    metrics.columns = ["Recency", "Frequency", "Monetary"]
    sns.heatmap(metrics, annot=True, fmt=",.1f", cmap="YlGnBu_r", ax=ax2)
    ax2.set_title("Average RFM per segment")
    st.pyplot(fig2)

# --- Scatter --------------------------------------------------------------- #
st.subheader("Recency × Monetary Scatter")
fig3, ax3 = plt.subplots(figsize=(10, 5))
for seg in SEGMENT_LABELS:
    mask = rfm["Segment"] == seg
    ax3.scatter(rfm.loc[mask, "Recency"], rfm.loc[mask, "Monetary"],
                label=seg, alpha=0.5, s=15)
ax3.set_xlabel("Recency (days)")
ax3.set_ylabel("Monetary (£)")
ax3.legend(title="Segment")
ax3.set_title("Customer segments — Recency vs Monetary")
st.pyplot(fig3)

# --- Customer drill-down --------------------------------------------------- #
st.subheader("🔍 Customer Lookup")
selected = st.selectbox("Select a customer", rfm.index.astype(str))
if selected:
    row = rfm.loc[int(selected)]
    st.write(f"**Segment:** {row['Segment']}  |  RFM Score: {row['RFM_Score']}")
    cols = st.columns(3)
    cols[0].metric("Recency", f"{row['Recency']:.0f} days")
    cols[1].metric("Frequency", f"{row['Frequency']:.0f} invoices")
    cols[2].metric("Monetary", f"£{row['Monetary']:,.2f}")
