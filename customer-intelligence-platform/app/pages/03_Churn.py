"""Churn prediction page — XGBoost + SHAP."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.visualizations import set_plot_style

set_plot_style()

# ---------------------------------------------------------------------------
# Access cached data via the shared _state module. The churn threshold comes
# from the home page's sidebar slider (stored in session_state) so the model
# is trained against whatever the user picked.
# ---------------------------------------------------------------------------
import _state  # noqa: E402

df = _state.get_df()
churn_days = _state.get_churn_days()
features, model, metrics, shap_df = _state.get_churn(df, churn_days=churn_days)

st.title("⚠️ Churn Prediction")
st.markdown(
    f"XGBoost classifier with SHAP explainability. A customer is **churned** "
    f"if they have not purchased in the last **{churn_days} days** "
    f"(adjust in the sidebar)."
)

# --- Key metrics ----------------------------------------------------------- #
col1, col2 = st.columns(2)
col1.metric("AUC-ROC", f"{metrics['auc_roc']:.4f}")
col2.metric("AUC-PR", f"{metrics['auc_pr']:.4f}")

st.text_area("Classification Report", metrics["classification_report"], height=200)

# --- Feature importance ----------------------------------------------------- #
st.subheader("Feature Importance (Gain)")
importance = pd.Series(
    model.feature_importances_, index=model.feature_names_in_
).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(9, 6))
importance.plot(kind="barh", ax=ax, color="#4C72B0")
ax.set_title("XGBoost feature importance")
ax.set_xlabel("Importance")
st.pyplot(fig)

# --- SHAP summary ---------------------------------------------------------- #
st.subheader("SHAP Explanations")
if not shap_df.empty:
    try:
        import shap as shap_lib

        fig2, ax2 = plt.subplots()
        shap_lib.summary_plot(
            shap_df.values,
            features=shap_df.columns,
            show=False,
        )
        plt.tight_layout()
        st.pyplot(fig2)
    except Exception:
        st.info("SHAP plot could not be rendered in this environment.")
else:
    st.warning("Install `shap` to see explanations: `pip install shap`")

# --- Customer drill-down --------------------------------------------------- #
st.subheader("🔍 Customer Lookup")
# Feature columns the model was trained on (excludes the target, which
# train_churn_model popped in-place). Predict from a labelled DataFrame so
# XGBoost validates feature names/order rather than relying on positional
# alignment of a raw numpy array.
feat_cols = list(model.feature_names_in_)
customer_ids = features.index.astype(str).tolist()
selected = st.selectbox("Select a customer", customer_ids, key="churn_lookup")
if selected:
    cid = int(selected)
    row = features.loc[cid, feat_cols]
    y_prob = float(model.predict_proba(row.to_frame().T)[0, 1])
    y_pred = "🔴 Churned" if y_prob > 0.5 else "🟢 Active"

    st.write(f"**Prediction:** {y_pred} (P = {y_prob:.2f})")
    st.dataframe(pd.DataFrame(row).T, use_container_width=True)
