# AdamAI-Systems — Data Analytics

Building intelligent business analytics systems that turn raw transactional data into actionable customer insights. This repository houses the data-science, business-intelligence, and predictive-analytics solutions developed by **AdamAI-Systems**.

## Projects in this Repository

### 1. [Customer Intelligence Platform](./customer-intelligence-platform)
An end-to-end customer analytics platform combining segmentation, lifetime-value forecasting, and churn prediction with an interactive dashboard.
* **Tech Stack:** Python, Pandas, scikit-learn, XGBoost, lifetimes (BG/NBD + Gamma-Gamma), SHAP, Streamlit, Plotly.
* **Key Features:**
  * **RFM segmentation** with K-Means clustering (5 customer segments).
  * **Probabilistic CLV** via BG/NBD + Gamma-Gamma models (3/6/12-month horizons).
  * **Churn prediction** with XGBoost + SHAP explainability.
  * **Multi-page Streamlit dashboard** with custom CSV upload and per-customer drill-down.

---

## Setup & General Instructions

To run any of the projects locally, navigate to the specific project directory and follow the instructions in its respective `README.md`.

Most projects follow a common workflow:

1. Create a virtual environment (`python -m venv .venv`) and activate it.
2. Install dependencies: `pip install -r requirements.txt`.
3. (Optional) Download the full dataset: `python scripts/download_data.py`.
4. Train the models: `python scripts/train_all.py`.
5. Launch the dashboard: `streamlit run app/streamlit_dashboard.py`.

> **Note:** Trained models (`*.pkl`) and raw datasets are intentionally **not committed** to this repository. Each project ships with a small sample dataset so the dashboard can be explored immediately.

---
© 2026 AdamAI-Systems. All rights reserved.
