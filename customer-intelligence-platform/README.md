# Customer Intelligence Platform

> **End-to-end customer analytics: segmentation, lifetime-value forecasting, and churn prediction — all in one interactive dashboard.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/dashboard-Streamlit-FF4B4B.svg)](https://streamlit.io)

## Overview

The **Customer Intelligence Platform** turns raw e-commerce transactions into three actionable layers of insight:

| Layer | Method | Output |
|-------|--------|--------|
| **Who are my customers?** | RFM scoring + K-Means clustering | 5 actionable segments (Champions / Loyal / Potential / At-Risk / Lost) |
| **How much are they worth?** | BG/NBD + Gamma-Gamma probabilistic models | Per-customer CLV for the next 3 / 6 / 12 months |
| **Who am I about to lose?** | XGBoost classifier + SHAP explanations | Churn probability + the *reason* behind each prediction |

All three layers are exposed through a multi-page **Streamlit dashboard** that supports custom CSV upload, so the platform works on **any** transactional dataset that follows the standard schema.

## Why This Project Stands Out

- **Probabilistic CLV** (BG/NBD) instead of the more common but less rigorous regression-based CLV.
- **SHAP-driven churn explanations** — every prediction is interpretable.
- **Custom-data upload** — the dashboard is not a static report; it is a working analytics tool.
- **Production-style layout** — separation of `src/`, `app/`, `notebooks/`, `tests/`, and `scripts/`.

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/AdamAI-Systems/Data-Analytics.git
cd Data-Analytics/customer-intelligence-platform

# 2. Set up the environment
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS / Linux: source .venv/bin/activate
pip install -r requirements.txt

# 3. (Optional) Download the full Online Retail II dataset
python scripts/download_data.py

# 4. (Optional) Pre-train & persist the LTV + churn models
python scripts/train_all.py

# 5. Launch the dashboard
streamlit run app/streamlit_dashboard.py
```

> The dashboard **auto-generates** a 5,000-row `data/sample_data.csv` on first
> launch and fits all models in-memory, so steps 3–4 are **optional** — you
> can launch the dashboard immediately after step 2.

## Project Structure

```text
customer-intelligence-platform/
├── data/                       # Sample data + README (raw/processed are gitignored)
├── notebooks/                  # 4 analysis notebooks (EDA → RFM → LTV → Churn)
├── src/                        # Reusable Python modules
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── rfm.py                  # RFM scoring + K-Means
│   ├── ltv_model.py            # BG/NBD + Gamma-Gamma
│   ├── churn_model.py          # XGBoost + SHAP
│   └── visualizations.py
├── app/                        # Multi-page Streamlit dashboard
│   ├── streamlit_dashboard.py  # entry point — sidebar + home page
│   ├── _state.py               # shared cached loaders (home ↔ pages)
│   └── pages/
│       ├── 01_Segmentation.py
│       ├── 02_Lifetime_Value.py
│       └── 03_Churn.py
├── scripts/
│   ├── download_data.py        # Fetch UCI dataset
│   ├── make_sample_data.py     # Generate synthetic sample
│   └── train_all.py            # Train + persist all models
├── tests/                      # Unit tests for src/
├── models/                     # Trained models (gitignored)
├── requirements.txt
├── Makefile                    # `make setup | data | train | dashboard | test`
├── LICENSE
└── README.md
```

## Dataset

**Online Retail II** — UCI Machine Learning Repository. ~1.07M transactions from a UK online retailer (2009–2011).

[https://archive.ics.uci.edu/dataset/502/online+retail+ii](https://archive.ics.uci.edu/dataset/502/online+retail+ii)

## Roadmap

- [x] Phase 1 — Project scaffolding
- [x] Phase 2 — Data loader + downloader + sample data
- [x] Phase 3 — Exploratory data analysis notebook
- [x] Phase 4 — RFM segmentation module + notebook
- [x] Phase 5 — Probabilistic LTV module + notebook
- [x] Phase 6 — Churn prediction module + notebook
- [x] Phase 7 — Multi-page Streamlit dashboard
- [ ] Phase 8 — Tests + final polish + screenshots

## License

MIT — see [LICENSE](LICENSE).

---
© 2026 AdamAI-Systems.
