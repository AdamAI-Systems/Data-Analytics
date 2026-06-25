"""Train and persist all three models: RFM segments, LTV (BG/NBD + GG), churn.

Orchestrates the platform's full training pipeline so a single command
reproduces every artefact the dashboard needs::

    python scripts/train_all.py

Steps:
    1. Load + clean the data (download / sample generation handled upstream).
    2. RFM  → score + K-Means segments → ``data/processed/rfm_segments.csv``
    3. LTV  → fit BG/NBD + Gamma-Gamma → ``models/bgf.pkl`` + ``models/ggf.pkl``
    4. Churn→ train XGBoost             → ``models/churn_model.json``

The LTV / churn artefacts are what the dashboard would otherwise fit on the
fly; persisting them keeps cold-start launches fast and the results stable.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make ``src`` importable when running ``python scripts/train_all.py`` from the
# project root, without installing the package.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_config  # noqa: E402
from src.preprocessing import preprocess  # noqa: E402


def main() -> None:
    cfg = get_config()
    models_dir: Path = cfg["paths"]["models_dir"]
    models_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Load + clean -------------------------------------------------- #
    print("=" * 70)
    print("1/4  Loading & cleaning transactions ...")
    print("=" * 70)
    df = preprocess(keep_returns=False, save=True)

    # --- 2. RFM ----------------------------------------------------------- #
    print("\n" + "=" * 70)
    print("2/4  RFM scoring + K-Means segmentation ...")
    print("=" * 70)
    from src.rfm import rfm_pipeline

    rfm = rfm_pipeline(df)

    segments_path = cfg["paths"]["processed_dir"] / "rfm_segments.csv"
    segments_path.parent.mkdir(parents=True, exist_ok=True)
    rfm.to_csv(segments_path, index=True)
    print(f"✅ RFM segments saved → {segments_path}")

    # --- 3. LTV (BG/NBD + Gamma-Gamma) ------------------------------------ #
    print("\n" + "=" * 70)
    print("3/4  LTV — BG/NBD + Gamma-Gamma ...")
    print("=" * 70)
    from src.ltv_model import fit_ltv_models, save_models

    summary, bgf, ggf = fit_ltv_models(df)
    save_models(bgf, ggf)

    # --- 4. Churn (XGBoost) ----------------------------------------------- #
    print("\n" + "=" * 70)
    print("4/4  Churn — XGBoost + SHAP ...")
    print("=" * 70)
    from src.churn_model import build_churn_features, save_model, train_churn_model

    features = build_churn_features(df)
    model, _metrics, _shap = train_churn_model(features)
    save_model(model)

    print("\n" + "=" * 70)
    print("🎉 All models trained and saved. Launch the dashboard with:")
    print("    streamlit run app/streamlit_dashboard.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
