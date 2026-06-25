.PHONY: help setup data train dashboard test clean

help:
	@echo "Available targets:"
	@echo "  setup     - Install dependencies"
	@echo "  data      - Download Online Retail II dataset"
	@echo "  train     - Train RFM, LTV, and Churn models"
	@echo "  dashboard - Launch Streamlit dashboard"
	@echo "  test      - Run unit tests"
	@echo "  clean     - Remove cached files and trained models"

setup:
	pip install -r requirements.txt

data:
	python scripts/download_data.py

train:
	python scripts/train_all.py

dashboard:
	streamlit run app/streamlit_dashboard.py

test:
	pytest tests/ -v

clean:
	rm -rf models/*.pkl models/*.joblib
	rm -rf data/processed/*
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} +
