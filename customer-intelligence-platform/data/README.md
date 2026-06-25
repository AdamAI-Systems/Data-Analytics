# Data Directory

## Layout

```
data/
├── raw/             # Original Online Retail II Excel/CSV (gitignored)
├── processed/       # Cleaned + feature-engineered parquet files (gitignored)
├── sample_data.csv  # 5,000-row synthetic sample (generated via scripts/make_sample_data.py)
└── README.md
```

## Full Dataset

The full dataset is **Online Retail II** from the UCI Machine Learning Repository (≈1.07M transactions from a UK online retailer, 2009–2011).

To download the full dataset, run:

```bash
python scripts/download_data.py
```

This will fetch the Excel file into `data/raw/online_retail_II.xlsx` and write a cleaned parquet to `data/processed/transactions.parquet`.

## Sample Data

`sample_data.csv` is a 5,000-row **synthetic** sample that follows the exact
canonical schema below (including ~5% cancellations/returns, so the cleaning
step has something to filter). It is **not** committed by default — generate it
locally with a single command:

```bash
python scripts/make_sample_data.py
```

The generator uses a fixed random seed (42) so the output is reproducible, and
it does **not** require a network connection. Once generated, the dashboard can
be launched immediately:

```bash
streamlit run app/streamlit_dashboard.py
```

## Schema normalization

The raw UCI `.xlsx` uses slightly different column names than the platform
expects. `src/data_loader.normalize_schema()` renames them automatically when
loading, so every downstream module (RFM / LTV / churn) works against one
canonical schema:

| Raw UCI column | Canonical column |
|----------------|------------------|
| `Invoice`      | `InvoiceNo`      |
| `Price`        | `UnitPrice`      |
| `Customer ID`  | `CustomerID`     |

The engineered `TotalPrice` column is added during normalization. The rename
map lives in `src/config.py` (`RENAME_MAP`).

## Schema (cleaned)

| Column        | Type    | Description                              |
|---------------|---------|------------------------------------------|
| InvoiceNo     | string  | Unique transaction id                    |
| StockCode     | string  | Product SKU                              |
| Description   | string  | Product name                             |
| Quantity      | int     | Units purchased (positive)               |
| InvoiceDate   | datetime| Transaction timestamp                    |
| UnitPrice     | float   | Price per unit (GBP)                     |
| CustomerID    | int     | Anonymized customer id                   |
| Country       | string  | Customer country                         |
| TotalPrice    | float   | Quantity × UnitPrice (engineered)        |

## Source

Chen, D. (2019). *Online Retail II*. UCI Machine Learning Repository.
[https://archive.ics.uci.edu/dataset/502/online+retail+ii](https://archive.ics.uci.edu/dataset/502/online+retail+ii)
