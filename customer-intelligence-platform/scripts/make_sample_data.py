"""Generate a realistic synthetic sample of Online Retail II transactions.

Because the full UCI dataset (~1.07M rows) is large and network-dependent, we
ship a small synthetic sample so the dashboard can be explored instantly. This
script fabricates ``sample_size`` rows (default 5,000) that follow the exact
canonical schema, including cancellations and returns, so the data *behaves*
like the real thing for RFM / LTV / churn demos.

It does NOT touch the network and uses a fixed random seed for
reproducibility.

Run it with::

    python scripts/make_sample_data.py
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# Make ``src`` importable when running ``python scripts/make_sample_data.py``
# directly from the project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_config  # noqa: E402  (path tweak above is required)


# --------------------------------------------------------------------------- #
# Synthetic population — plausibly-named SKUs / countries to look realistic.
# --------------------------------------------------------------------------- #
PRODUCTS: list[tuple[str, str, float]] = [
    # (StockCode, Description, typical unit price in GBP)
    ("85123A", "WHITE HANGING HEART T-LIGHT HOLDER", 2.55),
    ("22423", "REGENCY CAKESTAND 3 TIER", 12.75),
    ("85099B", "JUMBO BAG RED RETROSPOT", 1.95),
    ("47566B", "PARTY BUNTING", 4.25),
    ("84879", "ASSORTED COLOUR BIRD ORNAMENT", 1.65),
    ("22720", "SET OF 3 CAKE TINS PANTRY DESIGN", 6.35),
    ("20725", "LUNCH BAG RED RETROSPOT", 1.65),
    ("22375", "ROUND SNACK BOXES WOODLAND", 2.95),
    ("POST", "POSTAGE", 5.45),
    ("22197", "SMALL POPCORN HOLDER", 1.25),
    ("23298", "JUMBO STORAGE BAG SUKI", 2.45),
    ("22727", "ALARM CLOCK BAKELIKE RED", 5.95),
    ("84991", "60 CAKE CASES VINTAGE CHRISTMAS", 2.10),
    ("21212", "PACK OF 72 RETROSPOT CAKE CASES", 2.55),
    ("22666", "RECIPE BOX PANTRY YELLOW LABEL", 4.95),
    ("23084", "RABBIT NIGHT LIGHT", 7.45),
    ("22960", "JUMBO BAG VINTAGE LEAF", 1.95),
    ("16016", "SMALL FRENCH CHARM BRACELET", 2.95),
    ("21181", "PLEASE ONE PERSON  METAL SIGN", 2.10),
    ("21931", "JUMBO SCANDINAVIAN PEN BLUE", 2.25),
]

COUNTRIES: list[tuple[str, float]] = [
    # (country, probability of a customer being from there)
    ("United Kingdom", 0.92),
    ("Germany", 0.022),
    ("France", 0.020),
    ("EIRE", 0.016),
    ("Spain", 0.005),
    ("Netherlands", 0.005),
    ("Belgium", 0.004),
    ("Switzerland", 0.003),
    ("Portugal", 0.0025),
    ("Australia", 0.0025),
]


def _pick_countries(rng: np.random.Generator, n_customers: int) -> list[str]:
    """Assign each customer a country according to the weighted distribution."""
    names = [c for c, _ in COUNTRIES]
    probs = np.array([p for _, p in COUNTRIES])
    probs = probs / probs.sum()
    return list(rng.choice(names, size=n_customers, p=probs))


def generate_sample(
    sample_size: int | None = None,
    seed: int | None = None,
    out_path: Path | None = None,
) -> pd.DataFrame:
    """Fabricate a realistic Online Retail II-style transactional sample.

    The generated DataFrame follows the canonical schema
    (:data:`src.config.CANONICAL_COLUMNS`) and deliberately includes ~5%
    cancellations/returns so :func:`src.preprocessing.clean_transactions` has
    something to filter out.

    Args:
        sample_size: Number of rows to generate. Defaults to
            ``config.sampling.sample_size`` (5,000).
        seed: RNG seed for reproducibility. Defaults to
            ``config.sampling.random_seed`` (42).
        out_path: Where to write the CSV. Defaults to
            ``config.paths.sample_csv``.

    Returns:
        The generated DataFrame (also written to ``out_path``).
    """
    cfg = get_config()
    if sample_size is None:
        sample_size = cfg["sampling"]["sample_size"]
    if seed is None:
        seed = cfg["sampling"]["random_seed"]
    if out_path is None:
        out_path = cfg["paths"]["sample_csv"]
    out_path = Path(out_path)

    rng = np.random.default_rng(seed)

    # --- Population ------------------------------------------------------- #
    n_customers = 200
    customer_ids = rng.integers(10000, 20000, size=n_customers)
    countries = _pick_countries(rng, n_customers)
    customer_country = dict(zip(customer_ids, countries))

    # --- Dates: spread across Dec 2009 → Sep 2011 ------------------------- #
    start = datetime(2009, 12, 1)
    end = datetime(2011, 9, 9)
    span_days = (end - start).days
    # Cluster timestamps within business hours for realism, then sort ascending
    # so invoice numbers increase monotonically in time (as in real ledgers).
    offsets = rng.integers(0, span_days * 24, size=sample_size)
    dates = pd.to_datetime(
        [start + timedelta(hours=int(o)) for o in offsets]
    )
    # Push into 08:00–18:00 band.
    dates = dates.where(dates.dt.hour.between(8, 17), dates + pd.Timedelta(hours=9))
    dates = dates.sort_values(ignore_index=True)

    # --- Rows ------------------------------------------------------------- #
    # Invoice numbers strictly increase with time. Each invoice groups a few
    # consecutive rows (a basket of items bought together).
    rows = []
    invoice_counter = 570000
    rows_in_current_invoice = 0
    current_invoice_capacity = 0
    for i in range(sample_size):
        ts = dates.iloc[i]
        # Start a new basket once the current one is "full".
        if rows_in_current_invoice >= current_invoice_capacity:
            invoice_counter += 1
            rows_in_current_invoice = 0
            current_invoice_capacity = int(rng.integers(1, 6))
        rows_in_current_invoice += 1
        invoice_no = str(invoice_counter)

        cust = int(rng.choice(customer_ids))
        stock_code, desc, base_price = PRODUCTS[rng.integers(0, len(PRODUCTS))]

        # ~5% of rows are cancellations.
        is_return = rng.random() < 0.05
        if is_return:
            invoice_no = "C" + invoice_no
            quantity = int(rng.integers(1, 6)) * -1
            unit_price = round(base_price, 2)
        else:
            # Quantity: log-ish distribution so 1–12 dominates with long tail.
            quantity = int(rng.integers(1, 25))
            unit_price = round(max(0.05, base_price * rng.normal(1.0, 0.15)), 2)

        rows.append(
            {
                "InvoiceNo": invoice_no,
                "StockCode": stock_code,
                "Description": desc,
                "Quantity": quantity,
                "InvoiceDate": ts,
                "UnitPrice": unit_price,
                "CustomerID": cust,
                "Country": customer_country[cust],
            }
        )

    df = pd.DataFrame(rows)
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

    # Stable column order.
    df = df[cfg["schema"]["canonical_columns"]]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"✅ Generated {len(df):,} synthetic rows → {out_path}")
    print(f"   customers      : {df['CustomerID'].nunique():,}")
    print(f"   date range     : {df['InvoiceDate'].min()} → {df['InvoiceDate'].max()}")
    print(f"   cancellations  : {df['InvoiceNo'].str.startswith('C').sum():,}")
    return df


if __name__ == "__main__":
    df = generate_sample()
    print("\nPreview:")
    print(df.head())
