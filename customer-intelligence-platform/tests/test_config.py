"""Tests for the central configuration module."""

from pathlib import Path

from src.config import (
    CANONICAL_COLUMNS,
    RENAME_MAP,
    CANCELLED_INVOICE_PREFIX,
    get_config,
)


def test_config_returns_nested_dict():
    cfg = get_config()
    for key in ("paths", "schema", "uci", "preprocessing", "sampling"):
        assert key in cfg, f"missing top-level key: {key}"


def test_canonical_columns_complete():
    # Every downstream module assumes these exact names exist.
    expected = {
        "InvoiceNo", "StockCode", "Description", "Quantity", "InvoiceDate",
        "UnitPrice", "CustomerID", "Country", "TotalPrice",
    }
    assert set(CANONICAL_COLUMNS) == expected
    assert "TotalPrice" in CANONICAL_COLUMNS  # the engineered column


def test_rename_map_maps_uci_to_canonical():
    assert RENAME_MAP["Invoice"] == "InvoiceNo"
    assert RENAME_MAP["Price"] == "UnitPrice"
    assert RENAME_MAP["Customer ID"] == "CustomerID"


def test_cancelled_prefix_is_C():
    assert CANCELLED_INVOICE_PREFIX == "C"


def test_paths_are_absolute_and_inside_project():
    cfg = get_config()
    root: Path = cfg["paths"]["project_root"]
    assert root.is_absolute()
    for key, path in cfg["paths"].items():
        assert Path(path).is_absolute(), f"{key} path is not absolute"
        assert root in Path(path).resolve().parents or Path(path).resolve() == root


def test_sampling_seed_is_int_for_reproducibility():
    cfg = get_config()
    assert isinstance(cfg["sampling"]["random_seed"], int)
    assert cfg["sampling"]["sample_size"] > 0
