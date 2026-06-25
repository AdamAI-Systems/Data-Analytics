"""Download the Online Retail II dataset from the UCI Machine Learning Repository.

The dataset is distributed as a ``.zip`` containing ``online_retail_II.xlsx``
(~46 MB, ~1.07M transactions from a UK online retailer, 2009–2011). This script
fetches it, extracts the Excel file and writes it to ``data/raw/``.

Run it with::

    python scripts/download_data.py

Afterwards run the pipeline::

    python -c "from src.preprocessing import preprocess; preprocess()"
"""

from __future__ import annotations

import io
import sys
import urllib.request
import zipfile
from pathlib import Path

from tqdm import tqdm

# Allow ``python scripts/download_data.py`` from the project root by making
# ``src`` importable without installing the package.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_config  # noqa: E402  (path tweak above is required)


# Candidate download mirrors, tried in order. UCI's static file URL is the
# canonical one; the list is a small safety net in case the host changes.
MIRRORS: list[str] = [
    "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip",
]


def _download(url: str) -> bytes:
    """Download ``url`` with a progress bar and return its raw bytes."""
    with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310 — trusted URL
        total = int(response.headers.get("Content-Length", 0))
        block_size = 1024
        buffer = io.BytesIO()
        with tqdm(
            total=total or None,
            unit="B",
            unit_scale=True,
            desc=url.split("/")[-1],
        ) as bar:
            while True:
                chunk = response.read(block_size)
                if not chunk:
                    break
                buffer.write(chunk)
                bar.update(len(chunk))
    return buffer.getvalue()


def download_and_extract(dest_xlsx: Path | None = None) -> Path:
    """Fetch the Online Retail II zip and extract the ``.xlsx`` to ``data/raw/``.

    Tries each mirror in :data:`MIRRORS` until one succeeds. The downloaded
    archive is scanned for the first ``.xlsx`` entry and that single file is
    written to ``dest_xlsx``.

    Args:
        dest_xlsx: Where to write ``online_retail_II.xlsx``. Defaults to
            ``config.paths.raw_xlsx``.

    Returns:
        The path the Excel file was written to.

    Raises:
        RuntimeError: If every mirror fails.
    """
    cfg = get_config()
    if dest_xlsx is None:
        dest_xlsx = cfg["paths"]["raw_xlsx"]
    dest_xlsx = Path(dest_xlsx)
    dest_xlsx.parent.mkdir(parents=True, exist_ok=True)

    last_error: Exception | None = None
    for url in MIRRORS:
        try:
            print(f"⬇️  Downloading from {url}")
            data = _download(url)
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                xlsx_names = [n for n in archive.namelist() if n.lower().endswith(".xlsx")]
                if not xlsx_names:
                    raise RuntimeError(f"No .xlsx inside archive from {url}")
                target = xlsx_names[0]
                with archive.open(target) as src, open(dest_xlsx, "wb") as out:
                    out.write(src.read())
            print(f"✅ Extracted '{target}' → {dest_xlsx}")
            return dest_xlsx
        except Exception as exc:  # noqa: BLE001 — try the next mirror.
            last_error = exc
            print(f"   ✗ failed: {exc}")

    raise RuntimeError(
        "Could not download the dataset from any mirror. "
        "Download 'online_retail_II.xlsx' manually from "
        "https://archive.ics.uci.edu/dataset/502/online+retail+ii "
        f"and place it at {dest_xlsx}."
    ) from last_error


if __name__ == "__main__":
    out = download_and_extract()
    print("\nNext step: build the cleaned parquet with")
    print("    python -c \"from src.preprocessing import preprocess; preprocess()\"")
    print(f"Output written to: {out}")
