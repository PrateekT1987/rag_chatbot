"""CSV catalog loading, validation and hot-reload.

The catalog is a plain CSV file (see data/products.csv). To update products,
edit the CSV — no code changes and no restart required: the file is
re-checked (and re-indexed) automatically whenever its modification time
changes, or when POST /api/reindex is called.

Expected columns:
    sku, name, category, price, currency, availability, description, specs, tags

- `sku`, `name`, `price`, `description` are required.
- `specs` uses "Key: Value; Key: Value" pairs so one format works for any
  product type (e.g. "Battery life: 30h; Bluetooth: 5.3").
"""
from __future__ import annotations

import csv
import logging
import os
import re
import threading
from pathlib import Path

from .models import Product

log = logging.getLogger("catalog")

REQUIRED_COLUMNS = ("sku", "name", "price", "description")
OPTIONAL_COLUMNS = ("category", "currency", "availability", "specs", "tags", "url")


class CatalogError(Exception):
    """Raised when the CSV is missing or structurally unusable."""


def _clean(value: str | None) -> str:
    return (value or "").strip()


def _parse_price(value: str) -> float | None:
    cleaned = re.sub(r"[^\d.]", "", value or "")
    if not cleaned:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def _product_document(p: Product) -> str:
    """Flatten a product row into the text block used for indexing and for
    the LLM context. Kept close to the CSV so answers stay faithful."""
    parts = [
        f"SKU: {p.sku}",
        f"Product: {p.name}",
        f"Category: {p.category}",
        f"Price: {p.price:g} {p.currency}",
        f"Availability: {p.availability}",
        f"Description: {p.description}",
    ]
    if p.specs:
        parts.append(f"Specifications: {p.specs}")
    if p.tags:
        parts.append(f"Tags: {p.tags}")
    return " | ".join(parts)


class Catalog:
    """Thread-safe catalog snapshot with cheap mtime-based hot-reload."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._mtime: float | None = None
        self.products: list[Product] = []
        self.documents: list[str] = []
        self.load(force=True)

    # -- public -----------------------------------------------------------

    def load(self, force: bool = False) -> bool:
        """Reload the CSV if it changed (or if force=True).

        Returns True when the catalog was reloaded. On a *changed* file with
        fatal errors the old catalog is kept and the error is logged, so a
        bad edit never takes the service down.
        """
        with self._lock:
            if not self.path.exists():
                if force:
                    raise CatalogError(f"Catalog file not found: {self.path}")
                return False

            mtime = os.path.getmtime(self.path)
            if not force and self._mtime == mtime:
                return False

            try:
                products = self._read_csv()
            except CatalogError as exc:
                if force:
                    raise
                log.error("Reload failed, keeping previous catalog: %s", exc)
                return False

            self.products = products
            self.documents = [_product_document(p) for p in products]
            self._mtime = mtime
            log.info("Loaded %d products from %s", len(products), self.path)
            return True

    def get(self, sku: str) -> Product | None:
        sku = sku.strip().upper()
        for p in self.products:
            if p.sku.upper() == sku:
                return p
        return None

    # -- internal ---------------------------------------------------------

    def _read_csv(self) -> list[Product]:
        try:
            with self.path.open(newline="", encoding="utf-8-sig") as fh:
                reader = csv.DictReader(fh)
                if not reader.fieldnames:
                    raise CatalogError("CSV file is empty or has no header row.")
                header = {h.strip().lower() for h in reader.fieldnames if h}
                missing = [c for c in REQUIRED_COLUMNS if c not in header]
                if missing:
                    raise CatalogError(
                        f"CSV is missing required column(s): {', '.join(missing)}. "
                        f"Found: {', '.join(sorted(header))}"
                    )
                rows = list(reader)
        except (OSError, csv.Error) as exc:
            raise CatalogError(f"Could not read CSV: {exc}") from exc

        products: list[Product] = []
        seen_skus: set[str] = set()
        for line_no, row in enumerate(rows, start=2):  # start=2: row 1 is the header
            row = {(k or "").strip().lower(): (v or "").strip() for k, v in row.items()}
            sku = _clean(row.get("sku")).upper()
            name = _clean(row.get("name"))
            description = _clean(row.get("description"))

            if not any(row.values()):
                continue  # skip fully blank lines
            if not sku or not name:
                log.warning("Line %d skipped: missing sku or name", line_no)
                continue
            if sku in seen_skus:
                log.warning("Line %d skipped: duplicate SKU %s", line_no, sku)
                continue

            price = _parse_price(row.get("price", ""))
            if price is None:
                log.warning("Line %d (%s): missing/invalid price", line_no, sku)
                price = 0.0

            products.append(
                Product(
                    sku=sku,
                    name=name,
                    category=_clean(row.get("category")),
                    price=price,
                    currency=(_clean(row.get("currency")) or "USD").upper(),
                    availability=_clean(row.get("availability")),
                    description=description,
                    specs=_clean(row.get("specs")),
                    tags=_clean(row.get("tags")),
                    url=_clean(row.get("url")),
                )
            )

        if not products:
            raise CatalogError("CSV contains no valid product rows.")
        return products
