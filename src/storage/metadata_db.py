"""SQLite metadata store - the structured half of the data-storage layer.

Holds product metadata (brand, price, material, style), the cached colour histogram
and predicted tags per product, and the feedback event log + per-(style, product)
boost table. Keeping this cleanly separate from the FAISS vector index is the
"clear separation between AI inference and data storage" the NFRs ask for.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable

from src import config

_SCHEMA = """
CREATE TABLE IF NOT EXISTS products (
    product_id   TEXT PRIMARY KEY,
    image_path   TEXT NOT NULL,
    name         TEXT,
    brand        TEXT,
    price        REAL,
    material     TEXT,
    style        TEXT,
    product_url  TEXT,
    color_hist   TEXT,            -- json list[float]
    tags         TEXT             -- json dict
);

CREATE TABLE IF NOT EXISTS feedback (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    query_id    TEXT,
    product_id  TEXT,
    style       TEXT,
    relevant    INTEGER,          -- 1 relevant, 0 not-relevant
    ts          DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Aggregated boost per visual style so ranking improves from interaction.
CREATE TABLE IF NOT EXISTS boosts (
    style       TEXT,
    product_id  TEXT,
    score       REAL DEFAULT 0,
    PRIMARY KEY (style, product_id)
);
"""


class MetadataDB:
    def __init__(self, path: Path = config.SQLITE_PATH) -> None:
        self.path = path
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    # ------------------------------------------------------------------ #
    # Ingestion
    # ------------------------------------------------------------------ #
    def upsert_product(self, product: dict[str, Any]) -> None:
        self.conn.execute(
            """INSERT INTO products
               (product_id, image_path, name, brand, price, material, style,
                product_url, color_hist, tags)
               VALUES (:product_id, :image_path, :name, :brand, :price, :material,
                       :style, :product_url, :color_hist, :tags)
               ON CONFLICT(product_id) DO UPDATE SET
                 image_path=excluded.image_path, name=excluded.name,
                 brand=excluded.brand, price=excluded.price,
                 material=excluded.material, style=excluded.style,
                 product_url=excluded.product_url, color_hist=excluded.color_hist,
                 tags=excluded.tags""",
            {
                **product,
                "color_hist": json.dumps(product.get("color_hist", [])),
                "tags": json.dumps(product.get("tags", {})),
            },
        )
        self.conn.commit()

    # ------------------------------------------------------------------ #
    # Lookup + filtering
    # ------------------------------------------------------------------ #
    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        d = dict(row)
        d["color_hist"] = json.loads(d.get("color_hist") or "[]")
        d["tags"] = json.loads(d.get("tags") or "{}")
        return d

    def get_products(self, product_ids: Iterable[str]) -> dict[str, dict]:
        ids = list(product_ids)
        if not ids:
            return {}
        marks = ",".join("?" * len(ids))
        rows = self.conn.execute(
            f"SELECT * FROM products WHERE product_id IN ({marks})", ids
        ).fetchall()
        return {r["product_id"]: self._row_to_dict(r) for r in rows}

    def filter_ids(self, candidate_ids: list[str], *, price_min: float | None = None,
                   price_max: float | None = None, brand: str | None = None,
                   material: str | None = None) -> set[str]:
        """Return the subset of candidate_ids matching the structured filters."""
        if not candidate_ids:
            return set()
        clauses = [f"product_id IN ({','.join('?' * len(candidate_ids))})"]
        params: list[Any] = list(candidate_ids)
        if price_min is not None:
            clauses.append("price >= ?"); params.append(price_min)
        if price_max is not None:
            clauses.append("price <= ?"); params.append(price_max)
        if brand:
            clauses.append("LOWER(brand) = LOWER(?)"); params.append(brand)
        if material:
            clauses.append("LOWER(material) = LOWER(?)"); params.append(material)
        sql = f"SELECT product_id FROM products WHERE {' AND '.join(clauses)}"
        return {r["product_id"] for r in self.conn.execute(sql, params).fetchall()}

    def distinct(self, column: str) -> list[str]:
        assert column in {"brand", "material", "style"}
        rows = self.conn.execute(
            f"SELECT DISTINCT {column} FROM products WHERE {column} IS NOT NULL "
            f"ORDER BY {column}"
        ).fetchall()
        return [r[column] for r in rows]

    def price_bounds(self) -> tuple[float, float]:
        row = self.conn.execute("SELECT MIN(price) lo, MAX(price) hi FROM products").fetchone()
        return (float(row["lo"] or 0), float(row["hi"] or 0))

    def count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) c FROM products").fetchone()["c"]

    # ------------------------------------------------------------------ #
    # Feedback loop
    # ------------------------------------------------------------------ #
    def record_feedback(self, query_id: str, product_id: str, style: str,
                        relevant: bool) -> None:
        self.conn.execute(
            "INSERT INTO feedback (query_id, product_id, style, relevant) VALUES (?,?,?,?)",
            (query_id, product_id, style, int(relevant)),
        )
        delta = config.FEEDBACK_BOOST_PER_CLICK * (1 if relevant else -1)
        self.conn.execute(
            """INSERT INTO boosts (style, product_id, score) VALUES (?, ?, ?)
               ON CONFLICT(style, product_id) DO UPDATE SET
                 score = MAX(?, MIN(?, score + ?))""",
            (style, product_id, max(-config.FEEDBACK_BOOST_CAP,
                                    min(config.FEEDBACK_BOOST_CAP, delta)),
             -config.FEEDBACK_BOOST_CAP, config.FEEDBACK_BOOST_CAP, delta),
        )
        self.conn.commit()

    def get_boosts(self, style: str) -> dict[str, float]:
        rows = self.conn.execute(
            "SELECT product_id, score FROM boosts WHERE style = ?", (style,)
        ).fetchall()
        return {r["product_id"]: float(r["score"]) for r in rows}

    def close(self) -> None:
        self.conn.close()
