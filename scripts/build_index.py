"""Run the full ingestion pipeline over data/catalog.csv -> FAISS index + SQLite.

Usage:  python -m scripts.build_index
"""
from __future__ import annotations

from src import config
from src.pipeline.ingest import build_index
from src.utils.logging import get_logger

log = get_logger()


def main() -> None:
    if not config.CATALOG_CSV.exists():
        log.error("data/catalog.csv missing. Run scripts/collect_data.py first "
                  "(--download or --from-folder).")
        return
    summary = build_index()
    print("\n=== Ingestion summary ===")
    for k, v in summary.items():
        print(f"  {k:>14}: {v}")
    print(f"  FAISS index -> {config.FAISS_INDEX_PATH}")
    print(f"  SQLite db   -> {config.SQLITE_PATH}")


if __name__ == "__main__":
    main()
