"""Feedback loop (mandatory requirement 3.4).

Thin service over the metadata DB's feedback/boost tables. Records Relevant /
Not-Relevant clicks per (query style, product) and lets the search ranker read a
learned boost, so frequently-approved products for a given visual style surface
higher over time. Boosts are bounded (config.FEEDBACK_BOOST_CAP) so feedback nudges
rather than dominates the visual signal.
"""
from __future__ import annotations

from src.storage.metadata_db import MetadataDB


class FeedbackService:
    def __init__(self, db: MetadataDB | None = None) -> None:
        self.db = db or MetadataDB()

    def record(self, query_id: str, product_id: str, style: str, relevant: bool) -> dict:
        """Log a click and return the updated boost for that (style, product)."""
        self.db.record_feedback(query_id, product_id, style, relevant)
        new_boost = self.db.get_boosts(style).get(product_id, 0.0)
        return {"product_id": product_id, "style": style, "boost": round(new_boost, 4)}
