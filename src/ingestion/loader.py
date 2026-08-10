"""BrandPulse AI — Data Ingestion & Normalization.

Handles loading, normalizing, and filtering customer feedback
from multiple data sources.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from src.models.schemas import Channel, FeedbackRecord
from src.utils.helpers import DATA_DIR, load_sample_reviews, parse_date

logger = logging.getLogger("brandpulse.ingestion")


class FeedbackLoader:
    """Loads and normalizes customer feedback from various sources.

    Currently supports:
    - JSON file ingestion (sample data)
    - Extensible for API-based ingestion (Amazon, Flipkart, etc.)

    Usage:
        loader = FeedbackLoader()
        records = loader.load_all()
        filtered = loader.filter_by_brand(records, "glownest")
    """

    def __init__(self, data_dir: Path | None = None) -> None:
        """Initialize the loader.

        Args:
            data_dir: Optional custom data directory path.
        """
        self.data_dir = data_dir or DATA_DIR

    def load_from_json(
        self, filepath: Path | str | None = None
    ) -> list[dict[str, Any]]:
        """Load reviews from a JSON file.

        Args:
            filepath: Path to JSON file. Defaults to sample_reviews.json.

        Returns:
            List of normalized review dictionaries.
        """
        if filepath:
            path = Path(filepath)
            with open(path, "r", encoding="utf-8") as f:
                raw_reviews = json.load(f)
        else:
            raw_reviews = load_sample_reviews()

        logger.info(f"[Ingestion] Loaded {len(raw_reviews)} raw reviews")
        return self._normalize_records(raw_reviews)

    def load_all(self) -> list[dict[str, Any]]:
        """Load all available feedback from all sources.

        Returns:
            Consolidated list of normalized feedback records.
        """
        all_records = []

        # Load from default JSON
        try:
            json_records = self.load_from_json()
            all_records.extend(json_records)
        except FileNotFoundError:
            logger.warning("[Ingestion] No sample_reviews.json found")

        logger.info(
            f"[Ingestion] Total records loaded: {len(all_records)}"
        )
        return all_records

    def _normalize_records(
        self, raw_records: list[dict]
    ) -> list[dict[str, Any]]:
        """Normalize raw records to consistent schema.

        Args:
            raw_records: List of raw review dictionaries.

        Returns:
            List of normalized dictionaries matching FeedbackRecord schema.
        """
        normalized = []
        for raw in raw_records:
            try:
                record = self._normalize_single(raw)
                normalized.append(record)
            except Exception as e:
                logger.warning(
                    f"[Ingestion] Skipping malformed record "
                    f"{raw.get('id', '?')}: {e}"
                )
        return normalized

    def _normalize_single(self, raw: dict) -> dict[str, Any]:
        """Normalize a single raw record.

        Args:
            raw: Raw review dictionary.

        Returns:
            Normalized dictionary matching FeedbackRecord schema.
        """
        # Parse date
        date_val = raw.get("date")
        if isinstance(date_val, str):
            date_val = parse_date(date_val).isoformat()

        # Normalize channel
        channel = raw.get("channel", "other")
        valid_channels = {c.value for c in Channel}
        if channel not in valid_channels:
            channel = "amazon"  # Default fallback

        return {
            "id": raw.get("id", ""),
            "brand": raw.get("brand", "").lower().strip(),
            "product": raw.get("product", ""),
            "sku": raw.get("sku", ""),
            "channel": channel,
            "rating": raw.get("rating"),
            "title": raw.get("title", ""),
            "text": raw.get("text", ""),
            "author": raw.get("author", "anonymous"),
            "date": date_val,
            "verified_purchase": raw.get("verified_purchase", False),
            "language": raw.get("language", "en"),
        }

    # ── Filtering Methods ────────────────────────────────────────────

    @staticmethod
    def filter_by_brand(
        records: list[dict], brand: str
    ) -> list[dict]:
        """Filter records for a specific brand.

        Args:
            records: List of feedback records.
            brand: Brand slug to filter by.

        Returns:
            Filtered list of records.
        """
        return [r for r in records if r.get("brand") == brand.lower()]

    @staticmethod
    def filter_by_channel(
        records: list[dict], channel: str
    ) -> list[dict]:
        """Filter records for a specific channel.

        Args:
            records: List of feedback records.
            channel: Channel name to filter by.

        Returns:
            Filtered list of records.
        """
        return [r for r in records if r.get("channel") == channel]

    @staticmethod
    def filter_by_date_range(
        records: list[dict],
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> list[dict]:
        """Filter records within a date range.

        Args:
            records: List of feedback records.
            start_date: Inclusive start date.
            end_date: Inclusive end date.

        Returns:
            Filtered list of records.
        """
        filtered = []
        for r in records:
            try:
                record_date = parse_date(
                    r["date"][:10] if isinstance(r["date"], str) else str(r["date"])[:10]
                )
                if start_date and record_date < start_date:
                    continue
                if end_date and record_date > end_date:
                    continue
                filtered.append(r)
            except (ValueError, KeyError):
                continue
        return filtered

    @staticmethod
    def filter_negative(
        records: list[dict], max_rating: int = 2
    ) -> list[dict]:
        """Filter records with low ratings (likely negative feedback).

        Args:
            records: List of feedback records.
            max_rating: Maximum rating to include (inclusive).

        Returns:
            Filtered list of records.
        """
        return [
            r for r in records
            if r.get("rating") is not None and r["rating"] <= max_rating
        ]

    # ── Statistics ───────────────────────────────────────────────────

    @staticmethod
    def get_stats(records: list[dict]) -> dict[str, Any]:
        """Calculate basic statistics for a set of records.

        Args:
            records: List of feedback records.

        Returns:
            Dictionary with statistics.
        """
        total = len(records)
        if total == 0:
            return {"total": 0}

        brands = {}
        channels = {}
        ratings = []

        for r in records:
            brand = r.get("brand", "unknown")
            brands[brand] = brands.get(brand, 0) + 1

            channel = r.get("channel", "unknown")
            channels[channel] = channels.get(channel, 0) + 1

            if r.get("rating") is not None:
                ratings.append(r["rating"])

        avg_rating = sum(ratings) / len(ratings) if ratings else 0

        return {
            "total": total,
            "brands": brands,
            "channels": channels,
            "avg_rating": round(avg_rating, 2),
            "rated_count": len(ratings),
            "unrated_count": total - len(ratings),
        }
