"""BrandPulse AI — Pipeline Tests.

Tests for the data loading, schema validation, and pipeline components.

Run:
    python -m pytest tests/test_pipeline.py -v
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def test_schema_validation():
    """Test that all Pydantic models validate correctly."""
    from src.models.schemas import (
        Alert,
        AlertSeverity,
        BrandInsight,
        ClassificationResult,
        EntityResult,
        FeedbackRecord,
        OrchestratorResult,
        ProcessedFeedback,
        SentimentResult,
    )

    # FeedbackRecord
    record = FeedbackRecord(
        id="TEST-001",
        brand="glownest",
        product="Vitamin C Serum",
        channel="amazon",
        text="Great product!",
        rating=5,
    )
    assert record.id == "TEST-001"
    assert record.brand == "glownest"

    # ClassificationResult
    classification = ClassificationResult(
        primary_category="positive_feedback",
        urgency=1,
        confidence=0.95,
        reasoning="Clearly positive review",
    )
    assert classification.urgency == 1

    # SentimentResult
    sentiment = SentimentResult(
        score=0.85,
        label="positive",
        emotions=["joy", "satisfaction"],
        key_phrases=["great product"],
        confidence=0.9,
    )
    assert sentiment.score == 0.85

    # EntityResult
    entities = EntityResult(
        brand_mentioned="GlowNest",
        products_mentioned=["Vitamin C Serum"],
        competitor_mentions=["Mamaearth"],
    )
    assert len(entities.competitor_mentions) == 1

    # OrchestratorResult
    orchestrator = OrchestratorResult(
        summary="Positive feedback about Vitamin C Serum",
        action_required=False,
        tags=["positive", "serum"],
    )
    assert not orchestrator.action_required

    # Alert
    alert = Alert(
        id="ALT-test",
        severity=AlertSeverity.HIGH,
        brand="glownest",
        title="Test Alert",
        description="Test description",
        trigger_reason="Test trigger",
    )
    assert alert.severity == AlertSeverity.HIGH

    print("[PASS] All schema validations passed")


def test_data_loading():
    """Test that sample data loads and parses correctly."""
    from src.ingestion.loader import FeedbackLoader

    loader = FeedbackLoader()
    records = loader.load_all()

    assert len(records) > 0, "No records loaded"
    assert len(records) >= 100, f"Expected 100+ records, got {len(records)}"

    # Check schema of first record
    first = records[0]
    assert "id" in first
    assert "brand" in first
    assert "text" in first
    assert "channel" in first

    print(f"[PASS] Loaded {len(records)} records successfully")


def test_brand_filtering():
    """Test that brand filtering works correctly."""
    from src.ingestion.loader import FeedbackLoader

    loader = FeedbackLoader()
    records = loader.load_all()

    glownest = loader.filter_by_brand(records, "glownest")
    pureroots = loader.filter_by_brand(records, "pureroots")
    urbanmane = loader.filter_by_brand(records, "urbanmane")

    assert len(glownest) > 0, "No GlowNest records"
    assert len(pureroots) > 0, "No PureRoots records"
    assert len(urbanmane) > 0, "No UrbanMane records"
    assert len(glownest) + len(pureroots) + len(urbanmane) == len(records)

    print(
        f"[PASS] Brand filtering: GlowNest={len(glownest)}, "
        f"PureRoots={len(pureroots)}, UrbanMane={len(urbanmane)}"
    )


def test_statistics():
    """Test that statistics calculation works."""
    from src.ingestion.loader import FeedbackLoader

    loader = FeedbackLoader()
    records = loader.load_all()
    stats = loader.get_stats(records)

    assert stats["total"] > 0
    assert "brands" in stats
    assert "channels" in stats
    assert stats["avg_rating"] > 0

    print(f"[PASS] Statistics: {json.dumps(stats, indent=2)}")


def test_config_loading():
    """Test that brand configuration loads correctly."""
    from src.utils.helpers import (
        get_brand_names,
        get_brand_products,
        load_brands_config,
    )

    config = load_brands_config()
    assert "brands" in config
    assert "settings" in config

    names = get_brand_names()
    assert "glownest" in names
    assert "pureroots" in names
    assert "urbanmane" in names

    products = get_brand_products("glownest")
    assert len(products) > 0
    assert any("Vitamin C" in p["name"] for p in products)

    print(f"[PASS] Config loaded: {len(names)} brands, {len(products)} GlowNest products")


def test_id_generation():
    """Test unique ID generation."""
    from src.utils.helpers import generate_alert_id, generate_id, generate_insight_id

    ids = {generate_id() for _ in range(100)}
    assert len(ids) == 100, "Generated duplicate IDs"

    alert_id = generate_alert_id()
    assert alert_id.startswith("ALT-")

    insight_id = generate_insight_id()
    assert insight_id.startswith("INS-")

    print("[PASS] ID generation: unique and correctly prefixed")


def test_anomaly_detector_initialization():
    """Test that AnomalyDetector initializes with config."""
    from src.alerts.notifier import AnomalyDetector

    detector = AnomalyDetector()
    assert detector.sentiment_threshold > 0
    assert detector.spike_threshold > 0
    assert detector.min_reviews > 0

    print(
        f"[PASS] AnomalyDetector initialized: "
        f"sentiment_threshold={detector.sentiment_threshold}, "
        f"spike_threshold={detector.spike_threshold}"
    )


def test_graph_state_type():
    """Test that GraphState TypedDict works correctly."""
    from src.models.schemas import GraphState

    state: GraphState = {
        "record": {"id": "TEST-001", "brand": "test", "text": "hello"},
        "classification": None,
        "sentiment": None,
        "entities": None,
        "orchestrator": None,
        "alerts": [],
        "errors": [],
        "processed": False,
    }
    assert state["record"]["id"] == "TEST-001"
    assert not state["processed"]

    print("[PASS] GraphState TypedDict works correctly")


if __name__ == "__main__":
    print("=" * 60)
    print(" BrandPulse AI — Test Suite")
    print("=" * 60)
    print()

    tests = [
        test_schema_validation,
        test_data_loading,
        test_brand_filtering,
        test_statistics,
        test_config_loading,
        test_id_generation,
        test_anomaly_detector_initialization,
        test_graph_state_type,
    ]

    passed = 0
    failed = 0
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"[FAIL] {test_fn.__name__}: {e}")
            failed += 1

    print()
    print("=" * 60)
    print(f" Results: {passed} passed, {failed} failed")
    print("=" * 60)
