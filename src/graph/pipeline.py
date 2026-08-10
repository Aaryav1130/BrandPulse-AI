"""BrandPulse AI — LangGraph Processing Pipeline.

Wires all agents into a LangGraph StateGraph. Uses sequential
execution of classifier → sentiment → entity_extractor to respect
API rate limits, followed by the orchestrator.

Pipeline Flow:
  ingest → classifier → sentiment → entity_extractor → orchestrator → alert_check
"""

from __future__ import annotations

import logging
import time
from typing import Any

from langgraph.graph import END, StateGraph

from src.agents.classifier import classify_feedback
from src.agents.entity_extractor import extract_entities
from src.agents.orchestrator import orchestrate_feedback
from src.agents.sentiment import analyze_sentiment
from src.alerts.notifier import check_and_create_alerts
from src.models.schemas import (
    FeedbackRecord,
    GraphState,
    ProcessedFeedback,
)

logger = logging.getLogger("brandpulse.pipeline")

# Rate limit delay between API calls (seconds)
# Groq free tier: 30 RPM, 6000 tokens/min — 4s avoids most retries
# which is faster overall than 2s + frequent 3-5s retry waits
RATE_LIMIT_DELAY = 4


# ── Node Functions ───────────────────────────────────────────────────


def ingest_node(state: GraphState) -> dict[str, Any]:
    """Entry node: validate and prepare the feedback record."""
    record = state["record"]
    logger.info(
        f"[Pipeline] Processing {record['id']} — "
        f"{record['brand']}/{record['product']}"
    )
    return {
        "errors": [],
        "processed": False,
        "classification": None,
        "sentiment": None,
        "entities": None,
        "orchestrator": None,
        "alerts": [],
    }


def rate_limit_after_classify(state: GraphState) -> dict[str, Any]:
    """Add delay after classifier to respect API rate limits."""
    time.sleep(RATE_LIMIT_DELAY)
    return {}


def rate_limit_after_sentiment(state: GraphState) -> dict[str, Any]:
    """Add delay after sentiment to respect API rate limits."""
    time.sleep(RATE_LIMIT_DELAY)
    return {}


def rate_limit_after_entity(state: GraphState) -> dict[str, Any]:
    """Add delay after entity extraction to respect API rate limits."""
    time.sleep(RATE_LIMIT_DELAY)
    return {}


def alert_check_node(state: GraphState) -> dict[str, Any]:
    """Final node: check if alerts should be generated."""
    alerts = check_and_create_alerts(state)
    return {"alerts": alerts}


# ── Graph Builder ────────────────────────────────────────────────────


def build_pipeline() -> StateGraph:
    """Build and compile the LangGraph processing pipeline.

    Architecture (sequential to respect free-tier rate limits):
        ingest → classifier → delay → sentiment → delay
               → entity_extractor → delay → orchestrator → alert_check → END

    In production (with paid API), this can be switched to parallel
    fan-out/fan-in for 3x faster processing.

    Returns:
        Compiled LangGraph application.
    """
    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node("ingest", ingest_node)
    workflow.add_node("classifier", classify_feedback)
    workflow.add_node("delay_1", rate_limit_after_classify)
    workflow.add_node("sentiment", analyze_sentiment)
    workflow.add_node("delay_2", rate_limit_after_sentiment)
    workflow.add_node("entity_extractor", extract_entities)
    workflow.add_node("delay_3", rate_limit_after_entity)
    workflow.add_node("orchestrator", orchestrate_feedback)
    workflow.add_node("alert_check", alert_check_node)

    # Set entry point
    workflow.set_entry_point("ingest")

    # Sequential chain with rate limiting delays
    workflow.add_edge("ingest", "classifier")
    workflow.add_edge("classifier", "delay_1")
    workflow.add_edge("delay_1", "sentiment")
    workflow.add_edge("sentiment", "delay_2")
    workflow.add_edge("delay_2", "entity_extractor")
    workflow.add_edge("entity_extractor", "delay_3")
    workflow.add_edge("delay_3", "orchestrator")

    # Orchestrator → alert check → END
    workflow.add_edge("orchestrator", "alert_check")
    workflow.add_edge("alert_check", END)

    return workflow.compile()


# ── Pipeline Runner ──────────────────────────────────────────────────


class BrandPulsePipeline:
    """Main pipeline class for processing customer feedback.

    Usage:
        pipeline = BrandPulsePipeline()
        results = pipeline.process_batch(records)
    """

    def __init__(self) -> None:
        """Initialize the pipeline by building the LangGraph."""
        logger.info("[Pipeline] Building LangGraph pipeline...")
        self.app = build_pipeline()
        logger.info("[Pipeline] Pipeline ready")

    def process_single(self, record: dict) -> dict[str, Any]:
        """Process a single feedback record through the pipeline.

        Args:
            record: Dictionary matching FeedbackRecord schema.

        Returns:
            Final state dict with all agent outputs.
        """
        # Validate the record
        feedback = FeedbackRecord(**record)
        initial_state: GraphState = {
            "record": feedback.model_dump(mode="json"),
            "classification": None,
            "sentiment": None,
            "entities": None,
            "orchestrator": None,
            "alerts": [],
            "errors": [],
            "processed": False,
        }

        start = time.time()
        result = self.app.invoke(initial_state)
        elapsed = time.time() - start

        logger.info(
            f"[Pipeline] Completed {record.get('id', '?')} "
            f"in {elapsed:.1f}s"
        )
        return result

    def process_batch(
        self,
        records: list[dict],
        max_records: int | None = None,
    ) -> list[dict[str, Any]]:
        """Process a batch of feedback records.

        Args:
            records: List of dictionaries matching FeedbackRecord schema.
            max_records: Optional limit on number of records to process.

        Returns:
            List of final state dicts with all agent outputs.
        """
        if max_records:
            records = records[:max_records]

        total = len(records)
        logger.info(f"[Pipeline] Starting batch processing of {total} records")

        results = []
        for idx, record in enumerate(records, 1):
            logger.info(f"[Pipeline] Processing record {idx}/{total}")
            try:
                result = self.process_single(record)
                results.append(result)
            except Exception as e:
                logger.error(
                    f"[Pipeline] Failed to process {record.get('id', '?')}: {e}"
                )
                results.append({
                    "record": record,
                    "errors": [str(e)],
                    "processed": False,
                })

        # Summary statistics
        successful = sum(1 for r in results if r.get("processed", False))
        failed = total - successful
        logger.info(
            f"[Pipeline] Batch complete: "
            f"{successful} succeeded, {failed} failed"
        )

        return results

    def to_processed_feedback(
        self, results: list[dict]
    ) -> list[ProcessedFeedback]:
        """Convert raw pipeline results to ProcessedFeedback models.

        Args:
            results: Raw results from process_batch.

        Returns:
            List of validated ProcessedFeedback objects.
        """
        processed = []
        for result in results:
            if not result.get("processed"):
                continue
            try:
                pf = ProcessedFeedback(
                    record=FeedbackRecord(**result["record"]),
                    classification=result.get("classification"),
                    sentiment=result.get("sentiment"),
                    entities=result.get("entities"),
                    orchestrator=result.get("orchestrator"),
                    processing_errors=result.get("errors", []),
                )
                processed.append(pf)
            except Exception as e:
                logger.warning(f"Could not convert result: {e}")
        return processed
