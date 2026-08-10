"""BrandPulse AI — Orchestrator Agent.

The orchestrator aggregates results from all analysis agents,
detects anomalies, generates cross-brand insights, and decides
on alert creation. It is the 'brain' of the pipeline.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from tenacity import retry, stop_after_attempt, wait_exponential

from src.models.schemas import GraphState, OrchestratorResult
from src.utils.helpers import get_llm

logger = logging.getLogger("brandpulse.orchestrator")

# ── System Prompt ────────────────────────────────────────────────────

ORCHESTRATOR_SYSTEM_PROMPT = """You are the Orchestrator Agent for BrandPulse AI — the central intelligence system for Think9's 30+ consumer brand portfolio.

You receive the aggregated outputs of three specialist agents (Classifier, Sentiment Analyzer, Entity Extractor) for a single customer feedback item. Your job is to synthesize these signals into actionable intelligence.

## Your Responsibilities:

1. **Summarize**: Create a concise one-line summary capturing the essence of this feedback.

2. **Action Decision**: Determine if this feedback requires team action.
   - Action required if: urgency >= 4, safety concerns, repeated issues, quality control failures
   - No action required for: general positive feedback, minor preferences, informational queries

3. **Recommended Action**: If action is required, suggest a specific action:
   - "Escalate to QC team — investigate packaging defect"
   - "Flag for product team — reformulation needed"
   - "Route to customer service — immediate response needed"
   - "Add to trend watchlist — monitor for recurrence"

4. **Cross-Brand Relevance**: Identify if this feedback is relevant to OTHER brands in the portfolio.
   - A packaging issue at GlowNest might apply to PureRoots if they use similar packaging
   - An ingredient concern at UrbanMane might apply to GlowNest if they share ingredients
   - A delivery issue could affect all brands
   - Return the list of other brand slugs this applies to: ["pureroots", "urbanmane", etc.]
   - Return empty list if this is brand-specific only

5. **Tags**: Generate 3-5 short descriptive tags for this feedback for search and filtering.

## Context:
- Think9 operates 30+ brands. Currently monitoring: GlowNest (skincare), PureRoots (organic food), UrbanMane (haircare)
- Cross-brand patterns are extremely valuable — a problem in one brand often predicts the same issue in another
- Packaging, delivery, and customer service issues are usually cross-brand relevant
- Product-specific issues (effectiveness, taste) are usually brand-specific

Respond ONLY with valid JSON matching this schema:
{
  "summary": "string (one concise line)",
  "action_required": boolean,
  "recommended_action": "string (empty if no action required)",
  "cross_brand_relevance": ["brand_slug", ...],
  "tags": ["string", ...]
}"""


# ── Agent Function ───────────────────────────────────────────────────


def _get_llm():
    """Initialize the LLM for orchestration."""
    return get_llm(temperature=0.2, max_tokens=500)


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=4, max=30),
)
def _orchestrate(
    record: dict,
    classification: dict,
    sentiment: dict,
    entities: dict,
) -> dict:
    """Call Gemini to orchestrate and synthesize agent outputs."""
    llm = _get_llm()

    user_prompt = f"""Synthesize the following agent outputs for this feedback item:

## Original Feedback:
- ID: {record.get('id')}
- Brand: {record.get('brand')}
- Product: {record.get('product')}
- Channel: {record.get('channel')}
- Rating: {record.get('rating', 'N/A')}
- Title: {record.get('title', '')}
- Text: {record.get('text', '')}

## Classifier Output:
{json.dumps(classification, indent=2)}

## Sentiment Analysis Output:
{json.dumps(sentiment, indent=2)}

## Entity Extraction Output:
{json.dumps(entities, indent=2)}

Based on all signals, provide your orchestration output as JSON."""

    messages = [
        SystemMessage(content=ORCHESTRATOR_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]

    response = llm.invoke(messages)
    content = response.content.strip()

    # Strip markdown code fences if present
    if content.startswith("```"):
        content = content.split("\n", 1)[1] if "\n" in content else content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

    return json.loads(content)


def orchestrate_feedback(state: GraphState) -> dict[str, Any]:
    """LangGraph node: Orchestrate and synthesize all agent outputs.

    This is the final processing node that runs AFTER the classifier,
    sentiment, and entity extraction nodes have completed.

    Args:
        state: Current graph state with all agent results.

    Returns:
        Updated state dict with orchestrator results.
    """
    record = state["record"]
    classification = state.get("classification") or {}
    sentiment = state.get("sentiment") or {}
    entities = state.get("entities") or {}
    errors = list(state.get("errors", []))

    try:
        result = _orchestrate(
            record=record,
            classification=classification,
            sentiment=sentiment,
            entities=entities,
        )

        orchestrator = OrchestratorResult(**result)
        action_indicator = "⚡ ACTION" if orchestrator.action_required else "✓ OK"
        logger.info(
            f"[Orchestrator] {record['id']}: "
            f"{action_indicator} — {orchestrator.summary[:60]}"
        )
        return {
            "orchestrator": orchestrator.model_dump(),
            "processed": True,
        }

    except Exception as e:
        error_msg = (
            f"Orchestration failed for {record.get('id', '?')}: {e}"
        )
        logger.error(f"[Orchestrator] {error_msg}")
        errors.append(error_msg)
        return {
            "orchestrator": {
                "summary": f"Processing error: {e}",
                "action_required": False,
                "recommended_action": "",
                "cross_brand_relevance": [],
                "tags": ["processing_error"],
            },
            "processed": True,
            "errors": errors,
        }
