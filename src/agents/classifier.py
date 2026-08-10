"""BrandPulse AI — Classifier Agent.

Classifies customer feedback into issue categories with urgency levels
using Google Gemini via LangChain.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from tenacity import retry, stop_after_attempt, wait_exponential

from src.models.schemas import ClassificationResult, GraphState
from src.utils.helpers import get_llm

logger = logging.getLogger("brandpulse.classifier")

# ── System Prompt ────────────────────────────────────────────────────

CLASSIFIER_SYSTEM_PROMPT = """You are an expert customer feedback classifier for a consumer brands company managing 30+ brands across skincare, food, and haircare categories in the Indian market.

Your task is to classify customer feedback into ONE primary category and optionally a secondary category, and assign an urgency level.

## Categories:
- product_quality: Issues with the product itself (defects, inconsistency, authenticity concerns)
- packaging: Packaging damage, leaks, poor design, shipping damage
- delivery: Late delivery, wrong item, logistics issues
- pricing: Price complaints, value-for-money concerns
- customer_service: Support response time, resolution quality
- effectiveness: Product not working as expected/marketed
- side_effects: Allergic reactions, adverse effects, safety concerns
- taste_texture: Taste, texture, smell issues (for food/beverage products)
- value_for_money: Cost vs benefit analysis, competitor price comparisons
- positive_feedback: Genuine praise, satisfaction, repurchase intent
- other: Doesn't fit any above category

## Urgency Levels (1-5):
1 = Informational (positive feedback, general comments)
2 = Low (minor suggestions, preference-based complaints)
3 = Medium (product issues that affect experience but not safety)
4 = High (quality control failures, misleading claims, repeated issues)
5 = Critical (safety concerns, allergic reactions, legal implications)

## Rules:
- If the review mentions a safety issue (allergic reaction, side effects), urgency MUST be 4 or 5
- If the review is clearly positive/praise, category should be "positive_feedback" with urgency 1
- Provide a brief reasoning (1-2 sentences) for your classification
- Set requires_immediate_action=true for urgency 4 or 5

Respond ONLY with valid JSON matching this schema:
{
  "primary_category": "string",
  "secondary_category": "string or null",
  "urgency": integer (1-5),
  "requires_immediate_action": boolean,
  "confidence": float (0.0-1.0),
  "reasoning": "string"
}"""


# ── Agent Function ───────────────────────────────────────────────────


def _get_llm():
    """Initialize the LLM for classification."""
    return get_llm(temperature=0.1, max_tokens=500)


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=4, max=30),
)
def _classify_text(text: str, title: str, brand: str, product: str) -> dict:
    """Call Gemini to classify a single feedback item."""
    llm = _get_llm()

    user_prompt = f"""Classify this customer feedback:

Brand: {brand}
Product: {product}
Title: {title}
Review Text: {text}

Respond with JSON only."""

    messages = [
        SystemMessage(content=CLASSIFIER_SYSTEM_PROMPT),
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


def classify_feedback(state: GraphState) -> dict[str, Any]:
    """LangGraph node: Classify a feedback record.

    Args:
        state: Current graph state with the feedback record.

    Returns:
        Updated state dict with classification results.
    """
    record = state["record"]
    errors = list(state.get("errors", []))

    try:
        result = _classify_text(
            text=record.get("text", ""),
            title=record.get("title", ""),
            brand=record.get("brand", ""),
            product=record.get("product", ""),
        )

        # Validate through Pydantic
        classification = ClassificationResult(**result)
        logger.info(
            f"[Classifier] {record['id']}: "
            f"{classification.primary_category} "
            f"(urgency={classification.urgency})"
        )
        return {"classification": classification.model_dump()}

    except Exception as e:
        error_msg = f"Classification failed for {record.get('id', '?')}: {e}"
        logger.error(f"[Classifier] {error_msg}")
        errors.append(error_msg)
        return {
            "classification": {
                "primary_category": "other",
                "secondary_category": None,
                "urgency": 3,
                "requires_immediate_action": False,
                "confidence": 0.0,
                "reasoning": f"Classification failed: {e}",
            },
            "errors": errors,
        }
