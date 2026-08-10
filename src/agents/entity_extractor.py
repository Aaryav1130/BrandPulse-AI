"""BrandPulse AI — Entity Extractor Agent.

Extracts structured entities from customer feedback including
brand references, products, competitors, and ingredients
using Google Gemini.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from tenacity import retry, stop_after_attempt, wait_exponential

from src.models.schemas import EntityResult, GraphState
from src.utils.helpers import get_llm

logger = logging.getLogger("brandpulse.entity_extractor")

# ── System Prompt ────────────────────────────────────────────────────

ENTITY_EXTRACTOR_SYSTEM_PROMPT = """You are an expert entity extraction agent for consumer brand feedback analysis. You extract structured information from customer reviews to enable cross-brand intelligence.

Your task is to identify and extract the following entities from customer feedback:

## Entities to Extract:

1. **brand_mentioned**: The primary brand being reviewed (GlowNest, PureRoots, UrbanMane, or others)

2. **products_mentioned**: Specific products referenced in the review. Include the exact product name(s).

3. **competitor_mentions**: Any competitor brands or products mentioned for comparison. Common Indian market competitors include:
   - Skincare: Mamaearth, Minimalist, Dot & Key, Plum, Forest Essentials, Kama Ayurveda, The Ordinary, SkinCeuticals, Wow
   - Food: Coco Soul, KLF Nirmal, Dabur, Patanjali, Organic India, 24 Mantra
   - Haircare: Tresemme, Dove, Head & Shoulders, L'Oreal, Streax, Set Wet, Beardo, Wow

4. **ingredients_mentioned**: Any specific ingredients, compounds, or formulation elements mentioned (e.g., vitamin C, niacinamide, biotin, turmeric, argan oil, SLS, dimethicone)

5. **issues_mentioned**: Specific issues or problems described (e.g., "bottle leaked", "caused breakouts", "strong chemical smell", "found stones in mix")

6. **suggestions**: Any explicit suggestions or requests from the customer (e.g., "add bigger size", "change packaging to tube", "make a lighter version")

## Rules:
- Only extract entities that are EXPLICITLY mentioned in the text
- Do NOT infer or hallucinate entities not present in the review
- For competitor_mentions, include both the brand name and product name if both are mentioned
- For issues_mentioned, use concise phrases (3-8 words each)
- If no entities found for a field, return an empty list []

Respond ONLY with valid JSON matching this schema:
{
  "brand_mentioned": "string",
  "products_mentioned": ["string", ...],
  "competitor_mentions": ["string", ...],
  "ingredients_mentioned": ["string", ...],
  "issues_mentioned": ["string", ...],
  "suggestions": ["string", ...]
}"""


# ── Agent Function ───────────────────────────────────────────────────


def _get_llm():
    """Initialize the LLM for entity extraction."""
    return get_llm(temperature=0.0, max_tokens=500)


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=4, max=30),
)
def _extract_entities(
    text: str, title: str, brand: str, product: str
) -> dict:
    """Call Gemini to extract entities from a single feedback item."""
    llm = _get_llm()

    user_prompt = f"""Extract entities from this customer feedback:

Brand: {brand}
Product: {product}
Title: {title}
Review Text: {text}

Respond with JSON only."""

    messages = [
        SystemMessage(content=ENTITY_EXTRACTOR_SYSTEM_PROMPT),
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


def extract_entities(state: GraphState) -> dict[str, Any]:
    """LangGraph node: Extract entities from a feedback record.

    Args:
        state: Current graph state with the feedback record.

    Returns:
        Updated state dict with entity extraction results.
    """
    record = state["record"]
    errors = list(state.get("errors", []))

    try:
        result = _extract_entities(
            text=record.get("text", ""),
            title=record.get("title", ""),
            brand=record.get("brand", ""),
            product=record.get("product", ""),
        )

        entities = EntityResult(**result)

        extracted_count = (
            len(entities.products_mentioned)
            + len(entities.competitor_mentions)
            + len(entities.ingredients_mentioned)
            + len(entities.issues_mentioned)
            + len(entities.suggestions)
        )
        logger.info(
            f"[Entity] {record['id']}: "
            f"Extracted {extracted_count} entities"
        )
        return {"entities": entities.model_dump()}

    except Exception as e:
        error_msg = (
            f"Entity extraction failed for {record.get('id', '?')}: {e}"
        )
        logger.error(f"[Entity] {error_msg}")
        errors.append(error_msg)
        return {
            "entities": {
                "brand_mentioned": record.get("brand", ""),
                "products_mentioned": [],
                "competitor_mentions": [],
                "ingredients_mentioned": [],
                "issues_mentioned": [],
                "suggestions": [],
            },
            "errors": errors,
        }
