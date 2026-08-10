"""BrandPulse AI — Sentiment Analyzer Agent.

Analyzes customer feedback sentiment with emotion detection
and key phrase extraction using Google Gemini.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from tenacity import retry, stop_after_attempt, wait_exponential

from src.models.schemas import GraphState, SentimentResult
from src.utils.helpers import get_llm

logger = logging.getLogger("brandpulse.sentiment")

# ── System Prompt ────────────────────────────────────────────────────

SENTIMENT_SYSTEM_PROMPT = """You are an expert sentiment analysis agent for consumer brand feedback in the Indian market. You understand both English and Hinglish (Hindi-English mix) customer reviews.

Your task is to analyze the sentiment of customer feedback with nuance and precision.

## Sentiment Score Scale:
- -1.0 to -0.6: Very Negative (angry, disgusted, demanding refund)
- -0.6 to -0.3: Negative (disappointed, dissatisfied, complaining)
- -0.3 to -0.1: Slightly Negative (minor complaints mixed with some positives)
- -0.1 to +0.1: Neutral (factual, inquiry, neither positive nor negative)
- +0.1 to +0.3: Slightly Positive (satisfied but with reservations)
- +0.3 to +0.6: Positive (happy, recommending, good experience)
- +0.6 to +1.0: Very Positive (enthusiastic, loyal, strongly recommending)

## Sentiment Labels:
- "positive": Score > 0.1
- "negative": Score < -0.1
- "neutral": Score between -0.1 and 0.1
- "mixed": Contains both strong positive AND strong negative sentiments

## Emotions to detect (pick 1-3 most relevant):
joy, trust, satisfaction, anger, disappointment, frustration, surprise, fear, neutral

## Key Phrases:
Extract 2-5 short phrases that most strongly drive the sentiment (positive or negative).

## Rules:
- Consider the star rating as context but base your analysis on the TEXT content
- Hinglish reviews should be analyzed for sentiment the same way as English
- Emojis carry sentiment weight (😤 = frustrated, 💛 = positive, etc.)
- Support tickets/complaints typically have negative sentiment even without explicit negative words
- A reviewer can be positive about the product but negative about packaging/service — classify as "mixed"

Respond ONLY with valid JSON matching this schema:
{
  "score": float (-1.0 to 1.0),
  "label": "positive" | "negative" | "neutral" | "mixed",
  "emotions": ["string", ...],
  "key_phrases": ["string", ...],
  "confidence": float (0.0 to 1.0)
}"""


# ── Agent Function ───────────────────────────────────────────────────


def _get_llm():
    """Initialize the LLM for sentiment analysis."""
    return get_llm(temperature=0.1, max_tokens=400)


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=4, max=30),
)
def _analyze_sentiment(
    text: str, title: str, rating: int | None
) -> dict:
    """Call Gemini to analyze sentiment of a single feedback item."""
    llm = _get_llm()

    rating_context = f"Star Rating: {rating}/5" if rating else "No star rating"

    user_prompt = f"""Analyze the sentiment of this customer feedback:

{rating_context}
Title: {title}
Review Text: {text}

Respond with JSON only."""

    messages = [
        SystemMessage(content=SENTIMENT_SYSTEM_PROMPT),
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


def analyze_sentiment(state: GraphState) -> dict[str, Any]:
    """LangGraph node: Analyze sentiment of a feedback record.

    Args:
        state: Current graph state with the feedback record.

    Returns:
        Updated state dict with sentiment results.
    """
    record = state["record"]
    errors = list(state.get("errors", []))

    try:
        result = _analyze_sentiment(
            text=record.get("text", ""),
            title=record.get("title", ""),
            rating=record.get("rating"),
        )

        sentiment = SentimentResult(**result)
        logger.info(
            f"[Sentiment] {record['id']}: "
            f"{sentiment.label} ({sentiment.score:+.2f})"
        )
        return {"sentiment": sentiment.model_dump()}

    except Exception as e:
        error_msg = f"Sentiment analysis failed for {record.get('id', '?')}: {e}"
        logger.error(f"[Sentiment] {error_msg}")
        errors.append(error_msg)
        return {
            "sentiment": {
                "score": 0.0,
                "label": "neutral",
                "emotions": ["neutral"],
                "key_phrases": [],
                "confidence": 0.0,
            },
            "errors": errors,
        }
