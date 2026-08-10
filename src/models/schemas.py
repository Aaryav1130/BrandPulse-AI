"""BrandPulse AI — Data Models & Schemas.

Defines all Pydantic models for the feedback processing pipeline
and the LangGraph state definition.
"""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field
from typing_extensions import TypedDict


# ── Enums ────────────────────────────────────────────────────────────


class Channel(str, enum.Enum):
    """Supported feedback ingestion channels."""

    AMAZON = "amazon"
    FLIPKART = "flipkart"
    INSTAGRAM = "instagram"
    SUPPORT_TICKETS = "support_tickets"
    APP_STORE = "app_store"


class IssueCategory(str, enum.Enum):
    """Categories for classifying customer feedback."""

    PRODUCT_QUALITY = "product_quality"
    PACKAGING = "packaging"
    DELIVERY = "delivery"
    PRICING = "pricing"
    CUSTOMER_SERVICE = "customer_service"
    EFFECTIVENESS = "effectiveness"
    SIDE_EFFECTS = "side_effects"
    TASTE_TEXTURE = "taste_texture"
    VALUE_FOR_MONEY = "value_for_money"
    POSITIVE_FEEDBACK = "positive_feedback"
    OTHER = "other"


class AlertSeverity(str, enum.Enum):
    """Alert severity levels for routing and prioritization."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class Emotion(str, enum.Enum):
    """Detectable emotional states in feedback text."""

    JOY = "joy"
    TRUST = "trust"
    SATISFACTION = "satisfaction"
    ANGER = "anger"
    DISAPPOINTMENT = "disappointment"
    FRUSTRATION = "frustration"
    SURPRISE = "surprise"
    FEAR = "fear"
    NEUTRAL = "neutral"


# ── Input Models ─────────────────────────────────────────────────────


class FeedbackRecord(BaseModel):
    """Raw customer feedback record as ingested from any channel."""

    id: str = Field(..., description="Unique review identifier")
    brand: str = Field(..., description="Brand slug (e.g., 'glownest')")
    product: str = Field(..., description="Product name")
    sku: str = Field(default="", description="Product SKU code")
    channel: Channel = Field(..., description="Source channel")
    rating: Optional[int] = Field(
        None, ge=1, le=5, description="Star rating if available"
    )
    title: str = Field(default="", description="Review title/subject")
    text: str = Field(..., description="Full review text")
    author: str = Field(default="anonymous", description="Reviewer name/handle")
    date: datetime = Field(default_factory=datetime.now, description="Review date")
    verified_purchase: bool = Field(default=False)
    language: str = Field(default="en")

    model_config = {"use_enum_values": True}


# ── Agent Output Models ──────────────────────────────────────────────


class ClassificationResult(BaseModel):
    """Output from the Classifier Agent."""

    primary_category: str = Field(..., description="Primary issue category")
    secondary_category: Optional[str] = Field(
        None, description="Secondary category if applicable"
    )
    urgency: int = Field(
        ..., ge=1, le=5, description="Urgency level: 1=low, 5=critical"
    )
    requires_immediate_action: bool = Field(default=False)
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Classification confidence"
    )
    reasoning: str = Field(
        default="", description="Brief reasoning for classification"
    )


class SentimentResult(BaseModel):
    """Output from the Sentiment Analyzer Agent."""

    score: float = Field(
        ...,
        ge=-1.0,
        le=1.0,
        description="Sentiment score: -1=very negative, +1=very positive",
    )
    label: str = Field(
        ..., description="Sentiment label: positive/negative/neutral/mixed"
    )
    emotions: list[str] = Field(
        default_factory=list, description="Detected emotions"
    )
    key_phrases: list[str] = Field(
        default_factory=list, description="Key phrases driving sentiment"
    )
    confidence: float = Field(..., ge=0.0, le=1.0)


class EntityResult(BaseModel):
    """Output from the Entity Extractor Agent."""

    brand_mentioned: str = Field(default="", description="Brand name referenced")
    products_mentioned: list[str] = Field(
        default_factory=list, description="Products referenced"
    )
    competitor_mentions: list[str] = Field(
        default_factory=list, description="Competitor brands mentioned"
    )
    ingredients_mentioned: list[str] = Field(
        default_factory=list, description="Specific ingredients cited"
    )
    issues_mentioned: list[str] = Field(
        default_factory=list, description="Specific issues described"
    )
    suggestions: list[str] = Field(
        default_factory=list, description="Customer suggestions if any"
    )


class OrchestratorResult(BaseModel):
    """Output from the Orchestrator Agent."""

    summary: str = Field(..., description="One-line summary of this feedback item")
    action_required: bool = Field(default=False)
    recommended_action: str = Field(
        default="", description="Suggested action for the team"
    )
    cross_brand_relevance: list[str] = Field(
        default_factory=list,
        description="Other brands this insight applies to",
    )
    tags: list[str] = Field(default_factory=list, description="Auto-generated tags")


# ── Composite Models ─────────────────────────────────────────────────


class ProcessedFeedback(BaseModel):
    """Fully processed feedback item with all agent outputs."""

    record: FeedbackRecord
    classification: Optional[ClassificationResult] = None
    sentiment: Optional[SentimentResult] = None
    entities: Optional[EntityResult] = None
    orchestrator: Optional[OrchestratorResult] = None
    processed_at: datetime = Field(default_factory=datetime.now)
    processing_errors: list[str] = Field(default_factory=list)


class Alert(BaseModel):
    """Alert generated by the anomaly detection system."""

    id: str = Field(..., description="Alert identifier")
    severity: AlertSeverity
    brand: str
    title: str
    description: str
    trigger_reason: str = Field(..., description="What triggered this alert")
    affected_products: list[str] = Field(default_factory=list)
    review_ids: list[str] = Field(
        default_factory=list, description="Related review IDs"
    )
    recommended_action: str = Field(default="")
    created_at: datetime = Field(default_factory=datetime.now)
    acknowledged: bool = Field(default=False)

    model_config = {"use_enum_values": True}


class BrandInsight(BaseModel):
    """Cross-brand or single-brand insight."""

    id: str
    insight_type: str = Field(
        ..., description="Type: trend, anomaly, cross_brand, opportunity"
    )
    brands_affected: list[str]
    title: str
    description: str
    evidence: list[str] = Field(
        default_factory=list, description="Supporting review IDs"
    )
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=datetime.now)


class BrandHealthMetrics(BaseModel):
    """Aggregated health metrics for a single brand."""

    brand: str
    total_reviews: int = 0
    avg_sentiment: float = 0.0
    avg_rating: float = 0.0
    category_distribution: dict[str, int] = Field(default_factory=dict)
    sentiment_trend: list[dict[str, Any]] = Field(default_factory=list)
    top_issues: list[str] = Field(default_factory=list)
    top_praises: list[str] = Field(default_factory=list)
    active_alerts: int = 0


# ── LangGraph State ──────────────────────────────────────────────────


class GraphState(TypedDict, total=False):
    """State object passed through the LangGraph processing pipeline.

    Each key maps to the output of a specific agent node.
    Uses dicts instead of Pydantic models for LangGraph compatibility.
    """

    record: dict  # Serialized FeedbackRecord
    classification: dict | None  # ClassificationResult as dict
    sentiment: dict | None  # SentimentResult as dict
    entities: dict | None  # EntityResult as dict
    orchestrator: dict | None  # OrchestratorResult as dict
    alerts: list[dict]  # Generated alerts
    errors: list[str]  # Processing errors
    processed: bool  # Whether processing is complete
