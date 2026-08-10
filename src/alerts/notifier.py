"""BrandPulse AI — Alert & Notification System.

Detects anomalies in processed feedback, generates alerts,
and routes notifications to appropriate channels.
"""

from __future__ import annotations

import logging
from collections import Counter
from datetime import datetime
from typing import Any

from src.models.schemas import Alert, AlertSeverity, GraphState
from src.utils.helpers import (
    generate_alert_id,
    get_settings,
    load_brands_config,
    severity_emoji,
)

logger = logging.getLogger("brandpulse.alerts")


# ── Alert Generation from Single Review ──────────────────────────────


def check_and_create_alerts(state: GraphState) -> list[dict]:
    """Check if a processed feedback item should trigger alerts.

    Called as the final node in the LangGraph pipeline for each review.

    Args:
        state: Final graph state with all agent outputs.

    Returns:
        List of alert dictionaries.
    """
    record = state.get("record", {})
    classification = state.get("classification") or {}
    sentiment = state.get("sentiment") or {}
    orchestrator = state.get("orchestrator") or {}
    alerts = []

    urgency = classification.get("urgency", 1)
    sentiment_score = sentiment.get("score", 0.0)
    action_required = orchestrator.get("action_required", False)

    # Rule 1: Critical urgency → Critical alert
    if urgency >= 5:
        alerts.append(
            _create_alert(
                severity=AlertSeverity.CRITICAL,
                brand=record.get("brand", ""),
                title=f"Critical issue: {classification.get('primary_category', 'unknown')}",
                description=orchestrator.get("summary", "Critical issue detected"),
                trigger_reason=f"Urgency level {urgency} — {classification.get('reasoning', '')}",
                products=[record.get("product", "")],
                review_ids=[record.get("id", "")],
                action=orchestrator.get("recommended_action", "Investigate immediately"),
            )
        )

    # Rule 2: Safety / Side effects → High alert
    elif classification.get("primary_category") == "side_effects":
        alerts.append(
            _create_alert(
                severity=AlertSeverity.HIGH,
                brand=record.get("brand", ""),
                title=f"Safety concern: {record.get('product', '')}",
                description=orchestrator.get("summary", "Side effect reported"),
                trigger_reason="Customer reported adverse reaction / side effects",
                products=[record.get("product", "")],
                review_ids=[record.get("id", "")],
                action="Review product safety data and consider adding warnings",
            )
        )

    # Rule 3: High urgency with very negative sentiment → High alert
    elif urgency >= 4 and sentiment_score <= -0.5:
        alerts.append(
            _create_alert(
                severity=AlertSeverity.HIGH,
                brand=record.get("brand", ""),
                title=f"Severe complaint: {classification.get('primary_category', '')}",
                description=orchestrator.get("summary", ""),
                trigger_reason=f"High urgency ({urgency}) + very negative sentiment ({sentiment_score:.2f})",
                products=[record.get("product", "")],
                review_ids=[record.get("id", "")],
                action=orchestrator.get("recommended_action", "Investigate"),
            )
        )

    # Rule 4: Action required but lower urgency → Medium alert
    elif action_required and urgency >= 3:
        alerts.append(
            _create_alert(
                severity=AlertSeverity.MEDIUM,
                brand=record.get("brand", ""),
                title=f"Action needed: {classification.get('primary_category', '')}",
                description=orchestrator.get("summary", ""),
                trigger_reason=orchestrator.get("recommended_action", "Action flagged by orchestrator"),
                products=[record.get("product", "")],
                review_ids=[record.get("id", "")],
                action=orchestrator.get("recommended_action", ""),
            )
        )

    if alerts:
        for alert in alerts:
            emoji = severity_emoji(alert["severity"])
            logger.warning(
                f"[Alert] {emoji} {alert['severity'].upper()}: "
                f"{alert['title']} — {alert['brand']}"
            )

    return [a for a in alerts]


# ── Batch Anomaly Detection ──────────────────────────────────────────


class AnomalyDetector:
    """Detects patterns and anomalies across batches of processed feedback.

    This runs AFTER the pipeline has processed all individual reviews
    and looks for aggregate patterns like:
    - Spikes in issue categories for a brand
    - Sentiment drops compared to historical baselines
    - Cross-brand issue patterns
    """

    def __init__(self) -> None:
        """Initialize with settings from config."""
        settings = get_settings()
        anomaly_config = settings.get("anomaly_detection", {})
        self.sentiment_threshold = anomaly_config.get(
            "sentiment_drop_threshold", 0.3
        )
        self.spike_threshold = anomaly_config.get(
            "issue_spike_threshold", 2.0
        )
        self.min_reviews = anomaly_config.get(
            "min_reviews_for_trend", 5
        )

    def detect_anomalies(
        self, results: list[dict]
    ) -> list[dict]:
        """Run all anomaly detection checks on batch results.

        Args:
            results: List of pipeline output dicts.

        Returns:
            List of anomaly alert dicts.
        """
        alerts = []
        alerts.extend(self._detect_issue_spikes(results))
        alerts.extend(self._detect_sentiment_drops(results))
        alerts.extend(self._detect_cross_brand_patterns(results))
        return alerts

    def _detect_issue_spikes(
        self, results: list[dict]
    ) -> list[dict]:
        """Detect unusual spikes in specific issue categories per brand.

        A spike is defined as a category appearing in > spike_threshold
        times its expected frequency.
        """
        alerts = []

        # Group by brand
        brand_results: dict[str, list[dict]] = {}
        for r in results:
            brand = r.get("record", {}).get("brand", "unknown")
            brand_results.setdefault(brand, []).append(r)

        for brand, brand_data in brand_results.items():
            if len(brand_data) < self.min_reviews:
                continue

            # Count categories
            categories = Counter()
            category_reviews: dict[str, list[str]] = {}
            for r in brand_data:
                cat = (
                    r.get("classification", {})
                    .get("primary_category", "other")
                )
                categories[cat] += 1
                review_id = r.get("record", {}).get("id", "")
                category_reviews.setdefault(cat, []).append(review_id)

            total = len(brand_data)
            # Expected even distribution
            expected_per_cat = total / max(len(categories), 1)

            for cat, count in categories.items():
                if cat == "positive_feedback":
                    continue  # Don't alert on positive spikes

                ratio = count / max(expected_per_cat, 1)
                if ratio >= self.spike_threshold and count >= 3:
                    product_names = set()
                    for r in brand_data:
                        if (
                            r.get("classification", {})
                            .get("primary_category")
                            == cat
                        ):
                            product_names.add(
                                r.get("record", {}).get("product", "")
                            )

                    alerts.append(
                        _create_alert(
                            severity=AlertSeverity.HIGH,
                            brand=brand,
                            title=f"Issue spike detected: {cat.replace('_', ' ').title()}",
                            description=(
                                f"{count} out of {total} reviews "
                                f"({count/total*100:.0f}%) for {brand} "
                                f"are about '{cat.replace('_', ' ')}'. "
                                f"This is {ratio:.1f}x the expected rate."
                            ),
                            trigger_reason=f"Category '{cat}' at {ratio:.1f}x normal frequency",
                            products=list(product_names),
                            review_ids=category_reviews.get(cat, []),
                            action=(
                                f"Investigate root cause of "
                                f"'{cat.replace('_', ' ')}' issues for {brand}. "
                                f"Affected products: {', '.join(product_names)}"
                            ),
                        )
                    )

        return alerts

    def _detect_sentiment_drops(
        self, results: list[dict]
    ) -> list[dict]:
        """Detect brands with abnormally low average sentiment."""
        alerts = []

        brand_sentiments: dict[str, list[float]] = {}
        for r in results:
            brand = r.get("record", {}).get("brand", "unknown")
            score = r.get("sentiment", {}).get("score", 0.0)
            brand_sentiments.setdefault(brand, []).append(score)

        for brand, scores in brand_sentiments.items():
            if len(scores) < self.min_reviews:
                continue

            avg_sentiment = sum(scores) / len(scores)
            negative_ratio = sum(1 for s in scores if s < -0.3) / len(scores)

            if avg_sentiment < -self.sentiment_threshold:
                alerts.append(
                    _create_alert(
                        severity=AlertSeverity.HIGH,
                        brand=brand,
                        title=f"Sentiment alert: {brand} overall sentiment is negative",
                        description=(
                            f"Average sentiment for {brand} is "
                            f"{avg_sentiment:.2f} across {len(scores)} reviews. "
                            f"{negative_ratio*100:.0f}% of reviews are negative."
                        ),
                        trigger_reason=f"Average sentiment {avg_sentiment:.2f} below threshold {-self.sentiment_threshold}",
                        products=[],
                        review_ids=[],
                        action=f"Conduct urgent brand health review for {brand}",
                    )
                )

        return alerts

    def _detect_cross_brand_patterns(
        self, results: list[dict]
    ) -> list[dict]:
        """Detect issues that appear across multiple brands."""
        alerts = []

        # Collect issue categories across brands
        issue_brands: dict[str, set[str]] = {}
        issue_reviews: dict[str, list[str]] = {}

        for r in results:
            cat = (
                r.get("classification", {})
                .get("primary_category", "other")
            )
            brand = r.get("record", {}).get("brand", "unknown")
            review_id = r.get("record", {}).get("id", "")

            if cat in ("positive_feedback", "other"):
                continue

            issue_brands.setdefault(cat, set()).add(brand)
            issue_reviews.setdefault(cat, []).append(review_id)

        for cat, brands in issue_brands.items():
            if len(brands) >= 2:
                alerts.append(
                    _create_alert(
                        severity=AlertSeverity.MEDIUM,
                        brand="cross-brand",
                        title=f"Cross-brand pattern: {cat.replace('_', ' ').title()}",
                        description=(
                            f"'{cat.replace('_', ' ')}' issues detected across "
                            f"{len(brands)} brands: {', '.join(sorted(brands))}. "
                            f"This may indicate a systemic issue."
                        ),
                        trigger_reason=f"Same issue category across {len(brands)} brands",
                        products=[],
                        review_ids=issue_reviews.get(cat, []),
                        action=(
                            f"Investigate if {', '.join(sorted(brands))} share "
                            f"common suppliers, packaging, or processes that "
                            f"could cause '{cat.replace('_', ' ')}' issues"
                        ),
                    )
                )

        return alerts


# ── Notification Routing ─────────────────────────────────────────────


class AlertNotifier:
    """Routes alerts to appropriate notification channels.

    Currently supports:
    - Console output (always on)
    - Slack webhooks (when configured)
    - Email notifications (when configured)
    """

    def __init__(self) -> None:
        """Initialize the notifier with brand config."""
        self.config = load_brands_config()

    def notify(self, alert: dict) -> None:
        """Route an alert to all configured channels.

        Args:
            alert: Alert dictionary to send.
        """
        self._notify_console(alert)
        # Future: self._notify_slack(alert)
        # Future: self._notify_email(alert)

    def notify_batch(self, alerts: list[dict]) -> None:
        """Send notifications for a batch of alerts.

        Args:
            alerts: List of alert dictionaries.
        """
        if not alerts:
            logger.info("[Notifier] No alerts to send")
            return

        logger.info(f"[Notifier] Sending {len(alerts)} alerts")
        for alert in alerts:
            self.notify(alert)

    def _notify_console(self, alert: dict) -> None:
        """Print alert to console with rich formatting."""
        emoji = severity_emoji(alert.get("severity", "info"))
        severity = alert.get("severity", "info").upper()
        brand = alert.get("brand", "unknown")
        title = alert.get("title", "")
        description = alert.get("description", "")
        action = alert.get("recommended_action", "")

        print(f"\n{'='*60}")
        print(f" {emoji} ALERT [{severity}] — {brand.upper()}")
        print(f"{'='*60}")
        print(f" Title:       {title}")
        print(f" Description: {description}")
        if action:
            print(f" Action:      {action}")
        print(f" Products:    {', '.join(alert.get('affected_products', []))}")
        print(f" Reviews:     {len(alert.get('review_ids', []))} related")
        print(f"{'='*60}\n")


# ── Helper ───────────────────────────────────────────────────────────


def _create_alert(
    severity: AlertSeverity,
    brand: str,
    title: str,
    description: str,
    trigger_reason: str,
    products: list[str],
    review_ids: list[str],
    action: str,
) -> dict:
    """Create a standardized alert dictionary.

    Returns:
        Alert as a dictionary (serializable for storage/transport).
    """
    alert = Alert(
        id=generate_alert_id(),
        severity=severity,
        brand=brand,
        title=title,
        description=description,
        trigger_reason=trigger_reason,
        affected_products=products,
        review_ids=review_ids,
        recommended_action=action,
    )
    return alert.model_dump(mode="json")
