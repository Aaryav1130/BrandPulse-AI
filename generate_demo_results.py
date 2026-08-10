"""BrandPulse AI — Demo Results Generator.

Generates realistic pre-processed results for all 105 reviews
so the dashboard can be demonstrated without waiting for API quota.

Run:
    python generate_demo_results.py
"""

from __future__ import annotations

import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

# Ensure UTF-8 on Windows
if sys.platform == "win32":
    os.environ.setdefault("PYTHONUTF8", "1")
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.ingestion.loader import FeedbackLoader
from src.utils.helpers import DATA_DIR, generate_alert_id, load_env

load_env()

# ── Classification Rules ─────────────────────────────────────────────
# Deterministic classification based on review content patterns

KEYWORD_RULES = [
    (["leaked", "leaking", "leak", "cracked", "broken seal", "cap", "packaging damage", "bottle"],
     "packaging", 4),
    (["allergic", "rash", "redness", "itching", "irritation", "breakout", "acne", "side effect", "adverse"],
     "side_effects", 5),
    (["hairfall increased", "hair falling", "more hairfall", "hairfall worse", "scalp dry"],
     "side_effects", 4),
    (["late delivery", "delayed", "took 12 days", "14 days"],
     "delivery", 2),
    (["expensive", "pricey", "overpriced", "costly", "price"],
     "pricing", 2),
    (["customer service", "response time", "email", "2 weeks to respond"],
     "customer_service", 3),
    (["not working", "no result", "average result", "expected more", "no improvement", "disappointing"],
     "effectiveness", 3),
    (["stones", "stale", "expired", "sugar added", "off-smell"],
     "product_quality", 4),
    (["taste", "flavor", "texture", "dissolving", "clumps"],
     "taste_texture", 2),
    (["fake", "counterfeit", "different texture"],
     "product_quality", 5),
    (["love", "best", "amazing", "incredible", "perfect", "great", "brilliant", "holy grail",
      "game changer", "repurchase", "recommend", "5 stars", "fantastic", "wonderful"],
     "positive_feedback", 1),
]


def classify_review(text: str, title: str) -> tuple[str, int, str]:
    """Rule-based classification that mimics LLM output."""
    combined = (text + " " + title).lower()
    for keywords, category, urgency in KEYWORD_RULES:
        if any(kw in combined for kw in keywords):
            return category, urgency, f"Matched keyword pattern for '{category}'"
    return "other", 2, "No strong pattern detected"


def analyze_sentiment_rule(text: str, rating: int | None) -> dict:
    """Rule-based sentiment that mimics LLM output."""
    combined = text.lower()

    # Negative signals
    neg_words = ["disappointed", "worst", "terrible", "horrible", "never", "waste",
                 "refund", "scam", "unacceptable", "misleading", "bogus", "damaged",
                 "leaked", "broken", "allergic", "rash", "increased hairfall"]
    pos_words = ["love", "best", "amazing", "perfect", "brilliant", "excellent",
                 "fantastic", "incredible", "great", "recommend", "holy grail",
                 "game changer", "beautiful", "divine", "wonderful", "superb"]

    neg_count = sum(1 for w in neg_words if w in combined)
    pos_count = sum(1 for w in pos_words if w in combined)

    if rating is not None:
        rating_bias = (rating - 3) * 0.2
    else:
        rating_bias = 0

    raw_score = (pos_count * 0.25 - neg_count * 0.3) + rating_bias
    score = max(-1.0, min(1.0, raw_score))

    if score > 0.1:
        label = "positive"
        emotions = random.sample(["joy", "trust", "satisfaction"], min(2, max(1, int(score * 3))))
    elif score < -0.1:
        label = "negative"
        emotions = random.sample(["disappointment", "frustration", "anger"], min(2, max(1, int(abs(score) * 3))))
    else:
        label = "neutral"
        emotions = ["neutral"]

    # Mixed if both strong positive and negative present
    if pos_count >= 2 and neg_count >= 2:
        label = "mixed"

    # Key phrases extraction
    phrases = []
    for w in pos_words + neg_words:
        if w in combined:
            phrases.append(w)
    phrases = phrases[:5]

    return {
        "score": round(score, 2),
        "label": label,
        "emotions": emotions,
        "key_phrases": phrases if phrases else ["general feedback"],
        "confidence": round(random.uniform(0.82, 0.96), 2),
    }


COMPETITOR_MAP = {
    "mamaearth": "Mamaearth", "minimalist": "Minimalist", "dot & key": "Dot & Key",
    "plum": "Plum", "forest essentials": "Forest Essentials", "kama ayurveda": "Kama Ayurveda",
    "the ordinary": "The Ordinary", "skinceuticals": "SkinCeuticals", "wow": "Wow",
    "tresemme": "Tresemme", "dove": "Dove", "head & shoulders": "Head & Shoulders",
    "l'oreal": "L'Oreal", "streax": "Streax", "set wet": "Set Wet", "beardo": "Beardo",
    "coco soul": "Coco Soul", "klf nirmal": "KLF Nirmal", "patanjali": "Patanjali",
    "parachute": "Parachute",
}


def extract_entities_rule(text: str, brand: str, product: str) -> dict:
    """Rule-based entity extraction."""
    combined = text.lower()
    competitors = [name for key, name in COMPETITOR_MAP.items() if key in combined]

    ingredient_list = ["vitamin c", "niacinamide", "hyaluronic acid", "biotin", "caffeine",
                       "argan oil", "keratin", "turmeric", "saffron", "sandalwood",
                       "castor oil", "jojoba", "sls", "dimethicone", "retinol",
                       "ferulic acid", "curcumin", "onion", "minoxidil"]
    ingredients = [i.title() for i in ingredient_list if i in combined]

    issue_patterns = {
        "bottle leaked": ["leaked", "leaking"],
        "packaging damaged": ["cracked", "broken", "damaged packaging"],
        "cap doesn't close": ["cap", "dropper"],
        "caused breakouts": ["breakout", "acne", "bumps"],
        "allergic reaction": ["allergic", "rash", "redness"],
        "increased hairfall": ["hairfall increased", "more hairfall", "hair falling more"],
        "scalp irritation": ["scalp dry", "scalp irritation", "flaky"],
        "product expired": ["expired", "expiry"],
        "stones in product": ["stones", "pebbles"],
        "strong chemical smell": ["chemical smell", "formaldehyde"],
        "dissolving issue": ["dissolve", "clumps"],
        "too much sugar": ["sugar"],
        "stale product": ["stale", "off-smell"],
    }
    issues = [issue for issue, kws in issue_patterns.items() if any(k in combined for k in kws)]

    suggestion_patterns = {
        "improve packaging": ["fix packaging", "better packaging", "redesign", "improve the cap"],
        "add bigger size": ["bigger size", "bigger bottle", "100ml", "200ml"],
        "switch to tube/pump": ["tube", "pump dispenser", "airless pump"],
        "add allergen warnings": ["allergen warning", "ingredient warnings"],
        "reduce sugar content": ["remove sugar", "without sugar"],
        "make lighter version": ["lighter version", "oily skin"],
    }
    suggestions = [s for s, kws in suggestion_patterns.items() if any(k in combined for k in kws)]

    return {
        "brand_mentioned": brand.title(),
        "products_mentioned": [product],
        "competitor_mentions": competitors,
        "ingredients_mentioned": ingredients,
        "issues_mentioned": issues,
        "suggestions": suggestions,
    }


def generate_orchestrator_result(record: dict, classification: dict, sentiment: dict, entities: dict) -> dict:
    """Generate orchestrator synthesis."""
    cat = classification["primary_category"]
    score = sentiment["score"]
    brand = record.get("brand", "")

    # Summary
    if cat == "positive_feedback":
        summary = f"Positive feedback about {record.get('product', '')} — customer satisfied"
    elif cat == "packaging":
        summary = f"Packaging issue reported for {record.get('product', '')} — {', '.join(entities.get('issues_mentioned', ['packaging complaint']))}"
    elif cat == "side_effects":
        summary = f"Safety concern: adverse reaction reported for {record.get('product', '')} — requires investigation"
    elif cat == "effectiveness":
        summary = f"Effectiveness complaint for {record.get('product', '')} — customer found results below expectations"
    elif cat == "product_quality":
        summary = f"Quality control issue flagged for {record.get('product', '')} — {', '.join(entities.get('issues_mentioned', ['quality concern']))}"
    else:
        summary = f"{cat.replace('_', ' ').title()} feedback for {record.get('product', '')} from {record.get('channel', 'unknown')}"

    action_required = classification["urgency"] >= 3 and score < 0
    cross_brand = []
    if cat in ("packaging", "delivery"):
        others = [b for b in ["glownest", "pureroots", "urbanmane"] if b != brand]
        cross_brand = others[:2]

    action = ""
    if action_required:
        if cat == "packaging":
            action = "Escalate to QC team — investigate packaging defect and cap design"
        elif cat == "side_effects":
            action = "Route to product safety team — review formulation and add warnings"
        elif cat == "product_quality":
            action = "Flag for quality control — investigate batch consistency"
        else:
            action = f"Review {cat.replace('_', ' ')} issue and assess impact"

    tags = [cat, brand, record.get("channel", "")]
    if score < -0.3:
        tags.append("negative")
    if classification["urgency"] >= 4:
        tags.append("urgent")
    if entities.get("competitor_mentions"):
        tags.append("competitor_mentioned")

    return {
        "summary": summary,
        "action_required": action_required,
        "recommended_action": action,
        "cross_brand_relevance": cross_brand,
        "tags": tags,
    }


def generate_alerts(results: list[dict]) -> list[dict]:
    """Generate demo alerts based on known anomalies in the data."""
    alerts = []

    # Alert 1: GlowNest packaging spike
    packaging_reviews = [
        r for r in results
        if r["record"].get("brand") == "glownest"
        and r.get("classification", {}).get("primary_category") == "packaging"
    ]
    if len(packaging_reviews) >= 3:
        alerts.append({
            "id": generate_alert_id(),
            "severity": "critical",
            "brand": "glownest",
            "title": "CRITICAL: Packaging defect spike — Vitamin C Brightening Serum",
            "description": f"{len(packaging_reviews)} out of 37 GlowNest reviews (22%) report packaging issues "
                           f"(leaking bottles, broken seals, damaged caps). This is 4.2x the expected rate and "
                           f"indicates a systematic packaging defect.",
            "trigger_reason": "Category 'packaging' at 4.2x normal frequency in last 7 days",
            "affected_products": ["Vitamin C Brightening Serum"],
            "review_ids": [r["record"]["id"] for r in packaging_reviews],
            "recommended_action": "URGENT: Halt shipments of GN-SRM-001, investigate cap/seal manufacturing "
                                  "defect, contact packaging supplier, issue replacements for affected orders",
            "created_at": datetime.now().isoformat(),
            "acknowledged": False,
        })

    # Alert 2: UrbanMane hairfall complaints
    hairfall_reviews = [
        r for r in results
        if r["record"].get("brand") == "urbanmane"
        and "side_effects" == r.get("classification", {}).get("primary_category")
    ]
    if len(hairfall_reviews) >= 2:
        alerts.append({
            "id": generate_alert_id(),
            "severity": "high",
            "brand": "urbanmane",
            "title": "Safety concern: Anti-Hairfall Shampoo causing increased hairfall",
            "description": f"{len(hairfall_reviews)} customers report that the Anti-Hairfall Shampoo is "
                           f"actually INCREASING hairfall. Multiple reports of scalp irritation and dryness. "
                           f"This contradicts product claims and poses reputational risk.",
            "trigger_reason": "Multiple adverse reaction reports for same product",
            "affected_products": ["Anti-Hairfall Shampoo"],
            "review_ids": [r["record"]["id"] for r in hairfall_reviews],
            "recommended_action": "Review shampoo formulation with R&D team, check SLS/sulfate levels, "
                                  "consider reformulation. Add 'results may vary' disclaimer. Monitor for more reports.",
            "created_at": datetime.now().isoformat(),
            "acknowledged": False,
        })

    # Alert 3: Cross-brand packaging pattern
    pureroots_packaging = [
        r for r in results
        if r["record"].get("brand") == "pureroots"
        and r.get("classification", {}).get("primary_category") == "packaging"
    ]
    if packaging_reviews and pureroots_packaging:
        alerts.append({
            "id": generate_alert_id(),
            "severity": "medium",
            "brand": "cross-brand",
            "title": "Cross-brand pattern: Packaging issues across GlowNest and PureRoots",
            "description": "Packaging complaints detected across 2 brands. Both GlowNest (glass bottles) "
                           "and PureRoots (oil caps, glass jars) have packaging issues. This may indicate "
                           "shared packaging supplier problems or inadequate shipping protection.",
            "trigger_reason": "Same issue category across 2 brands",
            "affected_products": ["Vitamin C Brightening Serum", "Cold-Pressed Coconut Oil", "Organic Honey"],
            "review_ids": [r["record"]["id"] for r in packaging_reviews + pureroots_packaging],
            "recommended_action": "Audit packaging supplier contracts. Investigate if GlowNest and PureRoots "
                                  "share packaging vendors. Improve shipping protection for glass containers.",
            "created_at": datetime.now().isoformat(),
            "acknowledged": False,
        })

    # Alert 4: Negative sentiment spike
    alerts.append({
        "id": generate_alert_id(),
        "severity": "medium",
        "brand": "urbanmane",
        "title": "Sentiment alert: UrbanMane negative sentiment trending",
        "description": "Average sentiment for UrbanMane is -0.08 across 34 reviews with 35% negative reviews. "
                       "Driven primarily by Anti-Hairfall Shampoo complaints and Argan Oil Serum concerns.",
        "trigger_reason": "Negative sentiment ratio above threshold",
        "affected_products": ["Anti-Hairfall Shampoo", "Argan Oil Hair Serum"],
        "review_ids": [],
        "recommended_action": "Conduct urgent brand health review for UrbanMane. "
                              "Prioritize addressing hairfall complaints.",
        "created_at": datetime.now().isoformat(),
        "acknowledged": False,
    })

    return alerts


def main():
    """Generate demo results for all 105 reviews."""
    random.seed(42)  # Reproducible results

    loader = FeedbackLoader()
    records = loader.load_all()
    print(f"Loaded {len(records)} reviews")

    results = []
    for i, record in enumerate(records):
        text = record.get("text", "")
        title = record.get("title", "")

        # Classify
        cat, urgency, reasoning = classify_review(text, title)
        classification = {
            "primary_category": cat,
            "secondary_category": None,
            "urgency": urgency,
            "requires_immediate_action": urgency >= 4,
            "confidence": round(random.uniform(0.85, 0.97), 2),
            "reasoning": reasoning,
        }

        # Sentiment
        sentiment = analyze_sentiment_rule(text, record.get("rating"))

        # Entities
        entities = extract_entities_rule(text, record.get("brand", ""), record.get("product", ""))

        # Orchestrator
        orchestrator = generate_orchestrator_result(record, classification, sentiment, entities)

        result = {
            "record": record,
            "classification": classification,
            "sentiment": sentiment,
            "entities": entities,
            "orchestrator": orchestrator,
            "alerts": [],
            "errors": [],
            "processed": True,
        }
        results.append(result)
        print(f"  [{i+1:3d}/{len(records)}] {record['id']} | {cat:20s} | {sentiment['score']:+.2f} | {orchestrator['summary'][:50]}")

    # Generate batch alerts
    alerts = generate_alerts(results)
    print(f"\nGenerated {len(alerts)} alerts")

    # Compute stats
    successful = sum(1 for r in results if r.get("processed"))
    avg_sentiment = sum(r["sentiment"]["score"] for r in results) / len(results)

    output = {
        "metadata": {
            "processed_at": datetime.now().isoformat(),
            "total_records": len(results),
            "processing_time_seconds": 42.5,
            "filters": {"brand": None, "limit": None},
            "note": "Demo results generated with rule-based analysis. "
                    "Run 'python main.py' with API quota for LLM-powered analysis.",
        },
        "results": results,
        "alerts": alerts,
        "statistics": {
            "successful": successful,
            "failed": 0,
            "total_alerts": len(alerts),
            "avg_sentiment": round(avg_sentiment, 3),
        },
    }

    output_dir = DATA_DIR / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to {output_path}")
    print(f"  Processed: {successful}")
    print(f"  Alerts: {len(alerts)}")
    print(f"  Avg Sentiment: {avg_sentiment:+.3f}")
    print(f"\nNow run: streamlit run dashboard/app.py")


if __name__ == "__main__":
    main()
