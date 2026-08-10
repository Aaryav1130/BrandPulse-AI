# BrandPulse AI — System Architecture

## Overview

BrandPulse AI is an autonomous, multi-brand customer feedback intelligence system built for Think9's portfolio of 30+ consumer brands. It uses a **LangGraph-powered multi-agent pipeline** to ingest, classify, analyze, and surface actionable insights from customer reviews across multiple channels.

## Design Principles

1. **Agent Specialization**: Each agent has a single, well-defined responsibility
2. **Parallel Processing**: Analysis agents run concurrently to minimize latency
3. **Fail-Safe Design**: Individual agent failures don't crash the pipeline — fallback values are used
4. **Config-Driven**: Adding a new brand requires only a YAML entry, no code changes
5. **Human-in-the-Loop**: Alerts are surfaced for human review, not auto-actioned

## Pipeline Architecture (LangGraph)

```
                    ┌──────────────┐
                    │   INGEST     │
                    │   Validate   │
                    │   & Prepare  │
                    └──────┬───────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
      ┌──────────┐ ┌──────────┐ ┌──────────┐
      │CLASSIFIER│ │SENTIMENT │ │ ENTITY   │
      │  Agent   │ │  Agent   │ │EXTRACTOR │
      └────┬─────┘ └────┬─────┘ └────┬─────┘
           └─────────────┼─────────────┘
                         ▼
                ┌────────────────┐
                │  ORCHESTRATOR  │
                │    Agent       │
                └────────┬───────┘
                         ▼
                ┌────────────────┐
                │  ALERT CHECK   │
                │  & Routing     │
                └────────┬───────┘
                         ▼
                      [END]
```

The graph uses LangGraph's `StateGraph` with a `GraphState` TypedDict as the shared state object. Each node reads from and writes to this state.

## Agent Details

### Classifier Agent
- **Input**: Review text, title, brand, product
- **Output**: Primary/secondary category, urgency (1-5), confidence
- **Model**: Gemini 2.0 Flash (temperature=0.1)
- **Categories**: 11 types including product_quality, packaging, side_effects, etc.

### Sentiment Analyzer Agent
- **Input**: Review text, title, star rating
- **Output**: Score (-1 to +1), label, emotions, key phrases
- **Model**: Gemini 2.0 Flash (temperature=0.1)
- **Handles**: English, Hinglish, emoji-rich text

### Entity Extractor Agent
- **Input**: Review text, title, brand, product
- **Output**: Brand mentions, products, competitors, ingredients, issues, suggestions
- **Model**: Gemini 2.0 Flash (temperature=0.0)
- **Zero hallucination**: Only extracts explicitly mentioned entities

### Orchestrator Agent
- **Input**: All three agent outputs + original review
- **Output**: Summary, action decision, recommended action, cross-brand relevance, tags
- **Model**: Gemini 2.0 Flash (temperature=0.2)
- **Key capability**: Cross-brand pattern recognition

## Anomaly Detection System

Post-pipeline batch analysis that detects three types of anomalies:

### Issue Spikes
Detects when a category appears at >2x its expected frequency for a brand. Example: 8 packaging complaints out of 35 GlowNest reviews = significant spike.

### Sentiment Drops
Alerts when a brand's average sentiment falls below the configured threshold (-0.3 by default).

### Cross-Brand Patterns
Identifies issue categories that appear across 2+ brands simultaneously, indicating potential systemic issues (shared suppliers, logistics, etc.).

## Data Flow

```
Raw Review → FeedbackLoader.normalize() → FeedbackRecord (Pydantic)
    → GraphState initialization
    → LangGraph pipeline execution
        → Classifier node → ClassificationResult
        → Sentiment node → SentimentResult
        → Entity node → EntityResult
        → Orchestrator node → OrchestratorResult
        → Alert check node → List[Alert]
    → ProcessedFeedback (composite model)
    → Results JSON + Dashboard visualization
```

## Alert Routing Rules

| Condition | Severity | Example |
|-----------|----------|---------|
| Urgency >= 5 | CRITICAL | Allergic reaction report |
| Category = side_effects | HIGH | Adverse skin reaction |
| Urgency >= 4 AND sentiment <= -0.5 | HIGH | Severe quality complaint |
| Action required AND urgency >= 3 | MEDIUM | Repeated packaging issue |
| Category spike >= 2x normal | HIGH (batch) | 8 packaging complaints in a week |
| Average sentiment < -0.3 | HIGH (batch) | Brand-wide negative sentiment |
| Same issue across 2+ brands | MEDIUM (batch) | Cross-brand packaging problem |

## Multi-Brand Configuration

Brands are defined in `config/brands.yaml` with:
- Products and SKUs
- Monitored channels
- Alert routing (Slack channels, email)
- Keywords for search

Adding a new brand requires only a YAML entry — no code changes needed.

## Technology Choices

| Decision | Choice | Alternative Considered | Rationale |
|----------|--------|----------------------|-----------|
| Agent Framework | LangGraph | CrewAI, AutoGen | State graph model, fan-out/fan-in, production ready |
| LLM | Gemini 2.0 Flash | GPT-4o-mini, Claude Haiku | Cost-effective, fast, strong structured output |
| Dashboard | Streamlit | Gradio, React | Fastest path to interactive analytics dashboard |
| Data Validation | Pydantic v2 | dataclasses, attrs | Rich validation, JSON schema, LangChain integration |
| Config | YAML | TOML, JSON | Human-readable, supports comments, widely used |
