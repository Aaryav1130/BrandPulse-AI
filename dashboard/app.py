"""BrandPulse AI — Streamlit Analytics Dashboard.

Interactive dashboard for monitoring customer feedback intelligence
across Think9's brand portfolio. Clean professional light theme.

Run:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.helpers import load_brands_config, DATA_DIR


# ── Page Configuration ───────────────────────────────────────────────

st.set_page_config(
    page_title="BrandPulse AI — Customer Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS (Light Professional Theme) ────────────────────────────

st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
    .stApp { background: #FAFBFC; }
    .stApp, .stApp p, .stApp span, .stApp div, .stApp h1, .stApp h2, .stApp h3,
    .stApp h4, .stApp label, .stApp input, .stApp textarea, .stApp select,
    .stApp button, .stApp a, .stApp li, .stApp td, .stApp th,
    .stApp [data-testid="stMetricValue"], .stApp [data-testid="stMetricLabel"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    section[data-testid="stSidebar"] {
        background: #F4F5F7;
        border-right: 1px solid #DFE1E6;
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #172B4D !important;
    }

    div[data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #DFE1E6;
        border-radius: 10px;
        padding: 18px 20px;
        box-shadow: 0 1px 3px rgba(9,30,66,0.08);
    }
    div[data-testid="stMetric"]:hover {
        box-shadow: 0 4px 12px rgba(9,30,66,0.12);
    }
    div[data-testid="stMetric"] label { color: #6B778C !important; font-size: 0.85rem !important; }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: #172B4D !important; font-weight: 700 !important; }

    h1, h2, h3 { color: #172B4D !important; }
    p, span, li { color: #42526E; }

    .stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 2px solid #DFE1E6; }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: #6B778C;
        padding: 10px 20px;
        border-radius: 6px 6px 0 0;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background: #DEEBFF !important;
        color: #0052CC !important;
        font-weight: 600;
    }

    .alert-critical { background: #FFEBE6; border-left: 4px solid #DE350B; padding: 16px; border-radius: 0 8px 8px 0; margin: 8px 0; }
    .alert-high { background: #FFF7E6; border-left: 4px solid #FF991F; padding: 16px; border-radius: 0 8px 8px 0; margin: 8px 0; }
    .alert-medium { background: #FFFAE6; border-left: 4px solid #FFAB00; padding: 16px; border-radius: 0 8px 8px 0; margin: 8px 0; }
    .alert-low { background: #DEEBFF; border-left: 4px solid #0065FF; padding: 16px; border-radius: 0 8px 8px 0; margin: 8px 0; }

    .brand-tag { display: inline-block; background: #DEEBFF; color: #0052CC; padding: 3px 10px; border-radius: 12px; font-size: 0.8rem; margin: 2px; font-weight: 500; }
    .severity-badge { display: inline-block; padding: 3px 10px; border-radius: 12px; font-size: 0.75rem; font-weight: 600; color: white; }

    div[data-testid="stExpander"] { background: #FFFFFF; border: 1px solid #DFE1E6; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ── Color Palettes ───────────────────────────────────────────────────

BRAND_COLORS = {"glownest": "#6554C0", "pureroots": "#00875A", "urbanmane": "#0065FF", "cross-brand": "#FF991F"}

CATEGORY_COLORS = {
    "positive_feedback": "#00875A", "effectiveness": "#0065FF", "packaging": "#DE350B",
    "product_quality": "#FF991F", "delivery": "#FFAB00", "pricing": "#6554C0",
    "customer_service": "#E774BB", "side_effects": "#BF2600", "taste_texture": "#00A3BF",
    "value_for_money": "#5243AA", "other": "#97A0AF",
}

SEVERITY_COLORS = {"critical": "#DE350B", "high": "#FF991F", "medium": "#FFAB00", "low": "#0065FF", "info": "#97A0AF"}
SEVERITY_BG = {"critical": "#FFEBE6", "high": "#FFF7E6", "medium": "#FFFAE6", "low": "#DEEBFF", "info": "#F4F5F7"}
PLOTLY_TEMPLATE = "plotly_white"
CHART_FONT_COLOR = "#172B4D"
CHART_GRID_COLOR = "#EBECF0"


# ── Data Loading ─────────────────────────────────────────────────────

@st.cache_data(ttl=60)
def load_results() -> dict | None:
    results_path = DATA_DIR / "processed" / "results.json"
    if not results_path.exists():
        return None
    with open(results_path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_brand_config() -> dict:
    return load_brands_config()


def results_to_dataframe(results: list[dict]) -> pd.DataFrame:
    rows = []
    for r in results:
        if not r.get("processed"):
            continue
        record = r.get("record", {})
        classification = r.get("classification", {})
        sentiment = r.get("sentiment", {})
        entities = r.get("entities", {})
        orchestrator = r.get("orchestrator", {})
        rows.append({
            "id": record.get("id", ""), "brand": record.get("brand", ""),
            "product": record.get("product", ""), "channel": record.get("channel", ""),
            "rating": record.get("rating"),
            "date": record.get("date", "")[:10] if record.get("date") else "",
            "text": record.get("text", "")[:200], "title": record.get("title", ""),
            "author": record.get("author", ""),
            "category": classification.get("primary_category", ""),
            "urgency": classification.get("urgency", 0),
            "sentiment_score": sentiment.get("score", 0.0),
            "sentiment_label": sentiment.get("label", ""),
            "emotions": ", ".join(sentiment.get("emotions", [])),
            "key_phrases": ", ".join(sentiment.get("key_phrases", [])),
            "competitors": ", ".join(entities.get("competitor_mentions", [])),
            "issues": ", ".join(entities.get("issues_mentioned", [])),
            "suggestions": ", ".join(entities.get("suggestions", [])),
            "summary": orchestrator.get("summary", ""),
            "action_required": orchestrator.get("action_required", False),
            "recommended_action": orchestrator.get("recommended_action", ""),
            "tags": ", ".join(orchestrator.get("tags", [])),
        })
    return pd.DataFrame(rows)


def make_chart(fig):
    """Apply consistent clean styling to a Plotly figure."""
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=CHART_FONT_COLOR, family="Inter"),
        xaxis=dict(gridcolor=CHART_GRID_COLOR),
        yaxis=dict(gridcolor=CHART_GRID_COLOR),
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


# ── Sidebar ──────────────────────────────────────────────────────────

def render_sidebar(config: dict) -> dict:
    st.sidebar.markdown("## 📊 BrandPulse AI")
    st.sidebar.caption("Customer Intelligence Hub")
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Filters")

    brands = list(config.get("brands", {}).keys())
    selected_brand = st.sidebar.selectbox("Brand", ["All Brands"] + brands, index=0)
    channels = ["All Channels", "amazon", "flipkart", "instagram", "support_tickets", "app_store"]
    selected_channel = st.sidebar.selectbox("Channel", channels, index=0)
    sentiment_filter = st.sidebar.select_slider(
        "Sentiment Range",
        options=["Very Negative", "Negative", "Neutral", "Positive", "Very Positive"],
        value=("Very Negative", "Very Positive"),
    )
    st.sidebar.markdown("---")
    return {
        "brand": None if selected_brand == "All Brands" else selected_brand,
        "channel": None if selected_channel == "All Channels" else selected_channel,
        "sentiment_filter": sentiment_filter,
    }


# ── Tab: Overview ────────────────────────────────────────────────────

def render_overview(df: pd.DataFrame) -> None:
    if len(df) == 0:
        st.info("No data available.")
        return

    # Sentiment by Brand + Donut
    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("#### Sentiment by Brand")
        brand_sent = df.groupby("brand")["sentiment_score"].mean().reset_index()
        brand_sent.columns = ["Brand", "Avg Sentiment"]
        colors = [BRAND_COLORS.get(b, "#97A0AF") for b in brand_sent["Brand"]]
        fig = go.Figure(go.Bar(
            x=brand_sent["Brand"].str.title(), y=brand_sent["Avg Sentiment"],
            marker_color=colors, text=brand_sent["Avg Sentiment"].apply(lambda x: f"{x:+.2f}"),
            textposition="outside",
        ))
        fig.add_hline(y=0, line_dash="dash", line_color="#DFE1E6")
        make_chart(fig)
        fig.update_layout(height=350, yaxis_title="Avg Sentiment Score")
        st.plotly_chart(fig, width="stretch")

    with col2:
        st.markdown("#### Sentiment Breakdown")
        label_counts = df["sentiment_label"].value_counts()
        colors_map = {"positive": "#00875A", "negative": "#DE350B", "neutral": "#97A0AF", "mixed": "#FFAB00"}
        fig = go.Figure(go.Pie(
            labels=label_counts.index.str.title(), values=label_counts.values,
            hole=0.55, marker_colors=[colors_map.get(l, "#97A0AF") for l in label_counts.index],
            textinfo="label+percent", textfont_size=12,
        ))
        make_chart(fig)
        fig.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig, width="stretch")

    # Category Distribution
    st.markdown("#### Issue Categories")
    col1, col2 = st.columns([3, 2])
    with col1:
        cat_counts = df["category"].value_counts().head(10)
        colors = [CATEGORY_COLORS.get(c, "#97A0AF") for c in cat_counts.index]
        fig = go.Figure(go.Bar(
            y=cat_counts.index.str.replace("_", " ").str.title(), x=cat_counts.values,
            orientation="h", marker_color=colors,
            text=cat_counts.values, textposition="outside",
        ))
        make_chart(fig)
        fig.update_layout(height=400, yaxis=dict(autorange="reversed"), xaxis_title="Number of Reviews")
        st.plotly_chart(fig, width="stretch")

    with col2:
        st.markdown("#### Urgency Distribution")
        urg_counts = df["urgency"].value_counts().sort_index()
        urg_colors = {1: "#00875A", 2: "#0065FF", 3: "#FFAB00", 4: "#FF991F", 5: "#DE350B"}
        fig = go.Figure(go.Bar(
            x=[f"Level {u}" for u in urg_counts.index], y=urg_counts.values,
            marker_color=[urg_colors.get(u, "#97A0AF") for u in urg_counts.index],
            text=urg_counts.values, textposition="outside",
        ))
        make_chart(fig)
        fig.update_layout(height=400, xaxis_title="Urgency Level", yaxis_title="Count")
        st.plotly_chart(fig, width="stretch")


# ── Tab: Alerts ──────────────────────────────────────────────────────

def render_alerts(alerts: list[dict]) -> None:
    if not alerts:
        st.info("🎉 No active alerts. All brands are looking healthy!")
        return

    st.markdown(f"#### {len(alerts)} Active Alerts")

    # Sort by severity
    severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}
    sorted_alerts = sorted(alerts, key=lambda a: severity_order.get(a.get("severity", "info"), 5))

    for alert in sorted_alerts:
        severity = alert.get("severity", "info")
        bg = SEVERITY_BG.get(severity, "#F4F5F7")
        color = SEVERITY_COLORS.get(severity, "#97A0AF")
        emoji = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🔵", "info": "⚪"}.get(severity, "⚪")

        products = alert.get("affected_products", [])
        products_html = " ".join(f'<span class="brand-tag">{p}</span>' for p in products)

        st.markdown(f"""
        <div style="background:{bg}; border-left:4px solid {color}; padding:18px 20px; border-radius:0 10px 10px 0; margin:10px 0;">
            <div style="display:flex; align-items:center; gap:8px; margin-bottom:8px;">
                <span>{emoji}</span>
                <span style="background:{color}; color:white; padding:2px 10px; border-radius:12px; font-size:0.75rem; font-weight:600;">{severity.upper()}</span>
                <span style="color:#6B778C; font-size:0.85rem;">{alert.get('brand', '').title()}</span>
            </div>
            <div style="font-size:1.05rem; font-weight:600; color:#172B4D; margin-bottom:6px;">{alert.get('title', '')}</div>
            <div style="color:#42526E; font-size:0.9rem; margin-bottom:10px;">{alert.get('description', '')}</div>
            <div style="background:rgba(0,82,204,0.06); padding:10px 14px; border-radius:6px; margin-bottom:8px;">
                <span style="color:#0052CC; font-weight:600; font-size:0.85rem;">→ {alert.get('recommended_action', 'No action specified')}</span>
            </div>
            <div>{products_html}</div>
        </div>
        """, unsafe_allow_html=True)


# ── Tab: Brand Deep Dive ─────────────────────────────────────────────

def render_brand_deep_dive(df: pd.DataFrame, config: dict) -> None:
    if len(df) == 0:
        st.info("No data available.")
        return

    brands = df["brand"].unique().tolist()
    selected = st.selectbox("Select Brand", brands, format_func=lambda x: x.title())
    brand_df = df[df["brand"] == selected]

    if len(brand_df) == 0:
        st.info(f"No data for {selected.title()}")
        return

    # Brand Metrics
    col1, col2, col3, col4 = st.columns(4)
    avg_sent = brand_df["sentiment_score"].mean()
    with col1:
        st.metric("Reviews", len(brand_df))
    with col2:
        st.metric("Avg Sentiment", f"{avg_sent:+.2f}")
    with col3:
        neg_pct = (brand_df["sentiment_label"] == "negative").sum() / max(len(brand_df), 1) * 100
        st.metric("Negative %", f"{neg_pct:.0f}%")
    with col4:
        action_count = brand_df["action_required"].sum()
        st.metric("Actions Needed", int(action_count))

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Product Sentiment")
        prod_sent = brand_df.groupby("product")["sentiment_score"].agg(["mean", "count"]).reset_index()
        prod_sent.columns = ["Product", "Avg Sentiment", "Reviews"]
        prod_sent = prod_sent.sort_values("Avg Sentiment")
        colors = ["#DE350B" if s < -0.1 else "#00875A" if s > 0.1 else "#97A0AF" for s in prod_sent["Avg Sentiment"]]
        fig = go.Figure(go.Bar(
            y=prod_sent["Product"], x=prod_sent["Avg Sentiment"],
            orientation="h", marker_color=colors,
            text=prod_sent["Avg Sentiment"].apply(lambda x: f"{x:+.2f}"),
            textposition="outside",
        ))
        fig.add_vline(x=0, line_dash="dash", line_color="#DFE1E6")
        make_chart(fig)
        fig.update_layout(height=300)
        st.plotly_chart(fig, width="stretch")

    with col2:
        st.markdown("#### Top Issues")
        all_issues = []
        for issues_str in brand_df["issues"].dropna():
            if issues_str.strip():
                all_issues.extend([i.strip() for i in issues_str.split(",")])
        if all_issues:
            issue_counts = Counter(all_issues).most_common(8)
            fig = go.Figure(go.Bar(
                x=[c for _, c in issue_counts],
                y=[n for n, _ in issue_counts],
                orientation="h", marker_color="#FF991F",
            ))
            make_chart(fig)
            fig.update_layout(height=300, yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("No specific issues extracted for this brand.")

    # Competitor Mentions
    all_competitors = []
    for comp_str in brand_df["competitors"].dropna():
        if comp_str.strip():
            all_competitors.extend([c.strip() for c in comp_str.split(",")])
    if all_competitors:
        st.markdown("#### Competitor Mentions")
        comp_counts = Counter(all_competitors).most_common(10)
        comp_html = " ".join(f'<span class="brand-tag">{name} ({count})</span>' for name, count in comp_counts)
        st.markdown(comp_html, unsafe_allow_html=True)


# ── Tab: Review Explorer ─────────────────────────────────────────────

def render_review_explorer(df: pd.DataFrame) -> None:
    if len(df) == 0:
        st.info("No reviews to display.")
        return

    st.markdown(f"#### Showing {len(df)} Reviews")

    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        cat_filter = st.selectbox("Filter by Category", ["All"] + sorted(df["category"].unique().tolist()))
    with col2:
        sort_by = st.selectbox("Sort by", ["Urgency (High→Low)", "Sentiment (Low→High)", "Sentiment (High→Low)"])
    with col3:
        action_only = st.checkbox("Action Required Only", value=False)

    filtered = df.copy()
    if cat_filter != "All":
        filtered = filtered[filtered["category"] == cat_filter]
    if action_only:
        filtered = filtered[filtered["action_required"] == True]

    if sort_by == "Urgency (High→Low)":
        filtered = filtered.sort_values("urgency", ascending=False)
    elif sort_by == "Sentiment (Low→High)":
        filtered = filtered.sort_values("sentiment_score", ascending=True)
    else:
        filtered = filtered.sort_values("sentiment_score", ascending=False)

    for _, row in filtered.head(30).iterrows():
        score = row["sentiment_score"]
        score_color = "#DE350B" if score < -0.2 else "#00875A" if score > 0.2 else "#97A0AF"
        urgency = int(row["urgency"]) if pd.notna(row["urgency"]) else 0
        urg_color = {1: "#00875A", 2: "#0065FF", 3: "#FFAB00", 4: "#FF991F", 5: "#DE350B"}.get(urgency, "#97A0AF")

        rating_str = ""
        if pd.notna(row["rating"]):
            rating_str = f"⭐ {int(row['rating'])}/5"

        with st.expander(f"{row['id']} — {row['brand'].title()} / {row['product']} | {row['category'].replace('_',' ').title()}"):
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.markdown(f"**Sentiment:** <span style='color:{score_color}; font-weight:700;'>{score:+.2f}</span>", unsafe_allow_html=True)
            mc2.markdown(f"**Urgency:** <span style='color:{urg_color}; font-weight:700;'>Level {urgency}</span>", unsafe_allow_html=True)
            mc3.markdown(f"**Channel:** {row['channel']}")
            mc4.markdown(f"**{rating_str}**" if rating_str else "**No rating**")

            st.markdown(f"**Title:** {row['title']}")
            st.markdown(f"> {row['text']}")

            if row["summary"]:
                st.markdown(f"🧠 **AI Summary:** {row['summary']}")
            if row["recommended_action"]:
                st.info(f"⚡ **Action:** {row['recommended_action']}")
            if row["emotions"]:
                st.markdown(f"**Emotions:** {row['emotions']}")
            if row["issues"]:
                st.markdown(f"**Issues:** {row['issues']}")


# ── Tab: Channels ────────────────────────────────────────────────────

def render_channels(df: pd.DataFrame) -> None:
    if len(df) == 0:
        st.info("No data available.")
        return

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Reviews by Channel")
        channel_counts = df["channel"].value_counts()
        fig = go.Figure(go.Pie(
            labels=channel_counts.index.str.replace("_", " ").str.title(),
            values=channel_counts.values, hole=0.5,
            marker_colors=["#0065FF", "#00875A", "#6554C0", "#FF991F", "#00A3BF"],
            textinfo="label+percent",
        ))
        make_chart(fig)
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, width="stretch")

    with col2:
        st.markdown("#### Sentiment by Channel")
        ch_sent = df.groupby("channel")["sentiment_score"].mean().sort_values()
        colors = ["#DE350B" if s < -0.1 else "#00875A" if s > 0.1 else "#97A0AF" for s in ch_sent.values]
        fig = go.Figure(go.Bar(
            y=ch_sent.index.str.replace("_", " ").str.title(), x=ch_sent.values,
            orientation="h", marker_color=colors,
            text=[f"{s:+.2f}" for s in ch_sent.values], textposition="outside",
        ))
        fig.add_vline(x=0, line_dash="dash", line_color="#DFE1E6")
        make_chart(fig)
        fig.update_layout(height=400)
        st.plotly_chart(fig, width="stretch")


# ── Main Application ─────────────────────────────────────────────────

def main() -> None:
    config = load_brand_config()
    filters = render_sidebar(config)

    # Header
    st.markdown("### 👋 Welcome to BrandPulse AI")
    st.caption("Here's a summary of customer feedback intelligence across your brands.")

    # Load data
    data = load_results()
    if data is None:
        st.warning("⚠️ No results found. Run the pipeline first:")
        st.code("python main.py --limit 10\n# Or generate demo results:\npython generate_demo_results.py", language="bash")
        return

    results = data.get("results", [])
    alerts = data.get("alerts", [])
    metadata = data.get("metadata", {})
    df = results_to_dataframe(results)

    # Apply filters
    if filters["brand"]:
        df = df[df["brand"] == filters["brand"]]
    if filters["channel"]:
        df = df[df["channel"] == filters["channel"]]

    # Sidebar stats
    st.sidebar.markdown("### Quick Stats")
    st.sidebar.metric("Reviews Processed", len(df))
    st.sidebar.metric("Active Alerts", len(alerts))
    if len(df) > 0:
        st.sidebar.metric("Avg Sentiment", f"{df['sentiment_score'].mean():+.2f}")
    st.sidebar.metric("Processing Time", f"{metadata.get('processing_time_seconds', 0):.0f}s")
    st.sidebar.markdown("---")
    st.sidebar.caption(f"Last updated: {metadata.get('processed_at', 'N/A')[:19]}")

    # Top KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("✅ Reviews Processed", len(df))
    with col2:
        avg = df["sentiment_score"].mean() if len(df) > 0 else 0
        st.metric("📊 Avg Sentiment", f"{avg:+.2f}")
    with col3:
        issues = int(df["action_required"].sum()) if len(df) > 0 else 0
        st.metric("⚠️ Issues Detected", issues)
    with col4:
        st.metric("🔔 Active Alerts", len(alerts))

    st.markdown("")

    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Overview", "🔔 Alerts", "🏢 Brand Deep Dive", "🔍 Review Explorer", "📡 Channels"
    ])

    with tab1:
        render_overview(df)
    with tab2:
        render_alerts(alerts)
    with tab3:
        render_brand_deep_dive(df, config)
    with tab4:
        render_review_explorer(df)
    with tab5:
        render_channels(df)

    # Footer
    st.markdown("---")
    st.caption("Built with LangGraph + Groq/Gemini | BrandPulse AI v1.0 | Think9 AI Challenge")


if __name__ == "__main__":
    main()
