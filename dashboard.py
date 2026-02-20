"""
dashboard.py
Editorial Intelligence — Newspaper-Style Streamlit Dashboard

Shows a front-page manifest produced by `optimizer.py` from ONE ranked candidate pool, plus the offline
evaluation report of the loaded artifact. Scores are labeled honestly: `ranking_score` is a raw LambdaRank
score (comparable only within its pool), `composite_score` is the within-pool normalized utility. Nothing
here is a probability, a confidence, or a model explanation (no SHAP is computed).
"""

import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from newsrec.postrank import within_pool_minmax

# ─── Page Config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="The Editorial — AI Front Page",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Stylesheet ──────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;0,900;1,400;1,700&family=UnifrakturMaguntia&family=Source+Serif+4:ital,opsz,wght@0,8..60,300;0,8..60,400;0,8..60,600;1,8..60,300;1,8..60,400&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Reset & Base ── */
:root {
    --ink:        #0d0d0d;
    --ink-soft:   #2c2c2c;
    --ink-muted:  #666666;
    --paper:      #f7f4ef;
    --paper-dark: #ede9e1;
    --rule:       #1a1a1a;
    --rule-light: #c8bfaf;
    --accent:     #b91c1c;
    --accent-dim: #7f1d1d;
    --positive:   #166534;
    --negative:   #9f1239;
    --highlight:  #fef08a;
    --col-gap:    1px;
}

html, body, [class*="css"] {
    font-family: 'Source Serif 4', Georgia, serif;
    background-color: var(--paper);
    color: var(--ink);
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 2rem 2rem; max-width: 1400px; }

/* ── Masthead ── */
.masthead {
    border-top: 6px solid var(--rule);
    border-bottom: 3px double var(--rule);
    padding: 0.6rem 0 0.5rem;
    text-align: center;
    margin-bottom: 0;
    position: relative;
    background: var(--paper);
}
.masthead-kicker {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--ink-muted);
    margin-bottom: 0.25rem;
}
.masthead-title {
    font-family: 'UnifrakturMaguntia', cursive;
    font-size: clamp(2.8rem, 6vw, 5.2rem);
    line-height: 1;
    color: var(--ink);
    letter-spacing: 0.01em;
    margin: 0;
}
.masthead-rule {
    height: 2px;
    background: var(--rule);
    margin: 0.4rem 0 0.2rem;
}
.masthead-meta {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.58rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--ink-muted);
    border-top: 1px solid var(--rule-light);
    padding-top: 0.3rem;
}
.masthead-edition {
    background: var(--ink);
    color: var(--paper);
    padding: 1px 6px;
    font-size: 0.55rem;
}

/* ── Section Rules ── */
.section-rule {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin: 1.2rem 0 0.8rem;
    border-top: 3px solid var(--ink);
    padding-top: 0.3rem;
}
.section-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--ink-muted);
    white-space: nowrap;
}

/* ── Article Cards ── */
.article-hero {
    border-bottom: 2px solid var(--rule);
    padding-bottom: 1.2rem;
    margin-bottom: 1rem;
}
.article-hero .headline {
    font-family: 'Playfair Display', Georgia, serif;
    font-size: clamp(1.6rem, 3.2vw, 2.4rem);
    font-weight: 900;
    line-height: 1.15;
    color: var(--ink);
    margin: 0 0 0.5rem;
    letter-spacing: -0.02em;
}
.article-card {
    border-bottom: 1px solid var(--rule-light);
    padding: 0.8rem 0;
    position: relative;
}
.article-card:last-child { border-bottom: none; }
.article-card .headline {
    font-family: 'Playfair Display', Georgia, serif;
    font-size: 1.05rem;
    font-weight: 700;
    line-height: 1.25;
    color: var(--ink);
    margin: 0 0 0.3rem;
}
.article-card .headline.italic { font-style: italic; font-weight: 400; }

/* ── Byline / Meta ── */
.article-meta {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    flex-wrap: wrap;
    margin-bottom: 0.4rem;
}
.category-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.5rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    background: var(--ink);
    color: var(--paper);
    padding: 1px 5px;
}
.category-tag.accent { background: var(--accent); }
.slot-badge {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.5rem;
    letter-spacing: 0.1em;
    color: var(--ink-muted);
    border: 1px solid var(--rule-light);
    padding: 1px 4px;
}
.score-inline {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.55rem;
    color: var(--ink-muted);
}

/* ── Score Bar ── */
.score-bar-wrap {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-top: 0.35rem;
}
.score-bar-bg {
    flex: 1;
    height: 3px;
    background: var(--rule-light);
    position: relative;
}
.score-bar-fill {
    height: 100%;
    background: var(--ink);
    transition: width 0.4s ease;
}
.score-bar-fill.high   { background: var(--positive); }
.score-bar-fill.medium { background: var(--ink); }
.score-bar-fill.low    { background: var(--ink-muted); }
.score-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.58rem;
    color: var(--ink-muted);
    width: 2.8rem;
    text-align: right;
}

/* ── Visibility Pill ── */
.visibility-pill {
    display: inline-flex;
    align-items: center;
    gap: 3px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.5rem;
    letter-spacing: 0.08em;
    border: 1px solid var(--rule-light);
    padding: 1px 5px;
    color: var(--ink-muted);
}
.vis-dot {
    width: 5px; height: 5px;
    border-radius: 50%;
    background: var(--positive);
    display: inline-block;
}
.vis-dot.med  { background: #b45309; }
.vis-dot.low  { background: var(--ink-muted); }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--paper-dark);
    border-right: 2px solid var(--rule-light);
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    font-family: 'Playfair Display', serif;
}

/* ── Ticker Banner ── */
.ticker-wrap {
    background: var(--ink);
    color: var(--paper);
    overflow: hidden;
    white-space: nowrap;
    border-top: 1px solid var(--accent);
    border-bottom: 1px solid var(--accent);
    padding: 4px 0;
    margin-bottom: 1rem;
}
.ticker-label {
    display: inline-block;
    background: var(--accent);
    color: var(--paper);
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.55rem;
    letter-spacing: 0.15em;
    padding: 0 8px;
    margin-right: 12px;
    text-transform: uppercase;
    vertical-align: middle;
}
.ticker-content {
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.06em;
    animation: ticker 35s linear infinite;
}
@keyframes ticker {
    from { transform: translateX(100vw); }
    to   { transform: translateX(-100%); }
}

/* ── Divider column line ── */
.col-rule {
    border-left: 1px solid var(--rule-light);
    height: 100%;
    min-height: 200px;
}

/* ── Stats Metric ── */
.stat-box {
    border: 1px solid var(--rule-light);
    padding: 0.5rem 0.7rem;
    background: var(--paper);
}
.stat-val {
    font-family: 'Playfair Display', serif;
    font-size: 1.6rem;
    font-weight: 900;
    color: var(--ink);
    line-height: 1;
}
.stat-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.5rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--ink-muted);
    margin-top: 2px;
}
.stat-delta {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.55rem;
    color: var(--positive);
}
.stat-delta.neg { color: var(--negative); }

/* ── Explainability ── */
.explain-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 3px 0;
    border-bottom: 1px dotted var(--rule-light);
    font-size: 0.72rem;
}
.explain-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.58rem;
    letter-spacing: 0.05em;
    color: var(--ink-soft);
}
.explain-bar-bg {
    flex: 1;
    height: 4px;
    background: var(--rule-light);
    margin: 0 8px;
}
.explain-bar-fill {
    height: 100%;
    background: var(--ink);
}
.explain-bar-fill.accent { background: var(--accent); }
.explain-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.58rem;
    width: 2.5rem;
    text-align: right;
    color: var(--ink-muted);
}

/* ── Highlighted alert ── */
.alert-bar {
    background: var(--highlight);
    border-left: 3px solid var(--accent);
    padding: 4px 8px;
    font-size: 0.7rem;
    margin-bottom: 0.6rem;
    color: var(--ink);
}

/* ── Plotly override ── */
.js-plotly-plot .plotly { background: transparent !important; }
</style>
""", unsafe_allow_html=True)


# ─── Data Loading ─────────────────────────────────────────────────────────────

@st.cache_data
def load_manifest(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        st.error(f"Missing file: {path}")
        st.stop()
    return pd.read_parquet(path)


@st.cache_data
def load_ranked(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        st.error(f"Missing file: {path}")
        st.stop()
    return pd.read_parquet(path)


# ─── Helpers ─────────────────────────────────────────────────────────────────

CATEGORY_COLORS = {
    "politics":      "#b91c1c",
    "tech":          "#1d4ed8",
    "sports":        "#166534",
    "health":        "#0e7490",
    "finance":       "#92400e",
    "world":         "#4c1d95",
    "entertainment": "#be185d",
    "crime":         "#374151",
}

FEATURE_LABELS = {
    "ranking_score":          "Raw ranking score (within pool)",
    "composite_score":        "Within-pool composite",
    "hours_since_first_seen": "Hours since first exposure (proxy)",
    "sent_polarity":          "Sentiment (exploratory)",
    "cb_composite_score":     "Clickbait risk (exploratory)",
}

def visibility_class(v: float) -> str:
    if v >= 0.6: return "high"
    if v >= 0.3: return "medium"
    return "low"

def vis_dot_class(v: float) -> str:
    if v >= 0.6: return ""
    if v >= 0.3: return "med"
    return "low"

def score_class(s: float) -> str:
    if s >= 0.7: return "high"
    if s >= 0.4: return "medium"
    return "low"

def fmt_score(s: float) -> str:
    return f"{s:.3f}"

def cat_tag(cat: str, accent: bool = False) -> str:
    cls = "category-tag accent" if accent else "category-tag"
    return f'<span class="{cls}">{cat.upper()}</span>'

def score_bar_html(score: float, label: str = "") -> str:
    pct  = int(score * 100)
    cls  = score_class(score)
    return f"""
    <div class="score-bar-wrap">
        <div class="score-bar-bg"><div class="score-bar-fill {cls}" style="width:{pct}%"></div></div>
        <span class="score-val">{fmt_score(score)}</span>
    </div>"""

def vis_pill(v: float, zone: str) -> str:
    dc = vis_dot_class(v)
    return f'<span class="visibility-pill"><span class="vis-dot {dc}"></span>{zone.upper().replace("_"," ")} · {v:.2f}</span>'

def explain_bar_html(label: str, value: float, max_val: float = 1.0, accent: bool = False) -> str:
    pct  = int((abs(value) / max(abs(max_val), 1e-9)) * 100)
    cls  = "accent" if accent else ""
    sign = "+" if value > 0 else ""
    return f"""
    <div class="explain-row">
        <span class="explain-label">{label}</span>
        <div class="explain-bar-bg"><div class="explain-bar-fill {cls}" style="width:{pct}%"></div></div>
        <span class="explain-val">{sign}{value:.3f}</span>
    </div>"""


# ─── Sidebar ──────────────────────────────────────────────────────────────────

def render_sidebar(manifest: pd.DataFrame, ranked: pd.DataFrame):
    st.sidebar.markdown("## ⚙ Editorial Controls")
    st.sidebar.markdown("---")

    # Category filter
    st.sidebar.markdown("### Category Filter")
    all_cats = sorted(manifest["category"].dropna().unique().tolist())
    selected_cats = st.sidebar.multiselect(
        "Show categories", all_cats, default=all_cats, label_visibility="collapsed"
    )

    st.sidebar.markdown("---")

    # Score threshold
    st.sidebar.markdown("### Minimum Composite Score")
    min_score = st.sidebar.slider("", 0.0, 1.0, 0.0, 0.01, label_visibility="collapsed")

    st.sidebar.markdown("---")

    # Zone filter
    st.sidebar.markdown("### Layout Zones")
    all_zones = sorted(manifest["zone"].unique().tolist())
    selected_zones = st.sidebar.multiselect(
        "Zones", all_zones, default=all_zones, label_visibility="collapsed"
    )

    st.sidebar.markdown("---")

    # Explainability panel
    st.sidebar.markdown("### 🔍 Article Explainability")
    all_titles = manifest["title"].dropna().tolist()
    selected_title = st.sidebar.selectbox(
        "Inspect article", ["— select —"] + all_titles, label_visibility="collapsed"
    )

    if selected_title and selected_title != "— select —":
        row = manifest[manifest["title"] == selected_title].iloc[0]
        nid = row["news_id"]
        r_row = ranked[ranked["news_id"] == nid] if "news_id" in ranked.columns else pd.DataFrame()

        st.sidebar.markdown(f"**{row['title'][:60]}{'…' if len(row['title'])>60 else ''}**")
        st.sidebar.markdown(
            f'<div class="alert-bar">{cat_tag(row["category"])} &nbsp;'
            f'{vis_pill(row["visibility"], row["zone"])}</div>',
            unsafe_allow_html=True
        )

        # Score decomposition
        st.sidebar.markdown("**Score decomposition**")
        explain_items = [
            ("Composite Score",   row["composite_score"],  1.0,  False),
            ("Visibility Weight", row["visibility"],        1.0,  False),
            ("Display-weighted utility", row["display_weighted_utility"], 1.0, True),
        ]
        if not r_row.empty:
            rr = r_row.iloc[0]
            for feat, label in FEATURE_LABELS.items():
                if feat in rr and pd.notna(rr[feat]):
                    val = float(rr[feat])
                    mx  = 1.0 if feat not in ("read_avg_grade","hl_word_count","ner_total_count") else (20 if "grade" in feat else (20 if "word" in feat else 10))
                    explain_items.append((label, val, mx, feat == "cb_composite_score"))

        html_blocks = "".join(
            explain_bar_html(lbl, val, mx, accent)
            for lbl, val, mx, accent in explain_items
        )
        st.sidebar.markdown(html_blocks, unsafe_allow_html=True)
        st.sidebar.info(
            f"""
            Placement = order of the within-pool composite score after deduplication and
            category caps. Visibility is a hand-set display weight and does not change which
            articles are selected. This panel shows values, not a model explanation.
            """
        )

    return selected_cats, min_score, selected_zones


# ─── Masthead ─────────────────────────────────────────────────────────────────

def render_masthead(manifest: pd.DataFrame):
    from datetime import datetime
    now  = datetime.now()
    date_str = f"{now:%A, %B} {now.day}, {now:%Y}".upper()  # portable (no %-d on Windows)
    n_stories = len(manifest)

    st.markdown(f"""
    <div class="masthead">
        <div class="masthead-kicker">AI EDITORIAL INTELLIGENCE SYSTEM · MICROSOFT MIND</div>
        <div class="masthead-title">The Editorial</div>
        <div class="masthead-rule"></div>
        <div class="masthead-meta">
            <span>{date_str}</span>
            <span><span class="masthead-edition">FRONT PAGE</span></span>
            <span>{n_stories} STORIES RANKED · ONE CANDIDATE POOL</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─── Ticker ──────────────────────────────────────────────────────────────────

def render_ticker(manifest: pd.DataFrame):
    top = manifest.nlargest(8, "composite_score")["title"].tolist()
    ticker_text = "   ·   ".join(f"▸ {t}" for t in top)
    st.markdown(f"""
    <div class="ticker-wrap">
        <span class="ticker-label">Top Stories</span>
        <span class="ticker-content">{ticker_text}</span>
    </div>
    """, unsafe_allow_html=True)


# ─── Stats Row ───────────────────────────────────────────────────────────────

def render_stats_row(manifest: pd.DataFrame, ranked: pd.DataFrame):
    n_cats  = manifest["category"].nunique()
    avg_vis = manifest["visibility"].mean()
    avg_eng = manifest["composite_score"].mean()
    top_cat = manifest["category"].value_counts().idxmax()
    top_score = manifest["composite_score"].max()
    n_slots = len(manifest)
    cb_flag = (ranked["cb_composite_score"] > 0.6).sum() if "cb_composite_score" in ranked.columns else 0

    stats = [
        (str(n_slots),          "Slots Filled",          f"+{n_slots} total",        False),
        (str(n_cats),           "Categories",            f"{n_cats} distinct",        False),
        (f"{avg_vis:.2f}",      "Avg Visibility",        "weighted mean",             False),
        (f"{avg_eng:.3f}",      "Avg Engagement",        "composite score",           False),
        (top_cat.upper(),       "Lead Category",         "most represented",          False),
        (f"{top_score:.3f}",    "Top Score",             "highest composite score",   False),
        (str(cb_flag),          "Clickbait Flagged",     "score > 0.6",              cb_flag > 3),
    ]

    cols = st.columns(len(stats))
    for col, (val, label, delta, neg) in zip(cols, stats):
        dcls = "neg" if neg else ""
        col.markdown(f"""
        <div class="stat-box">
            <div class="stat-val">{val}</div>
            <div class="stat-label">{label}</div>
            <div class="stat-delta {dcls}">{delta}</div>
        </div>
        """, unsafe_allow_html=True)


# ─── Article Card Renderers ───────────────────────────────────────────────────

def render_hero(row: pd.Series):
    is_accent = row["category"] in ("politics", "world", "crime")
    st.markdown(f"""
    <div class="article-hero">
        <div class="article-meta">
            {cat_tag(row['category'], accent=is_accent)}
            <span class="slot-badge">SLOT 0 · {row['zone'].upper().replace('_',' ')}</span>
            {vis_pill(row['visibility'], row['zone'])}
        </div>
        <div class="headline">{row['title']}</div>

        <div class="score-inline">
            Within-pool score {row['composite_score']:.3f} · raw ranking score {row['ranking_score']:.3f}
        </div>

    {score_bar_html(row['composite_score'])}
    </div>
    """, unsafe_allow_html=True)


def render_article_card(row: pd.Series, rank_num: int):
    italic = row["category"] in ("entertainment", "sports")
    hl_cls = "headline italic" if italic else "headline"
    is_accent = row["category"] in ("politics", "world", "crime")
    st.markdown(f"""
    <div class="article-card">
        <div class="article-meta">
            {cat_tag(row['category'], accent=is_accent)}
            <span class="slot-badge">#{rank_num} · {row['zone'].upper().replace('_',' ')}</span>
            {vis_pill(row['visibility'], row['zone'])}
        </div>
        <div class="{hl_cls}">{row['title']}</div>

        <div class="score-inline">
            Within-pool score {row['composite_score']:.3f} · raw ranking score {row['ranking_score']:.3f}
        </div>

        {score_bar_html(row['composite_score'])}
    </div>
    """, unsafe_allow_html=True)


# ─── Engagement Chart ─────────────────────────────────────────────────────────

def render_engagement_chart(manifest: pd.DataFrame):
    st.markdown('<div class="section-rule"><span class="section-label">§ Engagement Intelligence</span></div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Score Distribution", "Category Breakdown", "Visibility Map"])

    with tab1:
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=manifest["composite_score"],
            nbinsx=20,
            marker_color="#0d0d0d",
            marker_line_color="#f7f4ef",
            marker_line_width=1,
            name="Composite Score",
        ))
        if "ranking_score" in manifest.columns:
            fig.add_trace(go.Histogram(
                x=within_pool_minmax(manifest["ranking_score"].to_numpy()),
                nbinsx=20,
                marker_color="#b91c1c",
                marker_line_color="#f7f4ef",
                marker_line_width=1,
                opacity=0.55,
                name="Ranking score (min-max within pool)",
            ))
        fig.update_layout(
            barmode="overlay",
            plot_bgcolor="#f7f4ef", paper_bgcolor="#f7f4ef",
            font=dict(family="JetBrains Mono", size=10, color="#0d0d0d"),
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(font=dict(size=9)),
            xaxis=dict(showgrid=False, title="Score"),
            yaxis=dict(showgrid=True, gridcolor="#c8bfaf", title="Articles"),
            height=240,
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with tab2:
        cat_stats = (
            manifest.groupby("category")
            .agg(count=("composite_score","count"), avg_score=("composite_score","mean"), avg_vis=("visibility","mean"))
            .reset_index().sort_values("avg_score", ascending=True)
        )
        colors = [CATEGORY_COLORS.get(c, "#374151") for c in cat_stats["category"]]
        fig2 = go.Figure(go.Bar(
            x=cat_stats["avg_score"], y=cat_stats["category"],
            orientation="h",
            marker_color=colors,
            text=[f"{v:.3f}" for v in cat_stats["avg_score"]],
            textposition="outside",
            textfont=dict(family="JetBrains Mono", size=9),
        ))
        fig2.update_layout(
            plot_bgcolor="#f7f4ef", paper_bgcolor="#f7f4ef",
            font=dict(family="JetBrains Mono", size=10, color="#0d0d0d"),
            margin=dict(l=10, r=40, t=10, b=10),
            xaxis=dict(showgrid=True, gridcolor="#c8bfaf", title="Avg Composite Score", range=[0, 1.1]),
            yaxis=dict(showgrid=False),
            height=240,
        )
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    with tab3:
        fig3 = go.Figure(go.Scatter(
            x=manifest["slot_id"],
            y=manifest["visibility"],
            mode="markers+lines",
            marker=dict(
                size=manifest["composite_score"] * 18 + 4,
                color=manifest["composite_score"],
                colorscale=[[0,"#c8bfaf"],[0.5,"#0d0d0d"],[1,"#b91c1c"]],
                showscale=True,
                colorbar=dict(title="Score", thickness=10, tickfont=dict(size=8)),
                line=dict(color="#f7f4ef", width=1),
            ),
            line=dict(color="#c8bfaf", width=1, dash="dot"),
            text=manifest["category"].str.upper(),
            hovertemplate="<b>Slot %{x}</b><br>Visibility: %{y:.2f}<br>Category: %{text}<extra></extra>",
        ))
        fig3.update_layout(
            plot_bgcolor="#f7f4ef", paper_bgcolor="#f7f4ef",
            font=dict(family="JetBrains Mono", size=10, color="#0d0d0d"),
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(showgrid=False, title="Slot Position"),
            yaxis=dict(showgrid=True, gridcolor="#c8bfaf", title="Visibility Weight"),
            height=240,
        )
        st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})


# ─── Feature Importance Chart ─────────────────────────────────────────────────

def render_feature_importance(ranked: pd.DataFrame):
    feat_cols = [c for c in FEATURE_LABELS if c in ranked.columns]
    if not feat_cols:
        return
    st.markdown('<div class="section-rule"><span class="section-label">§ Mean feature values (descriptive, not model importance)</span></div>', unsafe_allow_html=True)

    means = ranked[feat_cols].mean().rename(FEATURE_LABELS).sort_values(ascending=True)
    colors = ["#b91c1c" if "clickbait" in k.lower() or "risk" in k.lower() else "#0d0d0d" for k in means.index]

    fig = go.Figure(go.Bar(
        x=means.values, y=means.index,
        orientation="h",
        marker_color=colors,
        text=[f"{v:.3f}" for v in means.values],
        textposition="outside",
        textfont=dict(family="JetBrains Mono", size=9),
    ))
    fig.update_layout(
        plot_bgcolor="#f7f4ef", paper_bgcolor="#f7f4ef",
        font=dict(family="JetBrains Mono", size=10, color="#0d0d0d"),
        margin=dict(l=10, r=40, t=10, b=10),
        xaxis=dict(showgrid=True, gridcolor="#c8bfaf"),
        yaxis=dict(showgrid=False),
        height=260,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_evaluation(artifacts_dir: str):
    root = Path(artifacts_dir)
    current = root / "CURRENT"
    if not current.exists():
        return
    report_path = root / current.read_text().strip() / "evaluation.json"
    if not report_path.exists():
        return
    report = json.loads(report_path.read_text())
    st.markdown('<div class="section-rule"><span class="section-label">§ Offline evaluation (test period)</span></div>',
                unsafe_allow_html=True)
    if report.get("warning"):
        st.warning(report["warning"])
    rows = []
    for name, r in report["rankers"].items():
        rows.append({"model": name, "NDCG@10": r["ndcg@10"]["mean"], "NDCG@10 95% CI": r["ndcg@10"]["ci95"],
                     "MRR": r["mrr"]["mean"], "impressions": r["impressions_evaluated"]})
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
    st.caption(f"Split boundaries: {report['split_boundaries']} · data source: {report['data_source']}")


# ─── Front Page Grid ─────────────────────────────────────────────────────────

def render_front_page(manifest: pd.DataFrame):
    st.markdown('<div class="section-rule"><span class="section-label">§ Front Page · Ranked Slate</span></div>', unsafe_allow_html=True)

    if manifest.empty:
        st.info("No articles match the current filter criteria.")
        return

    manifest = manifest.sort_values("slot_id").reset_index(drop=True)

    # Hero
    hero = manifest[manifest["zone"] == "hero"]
    if not hero.empty:
        render_hero(hero.iloc[0])

    # Top rail
    top_rail = manifest[manifest["zone"] == "top_rail"]
    if not top_rail.empty:
        st.markdown('<div class="section-rule"><span class="section-label">Top Stories</span></div>', unsafe_allow_html=True)
        cols = st.columns(len(top_rail))
        for col, (_, row) in zip(cols, top_rail.iterrows()):
            with col:
                render_article_card(row, int(row["slot_id"]) + 1)

    # Mid + sidebar
    mid   = manifest[manifest["zone"] == "mid_rail"]
    side  = manifest[manifest["zone"] == "sidebar"]

    if not mid.empty or not side.empty:
        st.markdown('<div class="section-rule"><span class="section-label">Analysis & Features</span></div>', unsafe_allow_html=True)
        c1, c2 = st.columns([2, 1])
        with c1:
            for _, row in mid.iterrows():
                render_article_card(row, int(row["slot_id"]) + 1)
        with c2:
            st.markdown("**BRIEFING**")
            for _, row in side.iterrows():
                render_article_card(row, int(row["slot_id"]) + 1)

    # Bottom rail
    bot = manifest[manifest["zone"] == "bottom_rail"]
    if not bot.empty:
        st.markdown('<div class="section-rule"><span class="section-label">More Stories</span></div>', unsafe_allow_html=True)
        cols = st.columns(min(4, len(bot)))
        for col, (_, row) in zip(cols, bot.iterrows()):
            with col:
                render_article_card(row, int(row["slot_id"]) + 1)

    # Below fold
    bf = manifest[manifest["zone"] == "below_fold"]
    if not bf.empty:
        with st.expander(f"▸ {len(bf)} More Stories (Below Fold)", expanded=False):
            for _, row in bf.iterrows():
                render_article_card(row, int(row["slot_id"]) + 1)


# ─── Raw Table ────────────────────────────────────────────────────────────────

def render_data_table(manifest: pd.DataFrame):
    st.markdown('<div class="section-rule"><span class="section-label">§ Manifest Data</span></div>', unsafe_allow_html=True)
    display_cols = [c for c in ["slot_id","zone","position","visibility","category",
                                "title","ranking_score","composite_score","display_weighted_utility"]
                    if c in manifest.columns]
    styled = (
        manifest[display_cols]
        .sort_values("slot_id")
        .style
        .format({c: "{:.3f}" for c in ["visibility","ranking_score","composite_score","display_weighted_utility"] if c in display_cols})
    )
    st.dataframe(styled, use_container_width=True, height=320)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    MANIFEST_PATH = os.environ.get("NEWSREC_DASHBOARD_MANIFEST", "data/processed/front_page_manifest.parquet")
    RANKED_PATH   = os.environ.get("NEWSREC_DASHBOARD_RANKED", "data/processed/ranked_test.parquet")

    manifest = load_manifest(MANIFEST_PATH)
    ranked   = load_ranked(RANKED_PATH)

    # Sidebar — returns filters
    selected_cats, min_score, selected_zones = render_sidebar(manifest, ranked)

    # Apply filters
    filtered = manifest.copy()
    if selected_cats:
        filtered = filtered[filtered["category"].isin(selected_cats)]
    if selected_zones:
        filtered = filtered[filtered["zone"].isin(selected_zones)]
    filtered = filtered[filtered["composite_score"] >= min_score]

    # Masthead
    render_masthead(filtered)

    # Ticker
    render_ticker(filtered)

    # Stats row
    render_stats_row(filtered, ranked)

    # Engagement charts
    render_engagement_chart(filtered)

    # Front page grid
    render_front_page(filtered)

    # Feature values
    render_feature_importance(ranked)

    # Offline evaluation (from artifacts)
    render_evaluation(os.environ.get("NEWSREC_ARTIFACTS_DIR", "artifacts"))

    # Raw data table (collapsible)
    with st.expander("▸ Raw Manifest Table", expanded=False):
        render_data_table(filtered)

    # Footer
    st.markdown("""
    <div style="border-top:2px solid #0d0d0d; margin-top:2rem; padding-top:0.5rem;
                font-family:'JetBrains Mono',monospace; font-size:0.52rem;
                letter-spacing:0.12em; text-transform:uppercase; color:#666;
                display:flex; justify-content:space-between;">
        <span>The Editorial · AI Editorial Intelligence System</span>
        <span>Powered by LightGBM · Sentence Transformers · Microsoft MIND</span>
        <span>Offline evaluation shown above is from the artifact report</span>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()