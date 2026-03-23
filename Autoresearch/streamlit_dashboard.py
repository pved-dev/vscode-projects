"""
streamlit_dashboard.py — Live AutoResearch M5 Dashboard
Run with: streamlit run streamlit_dashboard.py
"""
import json
import time
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AutoResearch M5",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

LOG_FILE  = Path("logs/experiment_log.jsonl")
BASELINE  = 0.98144
REFRESH_S = 15

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    .metric-label { font-size: 0.75rem !important; }
    div[data-testid="metric-container"] {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 12px 16px;
    }
</style>
""", unsafe_allow_html=True)


# ── Data loader ───────────────────────────────────────────────────────────────
@st.cache_data(ttl=REFRESH_S)
def load_log() -> pd.DataFrame:
    if not LOG_FILE.exists():
        return pd.DataFrame()
    entries = []
    for line in LOG_FILE.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    if not entries:
        return pd.DataFrame()
    df = pd.DataFrame(entries)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["exp_num"]   = range(1, len(df) + 1)
    return df


# ── Header ────────────────────────────────────────────────────────────────────
col_title, col_refresh = st.columns([4, 1])
with col_title:
    st.markdown("## 🔬 AutoResearch M5 — Live Dashboard")
    st.caption("LightGBM / XGBoost autonomous experiment loop · auto-refreshes every 15s")
with col_refresh:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("⟳ Refresh now"):
        st.cache_data.clear()

df = load_log()

if df.empty:
    st.info("No experiments yet. Start the loop with `./run.sh loop`")
    st.stop()

# ── Derived fields ────────────────────────────────────────────────────────────
valid    = df[df["wrmsse"].notna()]
failed   = df[df["wrmsse"].isna()]
improved = df[df.get("improvement", False) == True] if "improvement" in df.columns else pd.DataFrame()
best_row = valid.loc[valid["wrmsse"].idxmin()] if not valid.empty else None
best_val = best_row["wrmsse"] if best_row is not None else None
delta_vs_baseline = ((best_val - BASELINE) / BASELINE * 100) if best_val else None
keep_rate = len(improved) / len(valid) * 100 if len(valid) > 0 else 0

# ── Metric cards ──────────────────────────────────────────────────────────────
st.markdown("---")
m1, m2, m3, m4, m5 = st.columns(5)

m1.metric("Total Runs", len(df))
m2.metric("Successful", len(valid), f"{len(failed)} failed")
m3.metric(
    "Best WRMSSE",
    f"{best_val:.5f}" if best_val else "—",
    f"{delta_vs_baseline:+.2f}% vs baseline" if delta_vs_baseline else None,
    delta_color="inverse",
)
m4.metric("Baseline", f"{BASELINE:.5f}")
m5.metric("Keep Rate", f"{keep_rate:.0f}%", f"{len(improved)} improvements")

st.markdown("---")

# ── Charts ────────────────────────────────────────────────────────────────────
chart_col, pie_col = st.columns([3, 1])

with chart_col:
    st.markdown("#### WRMSSE over experiments")

    fig = go.Figure()

    # Baseline reference line
    fig.add_hline(
        y=BASELINE, line_dash="dash", line_color="#f59e0b",
        annotation_text=f"Baseline {BASELINE}", annotation_position="top left",
        annotation_font_color="#f59e0b",
    )

    # All valid scores
    colors = []
    for _, row in valid.iterrows():
        if best_val and row["wrmsse"] == best_val:
            colors.append("#22c55e")
        elif row.get("improvement"):
            colors.append("#22c55e")
        else:
            colors.append("#3b82f6")

    fig.add_trace(go.Scatter(
        x=valid["exp_num"],
        y=valid["wrmsse"],
        mode="lines+markers",
        line=dict(color="#3b82f6", width=2),
        marker=dict(color=colors, size=8),
        hovertemplate=(
            "<b>Exp %{x}</b><br>"
            "WRMSSE: %{y:.5f}<br>"
            "<extra></extra>"
        ),
        name="WRMSSE",
    ))

    # Highlight best
    if best_row is not None:
        fig.add_trace(go.Scatter(
            x=[best_row["exp_num"]],
            y=[best_row["wrmsse"]],
            mode="markers",
            marker=dict(color="#22c55e", size=14, symbol="star"),
            name="Best",
            hovertemplate=f"<b>★ Best</b><br>WRMSSE: {best_row['wrmsse']:.5f}<extra></extra>",
        ))

    fig.update_layout(
        height=320,
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="Experiment #", gridcolor="#e9ecef"),
        yaxis=dict(title="WRMSSE", gridcolor="#e9ecef"),
        legend=dict(orientation="h", y=-0.2),
        showlegend=True,
    )
    st.plotly_chart(fig, use_container_width=True)

with pie_col:
    st.markdown("#### Outcome breakdown")
    counts = {
        "Improved":  len(improved),
        "No gain":   len(valid) - len(improved),
        "Failed":    len(failed),
    }
    pie_fig = px.pie(
        values=list(counts.values()),
        names=list(counts.keys()),
        color=list(counts.keys()),
        color_discrete_map={
            "Improved": "#22c55e",
            "No gain":  "#3b82f6",
            "Failed":   "#ef4444",
        },
        hole=0.5,
    )
    pie_fig.update_layout(
        height=320,
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=True,
        legend=dict(orientation="h", y=-0.1),
    )
    pie_fig.update_traces(textposition="inside", textinfo="percent+label")
    st.plotly_chart(pie_fig, use_container_width=True)

# ── WRMSSE over time ──────────────────────────────────────────────────────────
if len(valid) > 1:
    st.markdown("#### WRMSSE over time")
    time_fig = px.scatter(
        valid, x="timestamp", y="wrmsse",
        color="improvement" if "improvement" in valid.columns else None,
        color_discrete_map={True: "#22c55e", False: "#3b82f6"},
        hover_data=["run_id", "hypothesis"] if "hypothesis" in valid.columns else ["run_id"],
    )
    time_fig.add_hline(y=BASELINE, line_dash="dash", line_color="#f59e0b")
    time_fig.update_layout(
        height=250,
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor="#e9ecef"),
        yaxis=dict(title="WRMSSE", gridcolor="#e9ecef"),
        showlegend=False,
    )
    st.plotly_chart(time_fig, use_container_width=True)

# ── Experiment log table ──────────────────────────────────────────────────────
st.markdown("---")
st.markdown("#### Experiment log")

# Filter controls
filter_col1, filter_col2, filter_col3 = st.columns([1, 1, 2])
with filter_col1:
    show_filter = st.selectbox("Show", ["All", "Improvements only", "Failed only"])
with filter_col2:
    sort_order = st.selectbox("Sort", ["Newest first", "Oldest first", "Best WRMSSE first"])
with filter_col3:
    search = st.text_input("Search hypothesis", placeholder="e.g. lag, xgboost, rolling …")

display_df = df.copy()

if show_filter == "Improvements only":
    display_df = display_df[display_df.get("improvement", False) == True]
elif show_filter == "Failed only":
    display_df = display_df[display_df["wrmsse"].isna()]

if search:
    mask = display_df["hypothesis"].str.contains(search, case=False, na=False) if "hypothesis" in display_df.columns else pd.Series([True] * len(display_df))
    display_df = display_df[mask]

if sort_order == "Newest first":
    display_df = display_df.sort_values("timestamp", ascending=False)
elif sort_order == "Oldest first":
    display_df = display_df.sort_values("timestamp", ascending=True)
elif sort_order == "Best WRMSSE first":
    display_df = display_df.sort_values("wrmsse", ascending=True)

# Format for display
table_cols = ["exp_num", "run_id", "wrmsse", "duration_sec", "improvement", "hypothesis", "timestamp"]
table_cols = [c for c in table_cols if c in display_df.columns]
display_df = display_df[table_cols].copy()

if "wrmsse" in display_df.columns:
    display_df["wrmsse"] = display_df["wrmsse"].apply(
        lambda x: f"{x:.5f}" if pd.notna(x) else "FAILED"
    )
if "duration_sec" in display_df.columns:
    display_df["duration_sec"] = display_df["duration_sec"].apply(
        lambda x: f"{x:.0f}s" if pd.notna(x) else "—"
    )
if "improvement" in display_df.columns:
    display_df["improvement"] = display_df["improvement"].apply(
        lambda x: "★ YES" if x else ""
    )
if "hypothesis" in display_df.columns:
    display_df["hypothesis"] = display_df["hypothesis"].str[:80]
if "timestamp" in display_df.columns:
    display_df["timestamp"] = display_df["timestamp"].dt.strftime("%H:%M:%S")

display_df = display_df.rename(columns={
    "exp_num": "#", "run_id": "Run ID", "wrmsse": "WRMSSE",
    "duration_sec": "Duration", "improvement": "Improved",
    "hypothesis": "Hypothesis", "timestamp": "Time"
})

st.dataframe(display_df, use_container_width=True, height=400)

# ── Best hypothesis ───────────────────────────────────────────────────────────
if best_row is not None and "hypothesis" in best_row:
    st.markdown("---")
    st.markdown("#### ★ Best hypothesis so far")
    st.success(f"**WRMSSE {best_row['wrmsse']:.5f}** — {best_row['hypothesis']}")

# ── Auto-refresh ──────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(f"Auto-refreshing every {REFRESH_S}s · {len(df)} total experiments · last updated {pd.Timestamp.now().strftime('%H:%M:%S')}")
time.sleep(REFRESH_S)
st.rerun()