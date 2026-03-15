"""
HydroPredict AI — Climate-Aware Renewable Grid Optimizer
Streamlit Dashboard for the AI Hackathon 2026
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.data_generator import (
    DISTRICTS,
    HYDRO_PLANTS,
    generate_hourly_forecast_data,
)

# ── Page Config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="HydroPredict AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 1200px;
    }
    .stMetric {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        padding: 1rem 1.2rem;
        border-radius: 12px;
        border: 1px solid #334155;
    }
    .stMetric label {
        color: #94a3b8 !important;
        font-size: 0.85rem !important;
    }
    .stMetric [data-testid="stMetricValue"] {
        color: #f1f5f9 !important;
        font-weight: 600 !important;
    }
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    }
    div[data-testid="stSidebar"] .stMarkdown p,
    div[data-testid="stSidebar"] .stMarkdown li,
    div[data-testid="stSidebar"] label {
        color: #cbd5e1;
    }
    .alert-box-green {
        background: linear-gradient(135deg, #064e3b 0%, #065f46 100%);
        border: 1px solid #10b981;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        margin: 0.5rem 0;
    }
    .alert-box-yellow {
        background: linear-gradient(135deg, #78350f 0%, #92400e 100%);
        border: 1px solid #f59e0b;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        margin: 0.5rem 0;
    }
    .alert-box-red {
        background: linear-gradient(135deg, #7f1d1d 0%, #991b1b 100%);
        border: 1px solid #ef4444;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        margin: 0.5rem 0;
    }
    .hero-header {
        text-align: center;
        padding: 0.5rem 0 1rem 0;
    }
    .hero-header h1 {
        font-size: 2rem;
        margin-bottom: 0.3rem;
    }
    .hero-header p {
        color: #94a3b8;
        font-size: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Helpers ──────────────────────────────────────────────────────────────────

PLOT_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(15,23,42,0.8)",
    plot_bgcolor="rgba(15,23,42,0.4)",
    font=dict(family="Inter, sans-serif", color="#e2e8f0"),
    margin=dict(l=50, r=30, t=40, b=40),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)


def load_model_artifacts():
    model_path = Path("models/xgb_hydro_model.joblib")
    if model_path.exists():
        return {
            "model": joblib.load(model_path),
            "features": joblib.load("models/feature_columns.joblib"),
            "metrics": json.loads(Path("models/metrics.json").read_text()),
            "importance": json.loads(Path("models/feature_importance.json").read_text()),
        }
    return None


def get_alert_level(generation: float, capacity: float) -> tuple[str, str, str]:
    ratio = generation / capacity if capacity > 0 else 0
    if ratio >= 0.65:
        return "NORMAL", "green", "Grid operating within safe parameters. No action required."
    elif ratio >= 0.40:
        return "WARNING", "yellow", "Generation dropping — consider activating reserve capacity or demand-side management."
    else:
        return "CRITICAL", "red", "Severe generation deficit — activate backup power sources and initiate load shedding protocols."


# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚡ HydroPredict AI")
    st.markdown("*Climate-Aware Grid Optimizer*")
    st.markdown("---")

    district = st.selectbox(
        "Select District",
        list(DISTRICTS.keys()),
        index=0,
        help="Choose a district to view its hydropower forecast",
    )
    river = DISTRICTS[district]["river"]
    plant = HYDRO_PLANTS[river]

    st.markdown(f"**River System:** {river}")
    st.markdown(f"**Plant:** {plant['name']}")
    st.markdown(f"**Capacity:** {plant['capacity_mw']} MW")

    st.markdown("---")
    st.markdown("### 🔧 What-If Scenario")
    st.markdown("Adjust parameters to simulate conditions:")

    rain_modifier = st.slider(
        "Rainfall Adjustment (%)",
        min_value=-80, max_value=200, value=0, step=10,
        help="Simulate drought (-80%) or heavy monsoon (+200%)",
    )
    temp_modifier = st.slider(
        "Temperature Shift (°C)",
        min_value=-5.0, max_value=5.0, value=0.0, step=0.5,
        help="Simulate warming or cooling climate scenarios",
    )

    alert_threshold = st.slider(
        "Alert Threshold (% of capacity)",
        min_value=20, max_value=80, value=40, step=5,
        help="Set the threshold below which a grid alert triggers",
    )

    st.markdown("---")
    st.markdown("### 📊 About the Data")
    st.markdown("""
    - **Weather:** Nepal Multi-District (2020-2025)
    - **Hydrology:** River flow & generation
    - **Model:** XGBoost time-series
    - **Forecast:** 72-hour rolling window
    """)

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color:#64748b; font-size:0.75rem;'>"
        "AI Hackathon 2026 • Embark College"
        "</div>",
        unsafe_allow_html=True,
    )

# ── Generate Forecast Data ───────────────────────────────────────────────────

@st.cache_data(ttl=300)
def get_forecast(district_name: str, rain_mod: int, temp_mod: float):
    df = generate_hourly_forecast_data(
        base_date="2026-03-15",
        district=district_name,
        seed=99,
    )
    if rain_mod != 0:
        factor = 1 + rain_mod / 100
        df["rainfall_mm"] = np.clip(df["rainfall_mm"] * factor, 0, None).round(2)
        flow_adjustment = 1 + (rain_mod / 100) * 0.6
        df["river_flow_cumecs"] = np.clip(df["river_flow_cumecs"] * flow_adjustment, 5, 500).round(2)
    if temp_mod != 0:
        df["temperature_c"] = (df["temperature_c"] + temp_mod).round(1)
        snowmelt_effect = max(0, temp_mod * 0.02)
        df["river_flow_cumecs"] = (df["river_flow_cumecs"] * (1 + snowmelt_effect)).round(2)

    cap = HYDRO_PLANTS[DISTRICTS[district_name]["river"]]["capacity_mw"]
    flow_median = df["river_flow_cumecs"].median()
    normalized = df["river_flow_cumecs"] / (flow_median + 1e-6)
    efficiency = 1 - np.exp(-1.2 * normalized)
    df["predicted_generation_mw"] = np.clip(cap * efficiency, 0, cap).round(2)
    df["plant_capacity_mw"] = cap
    return df


forecast_df = get_forecast(district, rain_modifier, temp_modifier)

# ── Hero Header ──────────────────────────────────────────────────────────────

st.markdown(
    '<div class="hero-header">'
    '<h1>⚡ HydroPredict AI</h1>'
    '<p>Climate-Aware Renewable Grid Optimizer — Predicting hydropower generation to keep Nepal\'s grid stable</p>'
    '</div>',
    unsafe_allow_html=True,
)

# ── Current Status Metrics ───────────────────────────────────────────────────

current_gen = forecast_df["predicted_generation_mw"].iloc[0]
avg_gen_24h = forecast_df["predicted_generation_mw"].iloc[:24].mean()
min_gen_24h = forecast_df["predicted_generation_mw"].iloc[:24].min()
max_gen_24h = forecast_df["predicted_generation_mw"].iloc[:24].max()
capacity = forecast_df["plant_capacity_mw"].iloc[0]
utilization = (avg_gen_24h / capacity * 100) if capacity > 0 else 0

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Current Output", f"{current_gen:.1f} MW", f"{current_gen/capacity*100:.0f}% capacity")
col2.metric("24h Avg Forecast", f"{avg_gen_24h:.1f} MW", f"{utilization:.0f}% utilization")
col3.metric("24h Min", f"{min_gen_24h:.1f} MW")
col4.metric("24h Max", f"{max_gen_24h:.1f} MW")
col5.metric("Plant Capacity", f"{capacity:.1f} MW")

# ── Grid Alert ───────────────────────────────────────────────────────────────

alert_level, alert_color, alert_msg = get_alert_level(min_gen_24h, capacity)
threshold_mw = capacity * alert_threshold / 100
hours_below = (forecast_df["predicted_generation_mw"].iloc[:24] < threshold_mw).sum()

st.markdown(f"""
<div class="alert-box-{alert_color}">
    <strong style="font-size: 1.1rem; color: white;">
        {'🟢' if alert_color == 'green' else '🟡' if alert_color == 'yellow' else '🔴'}
        GRID STATUS: {alert_level}
    </strong>
    <br/>
    <span style="color: #e2e8f0;">{alert_msg}</span>
    <br/>
    <span style="color: #94a3b8; font-size: 0.85rem;">
        {hours_below} hour(s) in the next 24h predicted below threshold ({threshold_mw:.1f} MW)
    </span>
</div>
""", unsafe_allow_html=True)

# ── Main Forecast Chart ─────────────────────────────────────────────────────

st.markdown("### 📈 72-Hour Generation Forecast")

fig_main = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.08,
    row_heights=[0.65, 0.35],
    subplot_titles=("Predicted Hydropower Generation", "Weather Conditions"),
)

gen = forecast_df["predicted_generation_mw"]
colors = ["#ef4444" if g < threshold_mw else "#f59e0b" if g < capacity * 0.65 else "#10b981" for g in gen]

fig_main.add_trace(
    go.Bar(
        x=forecast_df["datetime"], y=gen,
        marker_color=colors, name="Generation (MW)",
        hovertemplate="<b>%{x|%b %d, %H:%M}</b><br>Generation: %{y:.1f} MW<extra></extra>",
    ),
    row=1, col=1,
)

fig_main.add_hline(
    y=capacity, line_dash="dash", line_color="#3b82f6", line_width=2,
    annotation_text=f"Capacity: {capacity} MW",
    annotation_position="top left",
    annotation_font_color="#3b82f6",
    row=1, col=1,
)

fig_main.add_hline(
    y=threshold_mw, line_dash="dot", line_color="#ef4444", line_width=1.5,
    annotation_text=f"Alert: {threshold_mw:.0f} MW",
    annotation_position="bottom left",
    annotation_font_color="#ef4444",
    row=1, col=1,
)

fig_main.add_trace(
    go.Scatter(
        x=forecast_df["datetime"], y=forecast_df["rainfall_mm"],
        fill="tozeroy", fillcolor="rgba(59,130,246,0.3)",
        line=dict(color="#3b82f6", width=1.5),
        name="Rainfall (mm)",
        hovertemplate="Rainfall: %{y:.1f} mm<extra></extra>",
    ),
    row=2, col=1,
)

fig_main.add_trace(
    go.Scatter(
        x=forecast_df["datetime"], y=forecast_df["temperature_c"],
        line=dict(color="#f97316", width=2),
        name="Temperature (°C)",
        yaxis="y4",
        hovertemplate="Temp: %{y:.1f}°C<extra></extra>",
    ),
    row=2, col=1,
)

fig_main.update_layout(
    **PLOT_LAYOUT,
    height=550,
    yaxis=dict(title="Generation (MW)", range=[0, capacity * 1.15]),
    yaxis2=dict(title="Rainfall (mm)"),
    hovermode="x unified",
)

st.plotly_chart(fig_main, use_container_width=True)

# ── River Flow & Generation Correlation ──────────────────────────────────────

st.markdown("### 🌊 River Flow vs Energy Generation")

col_left, col_right = st.columns(2)

with col_left:
    fig_flow = go.Figure()
    fig_flow.add_trace(go.Scatter(
        x=forecast_df["datetime"], y=forecast_df["river_flow_cumecs"],
        fill="tozeroy", fillcolor="rgba(6,182,212,0.2)",
        line=dict(color="#06b6d4", width=2),
        name="River Flow (cumecs)",
        hovertemplate="%{x|%b %d %H:%M}<br>Flow: %{y:.1f} cumecs<extra></extra>",
    ))
    fig_flow.update_layout(
        **PLOT_LAYOUT,
        height=350,
        title=dict(text=f"{river} River — Predicted Flow", font=dict(size=14)),
        yaxis_title="Flow (m³/s)",
    )
    st.plotly_chart(fig_flow, use_container_width=True)

with col_right:
    fig_scatter = go.Figure()
    fig_scatter.add_trace(go.Scatter(
        x=forecast_df["river_flow_cumecs"],
        y=forecast_df["predicted_generation_mw"],
        mode="markers",
        marker=dict(
            color=forecast_df["rainfall_mm"],
            colorscale="Blues",
            size=8,
            showscale=True,
            colorbar=dict(title="Rain (mm)"),
            line=dict(width=0.5, color="#1e293b"),
        ),
        hovertemplate="Flow: %{x:.1f} cumecs<br>Gen: %{y:.1f} MW<br><extra></extra>",
    ))
    fig_scatter.update_layout(
        **PLOT_LAYOUT,
        height=350,
        title=dict(text="Flow → Generation Relationship", font=dict(size=14)),
        xaxis_title="River Flow (cumecs)",
        yaxis_title="Generation (MW)",
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

# ── Model Performance (if trained) ───────────────────────────────────────────

artifacts = load_model_artifacts()
if artifacts:
    st.markdown("### 🤖 Model Performance")

    metrics = artifacts["metrics"]
    importance = artifacts["importance"]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("R² Score", f"{metrics['r2']:.4f}")
    m2.metric("MAE", f"{metrics['mae']:.3f} MW")
    m3.metric("RMSE", f"{metrics['rmse']:.3f} MW")
    m4.metric("CV MAE (±std)", f"{metrics['cv_mae_mean']:.3f} ± {metrics['cv_mae_std']:.3f}")

    fig_imp = go.Figure()
    sorted_imp = sorted(importance.items(), key=lambda x: x[1])
    fig_imp.add_trace(go.Bar(
        x=[v for _, v in sorted_imp],
        y=[k for k, _ in sorted_imp],
        orientation="h",
        marker=dict(
            color=[v for _, v in sorted_imp],
            colorscale="Viridis",
        ),
    ))
    fig_imp.update_layout(
        **PLOT_LAYOUT,
        height=400,
        title=dict(text="Top Feature Importances (XGBoost)", font=dict(size=14)),
        xaxis_title="Importance",
        yaxis_title="",
    )
    st.plotly_chart(fig_imp, use_container_width=True)

# ── Multi-District Overview ──────────────────────────────────────────────────

st.markdown("### 🗺️ Nepal Grid — All Districts Overview")

district_data = []
for d, props in DISTRICTS.items():
    df_d = generate_hourly_forecast_data(base_date="2026-03-15", district=d, seed=99)
    r = props["river"]
    p = HYDRO_PLANTS[r]
    avg = df_d["predicted_generation_mw"].iloc[:24].mean()
    mn = df_d["predicted_generation_mw"].iloc[:24].min()
    level, color, _ = get_alert_level(mn, p["capacity_mw"])
    district_data.append({
        "District": d,
        "River": r,
        "Plant": p["name"],
        "Capacity (MW)": p["capacity_mw"],
        "24h Avg (MW)": round(avg, 1),
        "24h Min (MW)": round(mn, 1),
        "Utilization": f"{avg / p['capacity_mw'] * 100:.0f}%",
        "Status": f"{'🟢' if color == 'green' else '🟡' if color == 'yellow' else '🔴'} {level}",
    })

overview_df = pd.DataFrame(district_data)

total_capacity = overview_df["Capacity (MW)"].sum()
total_avg = overview_df["24h Avg (MW)"].sum()
total_min = overview_df["24h Min (MW)"].sum()

o1, o2, o3 = st.columns(3)
o1.metric("Total Grid Capacity", f"{total_capacity:.0f} MW")
o2.metric("Total 24h Avg Generation", f"{total_avg:.1f} MW", f"{total_avg/total_capacity*100:.0f}%")
o3.metric("Total 24h Min Generation", f"{total_min:.1f} MW")

st.dataframe(
    overview_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Utilization": st.column_config.TextColumn(width="small"),
        "Status": st.column_config.TextColumn(width="medium"),
    },
)

# ── Footer ───────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#64748b; font-size:0.8rem; padding: 1rem 0;'>"
    "HydroPredict AI — AI Hackathon 2026, Embark College, Pulchowk<br/>"
    "Powered by XGBoost • Weather + Hydrology Data from Nepal Multi-District Dataset"
    "</div>",
    unsafe_allow_html=True,
)
