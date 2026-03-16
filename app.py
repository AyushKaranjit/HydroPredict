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
    page_icon="",
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
        background: #393E46;
        padding: 1rem 1.2rem;
        border-radius: 12px;
        border: 1px solid #393E46;
    }
    .stMetric label {
        color: #EEEEEE !important;
        font-size: 0.85rem !important;
    }
    .stMetric [data-testid="stMetricValue"] {
        color: #EEEEEE !important;
        font-weight: 600 !important;
    }
    div[data-testid="stSidebar"] {
        background: #222831;
    }
    div[data-testid="stSidebar"] .stMarkdown p,
    div[data-testid="stSidebar"] .stMarkdown li,
    div[data-testid="stSidebar"] label {
        color: #EEEEEE;
    }
    .alert-box-green {
        background: #393E46;
        border: 1px solid #00ADB5;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        margin: 0.5rem 0;
    }
    .alert-box-yellow {
        background: #393E46;
        border: 1px solid #00ADB5;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        margin: 0.5rem 0;
    }
    .alert-box-red {
        background: #393E46;
        border: 1px solid #00ADB5;
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
        color: #EEEEEE;
        font-size: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Helpers ──────────────────────────────────────────────────────────────────

DISTRICT_COORDS = {
    "Achham": {"lat": 29.12, "lon": 81.3}, "Arghakhanchi": {"lat": 27.95, "lon": 83.22},
    "Baglung": {"lat": 28.27, "lon": 83.61}, "Baitadi": {"lat": 29.53, "lon": 80.43},
    "Bajhang": {"lat": 29.72, "lon": 81.25}, "Bajura": {"lat": 29.51, "lon": 81.5},
    "Banke": {"lat": 28.05, "lon": 81.62}, "Bara": {"lat": 27.02, "lon": 85.05},
    "Bardiya": {"lat": 28.3, "lon": 81.5}, "Bhaktapur": {"lat": 27.67, "lon": 85.43},
    "Bhojpur": {"lat": 27.17, "lon": 87.03}, "Chitwan": {"lat": 27.53, "lon": 84.35},
    "Dadeldhura": {"lat": 29.3, "lon": 80.58}, "Dailekh": {"lat": 28.85, "lon": 81.7},
    "Dang": {"lat": 28.0, "lon": 82.3}, "Darchula": {"lat": 30.13, "lon": 80.58},
    "Dhading": {"lat": 27.85, "lon": 84.9}, "Dhankuta": {"lat": 26.98, "lon": 87.35},
    "Dhanusha": {"lat": 26.83, "lon": 86.03}, "Dolakha": {"lat": 27.66, "lon": 86.02},
    "Dolpa": {"lat": 29.08, "lon": 83.57}, "Doti": {"lat": 29.27, "lon": 80.93},
    "Rukum East": {"lat": 28.63, "lon": 82.47}, "Gorkha": {"lat": 28.0, "lon": 84.63},
    "Gulmi": {"lat": 28.08, "lon": 83.25}, "Humla": {"lat": 29.96, "lon": 81.83},
    "Ilam": {"lat": 26.91, "lon": 87.92}, "Jajarkot": {"lat": 28.7, "lon": 82.2},
    "Jhapa": {"lat": 26.63, "lon": 88.08}, "Jumla": {"lat": 29.27, "lon": 82.18},
    "Kailali": {"lat": 28.7, "lon": 80.63}, "Kalikot": {"lat": 29.13, "lon": 81.63},
    "Kanchanpur": {"lat": 28.83, "lon": 80.33}, "Kapilvastu": {"lat": 27.55, "lon": 83.05},
    "Kaski": {"lat": 28.21, "lon": 83.99}, "Kathmandu": {"lat": 27.71, "lon": 85.32},
    "Kavrepalanchok": {"lat": 27.63, "lon": 85.55}, "Khotang": {"lat": 27.2, "lon": 86.8},
    "Lalitpur": {"lat": 27.67, "lon": 85.32}, "Lamjung": {"lat": 28.1, "lon": 84.36},
    "Mahottari": {"lat": 26.65, "lon": 85.9}, "Makwanpur": {"lat": 27.43, "lon": 85.03},
    "Manang": {"lat": 28.65, "lon": 84.02}, "Morang": {"lat": 26.67, "lon": 87.45},
    "Mugu": {"lat": 29.52, "lon": 82.1}, "Mustang": {"lat": 28.83, "lon": 83.83},
    "Myagdi": {"lat": 28.38, "lon": 83.57}, "Nawalparasi East": {"lat": 27.7, "lon": 84.13},
    "Nuwakot": {"lat": 27.92, "lon": 85.15}, "Okhaldhunga": {"lat": 27.33, "lon": 86.5},
    "Palpa": {"lat": 27.9, "lon": 83.55}, "Panchthar": {"lat": 27.13, "lon": 87.8},
    "Nawalparasi West": {"lat": 27.55, "lon": 83.7}, "Parbat": {"lat": 28.23, "lon": 83.67},
    "Parsa": {"lat": 27.0, "lon": 84.88}, "Pyuthan": {"lat": 28.08, "lon": 82.87},
    "Ramechhap": {"lat": 27.33, "lon": 86.0}, "Rasuwa": {"lat": 28.1, "lon": 85.27},
    "Rautahat": {"lat": 26.93, "lon": 85.3}, "Rolpa": {"lat": 28.27, "lon": 82.83},
    "Rupandehi": {"lat": 27.52, "lon": 83.45}, "Salyan": {"lat": 28.37, "lon": 82.18},
    "Sankhuwasabha": {"lat": 27.57, "lon": 87.28}, "Saptari": {"lat": 26.6, "lon": 86.75},
    "Sarlahi": {"lat": 26.98, "lon": 85.55}, "Sindhuli": {"lat": 27.25, "lon": 85.97},
    "Sindhupalchok": {"lat": 27.85, "lon": 85.83}, "Siraha": {"lat": 26.65, "lon": 86.2},
    "Solukhumbu": {"lat": 27.67, "lon": 86.62}, "Sunsari": {"lat": 26.62, "lon": 87.3},
    "Surkhet": {"lat": 28.6, "lon": 81.63}, "Syangja": {"lat": 28.08, "lon": 83.87},
    "Tanahun": {"lat": 27.93, "lon": 84.25}, "Taplejung": {"lat": 27.35, "lon": 87.67},
    "Terhathum": {"lat": 27.12, "lon": 87.58}, "Udayapur": {"lat": 26.85, "lon": 86.67},
    "Rukum West": {"lat": 28.63, "lon": 82.45}
}

PLOT_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(34,40,49,0.8)",
    plot_bgcolor="rgba(34,40,49,0.4)",
    font=dict(family="Inter, sans-serif", color="#EEEEEE"),
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

artifacts = load_model_artifacts()



def get_alert_level(
    generation_mw: float, 
    capacity_mw: float, 
    critical_threshold_pct: float = 40.0
) -> tuple[str, str, str]:
    """
    Determine alert status based on generation level and user-defined threshold.
    """
    ratio = generation_mw / capacity_mw if capacity_mw > 0 else 0
    ratio_pct = ratio * 100
    
    # Define warning threshold as slightly above critical (e.g., +15% buffer)
    # If user sets critical at 40%, warning is between 40% and 55%
    # If user sets critical at 80%, warning is between 80% and 90% (capped at 90)
    warning_threshold_pct = min(critical_threshold_pct + 15, 90)
    
    if ratio_pct < critical_threshold_pct:
        return "CRITICAL", "red", f"Severe generation deficit (<{critical_threshold_pct}%) — activate backup power sources immediately."
    elif ratio_pct < warning_threshold_pct:
        return "WARNING", "yellow", f"Generation dropping (<{warning_threshold_pct}%) — consider demand-side management."
    else:
        return "NORMAL", "green", "Grid operating within safe parameters. No action required."


# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    district = st.selectbox(
        "Select District",
        sorted(DISTRICTS.keys()),
        index=0,
        help="Choose a district to view its hydropower forecast",
    )
    river = DISTRICTS[district]["river"]
    plant = HYDRO_PLANTS[river]

    st.markdown(f"**River System:** {river}")
    st.markdown(f"**Plant:** {plant['name']}")
    st.markdown(f"**Capacity:** {plant['capacity_mw']} MW")

    st.markdown("---")
    st.markdown("### What-If Scenario")
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

# ── Generate Forecast Data ───────────────────────────────────────────────────

@st.cache_data(ttl=300)
def get_forecast(district_name: str, rain_mod: int, temp_mod: float):
    # 1. Generate Weather & Hydrology Logic (Physics-based Simulation)
    df = generate_hourly_forecast_data(
        base_date="2026-03-15",
        district=district_name,
        seed=99,
    )
    
    # 2. Apply "What-If" Scenarios
    if rain_mod != 0:
        factor = 1 + rain_mod / 100
        df["rainfall_mm"] = np.clip(df["rainfall_mm"] * factor, 0, None).round(2)
        flow_adjustment = 1 + (rain_mod / 100) * 0.6
        df["river_flow_cumecs"] = np.clip(df["river_flow_cumecs"] * flow_adjustment, 5, 500).round(2)
    if temp_mod != 0:
        df["temperature_c"] = (df["temperature_c"] + temp_mod).round(1)
        snowmelt_effect = max(0, temp_mod * 0.02)
        df["river_flow_cumecs"] = (df["river_flow_cumecs"] * (1 + snowmelt_effect)).round(2)

    # 3. Calculate Physics-Based Generation (Baseline)
    cap = HYDRO_PLANTS[DISTRICTS[district_name]["river"]]["capacity_mw"]
    flow_median = df["river_flow_cumecs"].median()
    normalized = df["river_flow_cumecs"] / (flow_median + 1e-6)
    efficiency = 1 - np.exp(-1.2 * normalized)
    df["predicted_generation_mw"] = np.clip(cap * efficiency, 0, cap).round(2)
    df["plant_capacity_mw"] = cap
    
    # 4. AI MODEL INTEGRATION
    # If the trained model exists, use it to refine the generation forecast
    # This bridges the gap between the "Simulation" and "AI Prediction"
    if artifacts and artifacts["model"]:
        try:
            # Create a daily aggregate input for the model
            current_physics_gen = df["predicted_generation_mw"].mean()
            
            # Lookup coords
            coords = DISTRICT_COORDS.get(district_name, {"lat": 28.0, "lon": 84.0})

            daily_stats = {
                "rainfall_mm": df["rainfall_mm"].sum() / 3.0,  # 3 days avg daily
                "temperature_c": df["temperature_c"].mean(),
                "humidity_pct": df["humidity_pct"].mean(),
                "river_flow_cumecs": df["river_flow_cumecs"].mean(),
                "generation_mw": current_physics_gen,  # Use physics est as proxy for "current/recent" gen
                "day_of_year": df["datetime"].dt.dayofyear.iloc[0],
                "month": df["datetime"].dt.month.iloc[0],
                "Latitude": coords["lat"], 
                "Longitude": coords["lon"], 
            }
            
            # temporal features
            doy = daily_stats["day_of_year"]
            daily_stats["sin_doy"] = np.sin(2 * np.pi * doy / 365)
            daily_stats["cos_doy"] = np.cos(2 * np.pi * doy / 365)
            daily_stats["sin_month"] = np.sin(2 * np.pi * daily_stats["month"] / 12)
            daily_stats["cos_month"] = np.cos(2 * np.pi * daily_stats["month"] / 12)
            daily_stats["is_monsoon"] = 1 if daily_stats["month"] in [6, 7, 8, 9] else 0

            # Prepare feature vector (filling missing lag/rolling cols with current values)
            # This is a simplification for the dashboard demo to avoid needing full history
            input_data = pd.DataFrame([daily_stats])
            for col in artifacts["features"]:
                if col not in input_data.columns:
                    # Heuristic: fill lag/rolling features with the current daily mean
                    # e.g. generation_mw_lag1 -> generation_mw (current physics est)
                    base_col = col.split("_lag")[0].split("_rmean")[0].split("_rstd")[0].split("_cumsum")[0]
                    if base_col in daily_stats:
                        input_data[col] = daily_stats[base_col]
                    else:
                        input_data[col] = 0.0
            
            # Predict Daily Generation (AI)
            predicted_daily_gen = artifacts["model"].predict(input_data[artifacts["features"]])[0]
            predicted_daily_gen = np.clip(predicted_daily_gen, 0, cap)
            
            # Scale the hourly profile to match the AI's daily average prediction
            current_daily_avg = df["predicted_generation_mw"].mean()
            if current_daily_avg > 0:
                scale_factor = predicted_daily_gen / current_daily_avg
                # Blend: 80% AI Prediction + 20% Physics (for stability)
                df["predicted_generation_mw"] = (df["predicted_generation_mw"] * scale_factor * 0.8) + (df["predicted_generation_mw"] * 0.2)
                df["predicted_generation_mw"] = np.clip(df["predicted_generation_mw"], 0, cap).round(2)
                
        except Exception as e:
            print(f"Model prediction failed, falling back to physics: {e}")
            
    return df


forecast_df = get_forecast(district, rain_modifier, temp_modifier)

# ── Hero Header ──────────────────────────────────────────────────────────────

st.markdown(
    '<div class="hero-header">'
    '<h1>Voltide</h1>'
    '<p>Predicting hydropower generation to keep Nepal\'s grid stable</p>'
    '</div>',
    unsafe_allow_html=True,
)

# ── Current Status Metrics ───────────────────────────────────────────────────

current_gen = forecast_df["predicted_generation_mw"].iloc[0]
avg_gen_24h = forecast_df["predicted_generation_mw"].iloc[:24].mean()
min_gen_24h = forecast_df["predicted_generation_mw"].iloc[:24].min()
max_gen_24h = forecast_df["predicted_generation_mw"].iloc[:24].max()
capacity = forecast_df["plant_capacity_mw"].iloc[0]

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Current Output", f"{current_gen:.1f} MW")
col2.metric("24h Avg Forecast", f"{avg_gen_24h:.1f} MW")
col3.metric("24h Min", f"{min_gen_24h:.1f} MW")
col4.metric("24h Max", f"{max_gen_24h:.1f} MW")
col5.metric("Plant Capacity", f"{capacity:.1f} MW")

# Calculate Alert Level
alert_status, alert_color, alert_msg = get_alert_level(min_gen_24h, capacity, critical_threshold_pct=alert_threshold)

# Display Alert
st.markdown(f"""
<div class="alert-box-{alert_color}">
    <h3 style="margin: 0; padding: 0;">{alert_status} ALERT</h3>
    <p style="margin: 5px 0 0 0;">{alert_msg}</p>
</div>
""", unsafe_allow_html=True)

threshold_mw = capacity * alert_threshold / 100

# ── Main Forecast Chart ─────────────────────────────────────────────────────

st.markdown("### 72-Hour Generation Forecast")

fig_main = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.08,
    row_heights=[0.65, 0.35],
    subplot_titles=("Predicted Hydropower Generation", "Weather Conditions"),
)

gen = forecast_df["predicted_generation_mw"]

# Dynamic color logic matching Alert Thresholds
def get_bar_color(val, cap, thresh_pct):
    crit = cap * (thresh_pct / 100.0)
    warn = cap * (min(thresh_pct + 15, 90) / 100.0)
    if val < crit:
        return "#FF2E63"  # Red
    elif val < warn:
        return "#FFE98A"  # Yellow
    else:
        return "#00ADB5"  # Teal

colors = [get_bar_color(g, capacity, alert_threshold) for g in gen]

fig_main.add_trace(
    go.Bar(
        x=forecast_df["datetime"], y=gen,
        marker_color=colors, name="Generation (MW)",
        hovertemplate="<b>%{x|%b %d, %H:%M}</b><br>Generation: %{y:.1f} MW<extra></extra>",
    ),
    row=1, col=1,
)

fig_main.add_hline(
    y=capacity, line_dash="dash", line_color="#00ADB5", line_width=2,
    annotation_text=f"Capacity: {capacity} MW",
    annotation_position="top left",
    annotation_font_color="#00ADB5",
    row=1, col=1,
)

fig_main.add_hline(
    y=threshold_mw, line_dash="dot", line_color="#EEEEEE", line_width=1.5,
    annotation_text=f"Alert: {threshold_mw:.0f} MW",
    annotation_position="bottom left",
    annotation_font_color="#EEEEEE",
    row=1, col=1,
)

fig_main.add_trace(
    go.Scatter(
        x=forecast_df["datetime"], y=forecast_df["rainfall_mm"],
        fill="tozeroy", fillcolor="rgba(0,173,181,0.3)",
        line=dict(color="#00ADB5", width=1.5),
        name="Rainfall (mm)",
        hovertemplate="Rainfall: %{y:.1f} mm<extra></extra>",
    ),
    row=2, col=1,
)

fig_main.add_trace(
    go.Scatter(
        x=forecast_df["datetime"], y=forecast_df["temperature_c"],
        line=dict(color="#EEEEEE", width=2),
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

st.markdown("### River Flow vs Energy Generation")

col_left, col_right = st.columns(2)

with col_left:
    fig_flow = go.Figure()
    fig_flow.add_trace(go.Scatter(
        x=forecast_df["datetime"], y=forecast_df["river_flow_cumecs"],
        fill="tozeroy", fillcolor="rgba(0,173,181,0.2)",
        line=dict(color="#00ADB5", width=2),
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
            line=dict(width=0.5, color="#393E46"),
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
# Section disabled per request to hide Model Performance from UI.
#
# artifacts = load_model_artifacts()
# if artifacts:
#     st.markdown("### Model Performance")
#
#     metrics = artifacts["metrics"]
#     importance = artifacts["importance"]
#
#     m1, m2, m3, m4 = st.columns(4)
#     m1.metric("R² Score", f"{metrics['r2']:.4f}")
#     m2.metric("MAE", f"{metrics['mae']:.3f} MW")
#     m3.metric("RMSE", f"{metrics['rmse']:.3f} MW")
#     m4.metric("CV MAE (±std)", f"{metrics['cv_mae_mean']:.3f} ± {metrics['cv_mae_std']:.3f}")
#
#     fig_imp = go.Figure()
#     sorted_imp = sorted(importance.items(), key=lambda x: x[1])
#     fig_imp.add_trace(go.Bar(
#         x=[v for _, v in sorted_imp],
#         y=[k for k, _ in sorted_imp],
#         orientation="h",
#         marker=dict(
#             color=[v for _, v in sorted_imp],
#             colorscale="Viridis",
#         ),
#     ))
#     fig_imp.update_layout(
#         **PLOT_LAYOUT,
#         height=400,
#         title=dict(text="Top Feature Importances (XGBoost)", font=dict(size=14)),
#         xaxis_title="Importance",
#         yaxis_title="",
#     )
#     st.plotly_chart(fig_imp, use_container_width=True)

# ── Multi-District Overview ──────────────────────────────────────────────────

st.markdown("### Nepal Grid — All Districts Overview")

district_data = []
for d in sorted(DISTRICTS.keys()):
    if d == district:
        # Optimization: Reuse the forecast already calculated for the main view
        df_d = forecast_df
    else:
        # Apply the same scenario settings (rain/temp) to all districts for consistency
        df_d = get_forecast(d, rain_modifier, temp_modifier)
        
    props = DISTRICTS[d]
    r = props["river"]
    p = HYDRO_PLANTS[r]
    
    avg = df_d["predicted_generation_mw"].iloc[:24].mean()
    mn = df_d["predicted_generation_mw"].iloc[:24].min()
    level, color, _ = get_alert_level(mn, p["capacity_mw"], critical_threshold_pct=alert_threshold)
    
    district_data.append({
        "District": d,
        "River": r,
        "Plant": p["name"],
        "Capacity (MW)": p["capacity_mw"],
        "24h Avg (MW)": round(avg, 1),
        "24h Min (MW)": round(mn, 1),
        "Utilization %": f"{avg / p['capacity_mw'] * 100:.0f}%",
        "Min Gen %": f"{mn / p['capacity_mw'] * 100:.0f}%",
        "Status": level,
    })

overview_df = pd.DataFrame(district_data).sort_values("District", ascending=True).reset_index(drop=True)

total_capacity = overview_df["Capacity (MW)"].sum()
total_avg = overview_df["24h Avg (MW)"].sum()
total_min = overview_df["24h Min (MW)"].sum()

o1, o2, o3 = st.columns(3)
o1.metric("Total Grid Capacity", f"{total_capacity:.0f} MW")
o2.metric("Total 24h Avg Generation", f"{total_avg:.1f} MW")
o3.metric("Total 24h Min Generation", f"{total_min:.1f} MW")

st.dataframe(
    overview_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Utilization %": st.column_config.ProgressColumn(
            "Avg Utilization",
            format="%f%%",
            min_value=0,
            max_value=100,
        ),
        "Min Gen %": st.column_config.ProgressColumn(
            "Min Generation",
            format="%f%%",
            min_value=0,
            max_value=100,
            help="The lowest predicted generation percentage in the next 24h (Used for Alerts)",
        ),
        "Status": st.column_config.TextColumn(width="medium"),
    },
)

