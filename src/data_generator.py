"""
Generates realistic synthetic weather + hydrology data modeled on Nepal's climate.

Patterns encoded:
- Monsoon season (June-September): heavy rainfall, high river flow
- Winter (December-February): low rainfall, low flow, cold temperatures
- Pre-monsoon (March-May): rising temperatures, sporadic rain
- Post-monsoon (October-November): declining rain, moderate flow

Districts modeled: Kaski (Pokhara), Kathmandu, Chitwan, Sunsari, Kalikot
Each has distinct elevation-based climate profiles.
"""

import numpy as np
import pandas as pd
from pathlib import Path

DISTRICTS = {
    "Kaski": {"elevation": 827, "base_temp": 20, "rain_mult": 1.6, "river": "Seti"},
    "Kathmandu": {"elevation": 1400, "base_temp": 18, "rain_mult": 1.0, "river": "Bagmati"},
    "Chitwan": {"elevation": 150, "base_temp": 25, "rain_mult": 1.1, "river": "Narayani"},
    "Sunsari": {"elevation": 100, "base_temp": 26, "rain_mult": 1.2, "river": "Koshi"},
    "Kalikot": {"elevation": 1220, "base_temp": 16, "rain_mult": 0.8, "river": "Karnali"},
}

# Hydropower plants mapped to river systems (capacity in MW)
HYDRO_PLANTS = {
    "Seti": {"name": "Seti Hydropower", "capacity_mw": 22.5},
    "Bagmati": {"name": "Bagmati Small Hydro", "capacity_mw": 10.0},
    "Narayani": {"name": "Narayani Hydropower", "capacity_mw": 45.0},
    "Koshi": {"name": "Koshi Hydropower", "capacity_mw": 30.0},
    "Karnali": {"name": "Karnali Hydropower", "capacity_mw": 38.0},
}


def _seasonal_rainfall(day_of_year: np.ndarray, rain_mult: float) -> np.ndarray:
    """Model Nepal's monsoon-driven rainfall pattern."""
    monsoon_peak = 200  # ~mid-July
    monsoon_width = 50
    monsoon_component = 28 * rain_mult * np.exp(-0.5 * ((day_of_year - monsoon_peak) / monsoon_width) ** 2)

    pre_monsoon_peak = 130
    pre_monsoon = 6 * rain_mult * np.exp(-0.5 * ((day_of_year - pre_monsoon_peak) / 30) ** 2)

    base_rain = 1.0 * rain_mult
    return base_rain + monsoon_component + pre_monsoon


def _seasonal_temperature(day_of_year: np.ndarray, base_temp: float) -> np.ndarray:
    """Sinusoidal annual temperature cycle peaking in June."""
    peak_day = 170  # ~mid-June
    amplitude = 8
    return base_temp + amplitude * np.sin(2 * np.pi * (day_of_year - peak_day + 91) / 365)


def _river_flow(rainfall_mm: np.ndarray, temperature: np.ndarray, elevation: float) -> np.ndarray:
    """
    Simplified river flow model:
    flow = f(cumulative recent rainfall, snowmelt proxy, base flow)
    """
    kernel_size = 7
    kernel = np.exp(-np.arange(kernel_size) / 3)
    kernel /= kernel.sum()
    cumulative_rain = np.convolve(rainfall_mm, kernel, mode="same")

    snowmelt = np.where(temperature > 5, 0.3 * (temperature - 5) * (elevation / 3000), 0)

    base_flow = 15 + elevation * 0.005
    flow = base_flow + 2.5 * cumulative_rain + snowmelt
    return np.clip(flow, 5, 500)


def _hydro_generation(river_flow: np.ndarray, capacity_mw: float) -> np.ndarray:
    """
    Hydropower output as a function of river flow.
    Uses a saturating curve: output plateaus near plant capacity.
    """
    flow_median = np.median(river_flow)
    normalized = river_flow / (flow_median + 1e-6)
    efficiency = 1 - np.exp(-1.2 * normalized)
    generation = capacity_mw * efficiency
    return np.clip(generation, 0, capacity_mw)


def generate_weather_data(
    start_date: str = "2020-01-01",
    end_date: str = "2025-12-31",
    seed: int = 42,
) -> pd.DataFrame:
    """Generate multi-district daily weather data."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start_date, end_date, freq="D")
    day_of_year = dates.dayofyear.values.astype(float)

    rows = []
    for district, props in DISTRICTS.items():
        n = len(dates)
        rain_base = _seasonal_rainfall(day_of_year, props["rain_mult"])
        rainfall = np.maximum(0, rain_base + rng.normal(0, rain_base * 0.4, n))
        dry_mask = rng.random(n) < np.where(rain_base < 3, 0.6, 0.1)
        rainfall[dry_mask] = 0
        rainfall = np.round(rainfall, 1)

        temp_base = _seasonal_temperature(day_of_year, props["base_temp"])
        temperature = np.round(temp_base + rng.normal(0, 2, n), 1)

        humidity = np.clip(55 + 0.8 * rainfall + rng.normal(0, 5, n), 20, 100).round(1)
        wind_speed = np.clip(rng.gamma(2, 2, n) + 1, 0.5, 25).round(1)
        pressure = np.round(1013 - props["elevation"] * 0.12 + rng.normal(0, 3, n), 1)

        df_district = pd.DataFrame({
            "date": dates,
            "district": district,
            "temperature_c": temperature,
            "rainfall_mm": rainfall,
            "humidity_pct": humidity,
            "wind_speed_kmh": wind_speed,
            "pressure_hpa": pressure,
        })
        rows.append(df_district)

    return pd.concat(rows, ignore_index=True)


def generate_river_data(weather_df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Generate river flow data derived from weather patterns."""
    rng = np.random.default_rng(seed)
    rows = []
    for district, props in DISTRICTS.items():
        subset = weather_df[weather_df["district"] == district].copy()
        flow = _river_flow(
            subset["rainfall_mm"].values,
            subset["temperature_c"].values,
            props["elevation"],
        )
        flow += rng.normal(0, flow * 0.08)
        flow = np.clip(flow, 5, 500).round(2)

        river_name = props["river"]
        plant = HYDRO_PLANTS[river_name]
        generation = _hydro_generation(flow, plant["capacity_mw"])
        generation += rng.normal(0, generation * 0.03)
        generation = np.clip(generation, 0, plant["capacity_mw"]).round(2)

        df_river = pd.DataFrame({
            "date": subset["date"].values,
            "district": district,
            "river": river_name,
            "river_flow_cumecs": flow,
            "hydro_plant": plant["name"],
            "plant_capacity_mw": plant["capacity_mw"],
            "generation_mw": generation,
        })
        rows.append(df_river)

    return pd.concat(rows, ignore_index=True)


def generate_hourly_forecast_data(
    base_date: str = "2026-03-15",
    district: str = "Kaski",
    seed: int = 99,
) -> pd.DataFrame:
    """Generate 72-hour ahead hourly forecast data for dashboard demo."""
    rng = np.random.default_rng(seed)
    hours = pd.date_range(base_date, periods=72, freq="h")
    hour_of_day = hours.hour.values.astype(float)
    day_of_year = hours.dayofyear.values.astype(float)

    props = DISTRICTS[district]
    temp_daily = _seasonal_temperature(day_of_year, props["base_temp"])
    temp_hourly = temp_daily + 4 * np.sin(2 * np.pi * (hour_of_day - 14) / 24)
    temperature = np.round(temp_hourly + rng.normal(0, 1, len(hours)), 1)

    rain_base = _seasonal_rainfall(day_of_year, props["rain_mult"])
    rain_hourly = rain_base / 24
    rainfall = np.maximum(0, rain_hourly + rng.exponential(rain_hourly * 0.5, len(hours)))
    dry_mask = rng.random(len(hours)) < 0.5
    rainfall[dry_mask] = 0
    rainfall = np.round(rainfall, 2)

    humidity = np.clip(55 + 2.5 * rainfall + rng.normal(0, 3, len(hours)), 25, 100).round(1)

    flow = _river_flow(rainfall * 24, temperature, props["elevation"])
    flow += rng.normal(0, flow * 0.05)
    flow = np.clip(flow, 5, 500).round(2)

    plant = HYDRO_PLANTS[props["river"]]
    generation = _hydro_generation(flow, plant["capacity_mw"])
    generation += rng.normal(0, generation * 0.02)
    generation = np.clip(generation, 0, plant["capacity_mw"]).round(2)

    return pd.DataFrame({
        "datetime": hours,
        "temperature_c": temperature,
        "rainfall_mm": rainfall,
        "humidity_pct": humidity,
        "river_flow_cumecs": flow,
        "predicted_generation_mw": generation,
        "plant_capacity_mw": plant["capacity_mw"],
    })


def save_all_data(output_dir: str = "data"):
    """Generate and persist all datasets."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Generating weather data (2020-2025, 5 districts)...")
    weather = generate_weather_data()
    weather.to_csv(out / "nepal_weather_2020_2025.csv", index=False)
    print(f"  → {len(weather):,} rows saved to nepal_weather_2020_2025.csv")

    print("Generating river flow & hydro generation data...")
    river = generate_river_data(weather)
    river.to_csv(out / "nepal_hydro_generation.csv", index=False)
    print(f"  → {len(river):,} rows saved to nepal_hydro_generation.csv")

    print("Generating 72-hour forecast demo data...")
    forecast = generate_hourly_forecast_data()
    forecast.to_csv(out / "forecast_demo_72h.csv", index=False)
    print(f"  → {len(forecast)} rows saved to forecast_demo_72h.csv")

    print("\nAll data generated successfully!")
    return weather, river, forecast


if __name__ == "__main__":
    save_all_data()
