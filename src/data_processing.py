"""
Feature engineering and data processing for the hydro prediction model.

Merges weather and hydrology data, creates lag features, rolling statistics,
and seasonal encodings for time-series forecasting.
"""

import numpy as np
import pandas as pd


def load_and_merge(weather_path: str, hydro_path: str) -> pd.DataFrame:
    """Load weather and hydro CSVs and merge on date + district."""
    weather = pd.read_csv(weather_path, parse_dates=["date"])
    hydro = pd.read_csv(hydro_path, parse_dates=["date"])
    merged = weather.merge(hydro, on=["date", "district"], how="inner")
    return merged.sort_values(["district", "date"]).reset_index(drop=True)


def add_temporal_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Add calendar-based features."""
    df = df.copy()
    dt = df[date_col]
    df["day_of_year"] = dt.dt.dayofyear
    df["month"] = dt.dt.month
    df["week_of_year"] = dt.dt.isocalendar().week.astype(int)
    df["sin_doy"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["cos_doy"] = np.cos(2 * np.pi * df["day_of_year"] / 365)
    df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12)
    df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12)
    df["is_monsoon"] = df["month"].isin([6, 7, 8, 9]).astype(int)
    return df


def add_lag_features(
    df: pd.DataFrame,
    group_col: str = "district",
    target_col: str = "generation_mw",
    lags: list[int] | None = None,
) -> pd.DataFrame:
    """Add lagged values of the target and key weather variables."""
    df = df.copy()
    if lags is None:
        lags = [1, 2, 3, 7, 14]

    lag_cols = [target_col, "rainfall_mm", "river_flow_cumecs", "temperature_c"]
    for col in lag_cols:
        if col not in df.columns:
            continue
        for lag in lags:
            df[f"{col}_lag{lag}"] = df.groupby(group_col)[col].shift(lag)
    return df


def add_rolling_features(
    df: pd.DataFrame,
    group_col: str = "district",
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """Add rolling mean and std for key variables."""
    df = df.copy()
    if windows is None:
        windows = [3, 7, 14, 30]

    roll_cols = ["rainfall_mm", "river_flow_cumecs", "temperature_c", "generation_mw"]
    for col in roll_cols:
        if col not in df.columns:
            continue
        for w in windows:
            grouped = df.groupby(group_col)[col]
            df[f"{col}_rmean{w}"] = grouped.transform(lambda x: x.rolling(w, min_periods=1).mean())
            df[f"{col}_rstd{w}"] = grouped.transform(lambda x: x.rolling(w, min_periods=1).std().fillna(0))

    return df


def add_cumulative_rainfall(df: pd.DataFrame, group_col: str = "district") -> pd.DataFrame:
    """Cumulative rainfall over recent windows — key predictor for flow."""
    df = df.copy()
    for w in [3, 7, 14]:
        df[f"rain_cumsum_{w}d"] = df.groupby(group_col)["rainfall_mm"].transform(
            lambda x: x.rolling(w, min_periods=1).sum()
        )
    return df


def prepare_training_data(
    weather_path: str,
    hydro_path: str,
    target_col: str = "generation_mw",
) -> tuple[pd.DataFrame, list[str]]:
    """
    Full pipeline: load → merge → feature engineer → return ready-to-train DataFrame.
    Returns (df, feature_columns).
    """
    df = load_and_merge(weather_path, hydro_path)
    df = add_temporal_features(df)
    df = add_lag_features(df, target_col=target_col)
    df = add_rolling_features(df)
    df = add_cumulative_rainfall(df)

    df = df.dropna().reset_index(drop=True)

    exclude = {
        "date", "district", "river", "hydro_plant",
        target_col, "plant_capacity_mw",
    }
    feature_cols = [c for c in df.columns if c not in exclude and df[c].dtype in [np.float64, np.int64, np.int32, float, int]]

    return df, feature_cols
