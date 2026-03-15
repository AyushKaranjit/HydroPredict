"""
Model training, evaluation, and prediction for hydropower generation forecasting.

Uses XGBoost with time-series aware cross-validation (no future data leakage).
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

from src.data_processing import prepare_training_data


def train_model(
    weather_path: str = "data/nepal_weather_2020_2025.csv",
    hydro_path: str = "data/nepal_hydro_generation.csv",
    model_dir: str = "models",
    target_col: str = "generation_mw",
) -> dict:
    """Train XGBoost model and save artifacts."""
    print("Preparing training data...")
    df, feature_cols = prepare_training_data(weather_path, hydro_path, target_col)
    print(f"  Features: {len(feature_cols)}, Samples: {len(df):,}")

    X = df[feature_cols].values
    y = df[target_col].values
    districts = df["district"].values
    capacities = df["plant_capacity_mw"].values

    # Time-series split (no shuffling — prevents future leakage)
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        preds = model.predict(X_val)
        mae = mean_absolute_error(y_val, preds)
        cv_scores.append(mae)
        print(f"  Fold {fold + 1}: MAE = {mae:.3f} MW")

    print(f"  Mean CV MAE: {np.mean(cv_scores):.3f} MW")

    # Final model on all data
    print("Training final model on full dataset...")
    final_model = xgb.XGBRegressor(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
    )
    final_model.fit(X, y, verbose=False)

    y_pred = final_model.predict(X)
    metrics = {
        "mae": round(mean_absolute_error(y, y_pred), 4),
        "rmse": round(np.sqrt(mean_squared_error(y, y_pred)), 4),
        "r2": round(r2_score(y, y_pred), 4),
        "cv_mae_mean": round(np.mean(cv_scores), 4),
        "cv_mae_std": round(np.std(cv_scores), 4),
        "n_features": len(feature_cols),
        "n_samples": len(df),
    }

    # Feature importance
    importance = dict(zip(feature_cols, final_model.feature_importances_.tolist()))
    top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:15]

    out = Path(model_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, out / "xgb_hydro_model.joblib")
    joblib.dump(feature_cols, out / "feature_columns.joblib")

    with open(out / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(out / "feature_importance.json", "w") as f:
        json.dump(dict(top_features), f, indent=2)

    print(f"\nModel saved to {out}/")
    print(f"  R²: {metrics['r2']:.4f}")
    print(f"  MAE: {metrics['mae']:.4f} MW")
    print(f"  RMSE: {metrics['rmse']:.4f} MW")
    print(f"\nTop 5 features:")
    for name, imp in top_features[:5]:
        print(f"  {name}: {imp:.4f}")

    return metrics


if __name__ == "__main__":
    train_model()
