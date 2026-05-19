"""ML models for stock price prediction."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


@dataclass
class ModelMetrics:
    rmse: float
    mae: float
    r2: float

    def to_dict(self) -> dict:
        return {"rmse": round(self.rmse, 4), "mae": round(self.mae, 4), "r2": round(self.r2, 4)}


def get_model(name: str):
    name = name.lower()
    if name in ("linear", "linear_regression", "lr"):
        return LinearRegression()
    if name in ("random_forest", "rf"):
        return RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
    raise ValueError(f"Unknown model: {name}")


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> ModelMetrics:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return ModelMetrics(rmse, mae, r2)
