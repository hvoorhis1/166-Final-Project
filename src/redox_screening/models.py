from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.base import RegressorMixin
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class ModelSpec:
    name: str
    estimator: RegressorMixin


def build_models(random_state: int, n_jobs: int) -> list[ModelSpec]:
    return [
        ModelSpec(
            name="ridge",
            estimator=Pipeline(
                [
                    ("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                    ("model", Ridge(alpha=1.0, random_state=random_state)),
                ]
            ),
        ),
        ModelSpec(
            name="random_forest",
            estimator=Pipeline(
                [
                    ("impute", SimpleImputer(strategy="median")),
                    (
                        "model",
                        RandomForestRegressor(
                            n_estimators=500,
                            random_state=random_state,
                            n_jobs=n_jobs,
                            min_samples_leaf=1,
                        ),
                    ),
                ]
            ),
        ),
        ModelSpec(
            name="gradient_boosting",
            estimator=Pipeline(
                [
                    ("impute", SimpleImputer(strategy="median")),
                    (
                        "model",
                        GradientBoostingRegressor(
                            random_state=random_state,
                            n_estimators=300,
                            learning_rate=0.05,
                            max_depth=3,
                        ),
                    ),
                ]
            ),
        ),
    ]


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "r2": r2}


def crossval_summary(estimator, x: np.ndarray, y: np.ndarray, random_state: int) -> dict[str, float]:
    cv = KFold(n_splits=5, shuffle=True, random_state=random_state)
    scores = cross_validate(
        estimator,
        x,
        y,
        cv=cv,
        scoring=("neg_root_mean_squared_error", "neg_mean_absolute_error", "r2"),
        n_jobs=1,
    )
    rmse = -scores["test_neg_root_mean_squared_error"]
    mae = -scores["test_neg_mean_absolute_error"]
    r2 = scores["test_r2"]
    return {
        "cv_rmse_mean": float(rmse.mean()),
        "cv_rmse_std": float(rmse.std()),
        "cv_mae_mean": float(mae.mean()),
        "cv_mae_std": float(mae.std()),
        "cv_r2_mean": float(r2.mean()),
        "cv_r2_std": float(r2.std()),
    }
