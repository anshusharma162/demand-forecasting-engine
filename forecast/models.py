"""Model wrappers and the weighted ensemble.

All models are trained on log1p(visitors), so RMSE on that scale equals
RMSLE on the original scale - a metric that treats a 10% miss at a small
outlet the same as a 10% miss at a large one.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor


def _to_codes(X: pd.DataFrame) -> pd.DataFrame:
    """Random Forest needs numeric input: categories -> integer codes, NaN -> -1."""
    X = X.copy()
    for c in X.select_dtypes("category").columns:
        X[c] = X[c].cat.codes
    return X.fillna(-1)


class LGBMModel:
    name = "lightgbm"

    def __init__(self, seed: int = 0):
        self.model = lgb.LGBMRegressor(
            n_estimators=600,
            learning_rate=0.03,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=seed,
            verbose=-1,
        )

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def importance(self, cols):
        return pd.Series(self.model.feature_importances_, index=cols)


class XGBModel:
    name = "xgboost"

    def __init__(self, seed: int = 0):
        self.model = xgb.XGBRegressor(
            n_estimators=600,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            tree_method="hist",
            enable_categorical=True,
            random_state=seed,
        )

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def importance(self, cols):
        return pd.Series(self.model.feature_importances_, index=cols)


class RFModel:
    name = "random_forest"

    def __init__(self, seed: int = 0):
        self.model = RandomForestRegressor(
            n_estimators=300, min_samples_leaf=3, max_features=0.5, n_jobs=-1, random_state=seed
        )

    def fit(self, X, y):
        self.model.fit(_to_codes(X), y)
        return self

    def predict(self, X):
        return self.model.predict(_to_codes(X))

    def importance(self, cols):
        return pd.Series(self.model.feature_importances_, index=cols)


ALL_MODELS = [LGBMModel, XGBModel, RFModel]


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def ensemble_weights(scores: dict[str, float]) -> dict[str, float]:
    """Weights proportional to 1 / RMSE^2 from out-of-fold validation."""
    inv = {k: 1.0 / (v**2) for k, v in scores.items()}
    total = sum(inv.values())
    return {k: v / total for k, v in inv.items()}


def blend(preds: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    return sum(weights[k] * preds[k] for k in weights)
