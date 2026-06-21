"""Machine-learning match-result models with calibration.

These operate on an engineered **feature matrix** (see
:class:`soccer_betting.data.features.FeatureBuilder`) rather than on team names,
so their interface is ``fit(X, y)`` / ``predict_proba(X)``.

Provided estimators:

* :class:`MLResultModel` — a gradient-boosted classifier (XGBoost if available,
  otherwise scikit-learn's ``HistGradientBoostingClassifier``) with optional
  probability calibration on a held-out split.
* :class:`TorchMLP` — an optional feed-forward neural network (requires
  ``torch``). Imported lazily so the package works without it.
* :class:`EnsembleModel` — soft-voting ensemble averaging member probabilities,
  the consistent winner highlighted in the research.

The research hierarchy for tabular data: gradient-boosted trees lead, ensembles
are the most robust, and calibration matters more than raw accuracy.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from soccer_betting.calibration.calibrators import MultiClassCalibrator
from soccer_betting.models.base import OUTCOMES
from soccer_betting.utils.logging import get_logger

logger = get_logger(__name__)

_LABEL_TO_IDX = {"H": 0, "D": 1, "A": 2}


def _labels_to_idx(y) -> np.ndarray:
    y = np.asarray(y)
    if y.dtype.kind in {"U", "S", "O"}:
        return np.array([_LABEL_TO_IDX[str(v)] for v in y])
    return y.astype(int)


def _build_gbm(random_state: int, params: Optional[dict]):
    """Return a gradient-boosting classifier, preferring XGBoost."""
    params = params or {}
    try:
        from xgboost import XGBClassifier

        return XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            random_state=random_state,
            tree_method="hist",
            n_estimators=params.get("n_estimators", 400),
            max_depth=params.get("max_depth", 4),
            learning_rate=params.get("learning_rate", 0.05),
            subsample=params.get("subsample", 0.8),
            colsample_bytree=params.get("colsample_bytree", 0.8),
        ), "xgboost"
    except Exception:  # pragma: no cover - fallback path
        from sklearn.ensemble import HistGradientBoostingClassifier

        logger.warning("XGBoost unavailable; using HistGradientBoostingClassifier.")
        return HistGradientBoostingClassifier(
            random_state=random_state,
            max_depth=params.get("max_depth", 4),
            learning_rate=params.get("learning_rate", 0.05),
            max_iter=params.get("n_estimators", 400),
        ), "sklearn_hgb"


class MLResultModel:
    """Calibrated gradient-boosted 1X2 classifier.

    Parameters
    ----------
    calibration:
        ``"isotonic"``, ``"sigmoid"`` or ``"none"``.
    calib_fraction:
        Fraction of the (chronologically last) training data held out to fit the
        calibrator. Using the tail respects temporal ordering.
    random_state, params:
        Passed to the underlying estimator.
    """

    def __init__(
        self,
        calibration: str = "isotonic",
        calib_fraction: float = 0.2,
        random_state: int = 42,
        params: Optional[dict] = None,
    ):
        self.calibration = calibration
        self.calib_fraction = calib_fraction
        self.random_state = random_state
        self.params = params
        self.model_ = None
        self.backend_ = None
        self.calibrator_: Optional[MultiClassCalibrator] = None
        self.feature_names_: List[str] = []
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y) -> "MLResultModel":
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
            X_arr = X.to_numpy(dtype=float)
        else:
            X_arr = np.asarray(X, dtype=float)
        y_idx = _labels_to_idx(y)
        if len(X_arr) != len(y_idx):
            raise ValueError("X and y length mismatch.")

        self.model_, self.backend_ = _build_gbm(self.random_state, self.params)

        if self.calibration != "none" and len(X_arr) >= 50:
            split = int(len(X_arr) * (1 - self.calib_fraction))
            X_tr, X_cal = X_arr[:split], X_arr[split:]
            y_tr, y_cal = y_idx[:split], y_idx[split:]
            self.model_.fit(X_tr, y_tr)
            raw_cal = self._raw_proba(X_cal)
            self.calibrator_ = MultiClassCalibrator(self.calibration).fit(raw_cal, y_cal)
            # Refit on full data for the final point estimate of the trees.
            self.model_.fit(X_arr, y_idx)
        else:
            self.model_.fit(X_arr, y_idx)
            self.calibrator_ = None

        self.is_fitted = True
        logger.info("Fitted MLResultModel (%s, calib=%s) on %d rows, %d features",
                    self.backend_, self.calibration, len(X_arr), X_arr.shape[1])
        return self

    def _raw_proba(self, X_arr: np.ndarray) -> np.ndarray:
        proba = self.model_.predict_proba(X_arr)
        # Ensure column order matches H/D/A (classes_ are 0,1,2 by construction).
        return np.asarray(proba, dtype=float)

    def predict_proba(self, X) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("MLResultModel must be fitted first.")
        X_arr = X.to_numpy(dtype=float) if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=float)
        raw = self._raw_proba(X_arr)
        if self.calibrator_ is not None:
            return self.calibrator_.predict(raw)
        return raw

    def feature_importance(self) -> Optional[pd.Series]:
        """Return feature importances when the backend exposes them."""
        if not self.is_fitted:
            return None
        imp = getattr(self.model_, "feature_importances_", None)
        if imp is None or not self.feature_names_:
            return None
        return pd.Series(imp, index=self.feature_names_).sort_values(ascending=False)


class EnsembleModel:
    """Soft-voting ensemble over feature-based 1X2 models.

    Parameters
    ----------
    members:
        List of fitted/unfitted models exposing ``fit(X, y)`` / ``predict_proba(X)``.
    weights:
        Optional per-member weights (defaults to equal weighting).
    """

    def __init__(self, members: List, weights: Optional[List[float]] = None):
        if not members:
            raise ValueError("EnsembleModel needs at least one member.")
        self.members = members
        self.weights = weights
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y) -> "EnsembleModel":
        for m in self.members:
            m.fit(X, y)
        self.is_fitted = True
        return self

    def predict_proba(self, X) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("EnsembleModel must be fitted first.")
        probs = [m.predict_proba(X) for m in self.members]
        w = np.ones(len(probs)) if self.weights is None else np.asarray(self.weights, dtype=float)
        w = w / w.sum()
        stacked = np.tensordot(w, np.stack(probs), axes=(0, 0))
        return stacked / stacked.sum(axis=1, keepdims=True)
