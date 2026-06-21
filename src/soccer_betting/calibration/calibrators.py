"""Probability calibration.

The research elevates calibration above raw accuracy: a miscalibrated,
overconfident model inflates the perceived edge and leads to over-staking and
risk of ruin. This module provides:

* :class:`MultiClassCalibrator` — wraps per-class Platt (sigmoid) or isotonic
  calibration in a one-vs-rest fashion and renormalises so the three 1X2
  probabilities sum to 1.
* :func:`expected_calibration_error` — ECE diagnostic.
* :func:`reliability_curve` — data for reliability diagrams.

Calibrators are fit on a held-out calibration set (never the training set) and
the method with the lowest Brier/log-loss should be selected empirically.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from soccer_betting.models.base import OUTCOMES


class _SigmoidCalibrator:
    """Platt scaling for a single class (1-D probabilities -> calibrated)."""

    def __init__(self):
        self.lr = LogisticRegression(C=1e10, solver="lbfgs")
        self._constant = None

    def fit(self, p: np.ndarray, y: np.ndarray) -> "_SigmoidCalibrator":
        if len(np.unique(y)) < 2:
            # Degenerate calibration set: fall back to the empirical base rate.
            self._constant = float(np.mean(y))
            return self
        self.lr.fit(p.reshape(-1, 1), y)
        return self

    def predict(self, p: np.ndarray) -> np.ndarray:
        if self._constant is not None:
            return np.full_like(p, self._constant, dtype=float)
        return self.lr.predict_proba(p.reshape(-1, 1))[:, 1]


class _IsotonicCalibrator:
    """Isotonic regression for a single class."""

    def __init__(self):
        self.ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self._constant = None

    def fit(self, p: np.ndarray, y: np.ndarray) -> "_IsotonicCalibrator":
        if len(np.unique(y)) < 2:
            self._constant = float(np.mean(y))
            return self
        self.ir.fit(p, y)
        return self

    def predict(self, p: np.ndarray) -> np.ndarray:
        if self._constant is not None:
            return np.full_like(p, self._constant, dtype=float)
        return self.ir.predict(p)


class MultiClassCalibrator:
    """One-vs-rest calibrator for 1X2 probabilities.

    Parameters
    ----------
    method:
        ``"sigmoid"`` (Platt), ``"isotonic"`` or ``"none"`` (identity).
    """

    def __init__(self, method: str = "isotonic"):
        if method not in {"sigmoid", "isotonic", "none"}:
            raise ValueError("method must be 'sigmoid', 'isotonic' or 'none'")
        self.method = method
        self.calibrators_: List = []
        self.classes_ = list(OUTCOMES)

    def fit(self, probs: np.ndarray, labels) -> "MultiClassCalibrator":
        """Fit per-class calibrators.

        Parameters
        ----------
        probs:
            Uncalibrated probabilities, shape ``(n, 3)`` in H/D/A order.
        labels:
            True outcomes as 'H'/'D'/'A' strings or integer class indices.
        """
        probs = np.asarray(probs, dtype=float)
        y_idx = self._labels_to_idx(labels)
        if self.method == "none":
            return self
        self.calibrators_ = []
        for c in range(3):
            y_bin = (y_idx == c).astype(int)
            cal = _SigmoidCalibrator() if self.method == "sigmoid" else _IsotonicCalibrator()
            cal.fit(probs[:, c], y_bin)
            self.calibrators_.append(cal)
        return self

    def predict(self, probs: np.ndarray) -> np.ndarray:
        """Return calibrated, renormalised probabilities of shape ``(n, 3)``."""
        probs = np.asarray(probs, dtype=float)
        if self.method == "none" or not self.calibrators_:
            return probs
        out = np.column_stack([
            self.calibrators_[c].predict(probs[:, c]) for c in range(3)
        ])
        out = np.clip(out, 1e-9, None)
        return out / out.sum(axis=1, keepdims=True)

    @staticmethod
    def _labels_to_idx(labels) -> np.ndarray:
        labels = np.asarray(labels)
        if labels.dtype.kind in {"U", "S", "O"}:
            mapping = {"H": 0, "D": 1, "A": 2}
            return np.array([mapping[str(v)] for v in labels])
        return labels.astype(int)


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #
def expected_calibration_error(probs: np.ndarray, labels, n_bins: int = 10) -> float:
    """Multi-class ECE using the predicted top-class confidence."""
    probs = np.asarray(probs, dtype=float)
    y_idx = MultiClassCalibrator._labels_to_idx(labels)
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_idx).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(conf)
    for i in range(n_bins):
        mask = (conf > bins[i]) & (conf <= bins[i + 1])
        if mask.sum() == 0:
            continue
        acc = correct[mask].mean()
        avg_conf = conf[mask].mean()
        ece += (mask.sum() / n) * abs(acc - avg_conf)
    return float(ece)


def reliability_curve(
    probs_class: np.ndarray, y_binary: np.ndarray, n_bins: int = 10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (bin_mean_pred, bin_observed_freq, bin_count) for one class."""
    probs_class = np.asarray(probs_class, dtype=float)
    y_binary = np.asarray(y_binary, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    mean_pred, obs_freq, counts = [], [], []
    for i in range(n_bins):
        mask = (probs_class > bins[i]) & (probs_class <= bins[i + 1])
        if mask.sum() == 0:
            continue
        mean_pred.append(probs_class[mask].mean())
        obs_freq.append(y_binary[mask].mean())
        counts.append(int(mask.sum()))
    return np.array(mean_pred), np.array(obs_freq), np.array(counts)
