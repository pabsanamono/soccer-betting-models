"""Proper scoring rules for probabilistic 1X2 forecasts.

The research is unanimous that classification accuracy is the wrong metric. The
correct toolkit, all implemented here:

* **Ranked Probability Score (RPS)** — the standard benchmark for ordinal
  win/draw/loss forecasts; sensitive to ordinal distance (a draw is "closer" to
  a home win than an away win). Lower is better.
* **Log loss (cross-entropy)** — penalises confident-but-wrong predictions.
* **Brier score** — mean squared error of probabilities; captures both
  calibration and discrimination.
* **Accuracy** — reported only for reference, never as the objective.

Outcome order is fixed Home, Draw, Away.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

_LABEL_TO_IDX = {"H": 0, "D": 1, "A": 2}


def _labels_to_onehot(labels) -> np.ndarray:
    labels = np.asarray(labels)
    if labels.dtype.kind in {"U", "S", "O"}:
        idx = np.array([_LABEL_TO_IDX[str(v)] for v in labels])
    else:
        idx = labels.astype(int)
    onehot = np.zeros((len(idx), 3))
    onehot[np.arange(len(idx)), idx] = 1.0
    return onehot


def _validate(probs: np.ndarray, labels) -> tuple:
    probs = np.asarray(probs, dtype=float)
    if probs.ndim != 2 or probs.shape[1] != 3:
        raise ValueError("probs must have shape (n, 3) in H/D/A order.")
    onehot = _labels_to_onehot(labels)
    if len(probs) != len(onehot):
        raise ValueError("probs and labels length mismatch.")
    # Drop rows containing NaNs (e.g. fixtures the model could not score).
    valid = ~np.isnan(probs).any(axis=1)
    return probs[valid], onehot[valid]


def ranked_probability_score(probs: np.ndarray, labels) -> float:
    """Mean RPS over all forecasts (lower is better).

    RPS = 1/(r-1) * sum_i ( cumsum(p)_i - cumsum(a)_i )^2, with r = 3 outcomes.
    """
    probs, onehot = _validate(probs, labels)
    if len(probs) == 0:
        return float("nan")
    cum_p = np.cumsum(probs, axis=1)
    cum_a = np.cumsum(onehot, axis=1)
    rps = np.sum((cum_p - cum_a) ** 2, axis=1) / (probs.shape[1] - 1)
    return float(rps.mean())


def multiclass_log_loss(probs: np.ndarray, labels, eps: float = 1e-15) -> float:
    """Mean negative log-likelihood of the true outcome."""
    probs, onehot = _validate(probs, labels)
    if len(probs) == 0:
        return float("nan")
    p = np.clip(probs, eps, 1 - eps)
    return float(-np.mean(np.sum(onehot * np.log(p), axis=1)))


def multiclass_brier_score(probs: np.ndarray, labels) -> float:
    """Multi-class Brier score: mean squared error across the 3 outcomes."""
    probs, onehot = _validate(probs, labels)
    if len(probs) == 0:
        return float("nan")
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def accuracy(probs: np.ndarray, labels) -> float:
    """Top-1 accuracy (reported for reference only)."""
    probs, onehot = _validate(probs, labels)
    if len(probs) == 0:
        return float("nan")
    return float(np.mean(probs.argmax(axis=1) == onehot.argmax(axis=1)))


def evaluate_predictions(probs: np.ndarray, labels) -> Dict[str, float]:
    """Return all proper scoring metrics in one dict."""
    return {
        "n": int(len(_validate(probs, labels)[0])),
        "rps": ranked_probability_score(probs, labels),
        "log_loss": multiclass_log_loss(probs, labels),
        "brier": multiclass_brier_score(probs, labels),
        "accuracy": accuracy(probs, labels),
    }
