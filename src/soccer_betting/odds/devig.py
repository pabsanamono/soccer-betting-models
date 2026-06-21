"""Devigging: recovering 'true' probabilities from bookmaker odds.

Bookmakers quote odds whose implied probabilities sum to more than 1; the
excess is the *overround* (a.k.a. vig/juice). To compare a model against the
market you must strip this margin. As the research stresses, devigging is a
modeling decision and the right method should be chosen empirically (lowest
log-loss / Brier on held-out data). This module implements the main documented
approaches:

* ``multiplicative`` (naive normalisation) — divide each implied probability by
  their sum. Simple, but assumes a uniform margin.
* ``additive`` — subtract an equal share of the margin from each implied prob.
* ``power`` — find an exponent ``k`` so that ``sum(p_i ** k) == 1``; captures
  the favourite-longshot skew that the multiplicative method ignores.
* ``shin`` — Shin's model, which attributes the margin to insider trading and
  applies a heavier correction to longshots; widely regarded as a strong
  default for 1X2 markets.

All functions accept either a single vector of decimal odds or a 2-D array of
shape ``(n_matches, n_outcomes)`` and return probabilities of the same shape.
"""
from __future__ import annotations

from typing import Union

import numpy as np
from scipy.optimize import brentq

ArrayLike = Union[np.ndarray, list, tuple]


# --------------------------------------------------------------------------- #
# Basic conversions
# --------------------------------------------------------------------------- #
def odds_to_prob(odds: ArrayLike) -> np.ndarray:
    """Implied (margin-inclusive) probability = 1 / decimal odds."""
    odds = np.asarray(odds, dtype=float)
    if np.any(odds <= 1.0):
        raise ValueError("Decimal odds must be > 1.0")
    return 1.0 / odds


def prob_to_odds(prob: ArrayLike) -> np.ndarray:
    """Convert (fair) probabilities to decimal odds."""
    prob = np.asarray(prob, dtype=float)
    if np.any(prob <= 0) or np.any(prob > 1):
        raise ValueError("Probabilities must be in (0, 1].")
    return 1.0 / prob


def implied_probabilities(odds: ArrayLike) -> np.ndarray:
    """Alias for :func:`odds_to_prob` (raw implied probabilities)."""
    return odds_to_prob(odds)


def overround(odds: ArrayLike, axis: int = -1) -> np.ndarray:
    """Overround = (sum of implied probabilities) - 1."""
    return odds_to_prob(odds).sum(axis=axis) - 1.0


# --------------------------------------------------------------------------- #
# Internal helpers operating on a single match (1-D vector of implied probs)
# --------------------------------------------------------------------------- #
def _multiplicative(imp: np.ndarray) -> np.ndarray:
    return imp / imp.sum()


def _additive(imp: np.ndarray) -> np.ndarray:
    n = len(imp)
    margin = imp.sum() - 1.0
    p = imp - margin / n
    # Guard against tiny negatives from heavy margins on longshots.
    p = np.clip(p, 1e-9, None)
    return p / p.sum()


def _power(imp: np.ndarray) -> np.ndarray:
    """Solve for k such that sum(imp ** k) == 1."""
    def f(k: float) -> float:
        return np.sum(imp ** k) - 1.0

    # k > 1 shrinks probabilities; the root lies in a generous bracket.
    try:
        k = brentq(f, 0.5, 5.0, maxiter=200)
    except ValueError:
        return _multiplicative(imp)
    p = imp ** k
    return p / p.sum()


def _shin(imp: np.ndarray) -> np.ndarray:
    """Shin (1992) devig: estimate insider proportion z, then back out probs."""
    booksum = imp.sum()

    def implied_z(z: float) -> np.ndarray:
        # p_i recovered from Shin's quadratic given z.
        return (np.sqrt(z ** 2 + 4 * (1 - z) * imp ** 2 / booksum) - z) / (2 * (1 - z))

    def f(z: float) -> float:
        return implied_z(z).sum() - 1.0

    try:
        z = brentq(f, 1e-9, 0.2, maxiter=200)
    except ValueError:
        return _multiplicative(imp)
    p = implied_z(z)
    return p / p.sum()


_METHODS = {
    "multiplicative": _multiplicative,
    "additive": _additive,
    "power": _power,
    "shin": _shin,
}


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def devig(odds: ArrayLike, method: str = "multiplicative") -> np.ndarray:
    """Return fair probabilities from decimal odds.

    Parameters
    ----------
    odds:
        Decimal odds, shape ``(n_outcomes,)`` or ``(n_matches, n_outcomes)``.
    method:
        One of ``multiplicative``, ``additive``, ``power``, ``shin``.

    Returns
    -------
    np.ndarray
        Probabilities that sum to 1 along the last axis, same shape as input.
    """
    if method not in _METHODS:
        raise ValueError(f"Unknown devig method '{method}'. Options: {list(_METHODS)}")
    fn = _METHODS[method]
    arr = np.asarray(odds, dtype=float)
    imp = odds_to_prob(arr)

    if imp.ndim == 1:
        return fn(imp)
    if imp.ndim == 2:
        return np.vstack([fn(row) for row in imp])
    raise ValueError("odds must be 1-D or 2-D.")
