"""Bivariate Poisson model (Karlis & Ntzoufras, 2003).

Where Dixon-Coles patches the independence assumption with a local correction,
the bivariate Poisson handles correlation structurally. Goals are modelled as

    X = W1 + W3   (home goals)
    Y = W2 + W3   (away goals)

with W1 ~ Poisson(l1), W2 ~ Poisson(l2), W3 ~ Poisson(l3) independent. The
shared term W3 induces a positive covariance equal to ``l3``. When ``l3 = 0``
the model collapses to independent Poisson.

We parameterise ``l1`` and ``l2`` with the usual attack/defence/home structure
and treat ``l3`` as a single non-negative covariance parameter estimated by
maximum likelihood.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

from soccer_betting.models.base import BaseMatchModel
from soccer_betting.utils.logging import get_logger

logger = get_logger(__name__)


def bivariate_poisson_pmf(x: int, y: int, l1: float, l2: float, l3: float) -> float:
    """Probability mass for a single (x, y) under bivariate Poisson."""
    if l1 <= 0 or l2 <= 0:
        return 0.0
    # If there is no covariance term, this reduces to independent Poisson.
    if l3 <= 0:
        log_p = (
            -(l1 + l2)
            + x * np.log(l1) + y * np.log(l2)
            - gammaln(x + 1) - gammaln(y + 1)
        )
        return float(np.exp(log_p))

    kmax = min(int(x), int(y))
    # Stable summation in log space for each k term of the convolution.
    log_pref = -(l1 + l2 + l3) + x * np.log(l1) + y * np.log(l2) - gammaln(x + 1) - gammaln(y + 1)
    total = 0.0
    for k in range(kmax + 1):
        log_term = (
            gammaln(x + 1) - gammaln(k + 1) - gammaln(x - k + 1)
            + gammaln(y + 1) - gammaln(k + 1) - gammaln(y - k + 1)
            + gammaln(k + 1)
            + k * (np.log(l3) - np.log(l1) - np.log(l2))
        )
        if np.isfinite(log_term):
            total += np.exp(np.clip(log_term, -700, 700))
    value = np.exp(np.clip(log_pref, -700, 700)) * total
    return float(value) if np.isfinite(value) else 0.0


class BivariatePoissonModel(BaseMatchModel):
    """Maximum-likelihood bivariate Poisson goals model."""

    def __init__(self, max_goals: int = 10):
        self.max_goals = max_goals
        self.teams_: List[str] = []
        self.attack_: Dict[str, float] = {}
        self.defence_: Dict[str, float] = {}
        self.home_adv_: float = 0.0
        self.intercept_: float = 0.0
        self.l3_: float = 0.0

    def _neg_log_likelihood(self, params, h_idx, a_idx, hg, ag):
        n = len(self.teams_)
        attack = params[:n]
        defence = params[n:2 * n]
        home_adv = params[2 * n]
        intercept = params[2 * n + 1]
        log_l3 = params[2 * n + 2]
        l3 = np.exp(log_l3)

        l1 = np.exp(intercept + home_adv + attack[h_idx] - defence[a_idx])
        l2 = np.exp(intercept + attack[a_idx] - defence[h_idx])

        ll = 0.0
        for i in range(len(hg)):
            p = bivariate_poisson_pmf(int(hg[i]), int(ag[i]), l1[i], l2[i], l3)
            ll += np.log(max(p, 1e-12))
        return -ll

    def fit(self, matches: pd.DataFrame) -> "BivariatePoissonModel":
        required = {"home_team", "away_team", "home_goals", "away_goals"}
        if not required.issubset(matches.columns):
            raise ValueError(f"matches missing columns: {required - set(matches.columns)}")
        if matches.empty:
            raise ValueError("Cannot fit on an empty frame.")

        self.teams_ = sorted(set(matches["home_team"]) | set(matches["away_team"]))
        idx = {t: i for i, t in enumerate(self.teams_)}
        n = len(self.teams_)
        h_idx = matches["home_team"].map(idx).to_numpy()
        a_idx = matches["away_team"].map(idx).to_numpy()
        hg = matches["home_goals"].to_numpy(dtype=float)
        ag = matches["away_goals"].to_numpy(dtype=float)

        x0 = np.concatenate([
            np.zeros(n), np.zeros(n),
            np.array([0.25]),
            np.array([np.log(max(hg.mean(), 0.1))]),
            np.array([np.log(0.1)]),  # log l3
        ])
        constraints = ({"type": "eq", "fun": lambda p: np.sum(p[:n])},)

        res = minimize(
            self._neg_log_likelihood,
            x0,
            args=(h_idx, a_idx, hg, ag),
            method="SLSQP",
            constraints=constraints,
            options={"maxiter": 200, "ftol": 1e-6, "disp": False},
        )
        if not res.success:
            logger.warning("Bivariate Poisson MLE did not fully converge: %s", res.message)

        p = res.x
        self.attack_ = {t: p[i] for t, i in idx.items()}
        self.defence_ = {t: p[n + i] for t, i in idx.items()}
        self.home_adv_ = float(p[2 * n])
        self.intercept_ = float(p[2 * n + 1])
        self.l3_ = float(np.exp(p[2 * n + 2]))
        self.is_fitted = True
        logger.info("Fitted BivariatePoissonModel (l3=%.4f) on %d teams", self.l3_, n)
        return self

    def expected_goals(self, home_team: str, away_team: str) -> Tuple[float, float]:
        self._check_fitted()
        for t in (home_team, away_team):
            if t not in self.attack_:
                raise KeyError(f"Unknown team '{t}'.")
        l1 = np.exp(self.intercept_ + self.home_adv_ + self.attack_[home_team] - self.defence_[away_team])
        l2 = np.exp(self.intercept_ + self.attack_[away_team] - self.defence_[home_team])
        # Marginal means include the shared component.
        return float(l1 + self.l3_), float(l2 + self.l3_)

    def score_matrix(self, home_team: str, away_team: str) -> np.ndarray:
        self._check_fitted()
        l1 = np.exp(self.intercept_ + self.home_adv_ + self.attack_[home_team] - self.defence_[away_team])
        l2 = np.exp(self.intercept_ + self.attack_[away_team] - self.defence_[home_team])
        g = self.max_goals + 1
        matrix = np.zeros((g, g))
        for x in range(g):
            for y in range(g):
                matrix[x, y] = bivariate_poisson_pmf(x, y, l1, l2, self.l3_)
        s = matrix.sum()
        return matrix / s if s > 0 else matrix

    def predict_proba(self, home_team: str, away_team: str, **kwargs) -> np.ndarray:
        return self._outcome_probs_from_matrix(self.score_matrix(home_team, away_team))
