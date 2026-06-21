"""Dixon-Coles model (1997): low-score correction + time decay.

Extends the independent Poisson model with two signature modifications:

1. A **dependence parameter rho** that re-weights the four low-scoring
   scorelines (0-0, 1-0, 0-1, 1-1) via the Dixon-Coles ``tau`` function,
   correcting the independent model's systematic underestimation of draws.

2. **Exponential time decay** ``exp(-xi * age_in_days)`` applied to the
   log-likelihood so recent matches dominate the fit, capturing current form.

The model still produces a full scoreline matrix and therefore serves 1X2,
over/under, BTTS and correct-score markets.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

from soccer_betting.models.base import BaseMatchModel
from soccer_betting.utils.logging import get_logger

logger = get_logger(__name__)


def _tau(hg: np.ndarray, ag: np.ndarray, lam_h: np.ndarray, lam_a: np.ndarray, rho: float) -> np.ndarray:
    """Dixon-Coles low-score correction factor (vectorised)."""
    tau = np.ones_like(lam_h, dtype=float)
    m00 = (hg == 0) & (ag == 0)
    m01 = (hg == 0) & (ag == 1)
    m10 = (hg == 1) & (ag == 0)
    m11 = (hg == 1) & (ag == 1)
    tau[m00] = 1.0 - lam_h[m00] * lam_a[m00] * rho
    tau[m01] = 1.0 + lam_h[m01] * rho
    tau[m10] = 1.0 + lam_a[m10] * rho
    tau[m11] = 1.0 - rho
    return tau


class DixonColesModel(BaseMatchModel):
    """Maximum-likelihood Dixon-Coles model with optional time decay."""

    def __init__(self, max_goals: int = 10, xi: float = 0.0):
        self.max_goals = max_goals
        self.xi = xi  # time-decay rate per day; 0 disables decay
        self.teams_: List[str] = []
        self.attack_: Dict[str, float] = {}
        self.defence_: Dict[str, float] = {}
        self.home_adv_: float = 0.0
        self.intercept_: float = 0.0
        self.rho_: float = 0.0
        self.ref_date_: pd.Timestamp | None = None

    # --------------------------------------------------------------- weights
    def _time_weights(self, dates: pd.Series) -> np.ndarray:
        if self.xi <= 0 or dates is None:
            return np.ones(len(dates)) if dates is not None else None
        age_days = (self.ref_date_ - dates).dt.days.to_numpy(dtype=float)
        age_days = np.clip(age_days, 0, None)
        return np.exp(-self.xi * age_days)

    # ------------------------------------------------------------------- nll
    def _neg_log_likelihood(self, params, h_idx, a_idx, hg, ag, weights):
        n = len(self.teams_)
        attack = params[:n]
        defence = params[n:2 * n]
        home_adv = params[2 * n]
        intercept = params[2 * n + 1]
        rho = params[2 * n + 2]

        lam_h = np.exp(intercept + home_adv + attack[h_idx] - defence[a_idx])
        lam_a = np.exp(intercept + attack[a_idx] - defence[h_idx])

        tau = _tau(hg, ag, lam_h, lam_a, rho)
        # tau can go non-positive for extreme rho; guard the log.
        tau = np.clip(tau, 1e-10, None)
        ll = np.log(tau) + poisson.logpmf(hg, lam_h) + poisson.logpmf(ag, lam_a)
        return -np.sum(weights * ll)

    # ------------------------------------------------------------------- fit
    def fit(self, matches: pd.DataFrame) -> "DixonColesModel":
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

        if "date" in matches.columns and self.xi > 0:
            dates = pd.to_datetime(matches["date"])
            self.ref_date_ = dates.max()
            weights = self._time_weights(dates)
        else:
            weights = np.ones(len(matches))

        x0 = np.concatenate([
            np.zeros(n), np.zeros(n),
            np.array([0.25]),                          # home advantage
            np.array([np.log(max(hg.mean(), 0.1))]),   # intercept
            np.array([0.0]),                           # rho
        ])
        constraints = ({"type": "eq", "fun": lambda p: np.sum(p[:n])},)
        # Keep rho in a sensible range to maintain positive probabilities.
        bounds = [(None, None)] * (2 * n + 2) + [(-0.2, 0.2)]

        res = minimize(
            self._neg_log_likelihood,
            x0,
            args=(h_idx, a_idx, hg, ag, weights),
            method="SLSQP",
            constraints=constraints,
            bounds=bounds,
            options={"maxiter": 300, "ftol": 1e-7, "disp": False},
        )
        if not res.success:
            logger.warning("Dixon-Coles MLE did not fully converge: %s", res.message)

        p = res.x
        self.attack_ = {t: p[i] for t, i in idx.items()}
        self.defence_ = {t: p[n + i] for t, i in idx.items()}
        self.home_adv_ = float(p[2 * n])
        self.intercept_ = float(p[2 * n + 1])
        self.rho_ = float(p[2 * n + 2])
        self.is_fitted = True
        logger.info(
            "Fitted DixonColesModel (rho=%.4f, xi=%.4g) on %d teams, %d matches",
            self.rho_, self.xi, n, len(matches),
        )
        return self

    # --------------------------------------------------------------- predict
    def expected_goals(self, home_team: str, away_team: str) -> Tuple[float, float]:
        self._check_fitted()
        for t in (home_team, away_team):
            if t not in self.attack_:
                raise KeyError(f"Unknown team '{t}'.")
        lam_h = np.exp(self.intercept_ + self.home_adv_ + self.attack_[home_team] - self.defence_[away_team])
        lam_a = np.exp(self.intercept_ + self.attack_[away_team] - self.defence_[home_team])
        return float(lam_h), float(lam_a)

    def score_matrix(self, home_team: str, away_team: str) -> np.ndarray:
        lam_h, lam_a = self.expected_goals(home_team, away_team)
        goals = np.arange(self.max_goals + 1)
        ph = poisson.pmf(goals, lam_h)
        pa = poisson.pmf(goals, lam_a)
        matrix = np.outer(ph, pa)

        # Apply the tau correction to the 2x2 low-score block.
        lh = np.full(4, lam_h)
        la = np.full(4, lam_a)
        hgs = np.array([0, 0, 1, 1])
        ags = np.array([0, 1, 0, 1])
        corr = _tau(hgs, ags, lh, la, self.rho_)
        matrix[0, 0] *= corr[0]
        matrix[0, 1] *= corr[1]
        matrix[1, 0] *= corr[2]
        matrix[1, 1] *= corr[3]
        return matrix / matrix.sum()

    def predict_proba(self, home_team: str, away_team: str, **kwargs) -> np.ndarray:
        return self._outcome_probs_from_matrix(self.score_matrix(home_team, away_team))

    def prob_over_under(self, home_team: str, away_team: str, line: float = 2.5) -> Tuple[float, float]:
        matrix = self.score_matrix(home_team, away_team)
        goals_grid = np.add.outer(np.arange(self.max_goals + 1), np.arange(self.max_goals + 1))
        p_over = matrix[goals_grid > line].sum()
        return float(p_over), float(1.0 - p_over)

    def prob_btts(self, home_team: str, away_team: str) -> float:
        matrix = self.score_matrix(home_team, away_team)
        return float(matrix[1:, 1:].sum())
