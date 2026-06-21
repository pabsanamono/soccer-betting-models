"""Independent Poisson model (Maher, 1982).

Home goals ~ Poisson(lambda_h) and away goals ~ Poisson(lambda_a), with

    log(lambda_h) = mu + home_adv + attack[home] - defence[away]
    log(lambda_a) = mu + attack[away] - defence[home]

Parameters are estimated by maximum likelihood. A sum-to-zero constraint on the
attack parameters keeps the parameterisation identifiable. From the fitted
rates the full scoreline matrix is built and aggregated into 1X2 (and any goals
market) probabilities.
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


class PoissonModel(BaseMatchModel):
    """Maximum-likelihood independent Poisson goals model."""

    def __init__(self, max_goals: int = 10):
        self.max_goals = max_goals
        self.teams_: List[str] = []
        self.params_: Dict[str, float] = {}
        self.attack_: Dict[str, float] = {}
        self.defence_: Dict[str, float] = {}
        self.home_adv_: float = 0.0
        self.intercept_: float = 0.0

    # ------------------------------------------------------------------ fit
    def _pack_index(self) -> Tuple[Dict[str, int], int]:
        idx = {team: i for i, team in enumerate(self.teams_)}
        return idx, len(self.teams_)

    def _neg_log_likelihood(self, params: np.ndarray, h_idx, a_idx, hg, ag, weights) -> float:
        n = len(self.teams_)
        attack = params[:n]
        defence = params[n:2 * n]
        home_adv = params[2 * n]
        intercept = params[2 * n + 1]

        log_lam_h = intercept + home_adv + attack[h_idx] - defence[a_idx]
        log_lam_a = intercept + attack[a_idx] - defence[h_idx]
        lam_h = np.exp(log_lam_h)
        lam_a = np.exp(log_lam_a)

        ll = poisson.logpmf(hg, lam_h) + poisson.logpmf(ag, lam_a)
        return -np.sum(weights * ll)

    def _weights(self, matches: pd.DataFrame) -> np.ndarray:
        """Uniform weights (overridden by Dixon-Coles time decay)."""
        return np.ones(len(matches))

    def fit(self, matches: pd.DataFrame) -> "PoissonModel":
        required = {"home_team", "away_team", "home_goals", "away_goals"}
        if not required.issubset(matches.columns):
            raise ValueError(f"matches missing columns: {required - set(matches.columns)}")
        if matches.empty:
            raise ValueError("Cannot fit on an empty frame.")

        self.teams_ = sorted(set(matches["home_team"]) | set(matches["away_team"]))
        idx, n = self._pack_index()
        h_idx = matches["home_team"].map(idx).to_numpy()
        a_idx = matches["away_team"].map(idx).to_numpy()
        hg = matches["home_goals"].to_numpy(dtype=float)
        ag = matches["away_goals"].to_numpy(dtype=float)
        weights = self._weights(matches)

        # Initial guess: zero strengths, modest home advantage, log mean goals.
        x0 = np.concatenate([
            np.zeros(n),                       # attack
            np.zeros(n),                       # defence
            np.array([0.25]),                  # home advantage
            np.array([np.log(max(hg.mean(), 0.1))]),  # intercept
        ])

        # Sum-to-zero constraint on attack params for identifiability.
        constraints = ({
            "type": "eq",
            "fun": lambda p: np.sum(p[:n]),
        },)

        res = minimize(
            self._neg_log_likelihood,
            x0,
            args=(h_idx, a_idx, hg, ag, weights),
            method="SLSQP",
            constraints=constraints,
            options={"maxiter": 200, "ftol": 1e-7, "disp": False},
        )
        if not res.success:
            logger.warning("Poisson MLE did not fully converge: %s", res.message)

        p = res.x
        self.attack_ = {t: p[i] for t, i in idx.items()}
        self.defence_ = {t: p[n + i] for t, i in idx.items()}
        self.home_adv_ = float(p[2 * n])
        self.intercept_ = float(p[2 * n + 1])
        self.is_fitted = True
        logger.info("Fitted PoissonModel on %d teams, %d matches", n, len(matches))
        return self

    # -------------------------------------------------------------- predict
    def expected_goals(self, home_team: str, away_team: str) -> Tuple[float, float]:
        """Return ``(lambda_home, lambda_away)`` for a fixture."""
        self._check_fitted()
        for t in (home_team, away_team):
            if t not in self.attack_:
                raise KeyError(f"Unknown team '{t}'. Fit on data that includes it.")
        lam_h = np.exp(self.intercept_ + self.home_adv_ + self.attack_[home_team] - self.defence_[away_team])
        lam_a = np.exp(self.intercept_ + self.attack_[away_team] - self.defence_[home_team])
        return float(lam_h), float(lam_a)

    def score_matrix(self, home_team: str, away_team: str) -> np.ndarray:
        """Joint scoreline probability matrix (home goals x away goals)."""
        lam_h, lam_a = self.expected_goals(home_team, away_team)
        goals = np.arange(self.max_goals + 1)
        ph = poisson.pmf(goals, lam_h)
        pa = poisson.pmf(goals, lam_a)
        return np.outer(ph, pa)

    def predict_proba(self, home_team: str, away_team: str, **kwargs) -> np.ndarray:
        matrix = self.score_matrix(home_team, away_team)
        return self._outcome_probs_from_matrix(matrix)

    # -- derivative markets --------------------------------------------------
    def prob_over_under(self, home_team: str, away_team: str, line: float = 2.5) -> Tuple[float, float]:
        """Return ``(P(over line), P(under line))`` total goals."""
        matrix = self.score_matrix(home_team, away_team)
        total = matrix.sum()
        goals_grid = np.add.outer(
            np.arange(self.max_goals + 1), np.arange(self.max_goals + 1)
        )
        p_over = matrix[goals_grid > line].sum() / total
        return float(p_over), float(1.0 - p_over)

    def prob_btts(self, home_team: str, away_team: str) -> float:
        """Probability both teams score."""
        matrix = self.score_matrix(home_team, away_team)
        return float(matrix[1:, 1:].sum() / matrix.sum())
