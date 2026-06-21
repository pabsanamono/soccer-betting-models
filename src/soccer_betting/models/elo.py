"""Elo-rating 1X2 model.

A lightweight rating model used both as a standalone baseline and as a source
of features. It converts the home/away rating difference (plus a home-advantage
offset) into win/draw/loss probabilities. The draw probability is modelled with
a simple, well-known parametric form driven by how close the two ratings are
(draws are more likely between evenly matched teams).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from soccer_betting.data.features import EloRating
from soccer_betting.models.base import BaseMatchModel
from soccer_betting.utils.logging import get_logger

logger = get_logger(__name__)


class EloModel(BaseMatchModel):
    """Elo ratings mapped to 1X2 probabilities.

    Parameters
    ----------
    k_factor, home_advantage, initial:
        Standard Elo hyper-parameters.
    draw_base, draw_width:
        Control the draw probability curve. ``draw_base`` is the maximum draw
        probability for perfectly matched teams; ``draw_width`` controls how
        fast it decays as the rating gap widens.
    """

    def __init__(
        self,
        k_factor: float = 20,
        home_advantage: float = 65,
        initial: float = 1500,
        draw_base: float = 0.28,
        draw_width: float = 200.0,
    ):
        self.k_factor = k_factor
        self.home_advantage = home_advantage
        self.initial = initial
        self.draw_base = draw_base
        self.draw_width = draw_width
        self._elo = EloRating(k_factor, home_advantage, initial)

    def fit(self, matches: pd.DataFrame) -> "EloModel":
        required = {"home_team", "away_team", "result"}
        if not required.issubset(matches.columns):
            raise ValueError(f"matches missing columns: {required - set(matches.columns)}")
        self._elo = EloRating(self.k_factor, self.home_advantage, self.initial)
        df = matches.sort_values("date") if "date" in matches.columns else matches
        for r in df.itertuples(index=False):
            self._elo.update(r.home_team, r.away_team, r.result)
        self.is_fitted = True
        logger.info("Fitted EloModel over %d matches", len(df))
        return self

    def predict_proba(self, home_team: str, away_team: str, **kwargs) -> np.ndarray:
        self._check_fitted()
        rh = self._elo.get(home_team) + self.home_advantage
        ra = self._elo.get(away_team)
        # Probability the home side is 'better' on the day (excluding draws).
        p_home_raw = 1.0 / (1.0 + 10 ** ((ra - rh) / 400.0))
        # Draw probability peaks when ratings are equal.
        gap = abs(rh - ra)
        p_draw = self.draw_base * np.exp(-(gap ** 2) / (2 * self.draw_width ** 2))
        p_home = p_home_raw * (1.0 - p_draw)
        p_away = (1.0 - p_home_raw) * (1.0 - p_draw)
        probs = np.array([p_home, p_draw, p_away])
        return probs / probs.sum()
