"""Common interface for all match-probability models.

Every model exposes:

* ``fit(matches)`` — learn parameters from a canonical match frame.
* ``predict_proba(home_team, away_team, **kw)`` — return a length-3 array
  ``[P(home win), P(draw), P(away win)]``.
* ``predict_frame(matches)`` — vectorised prediction returning a DataFrame with
  ``prob_home/prob_draw/prob_away`` columns aligned to the input rows.

Outcome ordering is fixed as Home, Draw, Away everywhere in the package so that
probabilities, odds and labels are always comparable.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import numpy as np
import pandas as pd

OUTCOMES: List[str] = ["H", "D", "A"]


class BaseMatchModel(ABC):
    """Abstract base class for 1X2 match models."""

    #: Set to True by ``fit`` implementations.
    is_fitted: bool = False

    @abstractmethod
    def fit(self, matches: pd.DataFrame) -> "BaseMatchModel":
        """Fit the model on a canonical match frame."""

    @abstractmethod
    def predict_proba(self, home_team: str, away_team: str, **kwargs) -> np.ndarray:
        """Return ``[p_home, p_draw, p_away]`` for a single fixture."""

    # -- shared helpers ------------------------------------------------------
    def _check_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError(f"{type(self).__name__} must be fitted before predicting.")

    def predict_frame(self, matches: pd.DataFrame) -> pd.DataFrame:
        """Vectorised prediction over a frame with home_team/away_team columns."""
        self._check_fitted()
        if len(matches) == 0:
            return pd.DataFrame(columns=["prob_home", "prob_draw", "prob_away"])
        rows = []
        for r in matches.itertuples(index=False):
            try:
                p = self.predict_proba(r.home_team, r.away_team)
            except Exception:
                p = np.array([np.nan, np.nan, np.nan])
            rows.append(p)
        arr = np.vstack(rows)
        return pd.DataFrame(
            {"prob_home": arr[:, 0], "prob_draw": arr[:, 1], "prob_away": arr[:, 2]},
            index=matches.index,
        )

    @staticmethod
    def _outcome_probs_from_matrix(score_matrix: np.ndarray) -> np.ndarray:
        """Aggregate a scoreline probability matrix into H/D/A probabilities.

        Rows index home goals, columns index away goals.
        """
        p_home = np.tril(score_matrix, -1).sum()   # home goals > away goals
        p_away = np.triu(score_matrix, 1).sum()    # away goals > home goals
        p_draw = np.trace(score_matrix)            # equal goals
        total = p_home + p_draw + p_away
        if total <= 0:
            return np.array([np.nan, np.nan, np.nan])
        return np.array([p_home, p_draw, p_away]) / total
