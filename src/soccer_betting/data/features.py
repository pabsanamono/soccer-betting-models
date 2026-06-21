"""Feature engineering with strict leakage avoidance.

The single most important rule from the research is *walk-forward* feature
construction: every feature for a match must be computable using only
information available **before kick-off**. This module enforces that by:

* computing rolling team-form features with a ``shift(1)`` so the current
  match never contributes to its own features;
* maintaining Elo ratings that are recorded *before* each match is played and
  only updated afterwards;
* deriving market-implied probabilities by devigging the closing odds (these
  are known pre-match and are the single strongest predictor available).

The output is a model-ready feature matrix aligned with the canonical match
frame produced by :mod:`soccer_betting.data.preprocess`.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from soccer_betting.odds.devig import devig
from soccer_betting.utils.logging import get_logger

logger = get_logger(__name__)


class EloRating:
    """Classic Elo with home advantage, suitable for soccer 1X2.

    Ratings are exposed *before* each update so they can be used as leakage-free
    features. The expected score uses a logistic on the rating difference plus a
    home-advantage offset; draws are handled by treating the actual score as
    0.5 for both teams (a standard, robust simplification).
    """

    def __init__(self, k_factor: float = 20, home_advantage: float = 65, initial: float = 1500):
        self.k = k_factor
        self.home_advantage = home_advantage
        self.initial = initial
        self.ratings: Dict[str, float] = {}

    def get(self, team: str) -> float:
        return self.ratings.get(team, self.initial)

    @staticmethod
    def _expected(rating_a: float, rating_b: float) -> float:
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))

    def update(self, home: str, away: str, result: str) -> None:
        """Update ratings after a match given result in {'H','D','A'}."""
        rh, ra = self.get(home), self.get(away)
        exp_h = self._expected(rh + self.home_advantage, ra)
        score_h = {"H": 1.0, "D": 0.5, "A": 0.0}[result]
        self.ratings[home] = rh + self.k * (score_h - exp_h)
        self.ratings[away] = ra + self.k * ((1.0 - score_h) - (1.0 - exp_h))


class FeatureBuilder:
    """Build a leakage-free feature matrix from canonical match data.

    Parameters
    ----------
    rolling_windows:
        Window sizes (in matches) for recent-form aggregates.
    elo_params:
        Keyword args forwarded to :class:`EloRating`.
    include_market_features:
        If True, add devigged market probabilities as features.
    devig_method:
        Devigging method used for the market features.
    """

    def __init__(
        self,
        rolling_windows: Sequence[int] = (3, 5, 10),
        elo_params: Optional[Dict] = None,
        include_market_features: bool = True,
        devig_method: str = "multiplicative",
    ):
        self.rolling_windows = list(rolling_windows)
        self.elo_params = elo_params or {}
        self.include_market_features = include_market_features
        self.devig_method = devig_method
        self.feature_columns_: List[str] = []

    # ------------------------------------------------------------------ Elo
    def _add_elo(self, df: pd.DataFrame) -> pd.DataFrame:
        elo = EloRating(**self.elo_params)
        home_pre, away_pre = [], []
        for row in df.itertuples(index=False):
            home_pre.append(elo.get(row.home_team))
            away_pre.append(elo.get(row.away_team))
            elo.update(row.home_team, row.away_team, row.result)
        df = df.copy()
        df["elo_home"] = home_pre
        df["elo_away"] = away_pre
        df["elo_diff"] = df["elo_home"] - df["elo_away"]
        df["elo_exp_home"] = 1.0 / (
            1.0 + 10 ** ((df["elo_away"] - (df["elo_home"] + elo.home_advantage)) / 400.0)
        )
        return df

    # -------------------------------------------------------------- rolling
    def _add_rolling(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling team-form features computed from past matches only."""
        # Build a long, team-centric view so each team's history is contiguous.
        home = pd.DataFrame(
            {
                "match_id": df.index,
                "date": df["date"].values,
                "team": df["home_team"].values,
                "gf": df["home_goals"].values,
                "ga": df["away_goals"].values,
                "venue": "home",
            }
        )
        away = pd.DataFrame(
            {
                "match_id": df.index,
                "date": df["date"].values,
                "team": df["away_team"].values,
                "gf": df["away_goals"].values,
                "ga": df["home_goals"].values,
                "venue": "away",
            }
        )
        long = pd.concat([home, away], ignore_index=True)
        long["points"] = np.select(
            [long["gf"] > long["ga"], long["gf"] == long["ga"]], [3, 1], default=0
        )
        long = long.sort_values(["team", "date"]).reset_index(drop=True)

        grp = long.groupby("team", group_keys=False)
        for w in self.rolling_windows:
            # shift(1) => exclude the current match (no leakage).
            long[f"form_pts_{w}"] = grp["points"].apply(
                lambda s: s.shift(1).rolling(w, min_periods=1).mean()
            )
            long[f"form_gf_{w}"] = grp["gf"].apply(
                lambda s: s.shift(1).rolling(w, min_periods=1).mean()
            )
            long[f"form_ga_{w}"] = grp["ga"].apply(
                lambda s: s.shift(1).rolling(w, min_periods=1).mean()
            )
        # Days of rest since previous match (fatigue proxy).
        long["days_rest"] = grp["date"].apply(lambda s: s.diff().dt.days).fillna(7)

        feat_cols = [c for c in long.columns if c.startswith("form_")] + ["days_rest"]

        df = df.copy()
        for venue, prefix in (("home", "h"), ("away", "a")):
            sub = long[long["venue"] == venue].set_index("match_id")[feat_cols]
            sub = sub.rename(columns={c: f"{prefix}_{c}" for c in feat_cols})
            df = df.join(sub)

        # Differential features (home minus away) — usually the most predictive.
        for w in self.rolling_windows:
            df[f"diff_form_pts_{w}"] = df[f"h_form_pts_{w}"] - df[f"a_form_pts_{w}"]
            df[f"diff_form_gf_{w}"] = df[f"h_form_gf_{w}"] - df[f"a_form_gf_{w}"]
            df[f"diff_form_ga_{w}"] = df[f"h_form_ga_{w}"] - df[f"a_form_ga_{w}"]
        return df

    # --------------------------------------------------------------- market
    def _add_market(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        cols = ["odds_home", "odds_draw", "odds_away"]
        if not set(cols).issubset(df.columns):
            logger.warning("Odds columns missing; skipping market features.")
            return df
        valid = df[cols].notna().all(axis=1)
        probs = np.full((len(df), 3), np.nan)
        odds_arr = df.loc[valid, cols].to_numpy(dtype=float)
        if len(odds_arr):
            probs[valid.to_numpy()] = devig(odds_arr, method=self.devig_method)
        df["mkt_prob_home"] = probs[:, 0]
        df["mkt_prob_draw"] = probs[:, 1]
        df["mkt_prob_away"] = probs[:, 2]
        return df

    # ----------------------------------------------------------------- main
    def build(self, matches: pd.DataFrame) -> pd.DataFrame:
        """Return ``matches`` augmented with engineered feature columns."""
        if matches.empty:
            raise ValueError("Cannot build features from an empty match frame.")
        df = matches.sort_values("date").reset_index(drop=True)
        df = self._add_elo(df)
        df = self._add_rolling(df)
        if self.include_market_features:
            df = self._add_market(df)

        # Record the engineered feature columns (exclude identifiers/targets).
        non_features = {
            "date", "season", "division", "home_team", "away_team",
            "home_goals", "away_goals", "result",
            "odds_home", "odds_draw", "odds_away",
        }
        self.feature_columns_ = [
            c for c in df.columns
            if c not in non_features and pd.api.types.is_numeric_dtype(df[c])
        ]
        logger.info("Built %d feature columns", len(self.feature_columns_))
        return df

    def feature_matrix(self, featured: pd.DataFrame) -> pd.DataFrame:
        """Extract just the numeric feature columns (NaNs filled with 0)."""
        if not self.feature_columns_:
            raise RuntimeError("Call build() before feature_matrix().")
        return featured[self.feature_columns_].fillna(0.0)
