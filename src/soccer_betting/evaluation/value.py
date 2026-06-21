"""Value-bet identification: model probabilities vs bookmaker prices.

Profitability comes from the **edge** — the gap between the model's estimated
probability and the market-implied probability. For each outcome:

    edge      = model_prob * decimal_odds - 1     (expected return per unit)
    value     = model_prob - market_implied_prob

A bet is flagged as value when ``edge >= min_edge``. The market-implied
probabilities used for the value comparison are devigged so the comparison is
against the bookmaker's *fair* estimate, not the margin-inflated price — while
the *edge* (EV) deliberately uses the raw offered odds, because that is what you
actually get paid.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from soccer_betting.odds.devig import devig, odds_to_prob

_OUTCOME_COLS = {
    "home": ("prob_home", "odds_home"),
    "draw": ("prob_draw", "odds_draw"),
    "away": ("prob_away", "odds_away"),
}


def compare_to_market(
    predictions: pd.DataFrame,
    devig_method: str = "multiplicative",
) -> pd.DataFrame:
    """Attach market-implied probabilities, edges and value to predictions.

    Parameters
    ----------
    predictions:
        Frame with ``prob_home/prob_draw/prob_away`` and
        ``odds_home/odds_draw/odds_away`` columns.
    devig_method:
        Method used to compute fair market probabilities.

    Returns
    -------
    pd.DataFrame
        Copy of ``predictions`` with added ``mkt_prob_*``, ``edge_*`` and
        ``value_*`` columns for each outcome.
    """
    df = predictions.copy()
    odds_cols = ["odds_home", "odds_draw", "odds_away"]
    prob_cols = ["prob_home", "prob_draw", "prob_away"]
    for c in odds_cols + prob_cols:
        if c not in df.columns:
            raise ValueError(f"predictions missing required column '{c}'")

    valid = df[odds_cols].notna().all(axis=1) & df[prob_cols].notna().all(axis=1)
    mkt = np.full((len(df), 3), np.nan)
    if valid.any():
        mkt[valid.to_numpy()] = devig(df.loc[valid, odds_cols].to_numpy(float), method=devig_method)
    df["mkt_prob_home"], df["mkt_prob_draw"], df["mkt_prob_away"] = mkt.T

    for outcome, (pcol, ocol) in _OUTCOME_COLS.items():
        df[f"edge_{outcome}"] = df[pcol] * df[ocol] - 1.0
        df[f"value_{outcome}"] = df[pcol] - df[f"mkt_prob_{outcome}"]
    return df


def find_value_bets(
    predictions: pd.DataFrame,
    min_edge: float = 0.02,
    devig_method: str = "multiplicative",
    max_market_prob: Optional[float] = None,
) -> pd.DataFrame:
    """Return a tidy table of one row per value bet (selection-level).

    Parameters
    ----------
    predictions:
        As accepted by :func:`compare_to_market`. May also carry identifier
        columns (date, home_team, away_team) which are preserved.
    min_edge:
        Minimum expected return per unit stake to qualify as value.
    max_market_prob:
        Optional ceiling on the market probability of the selection — a simple
        way to avoid heavy favourites where edges are usually illusory.
    """
    df = compare_to_market(predictions, devig_method=devig_method)
    id_cols = [c for c in ("date", "home_team", "away_team", "season", "division") if c in df.columns]

    records = []
    for outcome, (pcol, ocol) in _OUTCOME_COLS.items():
        sub = df[df[f"edge_{outcome}"] >= min_edge].copy()
        if max_market_prob is not None:
            sub = sub[sub[f"mkt_prob_{outcome}"] <= max_market_prob]
        for r in sub.itertuples(index=True):
            rec = {c: getattr(r, c) for c in id_cols}
            rec.update(
                {
                    "index": r.Index,
                    "selection": outcome,
                    "model_prob": getattr(r, pcol),
                    "market_prob": getattr(r, f"mkt_prob_{outcome}"),
                    "odds": getattr(r, ocol),
                    "edge": getattr(r, f"edge_{outcome}"),
                    "value": getattr(r, f"value_{outcome}"),
                }
            )
            records.append(rec)
    out = pd.DataFrame(records)
    if not out.empty:
        out = out.sort_values("edge", ascending=False).reset_index(drop=True)
    return out
