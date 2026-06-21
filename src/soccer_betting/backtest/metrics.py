"""Betting performance metrics.

Given a settled bet log (one row per bet with stake, odds, won flag and the
profit), compute the metrics the research insists on: ROI/yield, total profit,
maximum drawdown, win rate, and Closing Line Value (CLV) when closing odds are
available. A paper-profitable strategy is useless if it cannot survive
drawdowns, so drawdown is reported alongside profit.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd


def max_drawdown(equity: np.ndarray) -> float:
    """Maximum peak-to-trough drop of an equity curve (absolute units)."""
    equity = np.asarray(equity, dtype=float)
    if len(equity) == 0:
        return 0.0
    running_max = np.maximum.accumulate(equity)
    drawdowns = running_max - equity
    return float(np.max(drawdowns))


def max_drawdown_pct(equity: np.ndarray) -> float:
    """Maximum drawdown as a fraction of the running peak."""
    equity = np.asarray(equity, dtype=float)
    if len(equity) == 0:
        return 0.0
    running_max = np.maximum.accumulate(equity)
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = np.where(running_max > 0, (running_max - equity) / running_max, 0.0)
    return float(np.max(dd))


def betting_performance(bet_log: pd.DataFrame, initial_bankroll: float = 0.0) -> Dict[str, float]:
    """Summarise a settled bet log.

    Required columns: ``stake``, ``profit`` (net win/loss for the bet),
    ``won`` (bool/int). Optional: ``bankroll_after`` for drawdown, ``odds``.
    """
    if bet_log is None or len(bet_log) == 0:
        return {
            "n_bets": 0, "total_staked": 0.0, "profit": 0.0, "roi": float("nan"),
            "yield": float("nan"), "win_rate": float("nan"),
            "avg_odds": float("nan"), "max_drawdown": 0.0, "max_drawdown_pct": 0.0,
        }

    total_staked = float(bet_log["stake"].sum())
    profit = float(bet_log["profit"].sum())
    n = int(len(bet_log))
    roi = profit / total_staked if total_staked > 0 else float("nan")

    if "bankroll_after" in bet_log.columns:
        equity = bet_log["bankroll_after"].to_numpy(dtype=float)
        if initial_bankroll:
            equity = np.concatenate([[initial_bankroll], equity])
    else:
        equity = initial_bankroll + np.cumsum(bet_log["profit"].to_numpy(dtype=float))

    return {
        "n_bets": n,
        "total_staked": total_staked,
        "profit": profit,
        "roi": roi,            # profit / turnover (a.k.a. yield)
        "yield": roi,
        "win_rate": float(bet_log["won"].mean()) if "won" in bet_log else float("nan"),
        "avg_odds": float(bet_log["odds"].mean()) if "odds" in bet_log else float("nan"),
        "max_drawdown": max_drawdown(equity),
        "max_drawdown_pct": max_drawdown_pct(equity),
        "final_bankroll": float(equity[-1]) if len(equity) else float("nan"),
    }


def closing_line_value(
    taken_odds: np.ndarray, closing_odds: np.ndarray
) -> Optional[float]:
    """Mean Closing Line Value: how much better than the close you got.

    CLV per bet = taken_odds / closing_odds - 1. Positive mean CLV is the
    field's preferred indicator of long-term edge (process over luck).
    """
    taken = np.asarray(taken_odds, dtype=float)
    closing = np.asarray(closing_odds, dtype=float)
    mask = (~np.isnan(taken)) & (~np.isnan(closing)) & (closing > 1.0)
    if mask.sum() == 0:
        return None
    return float(np.mean(taken[mask] / closing[mask] - 1.0))
