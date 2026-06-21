"""Backtesting: walk-forward simulation, Kelly staking and performance metrics."""
from soccer_betting.backtest.kelly import kelly_fraction, stake_size
from soccer_betting.backtest.metrics import betting_performance, closing_line_value
from soccer_betting.backtest.engine import Backtester, BacktestConfig, BacktestResult

__all__ = [
    "kelly_fraction",
    "stake_size",
    "betting_performance",
    "closing_line_value",
    "Backtester",
    "BacktestConfig",
    "BacktestResult",
]
