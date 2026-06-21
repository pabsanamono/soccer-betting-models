"""Soccer betting modeling toolkit.

A production-oriented framework for modeling soccer match probabilities and
identifying value against bookmaker prices. It implements:

* Data collection & feature engineering (``soccer_betting.data``)
* Statistical scoreline models — Poisson, Dixon-Coles, Bivariate Poisson, Elo
  (``soccer_betting.models``)
* Odds devigging (``soccer_betting.odds``)
* A machine-learning framework with probability calibration
  (``soccer_betting.models.ml_models`` and ``soccer_betting.calibration``)
* A walk-forward backtesting engine with Kelly staking
  (``soccer_betting.backtest``)
* Proper-scoring evaluation and value-bet identification
  (``soccer_betting.evaluation``)
"""

__version__ = "0.1.0"

from soccer_betting.config import Config, load_config  # noqa: E402

__all__ = ["Config", "load_config", "__version__"]
