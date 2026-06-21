"""Match-probability models.

Statistical scoreline models:
    * :class:`PoissonModel`            — independent Poisson (Maher, 1982)
    * :class:`DixonColesModel`         — rho correction + time decay (1997)
    * :class:`BivariatePoissonModel`   — explicit correlation (Karlis-Ntzoufras)

Rating model:
    * :class:`EloModel`                — Elo-based 1X2 probabilities

Machine learning:
    * :class:`MLResultModel`           — calibrated classifier (XGBoost/sklearn)
    * :class:`EnsembleModel`           — soft-voting ensemble
"""
from soccer_betting.models.base import BaseMatchModel, OUTCOMES
from soccer_betting.models.poisson import PoissonModel
from soccer_betting.models.dixon_coles import DixonColesModel
from soccer_betting.models.bivariate_poisson import BivariatePoissonModel
from soccer_betting.models.elo import EloModel

__all__ = [
    "BaseMatchModel",
    "OUTCOMES",
    "PoissonModel",
    "DixonColesModel",
    "BivariatePoissonModel",
    "EloModel",
]

# ML models are imported lazily to avoid importing heavy deps unless needed.
try:  # pragma: no cover - optional import
    from soccer_betting.models.ml_models import MLResultModel, EnsembleModel  # noqa: F401

    __all__ += ["MLResultModel", "EnsembleModel"]
except Exception:  # pragma: no cover
    pass
