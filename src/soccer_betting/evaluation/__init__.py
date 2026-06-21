"""Evaluation: proper scoring rules and value-bet identification."""
from soccer_betting.evaluation.metrics import (
    ranked_probability_score,
    multiclass_log_loss,
    multiclass_brier_score,
    accuracy,
    evaluate_predictions,
)
from soccer_betting.evaluation.value import (
    find_value_bets,
    compare_to_market,
)

__all__ = [
    "ranked_probability_score",
    "multiclass_log_loss",
    "multiclass_brier_score",
    "accuracy",
    "evaluate_predictions",
    "find_value_bets",
    "compare_to_market",
]
