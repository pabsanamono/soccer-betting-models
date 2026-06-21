import numpy as np

from soccer_betting.evaluation.metrics import (
    ranked_probability_score, multiclass_log_loss,
    multiclass_brier_score, evaluate_predictions,
)


def test_perfect_forecast_scores_zero():
    probs = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    labels = ["H", "A"]
    assert ranked_probability_score(probs, labels) == 0.0
    assert multiclass_brier_score(probs, labels) == 0.0
    assert multiclass_log_loss(probs, labels) < 1e-10


def test_rps_ordinal_sensitivity():
    # True outcome = Home. Putting mass on Draw (adjacent) should beat Away (far).
    near = np.array([[0.5, 0.5, 0.0]])
    far = np.array([[0.5, 0.0, 0.5]])
    assert ranked_probability_score(near, ["H"]) < ranked_probability_score(far, ["H"])


def test_evaluate_predictions_keys():
    probs = np.array([[0.4, 0.3, 0.3], [0.2, 0.3, 0.5]])
    out = evaluate_predictions(probs, ["H", "A"])
    for key in ("rps", "log_loss", "brier", "accuracy", "n"):
        assert key in out


def test_nan_rows_dropped():
    probs = np.array([[0.4, 0.3, 0.3], [np.nan, np.nan, np.nan]])
    out = evaluate_predictions(probs, ["H", "A"])
    assert out["n"] == 1
