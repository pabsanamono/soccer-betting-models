"""Integration tests: features, ML model, value bets and backtest."""
import numpy as np

from soccer_betting.data.features import FeatureBuilder
from soccer_betting.models.ml_models import MLResultModel, EnsembleModel
from soccer_betting.evaluation.value import find_value_bets
from soccer_betting.backtest.engine import (
    Backtester, BacktestConfig, StatisticalPredictor,
)
from soccer_betting.models import DixonColesModel


def test_feature_builder_no_leakage(synthetic_matches):
    fb = FeatureBuilder()
    featured = fb.build(synthetic_matches)
    # First match of each team has no prior history -> rolling features NaN/0.
    assert len(fb.feature_columns_) > 0
    X = fb.feature_matrix(featured)
    assert X.shape[0] == len(synthetic_matches)
    assert not np.isinf(X.to_numpy()).any()


def test_ml_model_trains_and_predicts(synthetic_matches):
    fb = FeatureBuilder()
    featured = fb.build(synthetic_matches)
    X = fb.feature_matrix(featured)
    y = featured["result"].to_numpy()
    split = int(len(X) * 0.8)
    model = MLResultModel(calibration="isotonic").fit(X.iloc[:split], y[:split])
    probs = model.predict_proba(X.iloc[split:])
    assert probs.shape == (len(X) - split, 3)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)


def test_ensemble_average(synthetic_matches):
    fb = FeatureBuilder()
    featured = fb.build(synthetic_matches)
    X = fb.feature_matrix(featured)
    y = featured["result"].to_numpy()
    ens = EnsembleModel([
        MLResultModel(calibration="none"),
        MLResultModel(calibration="sigmoid"),
    ]).fit(X.iloc[:400], y[:400])
    probs = ens.predict_proba(X.iloc[400:420])
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)


def test_find_value_bets(synthetic_matches):
    train = synthetic_matches.iloc[:500]
    test = synthetic_matches.iloc[500:].reset_index(drop=True)
    dc = DixonColesModel().fit(train)
    preds = dc.predict_frame(test).reset_index(drop=True)
    import pandas as pd
    frame = pd.concat([test[["date", "home_team", "away_team",
                             "odds_home", "odds_draw", "odds_away"]], preds], axis=1)
    bets = find_value_bets(frame, min_edge=0.02, devig_method="multiplicative")
    if not bets.empty:
        assert (bets["edge"] >= 0.02).all()
        assert set(bets["selection"]).issubset({"home", "draw", "away"})


def test_backtest_runs(synthetic_matches):
    bt = Backtester(BacktestConfig(train_min_matches=300, retrain_every=80, min_edge=0.03))
    result = bt.run(synthetic_matches, StatisticalPredictor(lambda: DixonColesModel()))
    assert "roi" in result.performance
    assert len(result.predictions) > 0
    assert result.equity_curve.iloc[0] == 1000.0
