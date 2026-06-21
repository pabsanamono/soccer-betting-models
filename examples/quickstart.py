#!/usr/bin/env python
"""End-to-end quickstart on synthetic data (no network required).

Runs the whole pipeline:

1. Generate a synthetic dataset (results + bookmaker odds).
2. Fit and compare the statistical models (Poisson, Dixon-Coles, Bivariate
   Poisson, Elo) and a calibrated ML model using proper scoring rules.
3. Devig the market and identify value bets.
4. Run a walk-forward Kelly-staked backtest and print performance + CLV-style
   summary.

Run with:  python examples/quickstart.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from soccer_betting.data.preprocess import make_synthetic_matches
from soccer_betting.data.features import FeatureBuilder
from soccer_betting.models import (
    PoissonModel, DixonColesModel, BivariatePoissonModel, EloModel,
)
from soccer_betting.models.ml_models import MLResultModel
from soccer_betting.evaluation.metrics import evaluate_predictions
from soccer_betting.evaluation.value import find_value_bets
from soccer_betting.backtest.engine import (
    Backtester, BacktestConfig, StatisticalPredictor,
)


def main():
    pd.set_option("display.width", 120)
    print("1) Generating synthetic data ...")
    matches = make_synthetic_matches(n_teams=16, n_seasons=4, seed=7)
    print(f"   {len(matches)} matches, {matches['home_team'].nunique()} teams")

    split = int(len(matches) * 0.8)
    train, test = matches.iloc[:split], matches.iloc[split:]

    print("\n2) Comparing models with proper scoring rules (lower RPS/log-loss/Brier = better):")
    stat_models = {
        "Poisson": PoissonModel(),
        "DixonColes": DixonColesModel(xi=0.0018),
        "BivariatePoisson": BivariatePoissonModel(),
        "Elo": EloModel(),
    }
    rows = []
    for name, model in stat_models.items():
        model.fit(train)
        preds = model.predict_frame(test)
        probs = preds[["prob_home", "prob_draw", "prob_away"]].to_numpy()
        m = evaluate_predictions(probs, test["result"].to_numpy())
        rows.append({"model": name, **{k: m[k] for k in ("rps", "log_loss", "brier", "accuracy")}})

    # Calibrated ML model on engineered features.
    fb = FeatureBuilder(include_market_features=True)
    featured = fb.build(matches)
    X = fb.feature_matrix(featured)
    y = featured["result"].to_numpy()
    ml = MLResultModel(calibration="isotonic").fit(X.iloc[:split], y[:split])
    ml_probs = ml.predict_proba(X.iloc[split:])
    m = evaluate_predictions(ml_probs, y[split:])
    rows.append({"model": "ML(GBM+cal)", **{k: m[k] for k in ("rps", "log_loss", "brier", "accuracy")}})

    # Market baseline (devigged odds) for reference.
    from soccer_betting.odds.devig import devig
    mkt = devig(test[["odds_home", "odds_draw", "odds_away"]].to_numpy(float), method="shin")
    m = evaluate_predictions(mkt, test["result"].to_numpy())
    rows.append({"model": "Market(devig)", **{k: m[k] for k in ("rps", "log_loss", "brier", "accuracy")}})

    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print("\n3) Identifying value bets (Dixon-Coles vs market) ...")
    dc = DixonColesModel(xi=0.0018).fit(train)
    preds = dc.predict_frame(test).reset_index(drop=True)
    pred_frame = pd.concat(
        [test.reset_index(drop=True)[["date", "home_team", "away_team",
                                      "odds_home", "odds_draw", "odds_away"]],
         preds], axis=1)
    value_bets = find_value_bets(pred_frame, min_edge=0.05, devig_method="shin")
    print(f"   Found {len(value_bets)} value selections (edge >= 5%). Top 5:")
    if not value_bets.empty:
        print(value_bets.head(5).to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    print("\n4) Walk-forward backtest (quarter-Kelly, Dixon-Coles) ...")
    bt = Backtester(BacktestConfig(
        initial_bankroll=1000.0, staking="fractional_kelly",
        kelly_fraction=0.25, min_edge=0.03, train_min_matches=300, retrain_every=60,
    ))
    result = bt.run(matches, StatisticalPredictor(lambda: DixonColesModel(xi=0.0018)))
    print(result.summary())


if __name__ == "__main__":
    main()
