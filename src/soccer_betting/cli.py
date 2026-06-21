"""Command-line entry points.

These thin wrappers wire the library modules together for the four core
workflows. They are exposed as console scripts (see ``setup.py``) and are also
called by the convenience scripts in ``scripts/``.

    sb-collect   download + standardise historical data
    sb-train     fit models and report forecast quality
    sb-backtest  run a walk-forward betting simulation
    sb-value     list current value bets from a predictions file
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from soccer_betting.config import load_config
from soccer_betting.utils.logging import get_logger

logger = get_logger("soccer_betting.cli")


# --------------------------------------------------------------------------- #
def _load_or_synth(cfg, use_synthetic: bool) -> pd.DataFrame:
    from soccer_betting.data.preprocess import make_synthetic_matches
    processed = cfg.path("paths.data_processed") / "matches.parquet"
    if use_synthetic:
        logger.info("Generating synthetic dataset.")
        return make_synthetic_matches(n_teams=16, n_seasons=4)
    if processed.exists():
        return pd.read_parquet(processed)
    raise FileNotFoundError(
        f"No processed data at {processed}. Run `sb-collect` first or use --synthetic."
    )


# --------------------------------------------------------------------------- #
def collect_main(argv=None):
    parser = argparse.ArgumentParser(description="Collect & standardise soccer data.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--force", action="store_true", help="Re-download cached files.")
    parser.add_argument("--synthetic", action="store_true", help="Generate synthetic data instead.")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)

    out = cfg.ensure_dir("paths.data_processed")
    if args.synthetic:
        from soccer_betting.data.preprocess import make_synthetic_matches
        matches = make_synthetic_matches(n_teams=16, n_seasons=4)
    else:
        from soccer_betting.data.collectors import FootballDataUKCollector
        from soccer_betting.data.preprocess import standardise_matches
        collector = FootballDataUKCollector(out_dir=cfg.ensure_dir("paths.data_raw"))
        paths = collector.download(
            seasons=cfg.get("data.seasons", []),
            divisions=cfg.get("data.divisions", []),
            force=args.force,
        )
        raw = collector.read_many(paths)
        matches = standardise_matches(raw)

    dest = out / "matches.parquet"
    matches.to_parquet(dest, index=False)
    logger.info("Saved %d matches to %s", len(matches), dest)
    print(f"Collected {len(matches)} matches -> {dest}")


# --------------------------------------------------------------------------- #
def train_main(argv=None):
    parser = argparse.ArgumentParser(description="Fit models and report forecast quality.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--model", default="dixon_coles",
                        choices=["poisson", "dixon_coles", "bivariate_poisson", "elo", "ml"])
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    matches = _load_or_synth(cfg, args.synthetic)

    matches = matches.sort_values("date").reset_index(drop=True)
    split = int(len(matches) * (1 - args.test_size))
    train, test = matches.iloc[:split], matches.iloc[split:]

    from soccer_betting.evaluation.metrics import evaluate_predictions

    if args.model == "ml":
        from soccer_betting.data.features import FeatureBuilder
        from soccer_betting.models.ml_models import MLResultModel
        fb = FeatureBuilder(
            rolling_windows=cfg.get("features.rolling_windows", [3, 5, 10]),
            elo_params=cfg.get("features.elo", {}),
            include_market_features=cfg.get("features.include_market_features", True),
            devig_method=cfg.get("devig.method", "multiplicative"),
        )
        featured_all = fb.build(matches)
        X_all = fb.feature_matrix(featured_all)
        y_all = featured_all["result"].to_numpy()
        model = MLResultModel(calibration=cfg.get("ml.calibration.method", "isotonic"),
                              params=cfg.get("ml.xgboost", {}))
        model.fit(X_all.iloc[:split], y_all[:split])
        probs = model.predict_proba(X_all.iloc[split:])
        metrics = evaluate_predictions(probs, y_all[split:])
    else:
        model = _build_stat_model(args.model, cfg).fit(train)
        preds = model.predict_frame(test)
        probs = preds[["prob_home", "prob_draw", "prob_away"]].to_numpy()
        metrics = evaluate_predictions(probs, test["result"].to_numpy())

    print(f"\n=== {args.model} forecast quality (n={metrics.get('n')}) ===")
    for k in ("rps", "log_loss", "brier", "accuracy"):
        print(f"{k:>10}: {metrics[k]:.4f}")


def _build_stat_model(name: str, cfg):
    from soccer_betting.models import (
        PoissonModel, DixonColesModel, BivariatePoissonModel, EloModel,
    )
    if name == "poisson":
        return PoissonModel(max_goals=cfg.get("models.poisson.max_goals", 10))
    if name == "dixon_coles":
        return DixonColesModel(max_goals=cfg.get("models.dixon_coles.max_goals", 10),
                               xi=cfg.get("models.dixon_coles.xi", 0.0))
    if name == "bivariate_poisson":
        return BivariatePoissonModel(max_goals=10)
    if name == "elo":
        return EloModel(**(cfg.get("models.elo", {}) or {}))
    raise ValueError(name)


# --------------------------------------------------------------------------- #
def backtest_main(argv=None):
    parser = argparse.ArgumentParser(description="Run a walk-forward betting backtest.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--model", default="dixon_coles",
                        choices=["poisson", "dixon_coles", "bivariate_poisson", "elo", "ml"])
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    matches = _load_or_synth(cfg, args.synthetic)

    from soccer_betting.backtest.engine import (
        Backtester, BacktestConfig, StatisticalPredictor, MLPredictor,
    )
    bt_cfg = BacktestConfig(
        initial_bankroll=cfg.get("backtest.initial_bankroll", 1000.0),
        staking=cfg.get("backtest.staking", "fractional_kelly"),
        kelly_fraction=cfg.get("backtest.kelly_fraction", 0.25),
        flat_stake=cfg.get("backtest.flat_stake", 10.0),
        min_edge=cfg.get("backtest.min_edge", 0.02),
        max_stake_fraction=cfg.get("backtest.max_stake_fraction", 0.05),
        commission=cfg.get("backtest.commission", 0.0),
        devig_method=cfg.get("devig.method", "multiplicative"),
    )

    if args.model == "ml":
        from soccer_betting.data.features import FeatureBuilder
        from soccer_betting.models.ml_models import MLResultModel
        predictor = MLPredictor(
            model_factory=lambda: MLResultModel(
                calibration=cfg.get("ml.calibration.method", "isotonic"),
                params=cfg.get("ml.xgboost", {})),
            feature_builder_factory=lambda: FeatureBuilder(
                rolling_windows=cfg.get("features.rolling_windows", [3, 5, 10]),
                elo_params=cfg.get("features.elo", {}),
                include_market_features=cfg.get("features.include_market_features", True),
                devig_method=cfg.get("devig.method", "multiplicative")),
        )
    else:
        predictor = StatisticalPredictor(lambda: _build_stat_model(args.model, cfg))

    result = Backtester(bt_cfg).run(matches, predictor)
    print(result.summary())

    out = cfg.ensure_dir("paths.data_processed")
    result.bet_log.to_csv(out / f"bets_{args.model}.csv", index=False)
    print(f"\nBet log saved to {out / f'bets_{args.model}.csv'}")


# --------------------------------------------------------------------------- #
def value_main(argv=None):
    parser = argparse.ArgumentParser(description="List value bets from a predictions CSV.")
    parser.add_argument("predictions", help="CSV with prob_* and odds_* columns.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--min-edge", type=float, default=None)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)

    from soccer_betting.evaluation.value import find_value_bets
    preds = pd.read_csv(args.predictions)
    min_edge = args.min_edge if args.min_edge is not None else cfg.get("backtest.min_edge", 0.02)
    bets = find_value_bets(preds, min_edge=min_edge, devig_method=cfg.get("devig.method", "multiplicative"))
    if bets.empty:
        print("No value bets found.")
    else:
        print(bets.to_string(index=False))


if __name__ == "__main__":  # pragma: no cover
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "train"
    {"collect": collect_main, "train": train_main,
     "backtest": backtest_main, "value": value_main}[cmd](sys.argv[2:])
