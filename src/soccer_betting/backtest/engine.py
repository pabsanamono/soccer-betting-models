"""Walk-forward backtesting engine.

Implements the disciplined backtest the research demands:

* **Walk-forward / out-of-sample only** — at each step the model is trained on
  matches strictly *before* the fixtures being predicted, eliminating leakage.
* **Periodic retraining** — the model is refit every ``retrain_every`` matches
  (or on each new prediction block) using all data available up to that point.
* **Value filtering + Kelly staking** — bets are placed only when the model
  edge clears ``min_edge``; stakes follow the configured (fractional) Kelly.
* **Honest accounting** — bankroll, drawdown, ROI and optional CLV are tracked.

The engine is model-agnostic: pass any ``predictor`` exposing
``fit_predict(train_matches, test_matches) -> DataFrame[prob_home,prob_draw,prob_away]``.
Convenience factories for the statistical and ML models are provided.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np
import pandas as pd

from soccer_betting.backtest.kelly import stake_size
from soccer_betting.backtest.metrics import betting_performance, closing_line_value
from soccer_betting.evaluation.metrics import evaluate_predictions
from soccer_betting.utils.logging import get_logger

logger = get_logger(__name__)

_OUTCOME_TO_RESULT = {"home": "H", "draw": "D", "away": "A"}
_OUTCOME_COLS = {
    "home": ("prob_home", "odds_home"),
    "draw": ("prob_draw", "odds_draw"),
    "away": ("prob_away", "odds_away"),
}


@dataclass
class BacktestConfig:
    """Backtest hyper-parameters."""
    initial_bankroll: float = 1000.0
    staking: str = "fractional_kelly"      # flat | kelly | fractional_kelly
    kelly_fraction: float = 0.25
    flat_stake: float = 10.0
    min_edge: float = 0.02
    max_stake_fraction: float = 0.05
    commission: float = 0.0                # exchange commission on net winnings
    devig_method: str = "multiplicative"
    train_min_matches: int = 200           # warm-up before betting starts
    retrain_every: int = 50                # refit cadence (matches)
    stop_on_ruin: bool = True


@dataclass
class BacktestResult:
    """Container for backtest outputs."""
    bet_log: pd.DataFrame
    predictions: pd.DataFrame
    equity_curve: pd.Series
    performance: dict = field(default_factory=dict)
    forecast_metrics: dict = field(default_factory=dict)

    def summary(self) -> str:
        lines = ["=== Backtest summary ==="]
        for k, v in self.performance.items():
            lines.append(f"{k:>18}: {v:.4f}" if isinstance(v, float) else f"{k:>18}: {v}")
        lines.append("--- Forecast quality (all predicted matches) ---")
        for k, v in self.forecast_metrics.items():
            lines.append(f"{k:>18}: {v:.4f}" if isinstance(v, float) else f"{k:>18}: {v}")
        return "\n".join(lines)


class Backtester:
    """Run a walk-forward simulation given a retrainable predictor."""

    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()

    def run(
        self,
        matches: pd.DataFrame,
        predictor: "Predictor",
    ) -> BacktestResult:
        cfg = self.config
        required = {"date", "home_team", "away_team", "result",
                    "odds_home", "odds_draw", "odds_away"}
        missing = required - set(matches.columns)
        if missing:
            raise ValueError(f"matches missing columns for backtest: {sorted(missing)}")

        df = matches.sort_values("date").reset_index(drop=True)
        n = len(df)
        if n <= cfg.train_min_matches:
            raise ValueError(
                f"Need more than train_min_matches ({cfg.train_min_matches}) rows; got {n}."
            )

        bankroll = cfg.initial_bankroll
        bet_records: List[dict] = []
        pred_records: List[dict] = []
        equity_points = [(df.loc[cfg.train_min_matches - 1, "date"], bankroll)]

        start = cfg.train_min_matches
        last_fit_size = -1
        cached_predictor = None

        for block_start in range(start, n, cfg.retrain_every):
            block_end = min(block_start + cfg.retrain_every, n)
            train = df.iloc[:block_start]
            test = df.iloc[block_start:block_end]

            # Refit on all data available so far (walk-forward).
            if last_fit_size != len(train):
                cached_predictor = predictor.fit(train)
                last_fit_size = len(train)
            probs = cached_predictor.predict(test)

            for local_i, (idx, row) in enumerate(test.iterrows()):
                p = probs.iloc[local_i]
                ph, pd_, pa = p["prob_home"], p["prob_draw"], p["prob_away"]
                pred_records.append({
                    "date": row["date"], "home_team": row["home_team"],
                    "away_team": row["away_team"], "result": row["result"],
                    "prob_home": ph, "prob_draw": pd_, "prob_away": pa,
                })
                if np.isnan([ph, pd_, pa]).any():
                    continue

                # Evaluate each outcome for value and place at most the best bet.
                best = self._best_value_bet(row, {"home": ph, "draw": pd_, "away": pa})
                if best is None:
                    continue

                outcome, prob, odds = best
                stake = stake_size(
                    prob=prob, odds=odds, bankroll=bankroll,
                    method=cfg.staking, kelly_fraction_mult=cfg.kelly_fraction,
                    flat_stake=cfg.flat_stake, max_stake_fraction=cfg.max_stake_fraction,
                    min_edge=cfg.min_edge,
                )
                if stake <= 0:
                    continue

                won = row["result"] == _OUTCOME_TO_RESULT[outcome]
                if won:
                    gross = stake * (odds - 1.0)
                    profit = gross * (1.0 - cfg.commission)
                else:
                    profit = -stake
                bankroll += profit

                bet_records.append({
                    "date": row["date"], "home_team": row["home_team"],
                    "away_team": row["away_team"], "selection": outcome,
                    "odds": odds, "model_prob": prob,
                    "edge": prob * odds - 1.0, "stake": stake,
                    "won": bool(won), "profit": profit,
                    "bankroll_after": bankroll,
                })
                equity_points.append((row["date"], bankroll))

                if cfg.stop_on_ruin and bankroll <= 0:
                    logger.warning("Bankroll exhausted at %s — stopping.", row["date"])
                    break
            if cfg.stop_on_ruin and bankroll <= 0:
                break

        bet_log = pd.DataFrame(bet_records)
        predictions = pd.DataFrame(pred_records)
        eq = pd.Series(
            [v for _, v in equity_points],
            index=pd.to_datetime([d for d, _ in equity_points]),
            name="bankroll",
        )

        perf = betting_performance(bet_log, initial_bankroll=cfg.initial_bankroll)
        # Forecast quality across every predicted match (not just bets).
        if not predictions.empty:
            prob_arr = predictions[["prob_home", "prob_draw", "prob_away"]].to_numpy()
            fmetrics = evaluate_predictions(prob_arr, predictions["result"].to_numpy())
        else:
            fmetrics = {}

        result = BacktestResult(
            bet_log=bet_log, predictions=predictions, equity_curve=eq,
            performance=perf, forecast_metrics=fmetrics,
        )
        logger.info("Backtest done: %d bets, ROI=%.3f, final bankroll=%.2f",
                    perf.get("n_bets", 0), perf.get("roi", float("nan")), bankroll)
        return result

    def _best_value_bet(self, row, probs: dict):
        """Pick the single highest-edge outcome that clears min_edge."""
        best = None
        best_edge = self.config.min_edge
        for outcome, prob in probs.items():
            _, ocol = _OUTCOME_COLS[outcome]
            odds = row[ocol]
            if pd.isna(odds) or odds <= 1.0:
                continue
            edge = prob * odds - 1.0
            if edge >= best_edge:
                best_edge = edge
                best = (outcome, prob, float(odds))
        return best


# --------------------------------------------------------------------------- #
# Predictor adapters
# --------------------------------------------------------------------------- #
class Predictor:
    """Interface: fit(train_matches) -> self; predict(test_matches) -> probs df."""

    def fit(self, train_matches: pd.DataFrame) -> "Predictor":  # pragma: no cover
        raise NotImplementedError

    def predict(self, test_matches: pd.DataFrame) -> pd.DataFrame:  # pragma: no cover
        raise NotImplementedError


class StatisticalPredictor(Predictor):
    """Wrap a team-based statistical model (Poisson, Dixon-Coles, Elo, ...).

    Parameters
    ----------
    model_factory:
        Zero-arg callable returning a fresh, unfitted model each retrain.
    """

    def __init__(self, model_factory: Callable[[], object]):
        self.model_factory = model_factory
        self.model_ = None

    def fit(self, train_matches: pd.DataFrame) -> "StatisticalPredictor":
        self.model_ = self.model_factory()
        self.model_.fit(train_matches)
        return self

    def predict(self, test_matches: pd.DataFrame) -> pd.DataFrame:
        return self.model_.predict_frame(test_matches).reset_index(drop=True)


class MLPredictor(Predictor):
    """Wrap a feature-based ML model with a feature builder.

    The feature builder is rebuilt over the *combined* train+test span at each
    block so rolling features for the test rows can use the latest history,
    while the target labels for test rows are never seen by the model.
    """

    def __init__(self, model_factory: Callable[[], object], feature_builder_factory: Callable[[], object]):
        self.model_factory = model_factory
        self.feature_builder_factory = feature_builder_factory
        self.model_ = None
        self._train_len = 0
        self._featured = None
        self._fb = None

    def fit(self, train_matches: pd.DataFrame) -> "MLPredictor":
        self._train = train_matches.reset_index(drop=True)
        self._fb = self.feature_builder_factory()
        featured = self._fb.build(self._train)
        X = self._fb.feature_matrix(featured)
        y = featured["result"].to_numpy()
        self.model_ = self.model_factory()
        self.model_.fit(X, y)
        return self

    def predict(self, test_matches: pd.DataFrame) -> pd.DataFrame:
        # Combine history + test to compute leakage-free rolling features.
        combined = pd.concat([self._train, test_matches], ignore_index=True)
        featured = self._fb.build(combined)
        X_all = self._fb.feature_matrix(featured)
        X_test = X_all.iloc[len(self._train):]
        probs = self.model_.predict_proba(X_test)
        return pd.DataFrame(
            {"prob_home": probs[:, 0], "prob_draw": probs[:, 1], "prob_away": probs[:, 2]}
        ).reset_index(drop=True)
