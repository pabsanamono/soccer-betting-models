#!/usr/bin/env python
"""Train the models on World Cup 2026 data and predict upcoming fixtures.

Run it after fetching the data:

    python scripts/fetch_worldcup_data.py     # do this first
    python scripts/predict_worldcup.py        # then this

What it does
------------
1. Loads the CSVs produced by ``fetch_worldcup_data.py``.
2. Trains the recommended Dixon-Coles model (and, for reference, compares a few
   other models on the historical data).
3. Predicts Home / Draw / Away probabilities for every upcoming fixture.
4. Compares those probabilities to the bookmaker odds and flags VALUE BETS.
5. Prints an easy-to-read table and saves the predictions + value bets to CSV.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from soccer_betting.models import (  # noqa: E402
    PoissonModel, DixonColesModel, BivariatePoissonModel, EloModel,
)
from soccer_betting.evaluation.metrics import evaluate_predictions  # noqa: E402
from soccer_betting.evaluation.value import find_value_bets  # noqa: E402
from soccer_betting.utils.logging import get_logger  # noqa: E402

logger = get_logger("predict_worldcup")

CONFIG_PATH = ROOT / "config" / "api_config.yaml"

_MODEL_REGISTRY = {
    "dixon_coles": lambda c: DixonColesModel(
        xi=float(c.get("dixon_coles", {}).get("xi", 0.0018)),
        max_goals=int(c.get("dixon_coles", {}).get("max_goals", 10)),
    ),
    "poisson": lambda c: PoissonModel(
        max_goals=int(c.get("poisson", {}).get("max_goals", 10)),
    ),
    "bivariate_poisson": lambda c: BivariatePoissonModel(),
    "elo": lambda c: EloModel(),
}


def load_api_config() -> dict:
    if not CONFIG_PATH.exists():
        return {}
    with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _print_header(text: str) -> None:
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


def _require_csv(path: Path, what: str) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(
            f"Could not find {what} at {path}.\n"
            f"Run 'python scripts/fetch_worldcup_data.py' first."
        )
    return pd.read_csv(path)


def attach_best_odds(fixtures: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
    """Ensure fixtures carry odds columns.

    If fixtures already have odds, keep them. Otherwise derive the best
    (highest = most bettor-friendly) odds per outcome from the long odds table.
    """
    fixtures = fixtures.copy()
    has_odds = all(c in fixtures.columns for c in ("odds_home", "odds_draw", "odds_away"))
    if has_odds and fixtures[["odds_home", "odds_draw", "odds_away"]].notna().any().any():
        return fixtures
    if odds is None or odds.empty:
        for c in ("odds_home", "odds_draw", "odds_away"):
            fixtures[c] = np.nan
        return fixtures
    best = (
        odds.groupby(["home_team", "away_team"], as_index=False)[
            ["odds_home", "odds_draw", "odds_away"]
        ].max()
    )
    return fixtures.merge(best, on=["home_team", "away_team"], how="left",
                          suffixes=("", "_book"))


def compare_models(historical: pd.DataFrame, model_names, model_cfg) -> None:
    """Print a quick scoring-rule comparison on a hold-out slice of history."""
    if len(historical) < 100 or "result" not in historical.columns:
        return
    split = int(len(historical) * 0.8)
    train, test = historical.iloc[:split], historical.iloc[split:]
    y = test["result"].to_numpy()
    rows = []
    print("\nComparing models (this can take a minute) ...")
    for name in model_names:
        builder = _MODEL_REGISTRY.get(name)
        if builder is None:
            continue
        try:
            print(f"  - fitting {name} ...", flush=True)
            model = builder(model_cfg).fit(train)
            preds = model.predict_frame(test)
            probs = preds[["prob_home", "prob_draw", "prob_away"]].to_numpy()
            mask = ~np.isnan(probs).any(axis=1)
            if mask.sum() == 0:
                continue
            m = evaluate_predictions(probs[mask], y[mask])
            rows.append({"model": name, **{k: m[k] for k in ("rps", "log_loss", "brier", "accuracy")}})
        except Exception as exc:  # pragma: no cover - robustness
            logger.warning("Model %s failed during comparison: %s", name, exc)
    if rows:
        _print_header("Model comparison on historical data (lower RPS/log-loss/Brier = better)")
        print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.4f}"))


def main() -> None:
    cfg = load_api_config()
    out_cfg = cfg.get("output", {}) or {}
    pred_cfg = cfg.get("prediction", {}) or {}
    vb_cfg = cfg.get("value_bets", {}) or {}

    data_dir = ROOT / out_cfg.get("data_dir", "data/worldcup")
    historical = _require_csv(data_dir / out_cfg.get("historical_file", "worldcup_historical.csv"),
                              "historical data")
    fixtures = _require_csv(data_dir / out_cfg.get("fixtures_file", "worldcup_fixtures.csv"),
                            "fixtures")
    odds_path = data_dir / out_cfg.get("odds_file", "worldcup_odds.csv")
    odds = pd.read_csv(odds_path) if odds_path.exists() else pd.DataFrame()

    if "date" in historical.columns:
        historical["date"] = pd.to_datetime(historical["date"], errors="coerce")
    historical = historical.sort_values("date").reset_index(drop=True)

    _print_header("World Cup 2026 — predictions")
    print(f"Historical matches loaded : {len(historical)}")
    print(f"Upcoming fixtures loaded  : {len(fixtures)}")

    # --- optional comparison table ----------------------------------------
    compare_models(historical, pred_cfg.get("compare_models", ["dixon_coles"]), pred_cfg)

    # --- train the headline model -----------------------------------------
    model_name = pred_cfg.get("model", "dixon_coles")
    builder = _MODEL_REGISTRY.get(model_name, _MODEL_REGISTRY["dixon_coles"])
    print(f"\nTraining the prediction model ({model_name}) on all historical data ...")
    logger.info("Training headline model: %s", model_name)
    model = builder(pred_cfg).fit(historical)

    # --- restrict fixtures to teams the model actually knows --------------
    known = set(getattr(model, "attack_", {}).keys()) or set(historical["home_team"]) | set(historical["away_team"])
    fixtures = attach_best_odds(fixtures, odds)
    mask = fixtures["home_team"].isin(known) & fixtures["away_team"].isin(known)
    skipped = (~mask).sum()
    fixtures = fixtures[mask].reset_index(drop=True)
    if skipped:
        logger.warning("Skipped %d fixtures with teams missing from training data.", skipped)
    if fixtures.empty:
        raise SystemExit("No predictable fixtures (teams not present in historical data).")

    # --- predict ----------------------------------------------------------
    preds = model.predict_frame(fixtures).reset_index(drop=True)
    id_cols = [c for c in ("date", "stage", "home_team", "away_team",
                           "odds_home", "odds_draw", "odds_away") if c in fixtures.columns]
    out = pd.concat([fixtures[id_cols].reset_index(drop=True), preds], axis=1)

    # Most likely outcome + its probability, for easy reading.
    prob_cols = ["prob_home", "prob_draw", "prob_away"]
    labels = np.array(["Home win", "Draw", "Away win"])
    out["prediction"] = labels[out[prob_cols].to_numpy().argmax(axis=1)]
    out["confidence"] = out[prob_cols].max(axis=1)

    # --- value bets -------------------------------------------------------
    value_bets = pd.DataFrame()
    if all(c in out.columns for c in ("odds_home", "odds_draw", "odds_away")) and \
            out[["odds_home", "odds_draw", "odds_away"]].notna().any().any():
        max_mp = vb_cfg.get("max_market_prob")
        value_bets = find_value_bets(
            out,
            min_edge=float(vb_cfg.get("min_edge", 0.05)),
            devig_method=vb_cfg.get("devig_method", "shin"),
            max_market_prob=float(max_mp) if max_mp is not None else None,
        )

    # --- display ----------------------------------------------------------
    _print_header(f"Predictions for {len(out)} upcoming fixtures  (model: {model_name})")
    display = out.copy()
    for c in prob_cols + ["confidence"]:
        display[c] = (display[c] * 100).round(1).astype(str) + "%"
    show_cols = [c for c in ("date", "stage", "home_team", "away_team",
                             "prob_home", "prob_draw", "prob_away",
                             "prediction", "confidence") if c in display.columns]
    with pd.option_context("display.width", 140, "display.max_rows", None, "display.max_columns", None):
        print(display[show_cols].rename(columns={
            "prob_home": "P(Home)", "prob_draw": "P(Draw)", "prob_away": "P(Away)",
        }).to_string(index=False))

    _print_header(f"Value bets (edge >= {float(vb_cfg.get('min_edge', 0.05)) * 100:.0f}%)")
    if value_bets.empty:
        print("No value bets found at the configured threshold.")
        print("Tip: lower 'min_edge' in config/api_config.yaml to see more selections.")
    else:
        vb = value_bets.copy()
        sel_label = {"home": "Home win", "draw": "Draw", "away": "Away win"}
        vb["selection"] = vb["selection"].map(sel_label).fillna(vb["selection"])
        for c in ("model_prob", "market_prob", "edge", "value"):
            if c in vb.columns:
                vb[c] = (vb[c] * 100).round(1).astype(str) + "%"
        vb_cols = [c for c in ("date", "home_team", "away_team", "selection",
                               "odds", "model_prob", "market_prob", "edge") if c in vb.columns]
        with pd.option_context("display.width", 140, "display.max_rows", None):
            print(vb[vb_cols].to_string(index=False))
        print(f"\nFound {len(vb)} value selection(s).")

    # --- save -------------------------------------------------------------
    pred_path = data_dir / out_cfg.get("predictions_file", "worldcup_predictions.csv")
    vb_path = data_dir / out_cfg.get("value_bets_file", "worldcup_value_bets.csv")
    out.to_csv(pred_path, index=False)
    value_bets.to_csv(vb_path, index=False)
    print(f"\nSaved predictions -> {pred_path.relative_to(ROOT)}")
    print(f"Saved value bets  -> {vb_path.relative_to(ROOT)}")
    _print_header("Done.")


if __name__ == "__main__":
    main()
