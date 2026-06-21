#!/usr/bin/env python
"""Download all World Cup 2026 data and save it as CSV files.

Just run it -- it handles everything:

    python scripts/fetch_worldcup_data.py

What it does
------------
1. Reads ``config/api_config.yaml`` for your API key and settings.
2. Tries TheStatsAPI (recommended). If no key is set, tries the free Bzzoiro
   backup. If neither works, falls back to a realistic built-in DEMO dataset
   so you can still see the whole pipeline run.
3. Saves three CSV files into ``data/worldcup/``:
       * worldcup_historical.csv  -- past results (used to train models)
       * worldcup_fixtures.csv    -- upcoming matches to predict
       * worldcup_odds.csv        -- bookmaker odds for those matches

After this finishes, run:  python scripts/predict_worldcup.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make the package importable when run directly from the repo.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from soccer_betting.data.api_client import (  # noqa: E402
    APIError,
    BzzoiroClient,
    FieldMap,
    TheStatsAPIClient,
    build_offline_worldcup_data,
)
from soccer_betting.utils.logging import get_logger  # noqa: E402

logger = get_logger("fetch_worldcup")

CONFIG_PATH = ROOT / "config" / "api_config.yaml"


def load_api_config() -> dict:
    if not CONFIG_PATH.exists():
        logger.warning("Config %s not found; using built-in defaults.", CONFIG_PATH)
        return {}
    with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _print_header(text: str) -> None:
    print("\n" + "=" * 68)
    print(text)
    print("=" * 68)


def fetch_from_provider(cfg: dict):
    """Return (historical, fixtures, odds, source_label) using the best source.

    Tries TheStatsAPI -> Bzzoiro -> demo fallback, in that order.
    """
    ts_cfg = cfg.get("thestatsapi", {}) or {}
    bz_cfg = cfg.get("bzzoiro", {}) or {}
    demo_cfg = cfg.get("demo", {}) or {}

    # --- 1) TheStatsAPI (primary) -----------------------------------------
    ts_client = TheStatsAPIClient(
        api_key=ts_cfg.get("api_key") or None,
        base_url=ts_cfg.get("base_url", "https://api.thestatsapi.com/v1"),
        competition=ts_cfg.get("competition", "world-cup-2026"),
        season=str(ts_cfg.get("season", "2026")),
        endpoints=ts_cfg.get("endpoints"),
        field_map=FieldMap.from_dict(ts_cfg.get("field_map")),
        rate_limit_per_min=int(ts_cfg.get("rate_limit_per_min", 120)),
        auth_style=ts_cfg.get("auth_style", "header"),
    )
    if ts_client.available:
        try:
            logger.info("Using TheStatsAPI as the data source.")
            return (
                ts_client.fetch_historical_matches(),
                ts_client.fetch_fixtures(),
                ts_client.fetch_odds(),
                "TheStatsAPI (live)",
            )
        except APIError as exc:
            logger.warning("TheStatsAPI failed: %s", exc)
    else:
        logger.info("No TheStatsAPI key configured; checking backup provider.")

    # --- 2) Bzzoiro (free backup) -----------------------------------------
    if bz_cfg.get("enabled", True):
        bz_client = BzzoiroClient(
            api_key=bz_cfg.get("api_key") or None,
            base_url=bz_cfg.get("base_url", "https://api.bzzoiro.com/v1"),
            competition=bz_cfg.get("competition", "world-cup-2026"),
        )
        try:
            logger.info("Trying Bzzoiro (free backup) ...")
            hist = bz_client.fetch_historical_matches()
            fix = bz_client.fetch_fixtures()
            odds = bz_client.fetch_odds()
            if not fix.empty:
                return hist, fix, odds, "Bzzoiro (live, free backup)"
            logger.warning("Bzzoiro returned no fixtures; falling back to demo.")
        except APIError as exc:
            logger.warning("Bzzoiro failed: %s", exc)

    # --- 3) Demo fallback --------------------------------------------------
    if not demo_cfg.get("allow_demo_fallback", True):
        raise SystemExit(
            "No live data source worked and demo fallback is disabled. "
            "Add a valid TheStatsAPI key to config/api_config.yaml."
        )
    logger.info("Falling back to built-in DEMO World Cup 2026 dataset.")
    hist, fix, odds = build_offline_worldcup_data(seed=int(demo_cfg.get("seed", 2026)))
    return hist, fix, odds, "DEMO (built-in synthetic data)"


def main() -> None:
    _print_header("World Cup 2026 — data download")
    cfg = load_api_config()

    out_cfg = cfg.get("output", {}) or {}
    data_dir = ROOT / out_cfg.get("data_dir", "data/worldcup")
    data_dir.mkdir(parents=True, exist_ok=True)

    historical, fixtures, odds, source = fetch_from_provider(cfg)

    hist_path = data_dir / out_cfg.get("historical_file", "worldcup_historical.csv")
    fix_path = data_dir / out_cfg.get("fixtures_file", "worldcup_fixtures.csv")
    odds_path = data_dir / out_cfg.get("odds_file", "worldcup_odds.csv")

    historical.to_csv(hist_path, index=False)
    fixtures.to_csv(fix_path, index=False)
    odds.to_csv(odds_path, index=False)

    print(f"\nData source: {source}")
    print("\nSaved files:")
    print(f"  - {hist_path.relative_to(ROOT)}   ({len(historical)} historical matches)")
    print(f"  - {fix_path.relative_to(ROOT)}     ({len(fixtures)} upcoming fixtures)")
    print(f"  - {odds_path.relative_to(ROOT)}         ({len(odds)} odds rows)")

    if not fixtures.empty:
        print("\nNext 5 upcoming fixtures:")
        preview_cols = [c for c in ("date", "stage", "home_team", "away_team",
                                    "odds_home", "odds_draw", "odds_away") if c in fixtures.columns]
        with pd.option_context("display.width", 120, "display.max_columns", None):
            print(fixtures[preview_cols].head(5).to_string(index=False))

    if "DEMO" in source:
        print(
            "\nNOTE: You are seeing DEMO data because no live API key was found.\n"
            "      To use real data, add your TheStatsAPI key to\n"
            "      config/api_config.yaml (see API_GUIDE.md), then run this again."
        )

    _print_header("Done — now run:  python scripts/predict_worldcup.py")


if __name__ == "__main__":
    main()
