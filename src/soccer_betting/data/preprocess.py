"""Preprocessing: turn raw football-data CSVs into a canonical schema.

Canonical match schema (one row per match):

    date          : pandas datetime
    season        : str (optional, propagated if present)
    division      : str (optional)
    home_team     : str
    away_team     : str
    home_goals    : int (full-time)
    away_goals    : int (full-time)
    result        : {'H', 'D', 'A'}
    odds_home     : float (closing decimal odds, best available source)
    odds_draw     : float
    odds_away     : float

Downstream models and the backtester rely on exactly these names.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from soccer_betting.utils.logging import get_logger

logger = get_logger(__name__)

# Preference order for closing odds columns in football-data files.
# B365C* / PSC* / *C* are *closing* odds; B365* etc. are opening odds.
_ODDS_PREFERENCE = {
    "home": ["B365CH", "PSCH", "MaxCH", "AvgCH", "B365H", "PSH", "AvgH", "BbAvH"],
    "draw": ["B365CD", "PSCD", "MaxCD", "AvgCD", "B365D", "PSD", "AvgD", "BbAvD"],
    "away": ["B365CA", "PSCA", "MaxCA", "AvgCA", "B365A", "PSA", "AvgA", "BbAvA"],
}


def _first_available(df: pd.DataFrame, candidates: List[str]) -> Optional[pd.Series]:
    for col in candidates:
        if col in df.columns:
            series = pd.to_numeric(df[col], errors="coerce")
            if series.notna().any():
                return series
    return None


def standardise_matches(raw: pd.DataFrame) -> pd.DataFrame:
    """Map a raw football-data.co.uk frame to the canonical schema."""
    required = {"HomeTeam", "AwayTeam", "FTHG", "FTAG"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Raw data missing required columns: {sorted(missing)}")

    out = pd.DataFrame()
    # football-data dates are dd/mm/yy or dd/mm/yyyy.
    out["date"] = pd.to_datetime(raw["Date"], dayfirst=True, errors="coerce")
    if "Div" in raw.columns:
        out["division"] = raw["Div"].astype(str)
    out["home_team"] = raw["HomeTeam"].astype(str).str.strip()
    out["away_team"] = raw["AwayTeam"].astype(str).str.strip()
    out["home_goals"] = pd.to_numeric(raw["FTHG"], errors="coerce")
    out["away_goals"] = pd.to_numeric(raw["FTAG"], errors="coerce")

    if "FTR" in raw.columns:
        out["result"] = raw["FTR"].astype(str).str.upper().str[0]
    else:
        out["result"] = np.select(
            [out["home_goals"] > out["away_goals"], out["home_goals"] == out["away_goals"]],
            ["H", "D"],
            default="A",
        )

    for side in ("home", "draw", "away"):
        series = _first_available(raw, _ODDS_PREFERENCE[side])
        out[f"odds_{side}"] = series if series is not None else np.nan

    # Drop rows without a valid date or score.
    before = len(out)
    out = out.dropna(subset=["date", "home_goals", "away_goals"]).copy()
    out["home_goals"] = out["home_goals"].astype(int)
    out["away_goals"] = out["away_goals"].astype(int)
    out = out[out["result"].isin(["H", "D", "A"])]
    out = out.sort_values("date").reset_index(drop=True)
    logger.info("Standardised %d/%d rows", len(out), before)
    return out


def load_matches(csv_paths: List[str]) -> pd.DataFrame:
    """Convenience: read raw CSVs and return a standardised match frame."""
    from soccer_betting.data.collectors import FootballDataUKCollector

    raw = FootballDataUKCollector.read_many(csv_paths)
    return standardise_matches(raw)


def make_synthetic_matches(
    n_teams: int = 12,
    n_seasons: int = 3,
    seed: int = 42,
    margin: float = 0.06,
) -> pd.DataFrame:
    """Generate a realistic synthetic dataset for tests/demos (offline).

    Each team gets latent attack/defence strengths; goals are drawn from a
    Poisson process with a home advantage, and bookmaker odds are produced from
    the *true* probabilities plus a multiplicative margin and a little noise so
    that value opportunities exist. This lets the whole pipeline run end-to-end
    without any network access.
    """
    rng = np.random.default_rng(seed)
    teams = [f"Team_{i:02d}" for i in range(n_teams)]
    attack = {t: rng.normal(0.2, 0.35) for t in teams}
    defence = {t: rng.normal(0.0, 0.35) for t in teams}
    home_adv = 0.30
    base = 0.1  # log base scoring rate

    start = pd.Timestamp("2021-08-01")
    rows = []
    match_day = start
    for season in range(n_seasons):
        fixtures = [(h, a) for h in teams for a in teams if h != a]
        rng.shuffle(fixtures)
        for i, (h, a) in enumerate(fixtures):
            lam_h = np.exp(base + home_adv + attack[h] - defence[a])
            lam_a = np.exp(base + attack[a] - defence[h])
            hg = int(rng.poisson(lam_h))
            ag = int(rng.poisson(lam_a))
            # True outcome probabilities via a quick Poisson grid (max 10 goals).
            gh = np.arange(11)
            ph = np.exp(-lam_h) * lam_h ** gh / _factorial(gh)
            pa = np.exp(-lam_a) * lam_a ** gh / _factorial(gh)
            joint = np.outer(ph, pa)
            p_home = float(np.tril(joint, -1).sum())
            p_away = float(np.triu(joint, 1).sum())
            p_draw = float(np.trace(joint))
            probs = np.array([p_home, p_draw, p_away])
            probs = probs / probs.sum()
            # Apply a margin + noise to construct bookmaker odds.
            noisy = probs * rng.normal(1.0, 0.05, size=3)
            noisy = np.clip(noisy, 1e-4, None)
            noisy = noisy / noisy.sum() * (1 + margin)
            # Guard against degenerate implied probs >= 1 (would give odds <= 1).
            noisy = np.clip(noisy, 1e-3, 0.95)
            odds = np.clip(1.0 / noisy, 1.01, None)
            result = "H" if hg > ag else ("D" if hg == ag else "A")
            rows.append(
                {
                    "date": match_day + pd.Timedelta(days=i // 6),
                    "season": f"S{season}",
                    "division": "SYN",
                    "home_team": h,
                    "away_team": a,
                    "home_goals": hg,
                    "away_goals": ag,
                    "result": result,
                    "odds_home": round(float(odds[0]), 3),
                    "odds_draw": round(float(odds[1]), 3),
                    "odds_away": round(float(odds[2]), 3),
                }
            )
        match_day = match_day + pd.Timedelta(days=300)
    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return df


def _factorial(arr: np.ndarray) -> np.ndarray:
    from scipy.special import factorial

    return factorial(arr)
