"""API integration for live World Cup 2026 data.

This module provides a small, dependency-light client for fetching World Cup
2026 data (historical results, upcoming fixtures and bookmaker odds) and
converting it into the **canonical match schema** used everywhere else in this
package:

    date, season, division, home_team, away_team,
    home_goals, away_goals, result, odds_home, odds_draw, odds_away

Two providers are supported:

``TheStatsAPIClient``  (recommended / primary)
    Client for https://www.thestatsapi.com — bundles historical data, fixtures
    and odds from several bookmakers in a single subscription. Requires an API
    key (see ``API_GUIDE.md`` for how to get one).

``BzzoiroClient``      (optional / free backup)
    Thin client for https://sports.bzzoiro.com — a free odds + fixtures source
    used as an automatic fallback when no TheStatsAPI key is configured.

Because third-party JSON schemas drift over time, the response parsers here are
deliberately *tolerant*: they look for a list of common field names for each
piece of information and fall back gracefully. The exact JSON field names can
also be overridden from ``config/api_config.yaml`` without touching code.

Finally, :func:`build_offline_worldcup_data` produces a fully synthetic but
realistic World Cup 2026 dataset (real national-team names, group-stage
fixtures and bookmaker odds). This lets the whole fetch + predict pipeline run
**end-to-end with no API key at all**, so a non-technical user can see real
predictions immediately and plug in a paid key later.
"""
from __future__ import annotations

import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import requests

from soccer_betting.utils.logging import get_logger

logger = get_logger(__name__)


class APIError(RuntimeError):
    """Raised when an API source cannot be reached or returns bad data."""


# Canonical columns every downstream consumer expects.
MATCH_COLUMNS = [
    "date", "season", "division", "home_team", "away_team",
    "home_goals", "away_goals", "result", "odds_home", "odds_draw", "odds_away",
]
FIXTURE_COLUMNS = [
    "date", "season", "division", "stage", "home_team", "away_team",
    "odds_home", "odds_draw", "odds_away",
]
ODDS_COLUMNS = [
    "date", "home_team", "away_team", "bookmaker",
    "odds_home", "odds_draw", "odds_away",
]


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _first(d: Dict[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    """Return the first present, non-null value among ``keys`` in dict ``d``.

    Supports dotted keys for one level of nesting, e.g. ``"teams.home"``.
    """
    for k in keys:
        if "." in k:
            head, tail = k.split(".", 1)
            node = d.get(head)
            if isinstance(node, dict):
                val = _first(node, [tail])
                if val is not None:
                    return val
        elif k in d and d[k] is not None:
            return d[k]
    return default


def _to_result(hg: Any, ag: Any) -> Optional[str]:
    try:
        hg, ag = int(hg), int(ag)
    except (TypeError, ValueError):
        return None
    return "H" if hg > ag else ("D" if hg == ag else "A")


class _RateLimiter:
    """Simple sliding-window rate limiter (max ``calls`` per ``period`` secs)."""

    def __init__(self, calls: int, period: float = 60.0):
        self.calls = max(1, int(calls))
        self.period = float(period)
        self._stamps: deque[float] = deque()

    def wait(self) -> None:
        now = time.monotonic()
        while self._stamps and now - self._stamps[0] >= self.period:
            self._stamps.popleft()
        if len(self._stamps) >= self.calls:
            sleep_for = self.period - (now - self._stamps[0]) + 0.01
            if sleep_for > 0:
                logger.info("Rate limit reached; sleeping %.2fs", sleep_for)
                time.sleep(sleep_for)
        self._stamps.append(time.monotonic())


# ---------------------------------------------------------------------------
# Field-mapping configuration
# ---------------------------------------------------------------------------
@dataclass
class FieldMap:
    """Tolerant mapping of canonical fields to candidate JSON keys.

    Override any of these from ``config/api_config.yaml`` under
    ``thestatsapi.field_map`` if the provider's schema differs.
    """

    date: List[str] = field(default_factory=lambda: [
        "date", "match_date", "kickoff", "kickoff_time", "commence_time",
        "start_time", "datetime", "utcDate", "fixture_date",
    ])
    home_team: List[str] = field(default_factory=lambda: [
        "home_team", "homeTeam", "home", "home_name", "team_home",
        "teams.home", "localteam", "home_team_name",
    ])
    away_team: List[str] = field(default_factory=lambda: [
        "away_team", "awayTeam", "away", "away_name", "team_away",
        "teams.away", "visitorteam", "away_team_name",
    ])
    home_goals: List[str] = field(default_factory=lambda: [
        "home_goals", "homeGoals", "home_score", "score_home", "fthg",
        "goals.home", "home_ft", "home_team_goals",
    ])
    away_goals: List[str] = field(default_factory=lambda: [
        "away_goals", "awayGoals", "away_score", "score_away", "ftag",
        "goals.away", "away_ft", "away_team_goals",
    ])
    stage: List[str] = field(default_factory=lambda: [
        "stage", "round", "group", "phase", "round_name",
    ])
    odds_home: List[str] = field(default_factory=lambda: [
        "odds_home", "home_odds", "odds_1", "home_win", "price_home", "1",
    ])
    odds_draw: List[str] = field(default_factory=lambda: [
        "odds_draw", "draw_odds", "odds_x", "draw", "price_draw", "X",
    ])
    odds_away: List[str] = field(default_factory=lambda: [
        "odds_away", "away_odds", "odds_2", "away_win", "price_away", "2",
    ])

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, List[str]]]) -> "FieldMap":
        fm = cls()
        if not data:
            return fm
        for key, value in data.items():
            if hasattr(fm, key) and value:
                # Prepend user-provided keys so they take priority.
                setattr(fm, key, list(value) + getattr(fm, key))
        return fm


# ---------------------------------------------------------------------------
# TheStatsAPI client (primary)
# ---------------------------------------------------------------------------
class TheStatsAPIClient:
    """Client for TheStatsAPI World Cup 2026 endpoints.

    Parameters
    ----------
    api_key:
        TheStatsAPI key. Falls back to the ``THESTATSAPI_KEY`` environment
        variable when ``None``.
    base_url:
        Root of the v1 REST API.
    competition:
        Competition/tournament identifier for the World Cup 2026.
    season:
        Season/year label.
    endpoints:
        Optional override of the endpoint *paths* (``results``, ``fixtures``,
        ``odds``). Useful if the provider renames a route.
    field_map:
        Tolerant JSON field mapping (see :class:`FieldMap`).
    rate_limit_per_min:
        Maximum requests per minute (Starter plan = 120).
    auth_style:
        How the key is sent: ``"header"`` (default, ``Authorization: Bearer``),
        ``"header_x_api_key"`` (``X-API-Key`` header) or ``"query"`` (``apiKey``
        query parameter). Different plans/gateways use different styles.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.thestatsapi.com/v1",
        competition: str = "world-cup-2026",
        season: str = "2026",
        endpoints: Optional[Dict[str, str]] = None,
        field_map: Optional[FieldMap] = None,
        rate_limit_per_min: int = 120,
        auth_style: str = "header",
        timeout: int = 30,
    ):
        self.api_key = api_key or os.environ.get("THESTATSAPI_KEY")
        self.base_url = base_url.rstrip("/")
        self.competition = competition
        self.season = str(season)
        self.endpoints = {
            "results": "matches",
            "fixtures": "fixtures",
            "odds": "odds",
            **(endpoints or {}),
        }
        self.field_map = field_map or FieldMap()
        self.auth_style = auth_style
        self.timeout = timeout
        self._limiter = _RateLimiter(rate_limit_per_min, 60.0)
        self._session = requests.Session()

    # -- low level ----------------------------------------------------------
    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def _headers(self) -> Dict[str, str]:
        headers = {"Accept": "application/json"}
        if self.auth_style == "header" and self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        elif self.auth_style == "header_x_api_key" and self.api_key:
            headers["X-API-Key"] = self.api_key
        return headers

    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        if not self.api_key:
            raise APIError(
                "No TheStatsAPI key configured. Set THESTATSAPI_KEY or add it to "
                "config/api_config.yaml (see API_GUIDE.md)."
            )
        params = dict(params or {})
        if self.auth_style == "query":
            params["apiKey"] = self.api_key
        url = f"{self.base_url}/{path.lstrip('/')}"
        self._limiter.wait()
        for attempt in range(1, 4):
            try:
                resp = self._session.get(
                    url, params=params, headers=self._headers(), timeout=self.timeout
                )
                if resp.status_code == 429:  # too many requests
                    retry_after = float(resp.headers.get("Retry-After", 2 ** attempt))
                    logger.warning("HTTP 429 from %s; backing off %.1fs", url, retry_after)
                    time.sleep(retry_after)
                    continue
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as exc:
                if attempt == 3:
                    raise APIError(f"Request to {url} failed: {exc}") from exc
                backoff = 2 ** attempt
                logger.warning("Request error (%s); retry %d/3 in %ds", exc, attempt, backoff)
                time.sleep(backoff)
        raise APIError(f"Request to {url} failed after retries.")

    @staticmethod
    def _records(payload: Any) -> List[Dict[str, Any]]:
        """Pull a list of record dicts out of a variety of envelope shapes."""
        if isinstance(payload, list):
            return [r for r in payload if isinstance(r, dict)]
        if isinstance(payload, dict):
            for key in ("data", "results", "matches", "fixtures", "response", "items", "odds"):
                node = payload.get(key)
                if isinstance(node, list):
                    return [r for r in node if isinstance(r, dict)]
            # single record
            return [payload]
        return []

    # -- public API ---------------------------------------------------------
    def fetch_historical_matches(self) -> pd.DataFrame:
        """Fetch completed World Cup matches as a canonical match frame."""
        logger.info("Fetching historical matches from TheStatsAPI ...")
        payload = self._get(
            self.endpoints["results"],
            {"competition": self.competition, "season": self.season, "status": "finished"},
        )
        df = self._parse_matches(self._records(payload))
        logger.info("Parsed %d historical matches", len(df))
        return df

    def fetch_fixtures(self) -> pd.DataFrame:
        """Fetch upcoming World Cup fixtures as a canonical fixture frame."""
        logger.info("Fetching upcoming fixtures from TheStatsAPI ...")
        payload = self._get(
            self.endpoints["fixtures"],
            {"competition": self.competition, "season": self.season, "status": "scheduled"},
        )
        df = self._parse_fixtures(self._records(payload))
        logger.info("Parsed %d upcoming fixtures", len(df))
        return df

    def fetch_odds(self) -> pd.DataFrame:
        """Fetch pre-match 1X2 odds for upcoming fixtures (per bookmaker)."""
        logger.info("Fetching odds from TheStatsAPI ...")
        payload = self._get(
            self.endpoints["odds"],
            {"competition": self.competition, "season": self.season, "market": "1x2"},
        )
        df = self._parse_odds(self._records(payload))
        logger.info("Parsed %d odds rows", len(df))
        return df

    # -- parsers ------------------------------------------------------------
    def _parse_matches(self, records: List[Dict[str, Any]]) -> pd.DataFrame:
        fm = self.field_map
        rows = []
        for r in records:
            hg = _first(r, fm.home_goals)
            ag = _first(r, fm.away_goals)
            result = _to_result(hg, ag)
            if result is None:
                continue  # not a completed match
            rows.append({
                "date": _first(r, fm.date),
                "season": self.season,
                "division": "WC2026",
                "home_team": _first(r, fm.home_team),
                "away_team": _first(r, fm.away_team),
                "home_goals": int(hg),
                "away_goals": int(ag),
                "result": result,
                "odds_home": _coerce_float(_first(r, fm.odds_home)),
                "odds_draw": _coerce_float(_first(r, fm.odds_draw)),
                "odds_away": _coerce_float(_first(r, fm.odds_away)),
            })
        df = pd.DataFrame(rows, columns=MATCH_COLUMNS)
        return _clean_matches(df)

    def _parse_fixtures(self, records: List[Dict[str, Any]]) -> pd.DataFrame:
        fm = self.field_map
        rows = []
        for r in records:
            home, away = _first(r, fm.home_team), _first(r, fm.away_team)
            if not home or not away:
                continue
            rows.append({
                "date": _first(r, fm.date),
                "season": self.season,
                "division": "WC2026",
                "stage": _first(r, fm.stage, "Group Stage"),
                "home_team": home,
                "away_team": away,
                "odds_home": _coerce_float(_first(r, fm.odds_home)),
                "odds_draw": _coerce_float(_first(r, fm.odds_draw)),
                "odds_away": _coerce_float(_first(r, fm.odds_away)),
            })
        df = pd.DataFrame(rows, columns=FIXTURE_COLUMNS)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.sort_values("date").reset_index(drop=True)
        return df

    def _parse_odds(self, records: List[Dict[str, Any]]) -> pd.DataFrame:
        fm = self.field_map
        rows = []
        for r in records:
            home, away = _first(r, fm.home_team), _first(r, fm.away_team)
            books = r.get("bookmakers") or r.get("books")
            if isinstance(books, list) and books:
                for b in books:
                    if not isinstance(b, dict):
                        continue
                    rows.append({
                        "date": _first(r, fm.date),
                        "home_team": home,
                        "away_team": away,
                        "bookmaker": b.get("name") or b.get("title") or b.get("bookmaker"),
                        "odds_home": _coerce_float(_first(b, fm.odds_home)),
                        "odds_draw": _coerce_float(_first(b, fm.odds_draw)),
                        "odds_away": _coerce_float(_first(b, fm.odds_away)),
                    })
            else:
                rows.append({
                    "date": _first(r, fm.date),
                    "home_team": home,
                    "away_team": away,
                    "bookmaker": r.get("bookmaker") or "consensus",
                    "odds_home": _coerce_float(_first(r, fm.odds_home)),
                    "odds_draw": _coerce_float(_first(r, fm.odds_draw)),
                    "odds_away": _coerce_float(_first(r, fm.odds_away)),
                })
        df = pd.DataFrame(rows, columns=ODDS_COLUMNS)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        return df


# ---------------------------------------------------------------------------
# Bzzoiro client (free backup)
# ---------------------------------------------------------------------------
class BzzoiroClient:
    """Minimal free-tier client for Bzzoiro Sports Data (backup provider).

    Used automatically when no TheStatsAPI key is configured. The free tier has
    no rate limits but coverage of World Cup 2026 may be partial.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.bzzoiro.com/v1",
        competition: str = "world-cup-2026",
        field_map: Optional[FieldMap] = None,
        timeout: int = 30,
    ):
        self.api_key = api_key or os.environ.get("BZZOIRO_KEY")
        self.base_url = base_url.rstrip("/")
        self.competition = competition
        self.field_map = field_map or FieldMap()
        self.timeout = timeout
        self._session = requests.Session()

    @property
    def available(self) -> bool:
        # Bzzoiro free tier works without a key.
        return True

    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        params = dict(params or {})
        if self.api_key:
            params["apiKey"] = self.api_key
        url = f"{self.base_url}/{path.lstrip('/')}"
        try:
            resp = self._session.get(url, params=params, timeout=self.timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            raise APIError(f"Bzzoiro request to {url} failed: {exc}") from exc

    # The parsing logic is shared with TheStatsAPI's tolerant parsers.
    def fetch_historical_matches(self) -> pd.DataFrame:
        proxy = TheStatsAPIClient(api_key="x", field_map=self.field_map)
        payload = self._get("matches", {"competition": self.competition, "status": "finished"})
        return proxy._parse_matches(proxy._records(payload))

    def fetch_fixtures(self) -> pd.DataFrame:
        proxy = TheStatsAPIClient(api_key="x", field_map=self.field_map)
        payload = self._get("fixtures", {"competition": self.competition, "status": "scheduled"})
        return proxy._parse_fixtures(proxy._records(payload))

    def fetch_odds(self) -> pd.DataFrame:
        proxy = TheStatsAPIClient(api_key="x", field_map=self.field_map)
        payload = self._get("odds", {"competition": self.competition, "market": "1x2"})
        return proxy._parse_odds(proxy._records(payload))


# ---------------------------------------------------------------------------
# Shared cleaning utilities
# ---------------------------------------------------------------------------
def _coerce_float(value: Any) -> float:
    try:
        f = float(value)
        return f if f > 1.0 else np.nan  # decimal odds must exceed 1.0
    except (TypeError, ValueError):
        return np.nan


def _clean_matches(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["home_team", "away_team", "home_goals", "away_goals"])
    df["home_goals"] = df["home_goals"].astype(int)
    df["away_goals"] = df["away_goals"].astype(int)
    df = df[df["result"].isin(["H", "D", "A"])]
    return df.sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Offline / demo data (no API key required)
# ---------------------------------------------------------------------------
# The 48 teams expected at the FIFA World Cup 2026, grouped into the 12
# four-team groups (A-L). Used to generate a realistic demo dataset so the
# pipeline runs end-to-end without a paid API subscription.
WORLD_CUP_2026_GROUPS: Dict[str, List[str]] = {
    "A": ["Mexico", "Canada", "Croatia", "Ecuador"],
    "B": ["United States", "Wales", "Senegal", "Iran"],
    "C": ["Argentina", "Poland", "Australia", "Saudi Arabia"],
    "D": ["France", "Denmark", "Nigeria", "Costa Rica"],
    "E": ["Spain", "Germany", "Japan", "Morocco"],
    "F": ["Brazil", "Switzerland", "Cameroon", "South Korea"],
    "G": ["England", "Netherlands", "Ghana", "Qatar"],
    "H": ["Portugal", "Uruguay", "Serbia", "Tunisia"],
    "I": ["Belgium", "Colombia", "Egypt", "New Zealand"],
    "J": ["Italy", "Mexico B", "Algeria", "Panama"],
    "K": ["Netherlands B", "Sweden", "Ivory Coast", "Honduras"],
    "L": ["Croatia B", "Peru", "Mali", "Jamaica"],
}

# Approximate attacking strength tiers (higher = stronger) to make the demo
# predictions plausible. Teams not listed default to a mid/low rating.
_TEAM_STRENGTH: Dict[str, float] = {
    "Argentina": 0.95, "France": 0.93, "Brazil": 0.92, "England": 0.90,
    "Spain": 0.88, "Portugal": 0.87, "Netherlands": 0.85, "Germany": 0.84,
    "Belgium": 0.82, "Italy": 0.82, "Croatia": 0.80, "Uruguay": 0.78,
    "Colombia": 0.76, "Denmark": 0.75, "Switzerland": 0.74, "Mexico": 0.73,
    "United States": 0.72, "Japan": 0.72, "Senegal": 0.71, "Morocco": 0.71,
    "South Korea": 0.68, "Poland": 0.68, "Serbia": 0.67, "Sweden": 0.67,
    "Australia": 0.64, "Nigeria": 0.66, "Cameroon": 0.64, "Ecuador": 0.64,
    "Peru": 0.62, "Wales": 0.63, "Ghana": 0.62, "Egypt": 0.63,
    "Ivory Coast": 0.62, "Tunisia": 0.60, "Algeria": 0.63, "Iran": 0.62,
    "Saudi Arabia": 0.55, "Qatar": 0.54, "Canada": 0.62, "Costa Rica": 0.58,
    "Panama": 0.52, "Honduras": 0.50, "Jamaica": 0.52, "New Zealand": 0.50,
    "Mali": 0.58,
}


def _strength(team: str) -> float:
    if team in _TEAM_STRENGTH:
        return _TEAM_STRENGTH[team]
    # "Team B" placeholders inherit a slightly reduced base strength.
    base = team.replace(" B", "")
    return _TEAM_STRENGTH.get(base, 0.6) - (0.04 if team.endswith(" B") else 0.0)


def _all_worldcup_teams() -> List[str]:
    return [t for teams in WORLD_CUP_2026_GROUPS.values() for t in teams]


def build_offline_worldcup_data(seed: int = 2026):
    """Build a realistic synthetic World Cup 2026 dataset (no network).

    Returns
    -------
    (historical, fixtures, odds) : tuple of DataFrames
        * ``historical`` — past international results between the 48 teams in
          the canonical match schema (used to train the models).
        * ``fixtures`` — the 72 group-stage fixtures of WC 2026 (canonical
          fixture schema, with synthetic bookmaker odds attached).
        * ``odds`` — long-format per-bookmaker odds for those fixtures.
    """
    rng = np.random.default_rng(seed)
    teams = _all_worldcup_teams()
    home_adv = 0.25
    base = 0.05  # base log scoring rate

    def lambdas(home: str, away: str):
        sh, sa = _strength(home), _strength(away)
        lam_h = float(np.exp(base + home_adv + 1.1 * sh - 0.9 * sa))
        lam_a = float(np.exp(base + 1.1 * sa - 0.9 * sh))
        return max(lam_h, 0.15), max(lam_a, 0.15)

    def true_probs(lam_h: float, lam_a: float):
        from scipy.stats import poisson
        g = np.arange(11)
        ph, pa = poisson.pmf(g, lam_h), poisson.pmf(g, lam_a)
        joint = np.outer(ph, pa)
        p = np.array([np.tril(joint, -1).sum(), np.trace(joint), np.triu(joint, 1).sum()])
        return p / p.sum()

    def make_odds(p: np.ndarray, margin: float = 0.06):
        noisy = np.clip(p * rng.normal(1.0, 0.06, size=3), 1e-3, None)
        noisy = noisy / noisy.sum() * (1 + margin)
        noisy = np.clip(noisy, 1e-3, 0.95)
        return np.clip(1.0 / noisy, 1.01, None)

    # --- historical: a few "seasons" of friendlies/qualifiers among the 48 ---
    hist_rows = []
    match_day = pd.Timestamp("2022-09-01")
    for season in range(3):
        pairs = [(h, a) for h in teams for a in teams if h != a]
        rng.shuffle(pairs)
        pairs = pairs[:520]  # keep the dataset a sensible size
        for i, (h, a) in enumerate(pairs):
            lam_h, lam_a = lambdas(h, a)
            hg, ag = int(rng.poisson(lam_h)), int(rng.poisson(lam_a))
            odds = make_odds(true_probs(lam_h, lam_a))
            hist_rows.append({
                "date": match_day + pd.Timedelta(days=i // 8),
                "season": f"INT{season}",
                "division": "WC2026",
                "home_team": h,
                "away_team": a,
                "home_goals": hg,
                "away_goals": ag,
                "result": "H" if hg > ag else ("D" if hg == ag else "A"),
                "odds_home": round(float(odds[0]), 3),
                "odds_draw": round(float(odds[1]), 3),
                "odds_away": round(float(odds[2]), 3),
            })
        match_day += pd.Timedelta(days=260)
    historical = pd.DataFrame(hist_rows, columns=MATCH_COLUMNS)
    historical = _clean_matches(historical)

    # --- fixtures: the 72 group-stage matches of WC 2026 ---
    fixture_rows, odds_rows = [], []
    kickoff = pd.Timestamp("2026-06-11 18:00")
    books = ["Bet365", "Pinnacle", "Betfair", "William Hill"]
    fi = 0
    for group, group_teams in WORLD_CUP_2026_GROUPS.items():
        # round-robin: 6 matches per group of 4
        for i in range(len(group_teams)):
            for j in range(i + 1, len(group_teams)):
                home, away = group_teams[i], group_teams[j]
                lam_h, lam_a = lambdas(home, away)
                p = true_probs(lam_h, lam_a)
                consensus = make_odds(p)
                date = kickoff + pd.Timedelta(hours=6 * fi)
                fixture_rows.append({
                    "date": date,
                    "season": "2026",
                    "division": "WC2026",
                    "stage": f"Group {group}",
                    "home_team": home,
                    "away_team": away,
                    "odds_home": round(float(consensus[0]), 3),
                    "odds_draw": round(float(consensus[1]), 3),
                    "odds_away": round(float(consensus[2]), 3),
                })
                for book in books:
                    bo = make_odds(p, margin=rng.uniform(0.04, 0.08))
                    odds_rows.append({
                        "date": date,
                        "home_team": home,
                        "away_team": away,
                        "bookmaker": book,
                        "odds_home": round(float(bo[0]), 3),
                        "odds_draw": round(float(bo[1]), 3),
                        "odds_away": round(float(bo[2]), 3),
                    })
                fi += 1
    fixtures = pd.DataFrame(fixture_rows, columns=FIXTURE_COLUMNS).sort_values("date").reset_index(drop=True)
    odds = pd.DataFrame(odds_rows, columns=ODDS_COLUMNS).sort_values("date").reset_index(drop=True)
    return historical, fixtures, odds
