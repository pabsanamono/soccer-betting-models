"""Data collectors for historical results and odds.

Two collectors are provided:

``FootballDataUKCollector``
    Downloads free historical CSVs from https://www.football-data.co.uk which
    include full-time results, half-time results, match statistics and both
    opening and closing bookmaker odds across many leagues and seasons. This is
    the recommended starting point for backtesting because the odds are the
    actual prices that were available.

``TheOddsAPICollector``
    A thin client for https://the-odds-api.com used to pull *forward-looking*
    odds for upcoming fixtures (for live value identification). Requires an API
    key, which should be supplied via the ``ODDS_API_KEY`` environment variable
    or the ``api_key`` argument — never hard-code secrets.

Both collectors fail gracefully: network problems raise a clear
``DataCollectionError`` rather than an opaque stack trace.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd
import requests

from soccer_betting.utils.logging import get_logger

logger = get_logger(__name__)


class DataCollectionError(RuntimeError):
    """Raised when a data source cannot be reached or returns bad data."""


class FootballDataUKCollector:
    """Download historical match + odds CSVs from football-data.co.uk.

    Parameters
    ----------
    out_dir:
        Directory where raw CSVs are cached.
    timeout:
        Per-request timeout in seconds.
    """

    BASE_URL = "https://www.football-data.co.uk/mmz4281"

    def __init__(self, out_dir: os.PathLike | str = "data/raw", timeout: int = 30):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout

    def _url(self, season: str, division: str) -> str:
        return f"{self.BASE_URL}/{season}/{division}.csv"

    def download_one(self, season: str, division: str, force: bool = False) -> Path:
        """Download a single season/division CSV, returning the local path.

        Files are cached locally; pass ``force=True`` to re-download.
        """
        dest = self.out_dir / f"{division}_{season}.csv"
        if dest.exists() and not force:
            logger.info("Using cached file %s", dest.name)
            return dest

        url = self._url(season, division)
        logger.info("Downloading %s", url)
        try:
            resp = requests.get(url, timeout=self.timeout)
            resp.raise_for_status()
        except requests.RequestException as exc:  # pragma: no cover - network
            raise DataCollectionError(f"Failed to download {url}: {exc}") from exc

        if not resp.content or b"," not in resp.content[:2048]:
            raise DataCollectionError(f"Unexpected (non-CSV) content from {url}")

        dest.write_bytes(resp.content)
        logger.info("Saved %s (%d bytes)", dest.name, len(resp.content))
        return dest

    def download(
        self,
        seasons: Iterable[str],
        divisions: Iterable[str],
        force: bool = False,
        polite_delay: float = 0.5,
    ) -> List[Path]:
        """Download many season/division combinations.

        A small ``polite_delay`` between requests avoids hammering the host.
        Individual failures are logged and skipped so one missing season does
        not abort the whole collection run.
        """
        paths: List[Path] = []
        for division in divisions:
            for season in seasons:
                try:
                    paths.append(self.download_one(season, division, force=force))
                except DataCollectionError as exc:
                    logger.warning("Skipping %s/%s: %s", division, season, exc)
                time.sleep(polite_delay)
        if not paths:
            raise DataCollectionError("No files were downloaded successfully.")
        return paths

    @staticmethod
    def read_many(paths: Iterable[os.PathLike | str]) -> pd.DataFrame:
        """Read and concatenate several football-data CSVs into one frame."""
        frames = []
        for p in paths:
            try:
                # football-data files occasionally contain trailing junk columns
                df = pd.read_csv(p, encoding="latin-1", on_bad_lines="skip")
                df = df.dropna(axis=1, how="all")
                frames.append(df)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Could not read %s: %s", p, exc)
        if not frames:
            raise DataCollectionError("No readable CSV files supplied.")
        return pd.concat(frames, ignore_index=True)


class TheOddsAPICollector:
    """Minimal client for the-odds-api.com (forward-looking odds).

    Parameters
    ----------
    api_key:
        API key. If ``None``, the ``ODDS_API_KEY`` environment variable is used.
    base_url, sport, regions, markets:
        See https://the-odds-api.com/liveapi/guides/v4/ for valid values.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.the-odds-api.com/v4",
        sport: str = "soccer_epl",
        regions: str = "eu",
        markets: str = "h2h",
        timeout: int = 30,
    ):
        self.api_key = api_key or os.environ.get("ODDS_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.sport = sport
        self.regions = regions
        self.markets = markets
        self.timeout = timeout

    def fetch_odds(self) -> pd.DataFrame:
        """Fetch current odds for upcoming fixtures as a tidy DataFrame.

        Returns one row per (fixture, bookmaker) with home/draw/away decimal
        odds. Raises ``DataCollectionError`` if no key is configured or the
        request fails.
        """
        if not self.api_key:
            raise DataCollectionError(
                "No Odds API key. Set ODDS_API_KEY or pass api_key=..."
            )
        url = f"{self.base_url}/sports/{self.sport}/odds"
        params = {
            "apiKey": self.api_key,
            "regions": self.regions,
            "markets": self.markets,
            "oddsFormat": "decimal",
        }
        try:
            resp = requests.get(url, params=params, timeout=self.timeout)
            resp.raise_for_status()
            payload = resp.json()
        except requests.RequestException as exc:  # pragma: no cover - network
            raise DataCollectionError(f"Odds API request failed: {exc}") from exc

        return self._flatten(payload)

    @staticmethod
    def _flatten(payload: list) -> pd.DataFrame:
        rows = []
        for event in payload:
            home = event.get("home_team")
            away = event.get("away_team")
            commence = event.get("commence_time")
            for book in event.get("bookmakers", []):
                book_name = book.get("title")
                for market in book.get("markets", []):
                    if market.get("key") != "h2h":
                        continue
                    odds = {o["name"]: o["price"] for o in market.get("outcomes", [])}
                    rows.append(
                        {
                            "commence_time": commence,
                            "home_team": home,
                            "away_team": away,
                            "bookmaker": book_name,
                            "odds_home": odds.get(home),
                            "odds_away": odds.get(away),
                            "odds_draw": odds.get("Draw"),
                        }
                    )
        return pd.DataFrame(rows)
