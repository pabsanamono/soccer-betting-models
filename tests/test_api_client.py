"""Tests for the World Cup 2026 API integration (no network required)."""
from __future__ import annotations

import pandas as pd

from soccer_betting.data.api_client import (
    FieldMap,
    TheStatsAPIClient,
    build_offline_worldcup_data,
    MATCH_COLUMNS,
    FIXTURE_COLUMNS,
)


def _client() -> TheStatsAPIClient:
    return TheStatsAPIClient(api_key="dummy")


def test_records_unwraps_common_envelopes():
    c = _client()
    assert len(c._records({"data": [{"a": 1}, {"b": 2}]})) == 2
    assert len(c._records([{"a": 1}])) == 1
    assert c._records({"x": 1}) == [{"x": 1}]


def test_parse_matches_tolerant_fields_and_result():
    c = _client()
    payload = {"data": [
        {"match_date": "2026-06-11", "home": "Mexico", "away": "Canada",
         "home_score": 2, "away_score": 1},
        {"kickoff": "2026-06-12", "homeTeam": "France", "awayTeam": "Brazil",
         "goals": {"home": 0, "away": 0}},
        {"home": "X", "away": "Y"},  # no score -> dropped
    ]}
    df = c._parse_matches(c._records(payload))
    assert list(df.columns) == MATCH_COLUMNS
    assert len(df) == 2
    assert set(df["result"]) <= {"H", "D", "A"}
    assert df.iloc[0]["result"] == "H"


def test_parse_fixtures_nested_teams_and_odds():
    c = _client()
    payload = [{
        "fixture_date": "2026-07-01",
        "teams": {"home": "Spain", "away": "Italy"},
        "round": "Final",
        "odds_1": 1.9, "odds_x": 3.4, "odds_2": 4.1,
    }]
    df = c._parse_fixtures(c._records(payload))
    assert list(df.columns) == FIXTURE_COLUMNS
    assert df.iloc[0]["home_team"] == "Spain"
    assert df.iloc[0]["stage"] == "Final"
    assert df.iloc[0]["odds_home"] == 1.9


def test_parse_odds_with_bookmaker_list():
    c = _client()
    payload = {"odds": [{
        "date": "2026-07-01", "home": "Spain", "away": "Italy",
        "bookmakers": [{"name": "Bet365", "home_odds": 1.9,
                        "draw_odds": 3.4, "away_odds": 4.1}],
    }]}
    df = c._parse_odds(c._records(payload))
    assert len(df) == 1
    assert df.iloc[0]["bookmaker"] == "Bet365"
    assert df.iloc[0]["odds_away"] == 4.1


def test_invalid_odds_become_nan():
    c = _client()
    payload = [{"date": "2026-07-01", "home": "A", "away": "B",
                "odds_1": 0.5, "odds_x": "n/a", "odds_2": 4.1}]
    df = c._parse_fixtures(c._records(payload))
    assert pd.isna(df.iloc[0]["odds_home"])   # 0.5 <= 1.0 rejected
    assert pd.isna(df.iloc[0]["odds_draw"])   # non-numeric
    assert df.iloc[0]["odds_away"] == 4.1


def test_field_map_override_prepends_keys():
    fm = FieldMap.from_dict({"home_team": ["localName"]})
    assert fm.home_team[0] == "localName"
    assert "home_team" in fm.home_team  # defaults retained


def test_no_api_key_marks_unavailable():
    c = TheStatsAPIClient(api_key=None)
    # Avoid reading a real env var that might exist in CI.
    c.api_key = None
    assert c.available is False


def test_offline_dataset_shapes_and_schema():
    historical, fixtures, odds = build_offline_worldcup_data(seed=1)
    assert len(fixtures) == 72          # 12 groups x 6 matches
    assert len(odds) == 72 * 4          # 4 bookmakers per fixture
    assert not historical.empty
    assert list(historical.columns) == MATCH_COLUMNS
    assert set(historical["result"]) <= {"H", "D", "A"}
    # odds must be valid decimal odds
    assert (fixtures["odds_home"] > 1.0).all()
