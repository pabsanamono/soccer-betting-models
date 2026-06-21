"""Data collection, preprocessing and feature engineering."""
from soccer_betting.data.collectors import (
    FootballDataUKCollector,
    TheOddsAPICollector,
)
from soccer_betting.data.preprocess import load_matches, standardise_matches
from soccer_betting.data.features import FeatureBuilder
from soccer_betting.data.api_client import (
    TheStatsAPIClient,
    BzzoiroClient,
    APIError,
    FieldMap,
    build_offline_worldcup_data,
    WORLD_CUP_2026_GROUPS,
)

__all__ = [
    "FootballDataUKCollector",
    "TheOddsAPICollector",
    "load_matches",
    "standardise_matches",
    "FeatureBuilder",
    "TheStatsAPIClient",
    "BzzoiroClient",
    "APIError",
    "FieldMap",
    "build_offline_worldcup_data",
    "WORLD_CUP_2026_GROUPS",
]
