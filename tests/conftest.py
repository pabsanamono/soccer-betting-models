"""Shared pytest fixtures and path setup."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from soccer_betting.data.preprocess import make_synthetic_matches  # noqa: E402


@pytest.fixture(scope="session")
def synthetic_matches():
    return make_synthetic_matches(n_teams=14, n_seasons=4, seed=1)
