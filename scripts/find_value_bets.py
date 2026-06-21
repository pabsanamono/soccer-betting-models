#!/usr/bin/env python
"""List value bets from a predictions CSV (prob_* + odds_* columns).

Example
-------
    python scripts/find_value_bets.py data/processed/predictions.csv --min-edge 0.03
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from soccer_betting.cli import value_main  # noqa: E402

if __name__ == "__main__":
    value_main()
