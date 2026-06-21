#!/usr/bin/env python
"""Run a walk-forward betting backtest.

Examples
--------
    python scripts/run_backtest.py --synthetic --model dixon_coles
    python scripts/run_backtest.py --model ml
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from soccer_betting.cli import backtest_main  # noqa: E402

if __name__ == "__main__":
    backtest_main()
