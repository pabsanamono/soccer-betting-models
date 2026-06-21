#!/usr/bin/env python
"""Download and standardise historical soccer data.

Examples
--------
    python scripts/collect_data.py                 # real data per config.yaml
    python scripts/collect_data.py --synthetic     # offline synthetic dataset
"""
import sys
from pathlib import Path

# Make the src/ package importable when running from a checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from soccer_betting.cli import collect_main  # noqa: E402

if __name__ == "__main__":
    collect_main()
