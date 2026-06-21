#!/usr/bin/env python
"""Fit a model and report proper-scoring forecast quality.

Examples
--------
    python scripts/train_models.py --synthetic --model dixon_coles
    python scripts/train_models.py --model ml
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from soccer_betting.cli import train_main  # noqa: E402

if __name__ == "__main__":
    train_main()
