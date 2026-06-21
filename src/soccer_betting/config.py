"""Configuration loading.

A thin, dependency-light wrapper around a YAML config file that provides
attribute-style and dict-style access plus a couple of convenience helpers for
resolving project-relative paths.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

# Project root = two levels up from this file (src/soccer_betting/config.py).
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


class Config:
    """Dictionary-backed configuration object with helpers.

    Supports ``cfg["models"]["dixon_coles"]`` style access and the convenience
    method :meth:`get` for dotted lookups, e.g. ``cfg.get("models.poisson.max_goals")``.
    """

    def __init__(self, data: Dict[str, Any], root: Optional[Path] = None):
        self._data = data
        self.root = Path(root) if root else PROJECT_ROOT

    # -- dict-like access ----------------------------------------------------
    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def to_dict(self) -> Dict[str, Any]:
        return dict(self._data)

    def get(self, dotted_key: str, default: Any = None) -> Any:
        """Look up a value using a dotted path, e.g. ``"ml.xgboost.max_depth"``."""
        node: Any = self._data
        for part in dotted_key.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                return default
        return node

    # -- path helpers --------------------------------------------------------
    def path(self, dotted_key: str, default: Optional[str] = None) -> Path:
        """Resolve a configured path relative to the project root."""
        value = self.get(dotted_key, default)
        if value is None:
            raise KeyError(f"No path configured at '{dotted_key}'")
        p = Path(value)
        return p if p.is_absolute() else self.root / p

    def ensure_dir(self, dotted_key: str) -> Path:
        """Resolve a configured directory path and create it if missing."""
        p = self.path(dotted_key)
        p.mkdir(parents=True, exist_ok=True)
        return p


def load_config(path: Optional[os.PathLike | str] = None) -> Config:
    """Load configuration from ``path`` (defaults to ``config/config.yaml``)."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    # Project root is two levels up from the config dir, i.e. its parent's parent.
    return Config(data, root=cfg_path.resolve().parent.parent)
