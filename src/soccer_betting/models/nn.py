"""Optional feed-forward neural network for 1X2 classification.

This module imports :mod:`torch` lazily so the rest of the package works
without it. Install the extra with ``pip install soccer-betting[nn]`` or
``pip install torch``. The network is a small MLP with dropout and a softmax
head trained with cross-entropy — deliberately modest, since the research notes
deep nets only earn their place with high-granularity tracking/time-series data.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from soccer_betting.utils.logging import get_logger

logger = get_logger(__name__)

_LABEL_TO_IDX = {"H": 0, "D": 1, "A": 2}


def _labels_to_idx(y) -> np.ndarray:
    y = np.asarray(y)
    if y.dtype.kind in {"U", "S", "O"}:
        return np.array([_LABEL_TO_IDX[str(v)] for v in y])
    return y.astype(int)


class TorchMLP:
    """A small PyTorch MLP classifier with a scikit-like interface."""

    def __init__(
        self,
        hidden: tuple = (64, 32),
        dropout: float = 0.3,
        lr: float = 1e-3,
        epochs: int = 100,
        batch_size: int = 64,
        weight_decay: float = 1e-4,
        random_state: int = 42,
    ):
        self.hidden = hidden
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.random_state = random_state
        self.net_ = None
        self.mean_ = None
        self.std_ = None
        self.feature_names_: List[str] = []
        self.is_fitted = False

    def _require_torch(self):
        try:
            import torch  # noqa: F401
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "TorchMLP requires PyTorch. Install with `pip install torch`."
            ) from exc
        return __import__("torch")

    def _build(self, torch, n_features: int):
        nn = torch.nn
        layers: list = []
        prev = n_features
        for h in self.hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(self.dropout)]
            prev = h
        layers += [nn.Linear(prev, 3)]
        return nn.Sequential(*layers)

    def fit(self, X: pd.DataFrame, y) -> "TorchMLP":
        torch = self._require_torch()
        torch.manual_seed(self.random_state)
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
            X_arr = X.to_numpy(dtype=float)
        else:
            X_arr = np.asarray(X, dtype=float)
        y_idx = _labels_to_idx(y)

        self.mean_ = X_arr.mean(axis=0)
        self.std_ = X_arr.std(axis=0) + 1e-8
        X_std = (X_arr - self.mean_) / self.std_

        self.net_ = self._build(torch, X_arr.shape[1])
        opt = torch.optim.Adam(self.net_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = torch.nn.CrossEntropyLoss()

        Xt = torch.tensor(X_std, dtype=torch.float32)
        yt = torch.tensor(y_idx, dtype=torch.long)
        n = len(Xt)
        self.net_.train()
        for epoch in range(self.epochs):
            perm = torch.randperm(n)
            for i in range(0, n, self.batch_size):
                idx = perm[i:i + self.batch_size]
                opt.zero_grad()
                out = self.net_(Xt[idx])
                loss = loss_fn(out, yt[idx])
                loss.backward()
                opt.step()
        self.is_fitted = True
        logger.info("Trained TorchMLP for %d epochs on %d rows", self.epochs, n)
        return self

    def predict_proba(self, X) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("TorchMLP must be fitted first.")
        torch = self._require_torch()
        X_arr = X.to_numpy(dtype=float) if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=float)
        X_std = (X_arr - self.mean_) / self.std_
        self.net_.eval()
        with torch.no_grad():
            logits = self.net_(torch.tensor(X_std, dtype=torch.float32))
            probs = torch.softmax(logits, dim=1).numpy()
        return probs
