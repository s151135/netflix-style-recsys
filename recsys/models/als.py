from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ALSConfig:
    factors: int = 32
    regularization: float = 0.1
    alpha: float = 20.0
    iterations: int = 10
    seed: int = 42


class ImplicitALS:
    """Small dense implicit ALS baseline for local prototypes and tests."""

    def __init__(self, config: ALSConfig | None = None) -> None:
        self.config = config or ALSConfig()
        self.user_factors: np.ndarray | None = None
        self.item_factors: np.ndarray | None = None

    def fit(self, confidence_matrix: np.ndarray) -> "ImplicitALS":
        rng = np.random.default_rng(self.config.seed)
        c = np.asarray(confidence_matrix, dtype=float)
        prefs = (c > 0).astype(float)
        n_users, n_items = c.shape
        f = self.config.factors
        self.user_factors = rng.normal(0, 0.01, size=(n_users, f))
        self.item_factors = rng.normal(0, 0.01, size=(n_items, f))
        reg_eye = np.eye(f) * self.config.regularization
        confidence = 1.0 + self.config.alpha * c
        for _ in range(self.config.iterations):
            yty = self.item_factors.T @ self.item_factors
            for u in range(n_users):
                cu = np.diag(confidence[u])
                a = yty + self.item_factors.T @ (cu - np.eye(n_items)) @ self.item_factors + reg_eye
                b = self.item_factors.T @ cu @ prefs[u]
                self.user_factors[u] = np.linalg.solve(a, b)
            xtx = self.user_factors.T @ self.user_factors
            for i in range(n_items):
                ci = np.diag(confidence[:, i])
                a = xtx + self.user_factors.T @ (ci - np.eye(n_users)) @ self.user_factors + reg_eye
                b = self.user_factors.T @ ci @ prefs[:, i]
                self.item_factors[i] = np.linalg.solve(a, b)
        return self

    def score_all_items(self, user_index: int) -> np.ndarray:
        if self.user_factors is None or self.item_factors is None:
            raise RuntimeError("Model has not been fit")
        return self.user_factors[user_index] @ self.item_factors.T
