from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BPRConfig:
    factors: int = 32
    learning_rate: float = 0.03
    regularization: float = 1e-4
    epochs: int = 8
    negatives_per_positive: int = 2
    seed: int = 42


class BPRMatrixFactorization:
    def __init__(self, config: BPRConfig | None = None) -> None:
        self.config = config or BPRConfig()
        self.user_factors: np.ndarray | None = None
        self.item_factors: np.ndarray | None = None
        self.item_bias: np.ndarray | None = None

    def fit(
        self,
        positive_pairs: list[tuple[int, int]],
        n_users: int,
        n_items: int,
        user_seen: dict[int, set[int]] | None = None,
    ) -> "BPRMatrixFactorization":
        rng = np.random.default_rng(self.config.seed)
        f = self.config.factors
        self.user_factors = rng.normal(0, 0.05, size=(n_users, f))
        self.item_factors = rng.normal(0, 0.05, size=(n_items, f))
        self.item_bias = np.zeros(n_items, dtype=float)
        user_seen = user_seen or {}
        pairs = list(positive_pairs)
        for _ in range(self.config.epochs):
            rng.shuffle(pairs)
            for u, i in pairs:
                seen = user_seen.get(u, set())
                for _neg in range(self.config.negatives_per_positive):
                    j = int(rng.integers(0, n_items))
                    attempts = 0
                    while j in seen and attempts < 50:
                        j = int(rng.integers(0, n_items))
                        attempts += 1
                    self._update(u, i, j)
        return self

    def _update(self, u: int, i: int, j: int) -> None:
        assert self.user_factors is not None
        assert self.item_factors is not None
        assert self.item_bias is not None
        lr = self.config.learning_rate
        reg = self.config.regularization
        user = self.user_factors[u].copy()
        pos = self.item_factors[i].copy()
        neg = self.item_factors[j].copy()
        x = self.item_bias[i] - self.item_bias[j] + user @ (pos - neg)
        grad = 1.0 / (1.0 + np.exp(x))
        self.user_factors[u] += lr * (grad * (pos - neg) - reg * user)
        self.item_factors[i] += lr * (grad * user - reg * pos)
        self.item_factors[j] += lr * (-grad * user - reg * neg)
        self.item_bias[i] += lr * (grad - reg * self.item_bias[i])
        self.item_bias[j] += lr * (-grad - reg * self.item_bias[j])

    def score_all_items(self, user_index: int) -> np.ndarray:
        if self.user_factors is None or self.item_factors is None or self.item_bias is None:
            raise RuntimeError("Model has not been fit")
        return self.user_factors[user_index] @ self.item_factors.T + self.item_bias
