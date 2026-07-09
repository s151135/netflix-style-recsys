from __future__ import annotations

from pathlib import Path

import numpy as np

from recsys.train.registry import load_factor_model


class RetrievalService:
    def __init__(
        self,
        user_factors: np.ndarray,
        item_factors: np.ndarray,
        item_bias: np.ndarray,
        users: list[str],
        items: list[str],
    ) -> None:
        self.user_factors = user_factors
        self.item_factors = item_factors
        self.item_bias = item_bias
        self.users = users
        self.items = items
        self.user_to_index = {u: i for i, u in enumerate(users)}

    @classmethod
    def load(cls, path: str | Path) -> "RetrievalService":
        artifact = load_factor_model(path)
        return cls(
            artifact["user_factors"],
            artifact["item_factors"],
            artifact["item_bias"],
            artifact["users"],
            artifact["items"],
        )

    def get_candidates(
        self,
        user_id: str,
        k: int = 200,
        exclude_item_ids: set[str] | None = None,
    ) -> list[dict[str, float | str]]:
        exclude_item_ids = exclude_item_ids or set()
        if user_id not in self.user_to_index:
            return self._popular_fallback(k, exclude_item_ids)
        uidx = self.user_to_index[user_id]
        scores = self.user_factors[uidx] @ self.item_factors.T + self.item_bias
        order = scores.argsort()[::-1]
        rows = []
        for idx in order:
            item_id = self.items[int(idx)]
            if item_id in exclude_item_ids:
                continue
            rows.append({"item_id": item_id, "retrieval_score": float(scores[int(idx)])})
            if len(rows) >= k:
                break
        return rows

    def _popular_fallback(self, k: int, exclude_item_ids: set[str]) -> list[dict[str, float | str]]:
        scores = self.item_bias
        order = scores.argsort()[::-1]
        rows = []
        for idx in order:
            item_id = self.items[int(idx)]
            if item_id in exclude_item_ids:
                continue
            rows.append({"item_id": item_id, "retrieval_score": float(scores[int(idx)])})
            if len(rows) >= k:
                break
        return rows
