from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SearchResult:
    item_id: str
    score: float


class NumpyANNIndex:
    def __init__(self, item_ids: list[str], embeddings: np.ndarray) -> None:
        if len(item_ids) != len(embeddings):
            raise ValueError("item_ids and embeddings must have the same length")
        self.item_ids = item_ids
        self.embeddings = _l2_normalise(np.asarray(embeddings, dtype=float))

    def search(self, query: np.ndarray, k: int = 100, exclude: set[str] | None = None) -> list[SearchResult]:
        exclude = exclude or set()
        q = _l2_normalise(np.asarray(query, dtype=float).reshape(1, -1))[0]
        scores = self.embeddings @ q
        order = scores.argsort()[::-1]
        results: list[SearchResult] = []
        for idx in order:
            item_id = self.item_ids[int(idx)]
            if item_id in exclude:
                continue
            results.append(SearchResult(item_id=item_id, score=float(scores[int(idx)])))
            if len(results) >= k:
                break
        return results


def _l2_normalise(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=-1, keepdims=True).clip(min=1e-12)
    return values / norms
