from __future__ import annotations

import numpy as np
import pandas as pd


class HeuristicRanker:
    """Dependency-light ranker; can be replaced by XGBoost LambdaMART."""

    def __init__(self, weights: dict[str, float] | None = None) -> None:
        self.weights = weights or {
            "retrieval_score": 1.0,
            "freshness_score": 0.25,
            "popularity_score": 0.15,
            "genre_affinity": 0.20,
        }

    def score(self, features: pd.DataFrame) -> np.ndarray:
        values = np.zeros(len(features), dtype=float)
        for name, weight in self.weights.items():
            if name in features.columns:
                col = pd.to_numeric(features[name], errors="coerce").fillna(0.0).to_numpy()
                values += weight * col
        return values

    def rank(self, features: pd.DataFrame, top_k: int = 20) -> pd.DataFrame:
        ranked = features.copy()
        ranked["rank_score"] = self.score(ranked)
        return ranked.sort_values("rank_score", ascending=False).head(top_k).reset_index(drop=True)
