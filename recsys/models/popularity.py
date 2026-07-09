from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


class PopularityModel:
    def __init__(self, scores: dict[str, float] | None = None) -> None:
        self.scores = scores or {}

    def fit(self, interactions: pd.DataFrame, item_col: str = "item_id") -> "PopularityModel":
        counts = interactions.groupby(item_col).size().sort_values(ascending=False)
        self.scores = {str(k): float(v) for k, v in counts.items()}
        return self

    def recommend(self, seen_items: set[str] | None = None, k: int = 20) -> list[tuple[str, float]]:
        seen_items = seen_items or set()
        ranked = ((item, score) for item, score in self.scores.items() if item not in seen_items)
        return sorted(ranked, key=lambda x: x[1], reverse=True)[:k]

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"scores": self.scores}, indent=2, sort_keys=True))

    @classmethod
    def load(cls, path: str | Path) -> "PopularityModel":
        return cls(json.loads(Path(path).read_text())["scores"])
