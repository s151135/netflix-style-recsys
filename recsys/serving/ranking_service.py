from __future__ import annotations

import pandas as pd

from recsys.models.rerankers import maximal_marginal_relevance
from recsys.models.xgb_ranker import HeuristicRanker
from recsys.serving.business_rules import apply_business_rules


class RankingService:
    def __init__(self, items: pd.DataFrame | None = None, ranker: HeuristicRanker | None = None) -> None:
        self.items = items if items is not None else pd.DataFrame(columns=["item_id"])
        self.ranker = ranker or HeuristicRanker()

    def rank(
        self,
        candidates: list[dict[str, float | str]],
        context: dict | None = None,
        top_k: int = 20,
        diversity_lambda: float = 0.2,
    ) -> list[dict[str, float | str]]:
        df = pd.DataFrame(candidates)
        if df.empty:
            return []
        if not self.items.empty:
            df = df.merge(self.items, on="item_id", how="left")
        df = apply_business_rules(df, context=context)
        ranked = self.ranker.rank(df, top_k=max(top_k * 2, top_k))
        reranked = maximal_marginal_relevance(
            ranked, top_k=top_k, diversity_lambda=diversity_lambda
        )
        return [
            {
                "item_id": str(row.item_id),
                "title": str(getattr(row, "title", row.item_id)),
                "genres": _normalise_genres(getattr(row, "genres", [])),
                "language": _optional_str(getattr(row, "language", None)),
                "release_year": _optional_int(getattr(row, "release_year", None)),
                "avg_rating": _optional_float(getattr(row, "avg_rating", None)),
                "rating_count": _optional_int(getattr(row, "rating_count", None)),
                "poster_seed": _optional_str(getattr(row, "poster_seed", row.item_id)),
                "score": float(getattr(row, "rank_score", 0.0)),
                "retrieval_score": float(getattr(row, "retrieval_score", 0.0)),
            }
            for row in reranked.itertuples(index=False)
        ]


def _normalise_genres(value) -> list[str]:
    if isinstance(value, list):
        return [str(x) for x in value]
    if isinstance(value, str):
        return [x.strip() for x in value.replace(",", "|").split("|") if x.strip()]
    return []


def _optional_str(value) -> str | None:
    if pd.isna(value):
        return None
    return str(value)


def _optional_int(value) -> int | None:
    if pd.isna(value):
        return None
    return int(value)


def _optional_float(value) -> float | None:
    if pd.isna(value):
        return None
    return float(value)
