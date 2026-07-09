from __future__ import annotations

import pandas as pd

from recsys.serving.ranking_service import RankingService
from recsys.serving.retrieval_service import RetrievalService


def materialize_homepage_candidates(
    user_ids: list[str],
    retriever: RetrievalService,
    ranker: RankingService,
    k: int = 50,
) -> pd.DataFrame:
    rows = []
    for user_id in user_ids:
        candidates = retriever.get_candidates(user_id=user_id, k=max(k * 10, 200))
        ranked = ranker.rank(candidates, top_k=k)
        rows.append(
            {
                "user_id": user_id,
                "ranked_item_ids": [row["item_id"] for row in ranked],
                "scores": [row["score"] for row in ranked],
            }
        )
    return pd.DataFrame(rows)
