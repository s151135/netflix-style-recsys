from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


def recall_at_k(recommended: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(recommended[:k]).intersection(relevant)) / len(relevant)


def mrr_at_k(recommended: Sequence[str], relevant: set[str], k: int) -> float:
    for rank, item_id in enumerate(recommended[:k], start=1):
        if item_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(recommended: Sequence[str], relevant: set[str], k: int) -> float:
    dcg = 0.0
    for rank, item_id in enumerate(recommended[:k], start=1):
        if item_id in relevant:
            dcg += 1.0 / math.log2(rank + 1)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return 0.0 if idcg == 0 else dcg / idcg


def evaluate_user_recommendations(
    recommendations: Mapping[str, Sequence[str]],
    holdout: Mapping[str, set[str]],
    k: int = 10,
) -> dict[str, float]:
    users = [u for u in holdout if u in recommendations]
    if not users:
        return {"recall": 0.0, "mrr": 0.0, "ndcg": 0.0, "users": 0.0}
    recalls = [recall_at_k(recommendations[u], holdout[u], k) for u in users]
    mrrs = [mrr_at_k(recommendations[u], holdout[u], k) for u in users]
    ndcgs = [ndcg_at_k(recommendations[u], holdout[u], k) for u in users]
    return {
        "recall": sum(recalls) / len(recalls),
        "mrr": sum(mrrs) / len(mrrs),
        "ndcg": sum(ndcgs) / len(ndcgs),
        "users": float(len(users)),
    }


def catalogue_coverage(recommendations: Mapping[str, Sequence[str]], catalogue_size: int) -> float:
    if catalogue_size <= 0:
        return 0.0
    exposed = {item for recs in recommendations.values() for item in recs}
    return len(exposed) / catalogue_size
