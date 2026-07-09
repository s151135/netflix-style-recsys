from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from recsys.data.splits import temporal_leave_last_k
from recsys.models.bpr import BPRConfig, BPRMatrixFactorization
from recsys.train.datasets import encode_interactions
from recsys.train.evaluate import catalogue_coverage, evaluate_user_recommendations
from recsys.train.registry import save_factor_model


@dataclass
class TrainingResult:
    artifact_dir: Path
    metrics: dict[str, float]


def train_bpr_baseline(
    interactions: pd.DataFrame,
    output_dir: str | Path,
    factors: int = 32,
    epochs: int = 8,
    k: int = 10,
) -> TrainingResult:
    train, holdout = temporal_leave_last_k(interactions, k=1, min_train_events=1)
    encoded = encode_interactions(train)
    model = BPRMatrixFactorization(BPRConfig(factors=factors, epochs=epochs))
    model.fit(
        encoded.positive_pairs,
        n_users=len(encoded.index_to_user),
        n_items=len(encoded.index_to_item),
        user_seen=encoded.user_seen,
    )
    recommendations: dict[str, list[str]] = {}
    for user_id, uidx in encoded.user_to_index.items():
        scores = model.score_all_items(uidx)
        seen = encoded.user_seen.get(uidx, set())
        ordered = [
            encoded.index_to_item[i]
            for i in scores.argsort()[::-1]
            if i not in seen
        ][:k]
        recommendations[user_id] = ordered
    holdout_map = (
        holdout.groupby("user_id")["item_id"].apply(lambda s: set(map(str, s))).to_dict()
        if not holdout.empty
        else {}
    )
    metrics = evaluate_user_recommendations(recommendations, holdout_map, k=k)
    metrics["catalogue_coverage"] = catalogue_coverage(recommendations, len(encoded.index_to_item))
    artifact_dir = Path(output_dir)
    save_factor_model(
        artifact_dir,
        model.user_factors,
        model.item_factors,
        model.item_bias,
        encoded.index_to_user,
        encoded.index_to_item,
        metadata={"model": "bpr", "metrics": metrics},
    )
    return TrainingResult(artifact_dir=artifact_dir, metrics=metrics)
