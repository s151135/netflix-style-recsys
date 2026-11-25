from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import pandas as pd

from recsys.data.splits import temporal_leave_last_k
from recsys.data.transforms import normalise_interactions
from recsys.models.bpr import BPRConfig, BPRMatrixFactorization
from recsys.models.popularity import PopularityModel
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
    interactions = normalise_interactions(interactions)
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
    popularity = PopularityModel().fit(train)
    popularity_recommendations: dict[str, list[str]] = {}
    # Exclude all observed training titles, including dislikes, from both models.
    observed = train.groupby("user_id")["item_id"].apply(lambda s: set(map(str, s))).to_dict()
    for user_id, uidx in encoded.user_to_index.items():
        scores = model.score_all_items(uidx)
        seen = observed[user_id]
        ordered = [
            encoded.index_to_item[i]
            for i in scores.argsort()[::-1]
            if encoded.index_to_item[i] not in seen
        ][:k]
        recommendations[user_id] = ordered
        popularity_recommendations[user_id] = [
            item for item, _ in popularity.recommend(seen_items=seen, k=len(popularity.scores))
            if item in encoded.item_to_index
        ][:k]
    # Negative feedback is not a relevant recommendation target.
    positive_holdout = holdout[holdout["label"] > 0]
    holdout_map = (
        positive_holdout.groupby("user_id")["item_id"].apply(lambda s: set(map(str, s))).to_dict()
        if not positive_holdout.empty
        else {}
    )
    metrics = evaluate_user_recommendations(recommendations, holdout_map, k=k)
    metrics["catalogue_coverage"] = catalogue_coverage(recommendations, len(encoded.index_to_item))
    baseline_metrics = evaluate_user_recommendations(popularity_recommendations, holdout_map, k=k)
    baseline_metrics["catalogue_coverage"] = catalogue_coverage(
        popularity_recommendations, len(encoded.index_to_item)
    )
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
    (artifact_dir / "evaluation.json").write_text(json.dumps({
        "protocol": {
            "split": "per-user temporal leave-last-one",
            "relevance": "positive event label; MovieLens ratings >= 2.5",
            "candidates": "all positive training items; all observed training titles excluded",
            "k": k,
            "seed": model.config.seed,
            "factors": factors,
            "epochs": epochs,
            "train_events": len(train),
            "holdout_events": len(holdout),
            "positive_holdout_events": len(positive_holdout),
            "unseen_holdout_items": int((~positive_holdout["item_id"].isin(encoded.item_to_index)).sum()),
            "limitations": [
                "Per-user split does not enforce a global time cutoff.",
                "Ratings are proxy interactions, not real streaming behaviour.",
                "Retrieval evaluation does not measure the curated UI or reranker.",
                "Training-catalogue cold-start targets remain in the denominator.",
            ],
        },
        "bpr": metrics,
        "popularity": baseline_metrics,
    }, indent=2, sort_keys=True) + "\n")
    return TrainingResult(artifact_dir=artifact_dir, metrics=metrics)
