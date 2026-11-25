"""Validation-selected BPR benchmark; final test data is scored only after selection."""
from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from recsys.data.splits import temporal_train_validation_test
from recsys.data.transforms import normalise_interactions
from recsys.models.bpr import BPRConfig, BPRMatrixFactorization
from recsys.models.popularity import PopularityModel
from recsys.train.datasets import encode_interactions
from recsys.train.evaluate import catalogue_coverage, evaluate_user_recommendations
from recsys.train.registry import save_factor_model


DEFAULT_TRIALS = (
    BPRConfig(factors=16, epochs=5),
    BPRConfig(factors=32, epochs=10),
    BPRConfig(factors=32, epochs=20, learning_rate=0.01, regularization=0.001),
)


def _fit(train: pd.DataFrame, config: BPRConfig):
    encoded = encode_interactions(train)
    if not encoded.positive_pairs:
        raise ValueError("Training data must contain positive interactions")
    model = BPRMatrixFactorization(config).fit(
        encoded.positive_pairs, len(encoded.index_to_user), len(encoded.index_to_item),
        encoded.user_seen,
    )
    # Both models learn from the same positive interactions and candidate items.
    popularity = PopularityModel().fit(train[train["label"] > 0])
    return model, encoded, popularity


def _evaluate(train, holdout, model, encoded, popularity, k):
    relevant_rows = holdout[holdout["label"] > 0]
    relevant = relevant_rows.groupby("user_id")["item_id"].apply(set).to_dict()
    seen = train.groupby("user_id")["item_id"].apply(set).to_dict()
    bpr_recs, popular_recs = {}, {}
    popular_order = [item for item, _ in popularity.recommend(k=len(popularity.scores))]
    for user in relevant:
        excluded = seen.get(user, set())
        fallback = [item for item in popular_order if item not in excluded][:k]
        popular_recs[user] = fallback
        if user not in encoded.user_to_index:
            bpr_recs[user] = fallback
            continue
        scores = model.score_all_items(encoded.user_to_index[user])
        order = np.argsort(-scores, kind="stable")
        bpr_recs[user] = [encoded.index_to_item[i] for i in order
                          if encoded.index_to_item[i] not in excluded][:k]
    result = {}
    for name, recs in [("bpr", bpr_recs), ("popularity", popular_recs)]:
        result[name] = evaluate_user_recommendations(recs, relevant, k)
        result[name]["catalogue_coverage"] = catalogue_coverage(recs, len(encoded.index_to_item))
    result["counts"] = {
        "holdout_events": len(holdout),
        "positive_holdout_events": len(relevant_rows),
        "evaluated_users": len(relevant),
        "cold_start_users": sum(user not in encoded.user_to_index for user in relevant),
        "cold_start_targets": int((~relevant_rows["item_id"].isin(encoded.item_to_index)).sum()),
        "already_seen_targets": sum(
            item in seen.get(user, set()) for user, targets in relevant.items() for item in targets
        ),
        "candidate_items": len(encoded.index_to_item),
    }
    return result


def tune_bpr(
    interactions: pd.DataFrame,
    output_dir: str | Path,
    configs: tuple[BPRConfig, ...] = DEFAULT_TRIALS,
    k: int = 10,
    rating_threshold: float = 4.0,
    progress: Callable[[str], None] = print,
) -> dict:
    if not configs or k < 1:
        raise ValueError("Provide at least one configuration and k >= 1")
    if "rating" not in interactions:
        raise ValueError("This MovieLens experiment requires explicit ratings")
    if not 0.5 <= rating_threshold <= 5.0:
        raise ValueError("rating_threshold must be between 0.5 and 5.0")
    # Keep low ratings in the history/split, but learn only from liked movies.
    data = interactions.copy()
    data["event_type"] = np.where(data["rating"] >= rating_threshold, "like", "dislike")
    data = normalise_interactions(data)
    train, validation, test = temporal_train_validation_test(data)
    if validation[validation["label"] > 0].empty:
        raise ValueError("Validation split needs at least one positive target")
    if test.empty:
        raise ValueError("At least one user needs three events for a test split")
    trials = []
    for index, config in enumerate(configs):
        progress(f"Validation trial {index + 1}/{len(configs)}: {asdict(config)}")
        model, encoded, popularity = _fit(train, config)
        scores = _evaluate(train, validation, model, encoded, popularity, k)
        trials.append({"config": asdict(config), "validation": scores})
        progress(f"Validation NDCG@{k}: {scores['bpr']['ndcg']:.6f}")
    # Stable tie-breaking: the first configuration wins an exact NDCG tie.
    best_index = max(range(len(trials)), key=lambda i: trials[i]["validation"]["bpr"]["ndcg"])
    selected = configs[best_index]
    progress(f"Selected trial {best_index + 1}; refitting on train + validation")
    final_train = pd.concat([train, validation], ignore_index=True)
    model, encoded, popularity = _fit(final_train, selected)
    final_scores = _evaluate(final_train, test, model, encoded, popularity, k)
    source = data[["user_id", "item_id", "rating", "timestamp"]].to_csv(index=False)
    report = {
        "environment": {
            "python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "platform": platform.platform(),
        },
        "protocol": {
            "split": "per-user last event test, penultimate validation; fewer than 3 events train-only",
            "rating_threshold": rating_threshold,
            "selection_metric": "validation bpr ndcg",
            "tie_break": "first configuration",
            "k": k,
            "dataset_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "train_events": len(train), "validation_events": len(validation), "test_events": len(test),
            "final_train_events": len(final_train),
            "limitations": [
                "Per-user chronology is not a global time cutoff; timestamp ties use item ID.",
                "One seed and one split; differences do not establish statistical significance.",
                "Cold-start targets remain as misses; unknown users use training popularity.",
                "Offline retrieval only; excludes curated UI and reranking quality.",
                "This dataset was explored previously; use a new dataset for an external confirmation.",
            ],
        },
        "trials": trials, "selected_trial": best_index + 1,
        "selected_config": asdict(selected), "test": final_scores,
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    save_factor_model(
        output / "selected_bpr", model.user_factors, model.item_factors, model.item_bias,
        encoded.index_to_user, encoded.index_to_item,
        metadata={"model": "bpr", "config": asdict(selected), "metrics": final_scores["bpr"]},
    )
    (output / "evaluation.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report
