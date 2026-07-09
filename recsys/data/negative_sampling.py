from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


def sample_negatives(
    positives: pd.DataFrame,
    all_item_ids: Iterable[str],
    negatives_per_positive: int = 4,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    item_ids = np.array(sorted(set(all_item_ids)))
    if len(item_ids) == 0:
        raise ValueError("Cannot sample negatives from an empty item universe")
    seen = positives.groupby("user_id")["item_id"].agg(set).to_dict()
    rows: list[dict[str, object]] = []
    for user_id, item_id in positives[["user_id", "item_id"]].itertuples(index=False):
        blocked = seen.get(user_id, set())
        sampled = 0
        attempts = 0
        while sampled < negatives_per_positive and attempts < negatives_per_positive * 50:
            neg = str(rng.choice(item_ids))
            attempts += 1
            if neg in blocked:
                continue
            rows.append({"user_id": user_id, "item_id": neg, "label": 0, "source_item_id": item_id})
            sampled += 1
    return pd.DataFrame(rows)
