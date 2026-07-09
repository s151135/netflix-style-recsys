from __future__ import annotations

import pandas as pd


def temporal_leave_last_k(
    interactions: pd.DataFrame,
    k: int = 1,
    min_train_events: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if k < 1:
        raise ValueError("k must be >= 1")
    df = interactions.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values(["user_id", "timestamp", "item_id"]).reset_index(drop=True)
    group_index = df.groupby("user_id").cumcount(ascending=False)
    user_sizes = df.groupby("user_id")["item_id"].transform("size")
    holdout_mask = (group_index < k) & (user_sizes > min_train_events)
    train = df.loc[~holdout_mask].reset_index(drop=True)
    test = df.loc[holdout_mask].reset_index(drop=True)
    return train, test
