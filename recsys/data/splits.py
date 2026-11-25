from __future__ import annotations

import pandas as pd


def temporal_train_validation_test(
    interactions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Reserve each eligible user's last two events for validation and test.

    Users with fewer than three events remain in training. Timestamp ties are
    ordered by item ID, as in temporal_leave_last_k; this is not a global cutoff.
    """
    sizes = interactions.groupby("user_id")["item_id"].transform("size")
    eligible = interactions.loc[sizes >= 3]
    remainder, test = temporal_leave_last_k(eligible)
    train, validation = temporal_leave_last_k(remainder)
    train = pd.concat([train, interactions.loc[sizes < 3]], ignore_index=True)
    return train, validation, test


def temporal_leave_last_k(
    interactions: pd.DataFrame,
    k: int = 1,
    min_train_events: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if k < 1:
        raise ValueError("k must be >= 1")
    if min_train_events < 1:
        raise ValueError("min_train_events must be >= 1")
    df = interactions.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values(["user_id", "timestamp", "item_id"]).reset_index(drop=True)
    group_index = df.groupby("user_id").cumcount(ascending=False)
    user_sizes = df.groupby("user_id")["item_id"].transform("size")
    holdout_counts = (user_sizes - min_train_events).clip(lower=0, upper=k)
    holdout_mask = group_index < holdout_counts
    train = df.loc[~holdout_mask].reset_index(drop=True)
    test = df.loc[holdout_mask].reset_index(drop=True)
    return train, test
