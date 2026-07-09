from __future__ import annotations

import pandas as pd


def item_feature_snapshot(interactions: pd.DataFrame, as_of: str | None = None) -> pd.DataFrame:
    df = interactions.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    if as_of is not None:
        df = df[df["timestamp"] <= pd.Timestamp(as_of, tz="UTC")]
    grouped = df.groupby("item_id", as_index=False).agg(
        item_events=("event_type", "count"),
        item_users=("user_id", "nunique"),
        last_event_ts=("timestamp", "max"),
    )
    return grouped


def user_feature_snapshot(interactions: pd.DataFrame, as_of: str | None = None) -> pd.DataFrame:
    df = interactions.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    if as_of is not None:
        df = df[df["timestamp"] <= pd.Timestamp(as_of, tz="UTC")]
    grouped = df.groupby("user_id", as_index=False).agg(
        user_events=("event_type", "count"),
        unique_items=("item_id", "nunique"),
        last_event_ts=("timestamp", "max"),
    )
    return grouped
