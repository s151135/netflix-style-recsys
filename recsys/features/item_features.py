from __future__ import annotations

import pandas as pd


def item_popularity_features(interactions: pd.DataFrame) -> pd.DataFrame:
    df = interactions.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    features = df.groupby("item_id", as_index=False).agg(
        impressions=("event_type", lambda s: int((s == "impression").sum())),
        clicks=("event_type", lambda s: int((s == "click").sum())),
        plays=("event_type", lambda s: int((s == "play").sum())),
        completes=("event_type", lambda s: int((s == "complete").sum())),
        unique_users=("user_id", "nunique"),
        last_event_ts=("timestamp", "max"),
    )
    features["ctr"] = features["clicks"] / features["impressions"].clip(lower=1)
    features["play_rate"] = features["plays"] / features["impressions"].clip(lower=1)
    return features
