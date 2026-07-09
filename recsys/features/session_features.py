from __future__ import annotations

import pandas as pd


def recent_session_sequences(interactions: pd.DataFrame, max_len: int = 100) -> dict[str, list[str]]:
    df = interactions.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values(["user_id", "timestamp"])
    return {
        user_id: group["item_id"].tail(max_len).astype(str).tolist()
        for user_id, group in df.groupby("user_id")
    }
