from __future__ import annotations

import pandas as pd


def user_genre_affinity(interactions: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
    if "genres" not in items.columns:
        return pd.DataFrame(columns=["user_id", "genre", "affinity"])
    df = interactions.merge(items[["item_id", "genres"]], on="item_id", how="left")
    df = df.explode("genres").dropna(subset=["genres"])
    if df.empty:
        return pd.DataFrame(columns=["user_id", "genre", "affinity"])
    grouped = df.groupby(["user_id", "genres"], as_index=False).size()
    grouped["affinity"] = grouped["size"] / grouped.groupby("user_id")["size"].transform("sum")
    return grouped.rename(columns={"genres": "genre"}).drop(columns=["size"])
