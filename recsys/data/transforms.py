from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd


REQUIRED_INTERACTION_COLUMNS = {"user_id", "item_id", "event_type", "timestamp"}
DEFAULT_EVENT_WEIGHTS = {
    "impression": 0.05,
    "click": 1.0,
    "play": 2.0,
    "complete": 4.0,
    "like": 5.0,
    "add_to_list": 3.0,
    "skip": -0.25,
    "dislike": -1.0,
}


def validate_interactions(interactions: pd.DataFrame) -> None:
    missing = REQUIRED_INTERACTION_COLUMNS.difference(interactions.columns)
    if missing:
        raise ValueError(f"Interactions missing required columns: {sorted(missing)}")
    if interactions[["user_id", "item_id", "event_type"]].isna().any().any():
        raise ValueError("Interactions contain null user_id, item_id, or event_type")


def normalise_interactions(
    interactions: pd.DataFrame,
    event_weights: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    validate_interactions(interactions)
    weights = dict(DEFAULT_EVENT_WEIGHTS if event_weights is None else event_weights)
    df = interactions.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["event_weight"] = df["event_type"].map(weights).fillna(0.0).astype(float)
    if "completion_ratio" in df.columns:
        df["event_weight"] = df["event_weight"] + df["completion_ratio"].fillna(0.0) * 0.5
    df["label"] = (df["event_weight"] > 0).astype(int)
    return df.sort_values(["user_id", "timestamp", "item_id"]).reset_index(drop=True)


def build_user_item_strength(interactions: pd.DataFrame) -> pd.DataFrame:
    df = normalise_interactions(interactions)
    agg = (
        df.groupby(["user_id", "item_id"], as_index=False)
        .agg(
            strength=("event_weight", "sum"),
            events=("event_id", "count") if "event_id" in df.columns else ("event_type", "count"),
            last_timestamp=("timestamp", "max"),
        )
        .query("strength > 0")
    )
    if agg.empty:
        return agg
    agg["confidence"] = np.log1p(agg["strength"].clip(lower=0.0))
    return agg.sort_values(["user_id", "last_timestamp"]).reset_index(drop=True)
