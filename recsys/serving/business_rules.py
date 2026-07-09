from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd


def apply_business_rules(
    candidates: pd.DataFrame,
    context: dict | None = None,
    now: datetime | None = None,
) -> pd.DataFrame:
    context = context or {}
    now_ts = pd.Timestamp(now or datetime.now(timezone.utc))
    df = candidates.copy()
    if "availability_start" in df.columns:
        start = pd.to_datetime(df["availability_start"], utc=True, errors="coerce")
        df = df[start.isna() | (start <= now_ts)]
    if "availability_end" in df.columns:
        end = pd.to_datetime(df["availability_end"], utc=True, errors="coerce")
        df = df[end.isna() | (end >= now_ts)]
    if context.get("is_kids_profile") and "maturity_rating" in df.columns:
        df = df[df["maturity_rating"].isin(["G", "PG", None]) | df["maturity_rating"].isna()]
    blocked = set(context.get("blocked_item_ids", []))
    if blocked:
        df = df[~df["item_id"].astype(str).isin(blocked)]
    return df.reset_index(drop=True)
