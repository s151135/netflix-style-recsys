from __future__ import annotations

import math
from datetime import datetime, timezone

import pandas as pd


def exponential_freshness_score(
    release_date: pd.Series,
    now: datetime | None = None,
    half_life_days: float = 21.0,
) -> pd.Series:
    now = now or datetime.now(timezone.utc)
    ts = pd.to_datetime(release_date, utc=True, errors="coerce")
    age_days = (pd.Timestamp(now) - ts).dt.total_seconds() / 86400.0
    decay = math.log(2.0) / max(half_life_days, 1e-6)
    return (-decay * age_days.clip(lower=0.0)).apply(math.exp).fillna(0.0)
