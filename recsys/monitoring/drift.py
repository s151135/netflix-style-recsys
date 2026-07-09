from __future__ import annotations

import pandas as pd


def null_rate_drift(reference: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    columns = sorted(set(reference.columns).intersection(current.columns))
    rows = []
    for col in columns:
        ref = float(reference[col].isna().mean())
        cur = float(current[col].isna().mean())
        rows.append({"column": col, "reference_null_rate": ref, "current_null_rate": cur, "delta": cur - ref})
    return pd.DataFrame(rows)
