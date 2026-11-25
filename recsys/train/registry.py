from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def save_factor_model(
    path: str | Path,
    user_factors: np.ndarray,
    item_factors: np.ndarray,
    item_bias: np.ndarray | None,
    users: list[str],
    items: list[str],
    metadata: dict[str, Any] | None = None,
) -> None:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        p / "factors.npz",
        user_factors=user_factors,
        item_factors=item_factors,
        item_bias=np.zeros(item_factors.shape[0]) if item_bias is None else item_bias,
    )
    (p / "metadata.json").write_text(
        json.dumps(
            {
                "users": users,
                "items": items,
                "metadata": metadata or {},
            },
            indent=2,
            sort_keys=True,
        )
    )


def load_factor_model(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    with np.load(p / "factors.npz", allow_pickle=False) as arrays:
        user_factors = arrays["user_factors"].copy()
        item_factors = arrays["item_factors"].copy()
        item_bias = arrays["item_bias"].copy()
    meta = json.loads((p / "metadata.json").read_text())
    users, items = meta["users"], meta["items"]
    if (not all(isinstance(value, str) for value in users + items)
            or len(set(users)) != len(users) or len(set(items)) != len(items)):
        raise ValueError("Model IDs must be unique strings")
    if (user_factors.ndim != 2 or item_factors.ndim != 2
            or user_factors.shape[0] != len(users) or item_factors.shape[0] != len(items)
            or user_factors.shape[1] != item_factors.shape[1]
            or item_bias.shape != (len(items),)
            or not len(users) or not len(items) or not user_factors.shape[1]):
        raise ValueError("Model factor shapes do not match its IDs")
    if not all(np.isfinite(value).all() for value in (user_factors, item_factors, item_bias)):
        raise ValueError("Model factors must be finite")
    return {
        "user_factors": user_factors,
        "item_factors": item_factors,
        "item_bias": item_bias,
        "users": users,
        "items": items,
        "metadata": meta.get("metadata", {}),
    }
