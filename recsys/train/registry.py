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
    arrays = np.load(p / "factors.npz")
    meta = json.loads((p / "metadata.json").read_text())
    return {
        "user_factors": arrays["user_factors"],
        "item_factors": arrays["item_factors"],
        "item_bias": arrays["item_bias"],
        "users": meta["users"],
        "items": meta["items"],
        "metadata": meta.get("metadata", {}),
    }
