from __future__ import annotations

from pathlib import Path

import numpy as np

from recsys.train.registry import load_factor_model


def export_item_embeddings(model_dir: str | Path, output_path: str | Path) -> None:
    artifact = load_factor_model(model_dir)
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, item_ids=np.array(artifact["items"]), embeddings=artifact["item_factors"])
