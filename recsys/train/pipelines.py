from __future__ import annotations

from pathlib import Path

from recsys.data.readers import write_table
from recsys.train.datasets import load_movielens_small
from recsys.train.trainer import TrainingResult, train_bpr_baseline


def run_local_smoke_pipeline(output_dir: str | Path) -> TrainingResult:
    out = Path(output_dir)
    interactions, items = load_movielens_small()
    write_table(interactions, out / "sample_interactions.csv")
    write_table(items, out / "sample_items.json")
    return train_bpr_baseline(interactions, out / "bpr_champion", factors=32, epochs=5, k=10)
