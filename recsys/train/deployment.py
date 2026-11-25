"""Package a selected model with the exact history it was trained on."""
from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from recsys.data.readers import write_table
from recsys.data.splits import temporal_train_validation_test
from recsys.data.transforms import normalise_interactions
from recsys.train.registry import load_factor_model, save_factor_model
from recsys.train.datasets import encode_interactions


def prepare_demo(interactions, items, experiment_dir, output_dir) -> Path:
    experiment = Path(experiment_dir)
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"Choose a new output directory; already exists: {output}")
    report = json.loads((experiment / "evaluation.json").read_text())
    threshold = report["protocol"]["rating_threshold"]
    data = interactions.copy()
    data["event_type"] = np.where(data["rating"] >= threshold, "like", "dislike")
    data = normalise_interactions(data)
    source = data[["user_id", "item_id", "rating", "timestamp"]].to_csv(index=False)
    if hashlib.sha256(source.encode()).hexdigest() != report["protocol"]["dataset_sha256"]:
        raise ValueError("Dataset does not match the selected experiment")
    train, validation, _ = temporal_train_validation_test(data)
    history = pd.concat([train, validation], ignore_index=True)
    model = load_factor_model(experiment / "selected_bpr")
    if model["metadata"].get("config") != report["selected_config"]:
        raise ValueError("Model configuration does not match the experiment")
    encoded = encode_interactions(history)
    if model["users"] != encoded.index_to_user or model["items"] != encoded.index_to_item:
        raise ValueError("Model IDs do not match the selected training history")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as tmp:
        stage = Path(tmp) / "bundle"
        save_factor_model(
            stage / "bpr_champion", model["user_factors"], model["item_factors"],
            model["item_bias"], model["users"], model["items"],
            metadata=model["metadata"] | {"validation_selected": True,
                                         "rating_threshold": threshold},
        )
        write_table(history, stage / "sample_interactions.csv")
        write_table(items, stage / "sample_items.json")
        shutil.copyfile(experiment / "evaluation.json", stage / "evaluation.json")
        notice = Path(__file__).resolve().parents[2] / "notices" / "MOVIELENS_README.txt"
        if notice.exists():
            shutil.copyfile(notice, stage / "MOVIELENS_README.txt")
        stage.rename(output)
    return output


def build_demo_from_report(interactions, items, report_path, output_dir) -> Path:
    """Reproduce the recorded configuration without searching or scoring test data."""
    from recsys.models.bpr import BPRConfig
    from recsys.train.tuning import _fit

    report_path = Path(report_path)
    report = json.loads(report_path.read_text())
    data = interactions.copy()
    data["event_type"] = np.where(
        data["rating"] >= report["protocol"]["rating_threshold"], "like", "dislike"
    )
    data = normalise_interactions(data)
    source = data[["user_id", "item_id", "rating", "timestamp"]].to_csv(index=False)
    if hashlib.sha256(source.encode()).hexdigest() != report["protocol"]["dataset_sha256"]:
        raise ValueError("Dataset does not match the recorded experiment; rerun tuning")
    train, validation, _ = temporal_train_validation_test(data)
    history = pd.concat([train, validation], ignore_index=True)
    config = BPRConfig(**report["selected_config"])
    model, encoded, _ = _fit(history, config)
    with tempfile.TemporaryDirectory() as tmp:
        experiment = Path(tmp)
        save_factor_model(
            experiment / "selected_bpr", model.user_factors, model.item_factors, model.item_bias,
            encoded.index_to_user, encoded.index_to_item,
            metadata={"model": "bpr", "config": report["selected_config"]},
        )
        shutil.copyfile(report_path, experiment / "evaluation.json")
        return prepare_demo(interactions, items, experiment, output_dir)
