import tempfile
import unittest
from pathlib import Path

import pandas as pd

from recsys.data.splits import temporal_train_validation_test
from recsys.models.bpr import BPRConfig
from recsys.train.datasets import make_synthetic_movie_dataset
from recsys.train.deployment import prepare_demo
from recsys.train.registry import load_factor_model
from recsys.train.tuning import tune_bpr


class DeploymentTests(unittest.TestCase):
    def test_bundle_preserves_model_and_excludes_final_test_history(self):
        interactions, items = make_synthetic_movie_dataset(n_users=4, n_items=12, events_per_user=6)
        interactions["rating"] = 5.0
        with tempfile.TemporaryDirectory() as tmp:
            experiment, output = Path(tmp) / "experiment", Path(tmp) / "demo"
            tune_bpr(interactions, experiment, configs=(BPRConfig(factors=2, epochs=1),),
                     progress=lambda _: None)
            prepare_demo(interactions, items, experiment, output)
            history = pd.read_csv(output / "sample_interactions.csv")
            _, _, test = temporal_train_validation_test(interactions)
            self.assertFalse(set(history["event_id"]) & set(test["event_id"]))
            model = load_factor_model(output / "bpr_champion")
            self.assertTrue(model["metadata"]["validation_selected"])
            self.assertEqual(model["metadata"]["rating_threshold"], 4.0)
            self.assertEqual(model["item_factors"].shape[1], 2)
            with self.assertRaises(FileExistsError):
                prepare_demo(interactions, items, experiment, output)
            changed = interactions.copy()
            changed.loc[0, "rating"] = 1.0
            with self.assertRaisesRegex(ValueError, "Dataset does not match"):
                prepare_demo(changed, items, experiment, Path(tmp) / "wrong")

