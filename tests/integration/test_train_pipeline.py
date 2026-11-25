import tempfile
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from recsys.train.datasets import make_synthetic_movie_dataset
from recsys.train.pipelines import run_local_smoke_pipeline


class TrainPipelineTests(unittest.TestCase):
    def test_smoke_pipeline_writes_model(self):
        dataset = make_synthetic_movie_dataset(
            n_users=8, n_items=20, events_per_user=8, seed=42
        )
        with tempfile.TemporaryDirectory() as tmp, patch(
            "recsys.train.pipelines.load_movielens_small", return_value=dataset
        ):
            result = run_local_smoke_pipeline(tmp)
            self.assertTrue((Path(result.artifact_dir) / "factors.npz").exists())
            self.assertIn("ndcg", result.metrics)
            self.assertTrue((Path(tmp) / "sample_interactions.csv").exists())
            self.assertTrue((Path(tmp) / "sample_items.json").exists())
            self.assertGreater(result.metrics["users"], 0)
            report = json.loads((Path(result.artifact_dir) / "evaluation.json").read_text())
            self.assertEqual(report["bpr"]["users"], report["popularity"]["users"])
            self.assertEqual(report["protocol"]["k"], 10)


if __name__ == "__main__":
    unittest.main()
