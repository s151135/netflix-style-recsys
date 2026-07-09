import tempfile
import unittest
from pathlib import Path

from recsys.train.pipelines import run_local_smoke_pipeline


class TrainPipelineTests(unittest.TestCase):
    def test_smoke_pipeline_writes_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = run_local_smoke_pipeline(tmp)
            self.assertTrue((Path(result.artifact_dir) / "factors.npz").exists())
            self.assertIn("ndcg", result.metrics)


if __name__ == "__main__":
    unittest.main()
