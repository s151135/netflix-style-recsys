import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from recsys.train.trainer import train_bpr_baseline


class TrainerEvaluationTests(unittest.TestCase):
    def test_disliked_holdouts_are_not_relevant_targets(self):
        interactions = pd.DataFrame([
            {"user_id": user, "item_id": item, "event_type": event,
             "timestamp": f"2020-01-0{day}T00:00:00Z"}
            for user, item, event, day in [
                ("u1", "a", "like", 1),
                ("u1", "b", "dislike", 2),
                ("u2", "b", "like", 1),
                ("u2", "a", "like", 2),
            ]
        ])
        with tempfile.TemporaryDirectory() as tmp:
            train_bpr_baseline(interactions, tmp, factors=2, epochs=1, k=1)
            report = json.loads((Path(tmp) / "evaluation.json").read_text())
        self.assertEqual(report["protocol"]["holdout_events"], 2)
        self.assertEqual(report["protocol"]["positive_holdout_events"], 1)
        for model in ["bpr", "popularity"]:
            self.assertEqual(report[model]["users"], 1)
            # u2 must receive a: its previously observed b must be excluded.
            self.assertEqual(report[model]["recall"], 1)
