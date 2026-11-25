import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from recsys.data.splits import temporal_train_validation_test
from recsys.models.bpr import BPRConfig
from recsys.train.tuning import _evaluate, tune_bpr


def ratings_fixture():
    return pd.DataFrame([
        {"event_id": f"{user}-{day}", "user_id": user,
         "item_id": f"item_{(offset + day) % 7}", "rating": 5.0,
         "timestamp": f"2020-01-0{day}T00:00:00Z"}
        for offset, user in enumerate(["a", "b", "c"])
        for day in range(1, 6)
    ])


class TuningTests(unittest.TestCase):
    def test_split_keeps_short_histories_and_preserves_event_partition(self):
        data = ratings_fixture()
        short = data.iloc[:2].assign(user_id="short", event_id=["short-1", "short-2"])
        data = pd.concat([data, short], ignore_index=True).sample(frac=1, random_state=9)
        train, validation, test = temporal_train_validation_test(data)
        self.assertEqual(set(validation["event_id"]), {f"{u}-4" for u in "abc"})
        self.assertEqual(set(test["event_id"]), {f"{u}-5" for u in "abc"})
        self.assertEqual(len(train[train["user_id"] == "short"]), 2)
        self.assertEqual(len(train) + len(validation) + len(test), len(data))
        self.assertFalse(set(train["event_id"]) & set(test["event_id"]))

    def test_test_events_are_only_scored_after_validation_selection(self):
        data = ratings_fixture()
        configs = (BPRConfig(factors=2, epochs=1), BPRConfig(factors=3, epochs=2))
        test_ids = {f"{u}-5" for u in "abc"}
        with tempfile.TemporaryDirectory() as tmp, patch(
            "recsys.train.tuning._evaluate", wraps=_evaluate
        ) as evaluate:
            report = tune_bpr(data, tmp, configs=configs, k=2, progress=lambda _: None)
            self.assertTrue((Path(tmp) / "selected_bpr" / "factors.npz").exists())
            self.assertTrue((Path(tmp) / "evaluation.json").exists())
            self.assertEqual(evaluate.call_count, 3)
            for call in evaluate.call_args_list:
                self.assertFalse(test_ids & set(call.args[0]["event_id"]))
            for call in evaluate.call_args_list[:-1]:
                self.assertFalse(test_ids & set(call.args[1]["event_id"]))
            self.assertEqual(set(evaluate.call_args_list[-1].args[1]["event_id"]), test_ids)
            expected = max(range(2), key=lambda i: report["trials"][i]["validation"]["bpr"]["ndcg"])
            self.assertEqual(report["selected_trial"], expected + 1)
            # Changing final-test labels must not affect model selection or validation scores.
            changed = data.copy()
            changed.loc[changed["event_id"].isin(test_ids), "rating"] = 1.0
            second = tune_bpr(changed, tmp, configs=configs, k=2, progress=lambda _: None)
            self.assertEqual(report["trials"], second["trials"])
            self.assertEqual(report["selected_config"], second["selected_config"])
            self.assertEqual(second["test"]["counts"]["evaluated_users"], 0)
