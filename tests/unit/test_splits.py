import unittest

import pandas as pd

from recsys.data.splits import temporal_leave_last_k


class SplitTests(unittest.TestCase):
    def test_leave_last_k_is_temporal_per_user(self):
        df = pd.DataFrame(
            [
                {"user_id": "u1", "item_id": "a", "timestamp": "2026-01-01T00:00:00Z"},
                {"user_id": "u1", "item_id": "b", "timestamp": "2026-01-02T00:00:00Z"},
                {"user_id": "u2", "item_id": "c", "timestamp": "2026-01-01T00:00:00Z"},
                {"user_id": "u2", "item_id": "d", "timestamp": "2026-01-03T00:00:00Z"},
            ]
        )
        train, test = temporal_leave_last_k(df, k=1)
        self.assertEqual(set(test["item_id"]), {"b", "d"})
        self.assertEqual(set(train["item_id"]), {"a", "c"})

    def test_large_holdout_preserves_minimum_training_history(self):
        data = pd.DataFrame([{"user_id": "u", "item_id": str(i),
                              "timestamp": f"2020-01-0{i + 1}"} for i in range(4)])
        train, test = temporal_leave_last_k(data, k=100, min_train_events=2)
        self.assertEqual(len(train), 2)
        self.assertEqual(len(test), 2)


if __name__ == "__main__":
    unittest.main()
