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


if __name__ == "__main__":
    unittest.main()
