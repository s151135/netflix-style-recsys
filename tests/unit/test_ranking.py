import unittest

import pandas as pd

from recsys.serving.api import _dedupe_candidates
from recsys.serving.ranking_service import RankingService


class RankingSignalTests(unittest.TestCase):
    def test_content_scores_cannot_replace_model_scores_or_claim_learning(self):
        items = pd.DataFrame([
            {"item_id": item, "title": item, "genres": ["Drama"], "release_year": 2000,
             "rating_count": 10} for item in ["a", "b", "c"]
        ])
        candidates = [
            {"item_id": "a", "retrieval_score": 2.0, "source": "collaborative"},
            {"item_id": "b", "retrieval_score": 1.0, "source": "collaborative"},
            {"item_id": "c", "retrieval_score": 10000.0, "source": "content"},
            {"item_id": "a", "retrieval_score": 10000.0, "source": "content"},
        ]
        ranked = RankingService(items).rank(_dedupe_candidates(candidates),
                                            context={"preferred_genres": []}, top_k=3)
        by_id = {row["item_id"]: row for row in ranked}
        self.assertEqual(ranked[0]["item_id"], "a")
        self.assertEqual(by_id["a"]["retrieval_score"], 2.0)
        self.assertEqual(by_id["a"]["source"], "collaborative")
        self.assertEqual(by_id["c"]["signals"]["learned_preference"], 0.0)
        self.assertIn("not scored by the learned model", by_id["c"]["reason"])
        self.assertNotIn("interest in", by_id["c"]["reason"])
