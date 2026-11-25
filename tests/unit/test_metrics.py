import unittest

from recsys.train.evaluate import (
    catalogue_coverage, evaluate_user_recommendations, mrr_at_k, ndcg_at_k, recall_at_k,
)


class MetricTests(unittest.TestCase):
    def test_ranking_metrics(self):
        recs = ["a", "b", "c"]
        relevant = {"b", "d"}
        self.assertEqual(recall_at_k(recs, relevant, 2), 0.5)
        self.assertEqual(mrr_at_k(recs, relevant, 3), 0.5)
        self.assertGreater(ndcg_at_k(recs, relevant, 3), 0.0)

    def test_coverage(self):
        recs = {"u1": ["a", "b"], "u2": ["b", "c"]}
        self.assertEqual(catalogue_coverage(recs, catalogue_size=4), 0.75)

    def test_missing_recommendations_count_as_misses(self):
        metrics = evaluate_user_recommendations({"a": ["x"]}, {"a": {"x"}, "b": {"y"}})
        self.assertEqual(metrics["users"], 2)
        self.assertEqual(metrics["recall"], 0.5)

    def test_duplicate_hits_do_not_inflate_ndcg(self):
        self.assertEqual(ndcg_at_k(["a", "a", "a"], {"a"}, 3), 1.0)
        for metric in [recall_at_k, mrr_at_k, ndcg_at_k]:
            with self.assertRaises(ValueError):
                metric(["a"], {"a"}, 0)


if __name__ == "__main__":
    unittest.main()
