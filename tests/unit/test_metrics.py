import unittest

from recsys.train.evaluate import catalogue_coverage, mrr_at_k, ndcg_at_k, recall_at_k


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


if __name__ == "__main__":
    unittest.main()
