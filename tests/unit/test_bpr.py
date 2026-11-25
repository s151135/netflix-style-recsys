import unittest
from unittest.mock import patch

from recsys.models.bpr import BPRConfig, BPRMatrixFactorization


class BPRSamplingTests(unittest.TestCase):
    def test_fully_observed_catalogue_has_no_valid_negative_updates(self):
        model = BPRMatrixFactorization(BPRConfig(epochs=2))
        with patch.object(model, "_update") as update:
            model.fit([(0, 0), (0, 1)], n_users=1, n_items=2)
        update.assert_not_called()

    def test_negative_samples_never_include_known_positives(self):
        model = BPRMatrixFactorization(BPRConfig(epochs=2, negatives_per_positive=4))
        pairs = [(0, i) for i in range(99)]
        with patch.object(model, "_update") as update:
            model.fit(pairs, n_users=1, n_items=100)
        self.assertEqual(update.call_count, 99 * 2 * 4)
        self.assertTrue(all(call.args[2] == 99 for call in update.call_args_list))
