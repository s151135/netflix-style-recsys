import unittest

import numpy as np

from recsys.serving.ann_index import NumpyANNIndex


class ANNIndexTests(unittest.TestCase):
    def test_search_excludes_seen_items(self):
        index = NumpyANNIndex(["a", "b"], np.array([[1.0, 0.0], [0.0, 1.0]]))
        results = index.search(np.array([1.0, 0.0]), k=1, exclude={"a"})
        self.assertEqual(results[0].item_id, "b")


if __name__ == "__main__":
    unittest.main()
