import tempfile
import unittest

import numpy as np

from recsys.train.registry import load_factor_model, save_factor_model


class RegistryValidationTests(unittest.TestCase):
    def test_malformed_models_are_rejected(self):
        for users, factors, message in [
            (["a", "a"], np.ones((2, 2)), "unique strings"),
            (["a"], np.ones((2, 2)), "factor shapes"),
            (["a"], np.array([[float("nan"), 0.0]]), "finite"),
        ]:
            with self.subTest(message=message), tempfile.TemporaryDirectory() as tmp:
                save_factor_model(tmp, factors, np.ones((2, 2)), np.zeros(2), users, ["x", "y"])
                with self.assertRaisesRegex(ValueError, message):
                    load_factor_model(tmp)
