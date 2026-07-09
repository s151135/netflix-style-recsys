import json
import unittest
from pathlib import Path


class ContractTests(unittest.TestCase):
    def test_contracts_are_valid_json(self):
        for path in Path("data_contracts").glob("*.schema.json"):
            with self.subTest(path=path):
                data = json.loads(path.read_text())
                self.assertIn("required", data)
                self.assertEqual(data["type"], "object")


if __name__ == "__main__":
    unittest.main()
