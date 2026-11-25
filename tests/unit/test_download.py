import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

from recsys.train.datasets import prepare_movielens_small


class DownloadTests(unittest.TestCase):
    def test_only_expected_files_are_extracted_to_custom_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "custom-root"
            with ZipFile(Path(tmp) / "ml-latest-small.zip", "w") as archive:
                for name in ["ratings.csv", "movies.csv", "links.csv", "README.txt"]:
                    archive.writestr(f"ml-latest-small/{name}", name)
                archive.writestr("../unexpected.txt", "must not be extracted")
            self.assertEqual(prepare_movielens_small(root), root)
            self.assertTrue((root / "ratings.csv").exists())
            self.assertFalse((Path(tmp) / "unexpected.txt").exists())
            self.assertEqual(len(list(root.iterdir())), 4)
