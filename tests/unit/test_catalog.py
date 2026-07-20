import unittest

import pandas as pd

from recsys.serving.catalog import enrich_catalogue_items


class CatalogTests(unittest.TestCase):
    def test_curated_titles_enrich_movies_and_add_series(self):
        items = pd.DataFrame(
            [
                {
                    "item_id": "ml_movie_2571",
                    "title": "Matrix, The",
                    "type": "movie",
                    "genres": ["Action"],
                    "maturity_rating": None,
                }
            ]
        )

        enriched = enrich_catalogue_items(items)

        matrix = enriched[enriched["item_id"] == "ml_movie_2571"].iloc[0]
        self.assertEqual(matrix["title"], "The Matrix")
        self.assertTrue(matrix["poster_url"].startswith("https://image.tmdb.org/"))
        self.assertIn("curated_stranger-things", set(enriched["item_id"]))


if __name__ == "__main__":
    unittest.main()
