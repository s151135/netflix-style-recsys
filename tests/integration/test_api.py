import os
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
from fastapi.testclient import TestClient

from recsys.serving.api import app
from recsys.train.datasets import make_synthetic_movie_dataset
from recsys.train.pipelines import run_local_smoke_pipeline


class APITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.artifacts = tempfile.TemporaryDirectory()
        dataset = make_synthetic_movie_dataset(n_users=8, n_items=20, events_per_user=8)
        # Match the rating column supplied by the MovieLens loader used by the UI.
        dataset[0]["rating"] = dataset[0]["completion_ratio"] * 5
        with patch("recsys.train.pipelines.load_movielens_small", return_value=dataset):
            run_local_smoke_pipeline(cls.artifacts.name)

    @classmethod
    def tearDownClass(cls):
        cls.artifacts.cleanup()

    def setUp(self):
        self.environment = patch.dict(os.environ, {"RECSYS_ARTIFACT_DIR": self.artifacts.name})
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.client = TestClient(app)
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)

    def test_demo_and_discovery_endpoints(self):
        for path in ["/", "/healthz", "/sample-users", "/browse-rows",
                     "/profile/user_0000", "/title/curated_stranger-things"]:
            with self.subTest(path=path):
                self.assertEqual(self.client.get(path).status_code, 200)

    def test_search_treats_regular_expression_symbols_as_literal_text(self):
        for query in ["[", ".*", "("]:
            with self.subTest(query=query):
                response = self.client.get("/catalog", params={"query": query})
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["items"], [])
        response = self.client.get("/catalog", params={"query": "stranger"})
        self.assertEqual(response.json()["items"][0]["title"], "Stranger Things")

    def test_recommendations_for_known_and_new_users(self):
        for user in ["user_0000", "new-user"]:
            with self.subTest(user=user):
                response = self.client.post("/recommend", json={
                    "user_id": user, "k": 5,
                    "context": {"seen_item_ids": ["curated_stranger-things"]},
                })
                self.assertEqual(response.status_code, 200)
                data = response.json()
                self.assertEqual(data["mode"], "personalized" if user == "user_0000" else "cold_start")
                if user == "user_0000":
                    self.assertTrue(any(item["source"] == "collaborative" for item in data["items"]))
                else:
                    self.assertTrue(all(item["source"] != "collaborative" for item in data["items"]))
                ids = [item["item_id"] for item in response.json()["items"]]
                self.assertTrue(ids)
                self.assertLessEqual(len(ids), 5)
                self.assertEqual(len(ids), len(set(ids)))
                self.assertNotIn("curated_stranger-things", ids)
                seen = set(app.state.interactions.loc[
                    app.state.interactions["user_id"] == user, "item_id"
                ])
                self.assertFalse(seen.intersection(ids))

    def test_invalid_recommendation_size_is_rejected(self):
        response = self.client.post("/recommend", json={"user_id": "new-user", "k": 0})
        self.assertEqual(response.status_code, 422)

    def test_missing_title_returns_not_found(self):
        self.assertEqual(self.client.get("/title/missing").status_code, 404)

    def test_model_info_and_positive_rating_profile(self):
        info = self.client.get("/model-info").json()
        self.assertTrue(info["ready"])
        self.assertEqual(info["model"], "BPR collaborative filtering")
        profile = self.client.get("/profile/user_0000").json()
        self.assertTrue(all(item["rating"] >= 4.0 for item in profile["history"]))

    def test_taste_counts_and_genres_exclude_low_ratings(self):
        interactions = pd.DataFrame([
            {"user_id": "user_0000", "item_id": "good", "rating": 5.0,
             "event_type": "like", "timestamp": "2020-01-01T00:00:00Z"},
            {"user_id": "user_0000", "item_id": "bad", "rating": 1.0,
             "event_type": "dislike", "timestamp": "2020-01-02T00:00:00Z"},
        ])
        items = pd.DataFrame([
            {"item_id": "good", "title": "Liked drama", "genres": ["Drama"]},
            {"item_id": "bad", "title": "Disliked comedy", "genres": ["Comedy"]},
        ])
        with patch.object(app.state, "interactions", interactions), patch.object(app.state, "items", items):
            profile = self.client.get("/profile/user_0000").json()
        self.assertEqual(profile["observed_count"], 2)
        self.assertEqual(profile["positive_count"], 1)
        self.assertEqual(profile["genre_sample_size"], 1)
        self.assertTrue(profile["has_learned_profile"])
        self.assertEqual(profile["history"][0]["item_id"], "good")
        self.assertEqual(profile["top_genres"], [{"genre": "Drama", "count": 1}])

    def test_new_viewer_taste_profile_has_no_invented_history(self):
        profile = self.client.get("/profile/demo-new-viewer").json()
        self.assertEqual(profile["observed_count"], 0)
        self.assertEqual(profile["positive_count"], 0)
        self.assertFalse(profile["has_learned_profile"])
        self.assertEqual(profile["history"], [])
        self.assertEqual(profile["top_genres"], [])

    def test_readiness_assets_metrics_and_headers(self):
        self.assertEqual(self.client.get("/readyz").status_code, 200)
        home = self.client.get("/")
        self.assertIn("script-src 'self'", home.headers["Content-Security-Policy"])
        self.assertIn("/assets/demo.js", home.text)
        script = self.client.get("/assets/demo.js")
        self.assertEqual(script.status_code, 200)
        self.assertIn("text/javascript", script.headers["content-type"])
        self.assertIn("loadRecommendations", script.text)
        response = self.client.post("/recommend", json={"user_id": "new-user", "k": 1})
        self.assertEqual(response.status_code, 200)
        self.assertIn("recsys_recommend_requests_total", self.client.get("/metrics").text)

    def test_invalid_context_and_bounds_return_validation_errors(self):
        for body in [
            {"user_id": " "}, {"user_id": "new", "k": 101},
            {"user_id": "new", "context": {"seen_item_ids": "not-a-list"}},
            {"user_id": "new", "context": {"seen_item_ids": [["nested"]]}},
            {"user_id": "new", "context": {"surface": "arbitrary-metric-label"}},
            {"user_id": "new", "context": {"unexpected": True}},
        ]:
            with self.subTest(body=body):
                self.assertEqual(self.client.post("/recommend", json=body).status_code, 422)
        self.assertEqual(self.client.get("/catalog", params={"query": "a" * 201}).status_code, 422)
        self.assertEqual(self.client.get("/catalog", params={"media_type": "invalid"}).status_code, 422)

    def test_missing_model_is_alive_but_not_ready(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"RECSYS_ARTIFACT_DIR": tmp}):
            with TestClient(app) as client:
                self.assertEqual(client.get("/healthz").status_code, 200)
                self.assertEqual(client.get("/readyz").status_code, 503)
                self.assertFalse(client.get("/model-info").json()["ready"])
                self.assertEqual(client.post("/recommend", json={"user_id": "new"}).status_code, 503)
