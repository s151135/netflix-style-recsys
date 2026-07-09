from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

import numpy as np
import pandas as pd

from recsys.data.transforms import build_user_item_strength, normalise_interactions


MOVIELENS_SMALL_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"


@dataclass
class EncodedInteractions:
    user_to_index: dict[str, int]
    item_to_index: dict[str, int]
    index_to_user: list[str]
    index_to_item: list[str]
    positive_pairs: list[tuple[int, int]]
    user_seen: dict[int, set[int]]
    confidence_matrix: np.ndarray


def encode_interactions(interactions: pd.DataFrame) -> EncodedInteractions:
    strength = build_user_item_strength(interactions)
    users = sorted(str(x) for x in strength["user_id"].unique())
    items = sorted(str(x) for x in strength["item_id"].unique())
    user_to_index = {u: i for i, u in enumerate(users)}
    item_to_index = {item: i for i, item in enumerate(items)}
    matrix = np.zeros((len(users), len(items)), dtype=float)
    pairs: list[tuple[int, int]] = []
    user_seen: dict[int, set[int]] = {}
    for row in strength.itertuples(index=False):
        u = user_to_index[str(row.user_id)]
        i = item_to_index[str(row.item_id)]
        matrix[u, i] = float(row.confidence)
        pairs.append((u, i))
        user_seen.setdefault(u, set()).add(i)
    return EncodedInteractions(user_to_index, item_to_index, users, items, pairs, user_seen, matrix)


def make_synthetic_movie_dataset(
    n_users: int = 80,
    n_items: int = 120,
    events_per_user: int = 18,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    genres = ["drama", "comedy", "thriller", "scifi", "documentary", "kids", "action"]
    item_rows = []
    for i in range(n_items):
        g = rng.choice(genres, size=int(rng.integers(1, 3)), replace=False).tolist()
        item_rows.append(
            {
                "item_id": f"item_{i:04d}",
                "title": f"Catalog Title {i:04d}",
                "type": "movie",
                "genres": g,
                "language": str(rng.choice(["en", "es", "fr", "ja"])),
                "release_year": int(rng.integers(1990, 2027)),
                "maturity_rating": str(rng.choice(["G", "PG", "PG-13", "R"])),
            }
        )
    items = pd.DataFrame(item_rows)
    event_types = np.array(["impression", "click", "play", "complete", "add_to_list"])
    event_probs = np.array([0.30, 0.25, 0.30, 0.10, 0.05])
    now = datetime.now(timezone.utc)
    rows = []
    for u in range(n_users):
        preferred = set(rng.choice(genres, size=2, replace=False))
        affinities = np.array(
            [1.5 if preferred.intersection(set(g)) else 0.3 for g in items["genres"].tolist()]
        )
        affinities = affinities / affinities.sum()
        for e in range(events_per_user):
            item_idx = int(rng.choice(np.arange(n_items), p=affinities))
            ts = now - timedelta(days=int(events_per_user - e), minutes=int(rng.integers(0, 1440)))
            event_type = str(rng.choice(event_types, p=event_probs))
            rows.append(
                {
                    "event_id": f"evt_{u:04d}_{e:04d}",
                    "user_id": f"user_{u:04d}",
                    "session_id": f"sess_{u:04d}_{e // 6:04d}",
                    "item_id": f"item_{item_idx:04d}",
                    "event_type": event_type,
                    "timestamp": ts.isoformat(),
                    "position": int(rng.integers(0, 40)),
                    "watch_time_sec": float(rng.integers(0, 7200)),
                    "completion_ratio": float(rng.random()),
                    "dwell_ms": float(rng.integers(100, 120000)),
                }
            )
    return normalise_interactions(pd.DataFrame(rows)), items


def prepare_movielens_small(root: str | Path = "work/ml-latest-small") -> Path:
    root_path = Path(root)
    if (root_path / "ratings.csv").exists() and (root_path / "movies.csv").exists():
        return root_path
    root_path.parent.mkdir(parents=True, exist_ok=True)
    zip_path = root_path.parent / "ml-latest-small.zip"
    if not zip_path.exists():
        urlretrieve(MOVIELENS_SMALL_URL, zip_path)
    with ZipFile(zip_path) as archive:
        archive.extractall(root_path.parent)
    return root_path


def load_movielens_small(root: str | Path = "work/ml-latest-small") -> tuple[pd.DataFrame, pd.DataFrame]:
    root_path = prepare_movielens_small(root)
    ratings = pd.read_csv(root_path / "ratings.csv")
    movies = pd.read_csv(root_path / "movies.csv")
    ratings["rating"] = ratings["rating"].astype(float)
    ratings["timestamp"] = pd.to_datetime(ratings["timestamp"], unit="s", utc=True)
    ratings["event_type"] = np.select(
        [ratings["rating"] >= 4.0, ratings["rating"] >= 3.0, ratings["rating"] < 2.5],
        ["complete", "play", "dislike"],
        default="click",
    )
    interactions = pd.DataFrame(
        {
            "event_id": "ml_" + ratings.index.astype(str),
            "user_id": "ml_user_" + ratings["userId"].astype(str),
            "session_id": "ml_session_" + ratings["userId"].astype(str),
            "item_id": "ml_movie_" + ratings["movieId"].astype(str),
            "event_type": ratings["event_type"],
            "timestamp": ratings["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "position": None,
            "watch_time_sec": None,
            "completion_ratio": (ratings["rating"] / 5.0).clip(0, 1),
            "dwell_ms": None,
            "rating": ratings["rating"],
        }
    )
    item_stats = ratings.groupby("movieId", as_index=False).agg(
        avg_rating=("rating", "mean"),
        rating_count=("rating", "count"),
    )
    enriched = movies.merge(item_stats, on="movieId", how="left")
    title_parts = enriched["title"].str.extract(r"^(?P<clean_title>.*?)(?:\s+\((?P<year>\d{4})\))?$")
    enriched["clean_title"] = title_parts["clean_title"].fillna(enriched["title"])
    enriched["release_year"] = pd.to_numeric(title_parts["year"], errors="coerce").astype("Int64")
    enriched["genres_list"] = enriched["genres"].fillna("").apply(
        lambda value: [] if value == "(no genres listed)" else value.split("|")
    )
    enriched["poster_seed"] = enriched["movieId"].apply(lambda movie_id: f"movie-{movie_id}")
    items = pd.DataFrame(
        {
            "item_id": "ml_movie_" + enriched["movieId"].astype(str),
            "title": enriched["clean_title"],
            "type": "movie",
            "genres": enriched["genres_list"],
            "language": "en",
            "release_year": enriched["release_year"],
            "maturity_rating": None,
            "avg_rating": enriched["avg_rating"].fillna(0.0).round(2),
            "rating_count": enriched["rating_count"].fillna(0).astype(int),
            "poster_seed": enriched["poster_seed"],
        }
    )
    return normalise_interactions(interactions), items
