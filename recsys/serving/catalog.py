"""Curated display metadata for the interactive discovery catalogue.

MovieLens supplies the real rating behavior used by the local model.  This module
adds editorial metadata for a small set of current, recognisable films and series
so the demo can also make cold-start recommendations outside of MovieLens.
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd


TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w780"


def _image(path: str) -> str:
    return f"{TMDB_IMAGE_BASE}{path}"


# Metadata is deliberately kept local: the app remains usable without asking a
# visitor for an API key. Poster paths follow TMDB's documented image URL format.
CURATED_TITLES: list[dict[str, Any]] = [
    {
        "catalog_id": "dune-part-two",
        "title": "Dune: Part Two",
        "type": "movie",
        "release_year": 2024,
        "genres": ["Sci-Fi", "Adventure", "Drama"],
        "maturity_rating": "PG-13",
        "runtime": "2h 46m",
        "tagline": "Long live the fighters.",
        "overview": "Paul Atreides joins Chani and the Fremen while seeking revenge against the conspirators who destroyed his family.",
        "credits": "Timothee Chalamet, Zendaya, Rebecca Ferguson",
        "creator": "Denis Villeneuve",
        "poster_url": _image("/1pdfLvkbY9ohJlCjQH2CZjjYVvJ.jpg"),
        "backdrop_url": _image("/xOMo8BRK7PfcJv9JCnx7s5hj0PX.jpg"),
        "popularity": 95,
    },
    {
        "catalog_id": "oppenheimer",
        "title": "Oppenheimer",
        "type": "movie",
        "release_year": 2023,
        "genres": ["Drama", "History", "Thriller"],
        "maturity_rating": "R",
        "runtime": "3h",
        "tagline": "The world forever changes.",
        "overview": "The story of physicist J. Robert Oppenheimer and his role in developing the atomic bomb.",
        "credits": "Cillian Murphy, Emily Blunt, Robert Downey Jr.",
        "creator": "Christopher Nolan",
        "poster_url": _image("/8Gxv8gSFCU0XGDykEGv7zR1n2ua.jpg"),
        "backdrop_url": _image("/fm6KqXpk3M2HVveHwCrBSSBaO0V.jpg"),
        "popularity": 93,
    },
    {
        "catalog_id": "everything-everywhere-all-at-once",
        "title": "Everything Everywhere All at Once",
        "type": "movie",
        "release_year": 2022,
        "genres": ["Adventure", "Sci-Fi", "Comedy"],
        "maturity_rating": "R",
        "runtime": "2h 20m",
        "tagline": "The universe is so much bigger than you realize.",
        "overview": "An exhausted laundromat owner is swept into a wild multiverse adventure where she alone may save existence.",
        "credits": "Michelle Yeoh, Ke Huy Quan, Stephanie Hsu",
        "creator": "Daniel Kwan and Daniel Scheinert",
        "poster_url": _image("/w3LxiVYdWWRvEVdn5RYq6jIqkb1.jpg"),
        "backdrop_url": _image("/r9PkFnRUIthgBp2JZZzD380MWZy.jpg"),
        "popularity": 91,
    },
    {
        "catalog_id": "spider-man-across-the-spider-verse",
        "title": "Spider-Man: Across the Spider-Verse",
        "type": "movie",
        "release_year": 2023,
        "genres": ["Animation", "Action", "Adventure"],
        "maturity_rating": "PG",
        "runtime": "2h 20m",
        "tagline": "It's how you wear the mask that matters.",
        "overview": "Miles Morales is catapulted across the Multiverse and must redefine what it means to be a hero.",
        "credits": "Shameik Moore, Hailee Steinfeld, Brian Tyree Henry",
        "creator": "Joaquim Dos Santos, Kemp Powers, Justin K. Thompson",
        "poster_url": _image("/8Vt6mWEReuy4Of61Lnj5Xj704m8.jpg"),
        "backdrop_url": _image("/4HodYYKEIsGOdinkGi2Ucz6X9i0.jpg"),
        "popularity": 89,
    },
    {
        "catalog_id": "the-batman",
        "title": "The Batman",
        "type": "movie",
        "release_year": 2022,
        "genres": ["Crime", "Mystery", "Thriller"],
        "maturity_rating": "PG-13",
        "runtime": "2h 56m",
        "tagline": "Unmask the truth.",
        "overview": "Batman ventures into Gotham City's underworld when a sadistic killer leaves a trail of cryptic clues.",
        "credits": "Robert Pattinson, Zoe Kravitz, Paul Dano",
        "creator": "Matt Reeves",
        "poster_url": _image("/74xTEgt7R36Fpooo50r9T25onhq.jpg"),
        "backdrop_url": _image("/b0PlSFdDwbyK0cf5RxwDpaOJQvQ.jpg"),
        "popularity": 87,
    },
    {
        "catalog_id": "past-lives",
        "title": "Past Lives",
        "type": "movie",
        "release_year": 2023,
        "genres": ["Drama", "Romance"],
        "maturity_rating": "PG-13",
        "runtime": "1h 46m",
        "tagline": "In a moment, everything can change.",
        "overview": "Two childhood friends are reunited in New York for one fateful week as they confront destiny and the choices that make a life.",
        "credits": "Greta Lee, Teo Yoo, John Magaro",
        "creator": "Celine Song",
        "poster_url": _image("/k3waqVXSnvCZWfJYNtdamTgTtTA.jpg"),
        "backdrop_url": _image("/8tU0Zlm4f8Qe7k54p9M6isFNY4W.jpg"),
        "popularity": 84,
    },
    {
        "catalog_id": "the-holdovers",
        "title": "The Holdovers",
        "type": "movie",
        "release_year": 2023,
        "genres": ["Comedy", "Drama"],
        "maturity_rating": "R",
        "runtime": "2h 13m",
        "tagline": "Discomfort and joy.",
        "overview": "A curmudgeonly instructor at a New England boarding school forms an unexpected bond with a stranded student and the school cook.",
        "credits": "Paul Giamatti, Da'Vine Joy Randolph, Dominic Sessa",
        "creator": "Alexander Payne",
        "poster_url": _image("/VHSzNBTwxV8vh7wylo7O9CLdac.jpg"),
        "backdrop_url": _image("/qgK7CrXrF0eT3M1Z2rXPOz8JmI7.jpg"),
        "popularity": 82,
    },
    {
        "catalog_id": "the-matrix",
        "title": "The Matrix",
        "aliases": ["Matrix, The"],
        "type": "movie",
        "release_year": 1999,
        "genres": ["Sci-Fi", "Action"],
        "maturity_rating": "R",
        "runtime": "2h 16m",
        "tagline": "Free your mind.",
        "overview": "A computer hacker learns that the world he knows is a sophisticated simulation and joins a rebellion against its machines.",
        "credits": "Keanu Reeves, Laurence Fishburne, Carrie-Anne Moss",
        "creator": "Lana and Lilly Wachowski",
        "poster_url": _image("/f89U3ADr1oiB1s9GkdPOEpXUk5H.jpg"),
        "backdrop_url": _image("/icmmSD4vTTDKOq2vvdulafOGw93.jpg"),
        "popularity": 88,
    },
    {
        "catalog_id": "stranger-things",
        "title": "Stranger Things",
        "type": "series",
        "release_year": 2016,
        "genres": ["Sci-Fi", "Horror", "Drama"],
        "maturity_rating": "TV-14",
        "seasons": 5,
        "tagline": "Every ending has a beginning.",
        "overview": "When a young boy vanishes, a small town uncovers secret experiments, terrifying supernatural forces and one strange girl.",
        "credits": "Winona Ryder, David Harbour, Millie Bobby Brown",
        "creator": "The Duffer Brothers",
        "poster_url": _image("/x2LSRK2Cm7MZhjluni1msVJ3wDF.jpg"),
        "backdrop_url": _image("/q4nD2QJVz4i8rFELX8z9anO3xZg.jpg"),
        "popularity": 96,
    },
    {
        "catalog_id": "arcane",
        "title": "Arcane",
        "type": "series",
        "release_year": 2021,
        "genres": ["Animation", "Action", "Fantasy"],
        "maturity_rating": "TV-14",
        "seasons": 2,
        "tagline": "The line between allies and enemies is blurred.",
        "overview": "Amid the stark conflict between twin cities, two sisters fight on rival sides of a war between magic and technology.",
        "credits": "Hailee Steinfeld, Ella Purnell, Kevin Alejandro",
        "creator": "Christian Linke and Alex Yee",
        "poster_url": _image("/fqldf2t8ztc9aiwn3k6mlX3tvRT.jpg"),
        "backdrop_url": _image("/xF5eX1lW9nZc8ZgK4s9dJrV9KjH.jpg"),
        "popularity": 92,
    },
    {
        "catalog_id": "wednesday",
        "title": "Wednesday",
        "type": "series",
        "release_year": 2022,
        "genres": ["Mystery", "Fantasy", "Comedy"],
        "maturity_rating": "TV-14",
        "seasons": 2,
        "tagline": "A new mystery is coming.",
        "overview": "Wednesday Addams investigates a murderous mystery while making new friends and enemies at Nevermore Academy.",
        "credits": "Jenna Ortega, Emma Myers, Catherine Zeta-Jones",
        "creator": "Alfred Gough and Miles Millar",
        "poster_url": _image("/9PFonBhy4cQy7Jz20NpMygczOkv.jpg"),
        "backdrop_url": _image("/iHSwvRVsRyxpX7FE7GbviaDkNys.jpg"),
        "popularity": 90,
    },
    {
        "catalog_id": "the-queens-gambit",
        "title": "The Queen's Gambit",
        "type": "series",
        "release_year": 2020,
        "genres": ["Drama"],
        "maturity_rating": "TV-MA",
        "seasons": 1,
        "tagline": "Every move is a move forward.",
        "overview": "In a 1950s orphanage, a young girl reveals an astonishing talent for chess and begins an unlikely journey to stardom.",
        "credits": "Anya Taylor-Joy, Bill Camp, Marielle Heller",
        "creator": "Scott Frank and Allan Scott",
        "poster_url": _image("/zU0htwkhNvBQdVSIKB9s6hgVeFK.jpg"),
        "backdrop_url": _image("/5N3X1aS9M8o8Tg3Vd5pR8tC9gBv.jpg"),
        "popularity": 83,
    },
    {
        "catalog_id": "the-last-of-us",
        "title": "The Last of Us",
        "type": "series",
        "release_year": 2023,
        "genres": ["Drama", "Sci-Fi", "Thriller"],
        "maturity_rating": "TV-MA",
        "seasons": 2,
        "tagline": "When you're lost in the darkness, look for the light.",
        "overview": "Twenty years after modern civilization has been destroyed, a hardened survivor must smuggle a teenage girl out of a quarantine zone.",
        "credits": "Pedro Pascal, Bella Ramsey, Gabriel Luna",
        "creator": "Craig Mazin and Neil Druckmann",
        "poster_url": _image("/uKvVjHNqB5VmOrdxqAt2F7J78ED.jpg"),
        "backdrop_url": _image("/uDgy6hyPd82kOHh6I95FLtLnj6p.jpg"),
        "popularity": 91,
    },
    {
        "catalog_id": "shogun",
        "title": "Shogun",
        "type": "series",
        "release_year": 2024,
        "genres": ["Drama", "History", "War"],
        "maturity_rating": "TV-MA",
        "seasons": 1,
        "tagline": "The rules of war have changed.",
        "overview": "In 1600 Japan, a shipwrecked English pilot and a powerful warlord are drawn into a pivotal struggle for power.",
        "credits": "Hiroyuki Sanada, Cosmo Jarvis, Anna Sawai",
        "creator": "Rachel Kondo and Justin Marks",
        "poster_url": _image("/7O4iVfOMQmdCSxhOg1WnzG1AgYT.jpg"),
        "backdrop_url": _image("/o2e5t1OmdYwIoK5pM1Qk3b0qzPf.jpg"),
        "popularity": 89,
    },
]


def _key(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()
    return " ".join(cleaned.split())


def enrich_catalogue_items(items: pd.DataFrame) -> pd.DataFrame:
    """Overlay curated metadata and append series that are not in MovieLens."""
    output = items.copy()
    metadata_fields = {
        field
        for title in CURATED_TITLES
        for field in title
        if field not in {"catalog_id", "aliases"}
    }
    # MovieLens stores all-empty columns such as maturity_rating as float NaN.
    # Convert the display metadata columns before assigning strings and lists.
    for field in metadata_fields:
        if field not in output:
            output[field] = None
        output[field] = output[field].astype("object")
    by_title: dict[str, dict[str, Any]] = {}
    for title in CURATED_TITLES:
        by_title[_key(title["title"])] = title
        for alias in title.get("aliases", []):
            by_title[_key(alias)] = title

    used_catalog_ids: set[str] = set()
    for index, row in output.iterrows():
        metadata = by_title.get(_key(str(row.get("title", ""))))
        if not metadata:
            continue
        used_catalog_ids.add(str(metadata["catalog_id"]))
        for field, value in metadata.items():
            if field not in {"catalog_id", "aliases"}:
                output.at[index, field] = value

    extra_rows = []
    for title in CURATED_TITLES:
        if title["catalog_id"] in used_catalog_ids:
            continue
        extra_rows.append(
            {
                "item_id": f"curated_{title['catalog_id']}",
                "language": "en",
                "avg_rating": None,
                "rating_count": 0,
                "poster_seed": title["catalog_id"],
                **{key: value for key, value in title.items() if key not in {"catalog_id", "aliases"}},
            }
        )
    if extra_rows:
        output = pd.concat([output, pd.DataFrame(extra_rows)], ignore_index=True, sort=False)
    return output


def curated_item_ids(items: pd.DataFrame) -> set[str]:
    return set(items.loc[items.get("poster_url").notna(), "item_id"].astype(str))
