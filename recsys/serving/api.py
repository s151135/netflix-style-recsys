from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import HTMLResponse
except Exception as exc:  # pragma: no cover
    raise RuntimeError("Install serving dependencies with `python -m pip install -e .`") from exc

from recsys.monitoring.metrics import record_recommendation_request
from recsys.serving.catalog import curated_item_ids, enrich_catalogue_items
from recsys.serving.ranking_service import RankingService
from recsys.serving.retrieval_service import RetrievalService


app = FastAPI(title="netflix-style-recsys", version="0.1.0")


class RecommendRequest(BaseModel):
    user_id: str
    session_id: str | None = None
    context: dict[str, Any] = Field(default_factory=dict)
    k: int = Field(default=20, ge=1, le=100)


class RecommendResponse(BaseModel):
    user_id: str
    items: list[dict[str, Any]]
    latency_ms: float


def _artifact_dir() -> Path:
    return Path(os.getenv("RECSYS_ARTIFACT_DIR", "artifacts/local")) / "bpr_champion"


def _load_items() -> pd.DataFrame:
    path = Path(os.getenv("RECSYS_ARTIFACT_DIR", "artifacts/local")) / "sample_items.json"
    if path.exists():
        return enrich_catalogue_items(pd.read_json(path))
    return enrich_catalogue_items(pd.DataFrame(columns=["item_id"]))


def _load_interactions() -> pd.DataFrame:
    path = Path(os.getenv("RECSYS_ARTIFACT_DIR", "artifacts/local")) / "sample_interactions.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame(columns=["user_id", "item_id", "rating", "timestamp"])


@app.on_event("startup")
def startup() -> None:
    artifact = _artifact_dir()
    items = _load_items()
    interactions = _load_interactions()
    app.state.retrieval = RetrievalService.load(artifact) if artifact.exists() else None
    app.state.items = items
    app.state.interactions = interactions
    app.state.ranking = RankingService(items=items)


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def demo_ui() -> str:
    html_path = Path(__file__).with_name("demo.html")
    return html_path.read_text()


@app.get("/sample-users")
def sample_users(limit: int = 12) -> dict[str, list[str]]:
    retrieval = getattr(app.state, "retrieval", None)
    if retrieval is None:
        return {"users": []}
    return {"users": retrieval.users[: max(1, min(limit, 50))]}


@app.get("/profile/{user_id}")
def profile(user_id: str) -> dict[str, Any]:
    interactions = getattr(app.state, "interactions", pd.DataFrame())
    items = getattr(app.state, "items", pd.DataFrame())
    if interactions.empty:
        return {"user_id": user_id, "history": [], "top_genres": []}
    user_rows = interactions[interactions["user_id"] == user_id].copy()
    if user_rows.empty:
        return {"user_id": user_id, "history": [], "top_genres": []}
    user_rows["timestamp"] = pd.to_datetime(user_rows["timestamp"], utc=True, errors="coerce")
    merged = user_rows.merge(items, on="item_id", how="left").sort_values(
        ["rating", "timestamp"], ascending=[False, False]
    )
    history = [
        _display_item(row)
        | {
            "rating": float(getattr(row, "rating", 0.0) or 0.0),
            "event_type": str(getattr(row, "event_type", "")),
        }
        for row in merged.head(12).itertuples(index=False)
    ]
    genre_counts: dict[str, int] = {}
    for genres in merged.head(50)["genres"]:
        for genre in _normalise_genres(genres):
            genre_counts[genre] = genre_counts.get(genre, 0) + 1
    top_genres = [
        {"genre": genre, "count": count}
        for genre, count in sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    ]
    return {"user_id": user_id, "history": history, "top_genres": top_genres}


@app.get("/browse-rows")
def browse_rows(limit: int = 18) -> dict[str, Any]:
    items = getattr(app.state, "items", pd.DataFrame())
    if items.empty:
        return {"rows": []}
    safe_limit = max(6, min(limit, 30))
    curated = items[items["poster_url"].notna()] if "poster_url" in items else items.iloc[0:0]
    rows = [
        {
            "title": "Trending now",
            "items": _items_to_display(curated.sort_values("popularity", ascending=False).head(safe_limit)),
        },
        {
            "title": "Series worth settling into",
            "items": _items_to_display(curated[curated["type"] == "series"].head(safe_limit)),
        },
    ]
    for genre in ["Sci-Fi", "Drama", "Action", "Mystery", "Animation", "Comedy"]:
        mask = curated["genres"].apply(lambda values: genre in _normalise_genres(values))
        genre_items = (
            curated[mask]
            .sort_values("popularity", ascending=False)
            .head(safe_limit)
        )
        if not genre_items.empty:
            rows.append({"title": f"{genre} picks", "items": _items_to_display(genre_items)})
    return {"rows": rows}


@app.get("/catalog")
def catalog(query: str = "", media_type: str = "all", genre: str = "", limit: int = 48) -> dict[str, Any]:
    items = getattr(app.state, "items", pd.DataFrame())
    if items.empty:
        return {"items": []}
    displayed = items[items.get("poster_url").notna()].copy()
    if media_type in {"movie", "series"}:
        displayed = displayed[displayed["type"] == media_type]
    if query:
        displayed = displayed[displayed["title"].str.contains(query, case=False, na=False)]
    if genre:
        displayed = displayed[displayed["genres"].apply(lambda values: genre in _normalise_genres(values))]
    return {"items": _items_to_display(displayed.sort_values("popularity", ascending=False).head(max(1, min(limit, 80))))}


@app.get("/title/{item_id}")
def title_detail(item_id: str) -> dict[str, Any]:
    items = getattr(app.state, "items", pd.DataFrame())
    selected = items[items["item_id"].astype(str) == item_id]
    if selected.empty:
        raise HTTPException(status_code=404, detail="Title not found")
    row = selected.iloc[0]
    genres = set(_normalise_genres(row.get("genres", [])))
    related = items[(items["item_id"].astype(str) != item_id) & items.get("poster_url").notna()].copy()
    related["similarity"] = related["genres"].apply(lambda values: len(genres.intersection(_normalise_genres(values))))
    related = related.sort_values(["similarity", "popularity"], ascending=False).head(12)
    return {
        "item": _display_item(row),
        "tagline": _optional_text(row.get("tagline")),
        "overview": _optional_text(row.get("overview")),
        "credits": _optional_text(row.get("credits")),
        "creator": _optional_text(row.get("creator")),
        "runtime": _optional_text(row.get("runtime")),
        "seasons": _optional_int(row.get("seasons")),
        "related": _items_to_display(related),
    }


@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest) -> RecommendResponse:
    started = time.perf_counter()
    if app.state.retrieval is None:
        raise HTTPException(status_code=503, detail="No model artifact available. Run scripts/train_local.py")
    context = dict(req.context)
    interactions = getattr(app.state, "interactions", pd.DataFrame())
    history = interactions[interactions["user_id"] == req.user_id] if not interactions.empty else pd.DataFrame()
    excluded = set(req.context.get("seen_item_ids", [])) | set(history.get("item_id", []))
    profile = _profile_data(req.user_id)
    context["preferred_genres"] = [entry["genre"] for entry in profile["top_genres"]]
    if profile["history"]:
        context["anchor_title"] = profile["history"][0]["title"]
    candidates = app.state.retrieval.get_candidates(
        user_id=req.user_id,
        k=max(req.k * 20, 200),
        exclude_item_ids=excluded,
    )
    candidates = [row for row in candidates if row["item_id"] in curated_item_ids(app.state.items)]
    candidates.extend(_content_candidates(app.state.items, excluded, context))
    candidates = _dedupe_candidates(candidates)
    ranked = app.state.ranking.rank(
        candidates,
        context=context,
        top_k=req.k,
    )
    latency_ms = (time.perf_counter() - started) * 1000
    record_recommendation_request(surface=str(context.get("surface", "unknown")), latency_ms=latency_ms)
    return RecommendResponse(user_id=req.user_id, items=ranked, latency_ms=latency_ms)


def _items_to_display(items: pd.DataFrame) -> list[dict[str, Any]]:
    return [_display_item(row) for row in items.itertuples(index=False)]


def _display_item(row) -> dict[str, Any]:
    return {
        "item_id": str(getattr(row, "item_id", "")),
        "title": str(getattr(row, "title", "Untitled")),
        "genres": _normalise_genres(getattr(row, "genres", [])),
        "release_year": _optional_int(getattr(row, "release_year", None)),
        "avg_rating": _optional_float(getattr(row, "avg_rating", None)),
        "rating_count": _optional_int(getattr(row, "rating_count", None)),
        "poster_seed": str(getattr(row, "poster_seed", getattr(row, "item_id", ""))),
        "poster_url": _optional_text(getattr(row, "poster_url", None)),
        "backdrop_url": _optional_text(getattr(row, "backdrop_url", None)),
        "maturity_rating": _optional_text(getattr(row, "maturity_rating", None)),
        "runtime": _optional_text(getattr(row, "runtime", None)),
        "seasons": _optional_int(getattr(row, "seasons", None)),
        "type": _optional_text(getattr(row, "type", "movie")) or "movie",
        "overview": _optional_text(getattr(row, "overview", None)),
    }


def _normalise_genres(value) -> list[str]:
    if isinstance(value, list):
        return [str(x) for x in value]
    if isinstance(value, str):
        if value.startswith("[") and value.endswith("]"):
            return [part.strip(" '\"") for part in value.strip("[]").split(",") if part.strip(" '\"")]
        return [part.strip() for part in value.replace(",", "|").split("|") if part.strip()]
    return []


def _optional_int(value) -> int | None:
    if pd.isna(value):
        return None
    return int(value)


def _optional_float(value) -> float | None:
    if pd.isna(value):
        return None
    return float(value)


def _optional_text(value) -> str | None:
    if value is None or pd.isna(value):
        return None
    return str(value)


def _profile_data(user_id: str) -> dict[str, Any]:
    interactions = getattr(app.state, "interactions", pd.DataFrame())
    items = getattr(app.state, "items", pd.DataFrame())
    if interactions.empty:
        return {"user_id": user_id, "history": [], "top_genres": []}
    user_rows = interactions[interactions["user_id"] == user_id].copy()
    if user_rows.empty:
        return {"user_id": user_id, "history": [], "top_genres": []}
    user_rows["timestamp"] = pd.to_datetime(user_rows["timestamp"], utc=True, errors="coerce")
    merged = user_rows.merge(items, on="item_id", how="left").sort_values(["rating", "timestamp"], ascending=[False, False])
    history = [_display_item(row) | {"rating": float(getattr(row, "rating", 0.0) or 0.0), "event_type": str(getattr(row, "event_type", ""))} for row in merged.head(12).itertuples(index=False)]
    genre_counts: dict[str, int] = {}
    for genres in merged.head(50)["genres"]:
        for genre in _normalise_genres(genres):
            genre_counts[genre] = genre_counts.get(genre, 0) + 1
    top_genres = [{"genre": genre, "count": count} for genre, count in sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)[:5]]
    return {"user_id": user_id, "history": history, "top_genres": top_genres}


def _content_candidates(items: pd.DataFrame, excluded: set[str], context: dict[str, Any]) -> list[dict[str, float | str]]:
    preferred = set(context.get("preferred_genres", []))
    candidates = []
    for row in items[items.get("poster_url").notna()].itertuples(index=False):
        item_id = str(row.item_id)
        if item_id in excluded:
            continue
        overlap = len(preferred.intersection(_normalise_genres(getattr(row, "genres", []))))
        score = 0.35 + (0.45 * overlap / max(1, len(preferred))) + (float(getattr(row, "popularity", 0) or 0) / 400)
        candidates.append({"item_id": item_id, "retrieval_score": score})
    return candidates


def _dedupe_candidates(candidates: list[dict[str, float | str]]) -> list[dict[str, float | str]]:
    best: dict[str, dict[str, float | str]] = {}
    for candidate in candidates:
        item_id = str(candidate["item_id"])
        if item_id not in best or float(candidate["retrieval_score"]) > float(best[item_id]["retrieval_score"]):
            best[item_id] = candidate
    return list(best.values())
