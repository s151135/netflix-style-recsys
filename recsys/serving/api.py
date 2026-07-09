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
        return pd.read_json(path)
    return pd.DataFrame(columns=["item_id"])


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
    rows = [
        {
            "title": "Popular on MovieLens",
            "items": _items_to_display(
                items.sort_values(["rating_count", "avg_rating"], ascending=False).head(safe_limit)
            ),
        },
        {
            "title": "Critically Loved",
            "items": _items_to_display(
                items[items["rating_count"] >= 40]
                .sort_values(["avg_rating", "rating_count"], ascending=False)
                .head(safe_limit)
            ),
        },
    ]
    for genre in ["Action", "Comedy", "Drama", "Sci-Fi", "Thriller", "Romance"]:
        mask = items["genres"].apply(lambda values: genre in _normalise_genres(values))
        genre_items = (
            items[mask]
            .sort_values(["avg_rating", "rating_count"], ascending=False)
            .head(safe_limit)
        )
        if not genre_items.empty:
            rows.append({"title": f"{genre} picks", "items": _items_to_display(genre_items)})
    return {"rows": rows}


@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest) -> RecommendResponse:
    started = time.perf_counter()
    if app.state.retrieval is None:
        raise HTTPException(status_code=503, detail="No model artifact available. Run scripts/train_local.py")
    excluded = set(req.context.get("seen_item_ids", []))
    candidates = app.state.retrieval.get_candidates(
        user_id=req.user_id,
        k=max(req.k * 20, 200),
        exclude_item_ids=excluded,
    )
    ranked = app.state.ranking.rank(
        candidates,
        context=req.context,
        top_k=req.k,
    )
    latency_ms = (time.perf_counter() - started) * 1000
    record_recommendation_request(surface=str(req.context.get("surface", "unknown")), latency_ms=latency_ms)
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
