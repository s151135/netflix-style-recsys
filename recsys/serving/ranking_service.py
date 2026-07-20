from __future__ import annotations

import pandas as pd

from recsys.models.rerankers import maximal_marginal_relevance
from recsys.models.xgb_ranker import HeuristicRanker
from recsys.serving.business_rules import apply_business_rules


class RankingService:
    def __init__(self, items: pd.DataFrame | None = None, ranker: HeuristicRanker | None = None) -> None:
        self.items = items if items is not None else pd.DataFrame(columns=["item_id"])
        self.ranker = ranker or HeuristicRanker()

    def rank(
        self,
        candidates: list[dict[str, float | str]],
        context: dict | None = None,
        top_k: int = 20,
        diversity_lambda: float = 0.2,
    ) -> list[dict[str, float | str]]:
        df = pd.DataFrame(candidates)
        if df.empty:
            return []
        if not self.items.empty:
            df = df.merge(self.items, on="item_id", how="left")
        df = self._add_personalization_features(df, context=context)
        df = apply_business_rules(df, context=context)
        ranked = self.ranker.rank(df, top_k=max(top_k * 2, top_k))
        reranked = maximal_marginal_relevance(
            ranked, top_k=top_k, diversity_lambda=diversity_lambda
        )
        return [
            {
                "item_id": str(row.item_id),
                "title": str(getattr(row, "title", row.item_id)),
                "genres": _normalise_genres(getattr(row, "genres", [])),
                "language": _optional_str(getattr(row, "language", None)),
                "release_year": _optional_int(getattr(row, "release_year", None)),
                "avg_rating": _optional_float(getattr(row, "avg_rating", None)),
                "rating_count": _optional_int(getattr(row, "rating_count", None)),
                "poster_seed": _optional_str(getattr(row, "poster_seed", row.item_id)),
                "poster_url": _optional_str(getattr(row, "poster_url", None)),
                "backdrop_url": _optional_str(getattr(row, "backdrop_url", None)),
                "maturity_rating": _optional_str(getattr(row, "maturity_rating", None)),
                "runtime": _optional_str(getattr(row, "runtime", None)),
                "seasons": _optional_int(getattr(row, "seasons", None)),
                "type": _optional_str(getattr(row, "type", "movie")) or "movie",
                "overview": _optional_str(getattr(row, "overview", None)),
                "score": float(getattr(row, "match_score", 0.0)),
                "retrieval_score": float(getattr(row, "retrieval_score", 0.0)),
                "reason": _optional_str(getattr(row, "reason", None)),
            }
            for row in reranked.itertuples(index=False)
        ]

    def _add_personalization_features(self, df: pd.DataFrame, context: dict | None) -> pd.DataFrame:
        context = context or {}
        output = df.copy()
        preferred_genres = {str(value) for value in context.get("preferred_genres", [])}
        output["genre_affinity"] = output.get("genres", pd.Series([[]] * len(output))).apply(
            lambda values: len(set(_normalise_genres(values)).intersection(preferred_genres))
            / max(1, len(preferred_genres))
        )
        raw_retrieval = pd.to_numeric(output.get("retrieval_score", 0.0), errors="coerce").fillna(0.0)
        output["retrieval_score"] = _scale(raw_retrieval)
        popularity = pd.to_numeric(output.get("popularity", output.get("rating_count", 0.0)), errors="coerce").fillna(0.0)
        output["popularity_score"] = _scale(popularity)
        output["freshness_score"] = _scale(pd.to_numeric(output.get("release_year", 0), errors="coerce").fillna(0.0))
        output["match_score"] = (
            0.52 * output["retrieval_score"]
            + 0.31 * output["genre_affinity"]
            + 0.11 * output["popularity_score"]
            + 0.06 * output["freshness_score"]
        ).clip(0, 1)
        output["retrieval_score"] = output["match_score"]
        anchor = str(context.get("anchor_title", "")).strip()
        def explanation(row) -> str:
            common = set(_normalise_genres(row.get("genres", []))).intersection(preferred_genres)
            genre = sorted(common)[0] if common else (next(iter(_normalise_genres(row.get("genres", []))), ""))
            if anchor and genre:
                return f"Because you rated {anchor} highly and often enjoy {genre}."
            if genre:
                return f"A strong {genre} match for this profile."
            return "Selected from collaborative rating patterns and catalogue quality."
        output["reason"] = output.apply(explanation, axis=1)
        return output


def _normalise_genres(value) -> list[str]:
    if isinstance(value, list):
        return [str(x) for x in value]
    if isinstance(value, str):
        return [x.strip() for x in value.replace(",", "|").split("|") if x.strip()]
    return []


def _optional_str(value) -> str | None:
    if pd.isna(value):
        return None
    return str(value)


def _optional_int(value) -> int | None:
    if pd.isna(value):
        return None
    return int(value)


def _optional_float(value) -> float | None:
    if pd.isna(value):
        return None
    return float(value)


def _scale(values: pd.Series) -> pd.Series:
    if values.empty:
        return values
    low, high = float(values.min()), float(values.max())
    if high <= low:
        return pd.Series([0.5] * len(values), index=values.index, dtype=float)
    return (values - low) / (high - low)
