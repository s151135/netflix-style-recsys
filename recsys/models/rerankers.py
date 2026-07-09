from __future__ import annotations

import pandas as pd


def maximal_marginal_relevance(
    ranked: pd.DataFrame,
    top_k: int,
    score_col: str = "rank_score",
    genre_col: str = "genres",
    diversity_lambda: float = 0.2,
) -> pd.DataFrame:
    if ranked.empty or genre_col not in ranked.columns:
        return ranked.head(top_k).reset_index(drop=True)
    remaining = ranked.copy().reset_index(drop=True)
    selected_rows: list[pd.Series] = []
    selected_genres: set[str] = set()
    while len(selected_rows) < top_k and not remaining.empty:
        best_idx = None
        best_score = float("-inf")
        for idx, row in remaining.iterrows():
            genres = _genre_set(row.get(genre_col))
            overlap = len(genres.intersection(selected_genres))
            adjusted = float(row.get(score_col, 0.0)) - diversity_lambda * overlap
            if adjusted > best_score:
                best_idx = idx
                best_score = adjusted
        row = remaining.loc[best_idx]
        selected_rows.append(row)
        selected_genres.update(_genre_set(row.get(genre_col)))
        remaining = remaining.drop(index=best_idx).reset_index(drop=True)
    return pd.DataFrame(selected_rows).reset_index(drop=True)


def _genre_set(value) -> set[str]:
    if isinstance(value, list):
        return {str(x) for x in value}
    if isinstance(value, str):
        return {x.strip() for x in value.split("|") if x.strip()}
    return set()
