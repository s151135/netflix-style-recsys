# Netflix-Style AI Recommendation System

This repository implements the local-first production blueprint from
`Production Blueprint for a Netflix-Style AI Recommendation System.pdf`.

The design follows the blueprint's staged architecture:

1. Immutable interaction/impression contracts.
2. Feature snapshots and temporal train/test splits.
3. Baseline retrieval with popularity, implicit ALS, and BPR-style matrix factorisation.
4. ANN-compatible item embedding search.
5. Lightweight ranking/reranking with freshness, diversity, and business filters.
6. FastAPI serving, batch materialisation, metrics hooks, and Docker Compose scaffolding.

The code is intentionally usable on a MacBook first. Optional heavier dependencies such as Torch,
XGBoost, MLflow, DuckDB, Great Expectations, and ANN libraries are declared as extras and can be added
when the workload justifies them.

## Quick Start

Create an environment with Python 3.11+ and install the package:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Run the local training smoke pipeline:

```bash
python scripts/train_local.py --output-dir artifacts/local
```

Run tests:

```bash
python -m unittest discover -s tests -v
```

Start the API after installing serving dependencies:

```bash
uvicorn recsys.serving.api:app --host 0.0.0.0 --port 8000
```

Or run the local stack:

```bash
docker compose up --build
```

## Discovery UI

The local UI at `http://127.0.0.1:8000` is a discovery experience, not a streaming player.
It includes browse rails, film/series filters, title search, rich click-through information, and
transparent recommendation reasons. The model blends local MovieLens collaborative signals with
genre affinity, curated catalogue popularity, and a diversity reranker so that it can suggest both
known movies and cold-start series.

The behavioral data comes from [MovieLens latest small](https://grouplens.org/datasets/movielens/)
and the display catalogue is a small curated metadata layer. Poster paths use the documented
[TMDB image URL format](https://developer.themoviedb.org/docs/image-basics); production use should
replace the local layer with a licensed metadata feed or a server-side TMDB integration and the
required attribution.

## Repository Map

- `configs/` - local Mac, GPU server, and experiment defaults.
- `data_contracts/` - JSON schemas for model-facing tables.
- `recsys/data/` - readers, temporal splits, transforms, negative sampling, feature snapshots.
- `recsys/features/` - user, item, session, and freshness features.
- `recsys/models/` - ALS/BPR baselines, two-tower placeholders, ranking, ANN, reranking.
- `recsys/train/` - datasets, evaluation, registry, and pipeline orchestration.
- `recsys/serving/` - retrieval/ranking services and FastAPI endpoint.
- `recsys/batch/` - offline materialisation jobs.
- `recsys/monitoring/` - serving metrics, drift checks, alert helpers.
- `tests/` - contract, unit, and integration smoke coverage.

## Production Notes

Use temporal splits for offline evaluation. Random splits leak future behaviour and inflate quality.
Keep direct identifiers outside the model tables, store deletion requests as replayable work items, and
write recommendation/impression logs for every online response so ranking, debiasing, interleaving,
and A/B tests have a reliable audit trail.
