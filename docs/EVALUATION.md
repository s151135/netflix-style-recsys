# Evaluation and model selection

Each training run writes `artifacts/local/bpr_champion/evaluation.json`, containing
BPR and popularity metrics, training settings, evaluation counts, and limitations.
Both models use the same candidate catalogue and exclude every title observed in
the user's training history, including disliked titles. Popularity counts training
interactions only. Disliked holdout titles are excluded from relevance targets.

Measured locally on MovieLens latest small on 25 November 2026, with 32 factors,
5 epochs, seed 42, and K=10:

| Metric | BPR | Popularity |
| --- | ---: | ---: |
| Recall@10 | 0.04461 | 0.04647 |
| MRR@10 | 0.01290 | 0.01578 |
| NDCG@10 | 0.02014 | 0.02295 |
| Training catalogue coverage@10 | 2.74% | 1.37% |

This run uses 100,226 training events and holds out the last event for each of
610 users. Of these holdouts, 538 have a positive event label and enter the
evaluation. The current MovieLens mapping treats ratings >=2.5 as positive;
this is a broad implicit-feedback definition rather than a strict "liked" label.
Twenty-one positive targets are absent from the positive training catalogue and
remain in the metric denominator as misses.

**The current BPR configuration does not outperform popularity on ranking metrics.**
Its recommendations cover more of the training catalogue. These are baseline
results from one seed, not evidence of a statistically significant quality gain.
The validation-selected experiment below uses a stricter relevance definition
and separates model selection from final evaluation.

The split is temporal within each user's history, but does not enforce a global
time cutoff across users. MovieLens ratings are proxy interactions, and this
benchmark measures retrieval over the training catalogue, not the curated UI's
genre blending, series recommendations, or diversity reranker.

### Validation-selected experiment

```bash
source .venv/bin/activate
python scripts/tune_local.py
# Or: make tune
```

This experiment reserves each user's final event for test and penultimate event
for validation, leaving users with fewer than three events in training only.
Ratings >=4 stars are positive for both training and evaluation. Lower ratings
remain in history for splitting and seen-title exclusion. BPR and popularity
use the same positive training events and candidate catalogue. Unknown users
receive training-popularity recommendations.

Three configurations are compared on validation NDCG@10, with the first
configuration winning an exact tie. The selected configuration is refitted on
train plus validation, then evaluated on test once. Changes to test labels
cannot change selection; this separation is covered by a regression test.

| Factors | Epochs | Learning rate | Regularization | Validation NDCG@10 |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 5 | 0.03 | 0.0001 | **0.03021** |
| 32 | 10 | 0.03 | 0.0001 | 0.02630 |
| 32 | 20 | 0.01 | 0.001 | 0.02994 |

Selected-model test results, measured on 25 November 2026 with seed 42:

| Metric | Selected BPR | Popularity |
| --- | ---: | ---: |
| Recall@10 | 0.06199 | 0.05391 |
| MRR@10 | 0.02508 | 0.02175 |
| NDCG@10 | 0.03373 | 0.02944 |
| Training catalogue coverage@10 | 1.59% | 1.48% |

There are 99,616 initial training events, 610 validation events and 610 test
events. Of the test targets, 371 are positive and enter evaluation; 23 are
absent from the final training catalogue of 6,275 items and count as misses.
BPR finds 23 targets in its top 10, versus 20 for popularity: a **15% relative
Recall@10 increase in this run**, or 0.81 percentage points. This is a small
absolute difference from one seed and split, not a statistical significance claim.
These numbers cannot be compared directly with the earlier benchmark because
the positive-rating threshold, training feedback and selection protocol changed.

The run writes `artifacts/experiments/bpr_validation/evaluation.json` and a
separate `selected_bpr/` model. It leaves the demo's serving model in place.
The report includes all trials, selected settings, dataset fingerprint, environment,
counts and limitations. A recorded run is in
[`docs/benchmarks/movielens-validation.json`](benchmarks/movielens-validation.json).
The dataset had already been explored earlier in this project; this is a
validation-selected experiment, not an independent external confirmation.
Per-user chronology still permits information from other users' later events;
a global time cutoff and repeated seeds are useful follow-up evaluations.

