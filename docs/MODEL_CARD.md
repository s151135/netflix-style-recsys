# Model card: BPR v1.0.0

## Intended use

Educational movie-recommendation experiments and a public portfolio demo using
anonymized MovieLens profiles. It is not a streaming service, an identity system,
or a model for decisions about people outside entertainment discovery.

## Model and training

Bayesian Personalized Ranking matrix factorization learns a user vector, item
vector and item bias. A score is a user/item dot product plus item bias. Training
optimizes pairwise preferences between a liked movie and an unobserved item.
Known positives are never sampled as negatives; fully observed catalogues skip
updates that have no valid negative.

The recorded model has 16 factors, 5 epochs, learning rate 0.03, regularization
0.0001, two negatives per positive and seed 42. Training and evaluation positives
are ratings of at least 4 stars. Lower ratings remain in the chronological split
and seen-title exclusion history. Absence of a rating is not evidence of dislike;
negative sampling is a modeling approximation.

## Data and evaluation

MovieLens latest small contains 100,836 ratings across 9,742 titles from 610 users.
The recorded snapshot's ratings span 1996–2018. Each eligible user's last event is
test, penultimate event validation, and earlier events training. Users with fewer
than three events remain in training only. Tied timestamps use deterministic item
ordering; this does not imply an actual preference order within a rating batch.

Three configurations are selected by validation NDCG@10. The chosen model is
refitted on train plus validation, then test is scored once. The final model has
609 learned users and 6,275 items. The test has 371 positive targets; 23 titles
absent from the training catalogue remain as misses. See [evaluation details](EVALUATION.md)
and [recorded settings and results](benchmarks/movielens-validation.json).

The demo history excludes the final test events. It is not a live user profile.
New users receive positive-training-rating popularity. New curated titles can
enter via genre discovery, but have no BPR item vector.

## Serving and explanations

The API blends normalized learned preference (0.60), genre affinity (0.25),
catalogue rating-count popularity (0.10), and release-year freshness (0.05),
followed by a genre diversity heuristic. Cold-start popularity replaces the
learned component for new users. These heuristic weights have not been optimized
on the held-out retrieval benchmark. The final UI ranking is not what the reported
retrieval metrics evaluate.

Explanation templates disclose the source and matching genres. They are not
causal attribution, calibrated confidence, or language-model-generated reasoning.
The UI therefore does not display a percentage likelihood that a viewer will like
a title. Genre charts summarize rating history, not latent embedding weights.

## Limits

- Per-user chronology can use other users' later events; there is no global cutoff.
- One seed and one split do not establish statistical significance. The dataset
  was explored earlier in development, so this is not external confirmation.
- MovieLens users and ratings are not a representative sample of all viewers.
  Popularity, selection bias, and historical catalogue limits affect recommendations.
- There are no demographic fairness claims, online A/B results, user studies, or
  production-scale performance claims.
- No online learning, account authentication, persistent personal ratings, streaming
  playback, or complete two-tower/SASRec implementation is included.
- Series and curated metadata are a static demonstration snapshot; availability,
  maturity labels and credits are not guaranteed current or suitable as parental controls.

Future research can use a global cutoff, multiple seeds, richer metadata, sequence
models and user studies. These are extensions, not release prerequisites.
