# Portfolio walkthrough

## Three-minute demonstration

1. Open Home. Explain that the BPR model learned user and movie embeddings from
   historical ratings and labels its output as **Learned pick**.
2. Open **My taste**. Show positive ratings, the genre summary and the list of
   highly rated movies. Explain that genre counts describe history rather than
   the model's internal latent features.
3. Switch to another profile. Compare its history and recommendation list. Some
   popular titles may overlap; personalization does not require every result to differ.
4. Open a recommended title's details and read its source explanation. Explain
   that the explanation is a transparent template, not generated reasoning.
5. Select **New viewer**. Show the empty history and training-popularity fallback.
6. Search for a title and switch between films and series. Note that modern series
   without training ratings are catalogue discoveries rather than learned BPR items.
7. Show the validation/test protocol and the popularity comparison in the README.

## CV wording

**Netflix-inspired AI Recommendation System — Python, NumPy, pandas, FastAPI, Docker**

- Built a movie discovery application using BPR collaborative filtering, heuristic
  reranking, cold-start popularity and transparent recommendation sources.
- Implemented per-user train/validation/test splits and compared BPR with popularity;
  achieved Recall@10 of 0.06199 versus 0.05391 in one held-out MovieLens experiment.
- Packaged the system with API validation, readiness checks, tests, CI and a
  reproducible container build.

Use a hosted demo link only after deployment is complete. Do not describe the
project as Netflix's own architecture, a production streaming platform, or a
deep-learning system. Do not imply that the 15% relative recall difference is a
statistically established general improvement.

## Interview topics to understand

- Why BPR learns pairwise preferences and how negative sampling works.
- Why a popularity baseline matters and why lower validation scores can favor
  a smaller model over a larger one.
- Why model selection is separated from final-test evaluation.
- Why a per-user temporal split can still leak global calendar information.
- Why cold-start users and items need fallbacks.
- Why offline retrieval metrics do not validate the final blended UI ranking.
- How artifact validation, wheel assets and readiness checks prevent deployment bugs.
