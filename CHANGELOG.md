# Changelog

## 1.0.0 — 25 November 2026

- Added validation-selected BPR training with a popularity comparison and recorded
  experiment settings, dataset fingerprint and evaluation limitations.
- Corrected negative sampling, minimum-training-history splits and ranking metrics
  for duplicate hits and missing recommendations.
- Connected the selected model to the UI, retained learned titles without posters,
  and distinguished collaborative, content and cold-start recommendation sources.
- Added the My taste dashboard, positive-rating summaries, source explanations and
  a new-viewer experience; removed uncalibrated match percentages.
- Added bounded request validation, readiness and metrics endpoints, model artifact
  validation, safer dataset extraction, and accessible UI loading/error handling.
- Added complete installed UI assets, dependency lock files, reproducible demo
  builds, a non-root Docker image and CI checks for source, wheel and container.
- Added evaluation, deployment, model-card and portfolio documentation plus data
  notices. Remote artwork is disabled by default.

This release is an educational recommendation-system demo. Hosted deployment is
separate from publication of the source and release assets.
