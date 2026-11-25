# Deployment

## Local release build

Use Python 3.12 and the lock files from the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.lock
python -m pip install --no-deps -e .
python scripts/build_demo.py
make demo
```

`build_demo.py` checks the dataset fingerprint in the recorded experiment,
recreates the selected training configuration, and writes model files, metadata,
train-plus-validation history, the display catalogue and the MovieLens notice to
`artifacts/demo`. It does not retune parameters or evaluate test targets. If the
upstream dataset changes, the build stops; rerun `tune_local.py` on that new data
and use its report rather than relabeling old results.

To package a new tuning experiment, use `prepare_demo.py`. Both commands require
a fresh output directory. The Docker image build starts from an empty context and
therefore does not copy a local development environment or unreviewed model.

## Docker

```bash
docker build -f infra/docker/Dockerfile -t netflix-ai-recsys:1.0.0 .
docker run --rm -p 127.0.0.1:8000:8000 netflix-ai-recsys:1.0.0
```

Or run `docker compose up --build`. The API is the only required service. The
multi-stage image installs locked dependencies, builds the selected model, and
runs the installed Python package as a non-root user. Dataset download and model
training happen at image build, not on each application restart. The healthcheck
uses `/readyz`; it fails if no model was loaded. Build time depends on network and CPU.

## Public hosting

Deploy the Docker image on a host that runs Python containers. Configure:

- Dockerfile: `infra/docker/Dockerfile`; build context: repository root.
- Listening port: 8000; readiness/health path: `/readyz`.
- `RECSYS_ARTIFACT_DIR=/app/artifacts/demo` (already set in the image).
- Start command: use the image's default CMD. For a platform requiring a dynamic
  port, override with `sh -c 'python -m uvicorn recsys.serving.api:app --host 0.0.0.0 --port ${PORT:-8000}'`.
- TLS, traffic limits and public-domain routing: configure at the hosting platform.

Keep `/metrics` behind the platform's internal monitoring route if publishing the
demo publicly. It contains service counters, not individual rating events. The demo
does not accept new ratings or provide authenticated accounts. Do not put private
user data into this educational configuration.

A hosted URL, hosting credentials and any hosting payment remain operator choices.
The repository and release package do not claim that a live public deployment has
already been created. GitHub Pages alone cannot execute this Python backend.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `RECSYS_ARTIFACT_DIR` | `artifacts/local` | Directory with `bpr_champion`, `sample_items.json` and `sample_interactions.csv` |
| `RECSYS_ENABLE_REMOTE_ARTWORK` | `0` | Opt in to legacy TMDB artwork references; see third-party notices first |

`make demo` sets the first variable to `artifacts/demo`. The API does not load `.env`
automatically. Export variables in your shell or configure them on the host.
Optional research configuration files under `configs/` are not a runtime control
panel for the maintained demo; its ranking weights live in `ranking_service.py`.

## Verification and troubleshooting

```bash
python -m unittest discover -s tests -v
ruff check recsys scripts tests
python scripts/smoke_api.py --base-url http://127.0.0.1:8000
python scripts/load_api.py --base-url http://127.0.0.1:8000 --requests 40 --workers 4
```

The load script is a small local measurement, not a production capacity test. It
reports failures and latency percentiles, exits nonzero on errors, and does not
log rating histories.

- `/healthz` 200 but `/readyz` 503: build the model and point the artifact variable
  at its bundle, then restart the API.
- Port already in use: stop the previous instance or select another port.
- Missing posters: text cards are the default; external artwork is optional.
- Model shape/ID/finiteness error: the bundle is malformed; rebuild in a fresh
  directory. Do not edit factor files by hand.
- First build cannot download: check outbound access to `files.grouplens.org`.
  Cached `work/ml-latest-small/ratings.csv` and `movies.csv` allow offline rebuilds.

## Release outputs

`python scripts/package_release.py --output-dir ../outputs/release-v1.0.0 --include-demo artifacts/release-demo`
creates a source ZIP, wheel, source distribution, optional model/data bundle and
SHA-256 checksums. Source exports exclude `.git`, virtual environments, secrets,
download caches and generated training artifacts. The optional demo bundle includes
the original MovieLens notice and must retain its research-use conditions.
