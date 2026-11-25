# HTTP smoke and small load checks

Run a release demo, then use `python scripts/smoke_api.py` and
`python scripts/load_api.py --requests 40 --workers 4` from the repository root.
The latter measures HTTP latency and errors, not production-scale capacity.
See `docs/DEPLOYMENT.md` for commands and `docs/benchmarks/serving-smoke.json`
for the recorded local result when present.
