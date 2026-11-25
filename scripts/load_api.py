#!/usr/bin/env python3
"""Small HTTP latency measurement; not a production capacity benchmark."""
from __future__ import annotations

import argparse
import json
import math
import platform
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from smoke_api import fetch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--requests", type=int, default=40)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output")
    args = parser.parse_args()
    if not 1 <= args.requests <= 500 or not 1 <= args.workers <= 8:
        parser.error("requests must be 1..500 and workers 1..8")
    users = fetch(args.base_url, "/sample-users")["users"]
    if not users:
        parser.error("API has no learned profiles")

    def measure(index):
        start = time.perf_counter()
        try:
            data = fetch(args.base_url, "/recommend", {"user_id": users[index % len(users)], "k": 18})
            if not data["items"]:
                raise ValueError("Empty recommendation response")
            return (time.perf_counter() - start) * 1000, None
        except Exception as error:
            return None, type(error).__name__

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(measure, range(args.requests)))
    latencies = sorted(latency for latency, error in results if error is None)
    errors = [error for _, error in results if error is not None]
    def percentile(fraction):
        return latencies[max(0, math.ceil(fraction * len(latencies)) - 1)] if latencies else None
    report = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "requests": args.requests, "workers": args.workers,
              "successful": len(latencies), "failed": len(errors), "error_types": sorted(set(errors)),
              "p50_ms": percentile(0.5), "p95_ms": percentile(0.95), "p99_ms": percentile(0.99),
              "elapsed_seconds": time.perf_counter() - start,
              "platform": platform.platform(), "python": platform.python_version(),
              "scope": "Small local HTTP measurement; no production capacity or latency guarantee"}
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    print(text, end="")
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
