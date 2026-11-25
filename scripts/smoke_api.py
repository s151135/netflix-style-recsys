#!/usr/bin/env python3
"""Exercise a running release API without downloading data or using a browser."""
from __future__ import annotations

import argparse
import json
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def fetch(base_url, path, body=None):
    request = Request(base_url.rstrip("/") + path,
                      data=None if body is None else json.dumps(body).encode(),
                      headers={} if body is None else {"Content-Type": "application/json"})
    with urlopen(request, timeout=30) as response:
        content = response.read()
        return json.loads(content) if "application/json" in response.headers.get("Content-Type", "") else content


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    base = args.base_url
    assert fetch(base, "/healthz")["status"] == "ok"
    assert fetch(base, "/readyz")["status"] == "ready"
    assert b"Netflix AI Recommendation System" in fetch(base, "/")
    assert b"loadRecommendations" in fetch(base, "/assets/demo.js")
    assert fetch(base, "/model-info")["ready"]
    users = fetch(base, "/sample-users")["users"]
    assert users, "No learned profiles available"
    for user, mode in [(users[0], "personalized"), ("smoke-new-viewer", "cold_start")]:
        data = fetch(base, "/recommend", {"user_id": user, "k": 10})
        assert data["mode"] == mode
        ids = [item["item_id"] for item in data["items"]]
        assert ids and len(ids) == len(set(ids)) and len(ids) <= 10
        assert all(item.get("reason") and item.get("source") for item in data["items"])
        assert fetch(base, f"/title/{ids[0]}")["item"]["item_id"] == ids[0]
        blocked = fetch(base, "/recommend", {"user_id": user, "k": 10,
                                              "context": {"blocked_item_ids": [ids[0]]}})
        assert ids[0] not in [item["item_id"] for item in blocked["items"]]
    assert isinstance(fetch(base, "/catalog?query=%5B")["items"], list)
    try:
        fetch(base, "/recommend", {"user_id": " ", "k": 0})
    except HTTPError as error:
        assert error.code == 422
    else:
        raise AssertionError("Invalid request was accepted")
    assert b"recsys_recommend_requests_total" in fetch(base, "/metrics")
    print("Release API smoke checks passed")


if __name__ == "__main__":
    main()
