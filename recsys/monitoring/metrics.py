from __future__ import annotations

try:
    from prometheus_client import Counter, Histogram
except Exception:  # pragma: no cover
    Counter = None
    Histogram = None


if Counter is not None:
    REQUESTS = Counter("recsys_recommend_requests_total", "Recommendation requests", ["surface"])
    LATENCY = Histogram("recsys_recommend_latency_ms", "Recommendation latency in milliseconds", ["surface"])
else:  # pragma: no cover
    REQUESTS = None
    LATENCY = None


def record_recommendation_request(surface: str, latency_ms: float) -> None:
    if REQUESTS is not None and LATENCY is not None:
        REQUESTS.labels(surface=surface).inc()
        LATENCY.labels(surface=surface).observe(latency_ms)
