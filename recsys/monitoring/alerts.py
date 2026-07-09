from __future__ import annotations


def latency_alert(latency_ms: float, threshold_ms: float = 150.0) -> dict[str, object]:
    return {
        "alert": "high_recommendation_latency",
        "active": latency_ms > threshold_ms,
        "latency_ms": latency_ms,
        "threshold_ms": threshold_ms,
    }
