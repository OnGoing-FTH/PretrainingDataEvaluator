"""从独立 OK_Audit 的近邻距离分布自动标定异常阈值。"""

from __future__ import annotations

from typing import Any

import numpy as np


def calibrate_thresholds(audit_distances: np.ndarray, warning_percentile: float, high_percentile: float, high_multiplier: float) -> dict[str, Any]:
    """以 Audit 到 Bank 的 1-NN 距离得到可追溯的动态阈值。"""
    values = np.asarray(audit_distances, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("Audit distance distribution is empty.")
    if not 0 < warning_percentile < high_percentile <= 100:
        raise ValueError("Invalid percentile configuration.")
    warning = float(np.percentile(values, warning_percentile))
    high_base = float(np.percentile(values, high_percentile))
    high = high_base * high_multiplier
    return {
        "sample_count": int(values.size),
        "distance_metric": "cosine_distance",
        "warning_percentile": warning_percentile,
        "high_percentile": high_percentile,
        "high_multiplier": high_multiplier,
        "warning_threshold": round(warning, 8),
        "high_threshold": round(high, 8),
        "audit_distance_summary": {
            "min": round(float(values.min()), 8),
            "median": round(float(np.median(values)), 8),
            "mean": round(float(values.mean()), 8),
            "max": round(float(values.max()), 8),
        },
    }


def classify_distance(distance: float, thresholds: dict[str, Any]) -> str:
    if distance > thresholds["high_threshold"]:
        return "high"
    if distance > thresholds["warning_threshold"]:
        return "warning"
    return "normal"
