"""OK 整图 CLS 的自适应聚类及非 OK 图簇归属。"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def adaptive_ok_clustering(vectors: np.ndarray, min_cluster_size: int, max_k: int, seed: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """遍历可行 K，以 silhouette 最大值选择 OK 图簇。"""
    count = len(vectors)
    upper = min(max_k, count // min_cluster_size)
    if upper < 2:
        labels = np.zeros(count, dtype=np.int32)
        return labels, np.mean(vectors, axis=0, keepdims=True), {"selected_k": 1, "silhouette_scores": {}, "fallback_reason": "insufficient_samples_for_multiple_clusters"}
    candidates: dict[int, float] = {}
    models: dict[int, KMeans] = {}
    for k in range(2, upper + 1):
        model = KMeans(n_clusters=k, random_state=seed, n_init=20)
        labels = model.fit_predict(vectors)
        if min(np.bincount(labels)) < min_cluster_size:
            continue
        candidates[k] = float(silhouette_score(vectors, labels, metric="cosine"))
        models[k] = model
    if not models:
        labels = np.zeros(count, dtype=np.int32)
        return labels, np.mean(vectors, axis=0, keepdims=True), {"selected_k": 1, "silhouette_scores": candidates, "fallback_reason": "no_valid_cluster_size"}
    selected_k = max(candidates, key=candidates.get)
    model = models[selected_k]
    centers = _normalize(model.cluster_centers_)
    return model.labels_.astype(np.int32), centers, {"selected_k": selected_k, "silhouette_scores": {str(k): round(value, 8) for k, value in candidates.items()}, "fallback_reason": None}


def assign_to_ok_clusters(vectors: np.ndarray, centers: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """非 OK 图只跟随最近 OK 聚类中心，不参与聚类边界更新。"""
    distances = 1.0 - np.clip(vectors @ centers.T, -1.0, 1.0)
    order = np.argsort(distances, axis=1)
    assigned = order[:, 0]
    nearest = distances[np.arange(len(vectors)), assigned]
    second = distances[np.arange(len(vectors)), order[:, 1]] if centers.shape[0] > 1 else np.full(len(vectors), np.nan)
    return assigned.astype(np.int32), nearest, second


def _normalize(vectors: np.ndarray) -> np.ndarray:
    return vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
