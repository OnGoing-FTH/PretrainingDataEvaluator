"""位置约束 OK Patch 原型库、Patch 异常分数与独立 Audit 标定。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .dinov2_extractor import DinoV2Extractor
from .ok_bank import Tile, make_fixed_tiles
from .threshold_calibration import calibrate_thresholds


def collect_tiles(image_paths: list[Path], tile_size: int, stride: int) -> list[Tile]:
    return [tile for image_path in image_paths for tile in make_fixed_tiles(image_path, tile_size, stride)]


def build_patch_prototypes(tiles: list[Tile], extractor: DinoV2Extractor) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """按相对 Patch 网格位置流式聚合正常特征，并重新 L2 标准化。"""
    prototype_sum: np.ndarray | None = None
    tile_count = 0
    for _, patch_features in extractor.iter_bgr_batches_with_patches(tile.bgr for tile in tiles):
        batch_sum = patch_features.sum(axis=0, dtype=np.float64)
        prototype_sum = batch_sum if prototype_sum is None else prototype_sum + batch_sum
        tile_count += len(patch_features)
    if prototype_sum is None or tile_count == 0:
        raise ValueError("No OK tiles were available for Patch Bank construction.")
    prototypes = (prototype_sum / tile_count).astype(np.float32)
    prototypes /= np.maximum(np.linalg.norm(prototypes, axis=-1, keepdims=True), 1e-12)
    metadata = [
        {"image_path": str(tile.image_path), "x": tile.x, "y": tile.y, "width": tile.width, "height": tile.height, "tile_size": tile.bgr.shape[0]}
        for tile in tiles
    ]
    return prototypes.astype(np.float32), metadata


def score_patch_features(patch_features: np.ndarray, prototypes: np.ndarray, patch_radius: int = 0) -> np.ndarray:
    """查询相对位置邻域内的正常原型，取最相似者的余弦距离。"""
    if patch_radius < 0:
        raise ValueError("patch_radius must be non-negative")
    height, width = prototypes.shape[:2]
    best_similarity = np.full(patch_features.shape[:3], -1.0, dtype=np.float32)
    for dy in range(-patch_radius, patch_radius + 1):
        source_y = np.clip(np.arange(height) + dy, 0, height - 1)
        for dx in range(-patch_radius, patch_radius + 1):
            source_x = np.clip(np.arange(width) + dx, 0, width - 1)
            neighbor = prototypes[source_y[:, None], source_x[None, :]]
            similarity = np.einsum("nhwd,hwd->nhw", patch_features, neighbor, optimize=True)
            best_similarity = np.maximum(best_similarity, similarity)
    return np.clip(1.0 - best_similarity, 0.0, 2.0).astype(np.float32)


def calibrate_patch_thresholds(audit_tiles: list[Tile], extractor: DinoV2Extractor, prototypes: np.ndarray, thresholds: dict[str, float], patch_radius: int = 0) -> dict[str, Any]:
    score_batches = []
    for _, audit_features in extractor.iter_bgr_batches_with_patches(tile.bgr for tile in audit_tiles):
        score_batches.append(score_patch_features(audit_features, prototypes, patch_radius).reshape(-1))
    result = calibrate_thresholds(np.concatenate(score_batches), **thresholds)
    result["feature_level"] = "patch_token"
    result["patch_grid"] = int(prototypes.shape[0])
    result["position_patch_radius"] = patch_radius
    result["audit_tiles"] = len(audit_tiles)
    return result
