"""Stage 1 同簇、同位置多 OK 参考的 DINOv2 Patch 差异与特征缓存。"""

from __future__ import annotations

import numpy as np

from .dinov2_extractor import DinoV2Extractor


def build_mean_reference_patch(extractor: DinoV2Extractor, reference_tiles: list[np.ndarray]) -> np.ndarray:
    """聚合一个簇、一个归一化位置的全部 OK 参考 Patch。

    对 L2 归一化特征，Query 与此均值的内积等价于 Query 对全部参考
    余弦相似度的均值，因此 1-q·mean(r) 等价于多参考平均余弦距离。
    """
    if not reference_tiles:
        raise ValueError("At least one same-cluster OK reference tile is required.")
    total: np.ndarray | None = None
    count = 0
    for _, patches in extractor.iter_bgr_batches_with_patches(reference_tiles):
        batch_sum = patches.sum(axis=0, dtype=np.float64)
        total = batch_sum if total is None else total + batch_sum
        count += len(patches)
    return (total / count).astype(np.float32)


def score_query_against_mean_reference(query_patches: np.ndarray, mean_reference: np.ndarray) -> np.ndarray:
    """返回同簇、同位置全部 OK 参考的平均余弦距离。"""
    similarity = np.einsum("hwd,hwd->hw", query_patches, mean_reference, optimize=True)
    return np.clip(1.0 - similarity, 0.0, 2.0).astype(np.float32)


def mean_paired_patch_distance(extractor: DinoV2Extractor, query_tile: np.ndarray, reference_tiles: list[np.ndarray]) -> np.ndarray:
    """兼容便捷接口：同簇、同位置多参考平均 Patch 距离。"""
    _, query_features = extractor.embed_bgr_images_with_patches([query_tile])
    return score_query_against_mean_reference(query_features[0], build_mean_reference_patch(extractor, reference_tiles))
