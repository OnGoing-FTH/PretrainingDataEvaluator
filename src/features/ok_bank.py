"""固定物理视野 Tile 的 OK 记忆库构建与余弦近邻查询。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .dinov2_extractor import DinoV2Extractor, read_bgr


@dataclass(frozen=True)
class Tile:
    image_path: Path
    x: int
    y: int
    width: int
    height: int
    bgr: np.ndarray


def crop_centered_tile(image: np.ndarray, center_x: float, center_y: float, tile_size: int) -> tuple[np.ndarray, dict[str, int]]:
    """提取固定物理视野上下文；越界部分使用反射填充。"""
    height, width = image.shape[:2]
    left = int(round(center_x - tile_size / 2))
    top = int(round(center_y - tile_size / 2))
    right, bottom = left + tile_size, top + tile_size
    source_left, source_top = max(0, left), max(0, top)
    source_right, source_bottom = min(width, right), min(height, bottom)
    crop = image[source_top:source_bottom, source_left:source_right]
    pad_left, pad_top = source_left - left, source_top - top
    pad_right, pad_bottom = right - source_right, bottom - source_bottom
    padded = cv2.copyMakeBorder(crop, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT_101)
    return padded, {"x": source_left, "y": source_top, "width": source_right - source_left, "height": source_bottom - source_top}


def make_fixed_tiles(image_path: Path, tile_size: int, stride: int) -> list[Tile]:
    """以固定物理视野生成覆盖全图的 Tile，末端强制贴边。"""
    image = read_bgr(image_path)
    height, width = image.shape[:2]
    if width <= tile_size and height <= tile_size:
        canvas = cv2.copyMakeBorder(image, 0, tile_size - height, 0, tile_size - width, cv2.BORDER_REFLECT_101)
        return [Tile(image_path, 0, 0, width, height, canvas)]
    x_positions = _positions(width, tile_size, stride)
    y_positions = _positions(height, tile_size, stride)
    tiles = []
    for y in y_positions:
        for x in x_positions:
            tile = image[y:min(y + tile_size, height), x:min(x + tile_size, width)]
            padded = cv2.copyMakeBorder(tile, 0, tile_size - tile.shape[0], 0, tile_size - tile.shape[1], cv2.BORDER_REFLECT_101)
            tiles.append(Tile(image_path, x, y, min(tile_size, width - x), min(tile_size, height - y), padded))
    return tiles


def _positions(length: int, tile_size: int, stride: int) -> list[int]:
    positions = list(range(0, max(1, length - tile_size + 1), stride))
    edge = max(0, length - tile_size)
    if positions[-1] != edge:
        positions.append(edge)
    return positions


def build_ok_bank(image_paths: list[Path], extractor: DinoV2Extractor, tile_size: int, stride: int) -> tuple[np.ndarray, list[dict[str, Any]]]:
    tiles = [tile for image_path in image_paths for tile in make_fixed_tiles(image_path, tile_size, stride)]
    vectors = extractor.embed_bgr_images(tile.bgr for tile in tiles)
    metadata = [
        {"image_path": str(tile.image_path), "x": tile.x, "y": tile.y, "width": tile.width, "height": tile.height, "tile_size": tile_size}
        for tile in tiles
    ]
    return vectors, metadata


def cosine_nearest(query_vectors: np.ndarray, bank_vectors: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """精确余弦距离 KNN；Stage 1 规模下无须引入 Faiss。"""
    if bank_vectors.size == 0:
        raise ValueError("OK bank is empty.")
    k = max(1, min(k, len(bank_vectors)))
    similarities = np.clip(query_vectors @ bank_vectors.T, -1.0, 1.0)
    distances = 1.0 - similarities
    nearest = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]
    nearest_distances = np.take_along_axis(distances, nearest, axis=1)
    order = np.argsort(nearest_distances, axis=1)
    return np.take_along_axis(nearest_distances, order, axis=1), np.take_along_axis(nearest, order, axis=1)
