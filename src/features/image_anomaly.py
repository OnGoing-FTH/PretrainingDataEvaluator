"""Stage 2 流 B：整图滑窗 Patch 异常图、漏标与 OK 污染候选。"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.geometry.polygons import rasterize_polygon

from .dinov2_extractor import DinoV2Extractor, read_bgr
from .ok_bank import make_fixed_tiles
from .patch_bank import score_patch_features


def analyze_images(image_paths: list[Path], extractor: DinoV2Extractor, prototypes: np.ndarray, config: dict[str, Any]) -> dict[str, np.ndarray]:
    maps: dict[str, np.ndarray] = {}
    for image_path in image_paths:
        image = read_bgr(image_path)
        height, width = image.shape[:2]
        tiles = make_fixed_tiles(image_path, config["tiles"]["size"], config["tiles"]["stride"])
        accumulator = np.full((height, width), np.inf, dtype=np.float32)
        cursor = 0
        for _, features in extractor.iter_bgr_batches_with_patches(tile.bgr for tile in tiles):
            scores = score_patch_features(features, prototypes, config["position_constraint"]["patch_radius"])
            for tile, score_grid in zip(tiles[cursor:cursor + len(scores)], scores):
                tile_map = cv2.resize(score_grid, (config["tiles"]["size"], config["tiles"]["size"]), interpolation=cv2.INTER_LINEAR)
                tile_map = tile_map[:tile.height, :tile.width]
                current = accumulator[tile.y:tile.y + tile.height, tile.x:tile.x + tile.width]
                accumulator[tile.y:tile.y + tile.height, tile.x:tile.x + tile.width] = np.minimum(current, tile_map)
            cursor += len(scores)
        maps[str(image_path)] = accumulator.astype(np.float16)
    return maps


def find_missing_annotation_candidates(image_maps: dict[str, np.ndarray], records: list[dict[str, Any]], high_thresholds: float | dict[str, float], config: dict[str, Any]) -> list[dict[str, Any]]:
    records_by_image: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        records_by_image[record["image_path"]].append(record)
    candidates: list[dict[str, Any]] = []
    for image_path, anomaly_map in image_maps.items():
        mask = np.zeros(anomaly_map.shape, dtype=np.uint8)
        for record in records_by_image.get(image_path, []):
            points = np.asarray(record["polygon_points"], dtype=np.float32)
            mask = cv2.bitwise_or(mask, rasterize_polygon(points, anomaly_map.shape[1], anomaly_map.shape[0]))
        dilation = config["missing_annotation_scan"]["polygon_dilation_px"]
        if dilation:
            mask = cv2.dilate(mask, np.ones((dilation * 2 + 1, dilation * 2 + 1), dtype=np.uint8))
        edge_margin = config["missing_annotation_scan"].get("edge_exclusion_px", 0)
        if edge_margin:
            mask[:edge_margin, :] = 255
            mask[-edge_margin:, :] = 255
            mask[:, :edge_margin] = 255
            mask[:, -edge_margin:] = 255
        high_threshold = high_thresholds[image_path] if isinstance(high_thresholds, dict) else high_thresholds
        active = ((anomaly_map > high_threshold) & (mask == 0)).astype(np.uint8)
        component_count, labels, stats, _ = cv2.connectedComponentsWithStats(active)
        for component_id in range(1, component_count):
            x, y, width, height, area = stats[component_id]
            if area < config["missing_annotation_scan"]["minimum_component_area_px"]:
                continue
            region = anomaly_map[labels == component_id]
            candidates.append({"image_path": image_path, "bbox_xywh": [int(x), int(y), int(width), int(height)], "area_px": int(area), "peak_anomaly": round(float(region.max()), 8), "mean_anomaly": round(float(region.mean()), 8)})
    return sorted(candidates, key=lambda item: item["peak_anomaly"], reverse=True)


def find_ok_contamination_candidates(image_maps: dict[str, np.ndarray], high_thresholds: float | dict[str, float], config: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = []
    minimum = config["missing_annotation_scan"]["minimum_component_area_px"]
    for image_path, anomaly_map in image_maps.items():
        high_threshold = high_thresholds[image_path] if isinstance(high_thresholds, dict) else high_thresholds
        active = (anomaly_map > high_threshold).astype(np.uint8)
        component_count, labels, stats, _ = cv2.connectedComponentsWithStats(active)
        for component_id in range(1, component_count):
            x, y, width, height, area = stats[component_id]
            if area < minimum:
                continue
            region = anomaly_map[labels == component_id]
            candidates.append({"image_path": image_path, "bbox_xywh": [int(x), int(y), int(width), int(height)], "area_px": int(area), "peak_anomaly": round(float(region.max()), 8), "mean_anomaly": round(float(region.mean()), 8)})
    return sorted(candidates, key=lambda item: item["peak_anomaly"], reverse=True)
