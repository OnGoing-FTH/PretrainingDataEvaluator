"""Stage 2 流 A：实例 Patch 异常图与标注质量指标。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.geometry.polygons import rasterize_polygon

from .dinov2_extractor import DinoV2Extractor, read_bgr
from .ok_bank import crop_centered_tile
from .patch_bank import score_patch_features


def analyze_instances(records: list[dict[str, Any]], extractor: DinoV2Extractor, prototypes: np.ndarray, patch_thresholds: dict[str, Any], config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    tile_size = config["tiles"]["size"]
    query_tiles: list[np.ndarray] = []
    payloads: list[dict[str, Any]] = []
    for record in records:
        image = read_bgr(Path(record["image_path"]))
        height, width = image.shape[:2]
        x1, y1, x2, y2 = record["geometry"]["bbox_xyxy"]
        center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
        tile, source_tile = crop_centered_tile(image, center_x, center_y, tile_size)
        virtual_left, virtual_top = int(round(center_x - tile_size / 2)), int(round(center_y - tile_size / 2))
        local_points = np.asarray(record["polygon_points"], dtype=np.float32) - np.asarray([virtual_left, virtual_top], dtype=np.float32)
        local_mask = rasterize_polygon(local_points, tile_size, tile_size)
        query_tiles.append(tile)
        payloads.append({"record": record, "tile": source_tile, "tile_virtual_origin": {"x": virtual_left, "y": virtual_top}, "mask": local_mask})

    results: list[dict[str, Any]] = []
    maps: dict[str, np.ndarray] = {}
    warning = patch_thresholds["warning_threshold"]
    high = patch_thresholds["high_threshold"]
    cursor = 0
    for _, patch_features in extractor.iter_bgr_batches_with_patches(query_tiles):
        patch_scores = score_patch_features(patch_features, prototypes, config["position_constraint"]["patch_radius"])
        for payload, score_grid in zip(payloads[cursor:cursor + len(patch_scores)], patch_scores):
            record, mask = payload["record"], payload["mask"]
            anomaly_map = cv2.resize(score_grid, (tile_size, tile_size), interpolation=cv2.INTER_LINEAR)
            metrics = _measure_mask(anomaly_map, mask, warning, high, config["instance_rules"])
            candidate_types = []
            if metrics["mask_mean_anomaly"] < warning and metrics["mask_normal_pixel_ratio"] >= config["instance_rules"]["weak_normal_ratio"]:
                candidate_types.append("suspected_weak_annotation")
            if metrics["mask_normal_pixel_ratio"] >= config["instance_rules"]["coarse_normal_ratio"] or metrics["high_components"] >= 2:
                candidate_types.append("suspected_coarse_or_multiple")
            if metrics["outer_ring_anomaly"] >= metrics["inner_ring_anomaly"] + config["instance_rules"]["boundary_margin"]:
                candidate_types.append("suspected_boundary_misalignment")
            if metrics["mask_mean_anomaly"] > high:
                candidate_types.append("strong_visual_anomaly")
            results.append({
                "instance_id": record["instance_id"], "image_path": record["image_path"], "label": record["label"],
                "stage0_risk": record["risk"]["level"], "stage0_proto_iou": record["proto"]["iou_proto"],
                "query_tile": payload["tile"], "query_tile_virtual_origin": payload["tile_virtual_origin"], "polygon_points": record["polygon_points"], "candidate_types": candidate_types, **metrics,
            })
            maps[record["instance_id"]] = anomaly_map.astype(np.float16)
        cursor += len(patch_scores)
    return sorted(results, key=lambda item: item["mask_mean_anomaly"], reverse=True), maps


def _measure_mask(anomaly_map: np.ndarray, mask: np.ndarray, warning: float, high: float, rules: dict[str, float]) -> dict[str, Any]:
    binary = mask > 0
    if not binary.any():
        return {"mask_mean_anomaly": 0.0, "mask_normal_pixel_ratio": 1.0, "bbox_background_mean_anomaly": 0.0, "inner_ring_anomaly": 0.0, "outer_ring_anomaly": 0.0, "high_components": 0}
    kernel_size = max(3, int(rules["ring_width_px"]) * 2 + 1)
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    eroded = cv2.erode(mask, kernel)
    dilated = cv2.dilate(mask, kernel)
    inner = (binary & ~(eroded > 0))
    outer = ((dilated > 0) & ~binary)
    x, y, width, height = cv2.boundingRect(mask)
    bbox_mask = np.zeros_like(binary)
    bbox_mask[y:y + height, x:x + width] = True
    background = bbox_mask & ~binary
    high_binary = ((anomaly_map > high) & binary).astype(np.uint8)
    components, _ = cv2.connectedComponents(high_binary)
    return {
        "mask_mean_anomaly": round(float(anomaly_map[binary].mean()), 8),
        "mask_normal_pixel_ratio": round(float((anomaly_map[binary] < warning).mean()), 8),
        "bbox_background_mean_anomaly": round(float(anomaly_map[background].mean()) if background.any() else 0.0, 8),
        "inner_ring_anomaly": round(float(anomaly_map[inner].mean()) if inner.any() else 0.0, 8),
        "outer_ring_anomaly": round(float(anomaly_map[outer].mean()) if outer.any() else 0.0, 8),
        "high_components": max(0, int(components - 1)),
    }
