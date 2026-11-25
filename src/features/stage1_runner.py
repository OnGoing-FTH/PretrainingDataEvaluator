"""Stage 1：OK 记忆库、独立 Audit 标定与缺陷实例 ROI 查询。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.geometry.letterbox import make_letterbox_transform

from .dinov2_extractor import DinoV2Extractor, DinoV2Settings, read_bgr
from .ok_bank import build_ok_bank, cosine_nearest, crop_centered_tile, make_fixed_tiles
from .ok_selection import discover_ok_images, split_ok_images
from .threshold_calibration import calibrate_thresholds, classify_distance


def run_stage1(input_dir: Path, stage0_records: list[dict[str, Any]], config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any], np.ndarray, list[dict[str, Any]]]:
    """建立 OK_Bank，使用独立 Audit 标定，再对 Stage 0 实例 ROI 查询。"""
    ok_images = discover_ok_images(input_dir)
    split = config["ok_split"]
    bank_images, audit_images, split_metadata = split_ok_images(ok_images, split["audit_fraction"], split["seed"])
    feature = config["feature"]
    extractor = DinoV2Extractor(DinoV2Settings(**feature))
    tile_config = config["tiles"]
    bank_vectors, bank_metadata = build_ok_bank(bank_images, extractor, tile_config["size"], tile_config["stride"])

    audit_tiles = [tile for image_path in audit_images for tile in make_fixed_tiles(image_path, tile_config["size"], tile_config["stride"])]
    audit_vectors = extractor.embed_bgr_images(tile.bgr for tile in audit_tiles)
    audit_distances, _ = cosine_nearest(audit_vectors, bank_vectors, config["query"]["k"])
    thresholds = calibrate_thresholds(audit_distances[:, 0], **config["thresholds"])

    queries = _build_instance_queries(stage0_records, extractor, bank_vectors, bank_metadata, thresholds, config)
    summary = {
        "stage": "stage1",
        "ok_selection_rule": "Images whose paired LabelMe JSON contains no valid polygon are treated as OK.",
        "ok_images": len(ok_images),
        "unique_ok_image_groups": split_metadata["unique_ok_image_groups"],
        "duplicate_ok_images_excluded_from_bank": split_metadata["duplicate_ok_images_excluded_from_bank"],
        "ok_bank_images": len(bank_images),
        "ok_audit_images": len(audit_images),
        "ok_bank_tiles": len(bank_metadata),
        "ok_audit_tiles": len(audit_tiles),
        "feature_extractor": extractor.describe(),
        "tile_strategy": tile_config,
        "thresholds": thresholds,
        "instance_queries": len(queries),
        "query_risk_counts": {level: sum(query["stage1_risk"] == level for query in queries) for level in ("high", "warning", "normal")},
    }
    return queries, summary, bank_vectors, bank_metadata


def _build_instance_queries(records: list[dict[str, Any]], extractor: DinoV2Extractor, bank_vectors: np.ndarray, bank_metadata: list[dict[str, Any]], thresholds: dict[str, Any], config: dict[str, Any]) -> list[dict[str, Any]]:
    margin_ratio = config["query"]["context_margin_ratio"]
    images: list[np.ndarray] = []
    payloads: list[dict[str, Any]] = []
    for record in records:
        image = read_bgr(Path(record["image_path"]))
        height, width = image.shape[:2]
        x1, y1, x2, y2 = record["geometry"]["bbox_xyxy"]
        box_width, box_height = max(1, x2 - x1), max(1, y2 - y1)
        margin = int(max(box_width, box_height) * margin_ratio)
        left, top = max(0, x1 - margin), max(0, y1 - margin)
        right, bottom = min(width, x2 + margin + 1), min(height, y2 + margin + 1)
        center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
        query_tile, query_tile_source = crop_centered_tile(image, center_x, center_y, config["tiles"]["size"])
        images.append(query_tile)
        transform = make_letterbox_transform(width, height, record["letterbox"]["target_size"])
        mapped_left = round(left * transform.scale + transform.pad_left)
        mapped_top = round(top * transform.scale + transform.pad_top)
        mapped_right = round(right * transform.scale + transform.pad_left)
        mapped_bottom = round(bottom * transform.scale + transform.pad_top)
        payloads.append({"record": record, "roi": {"x": left, "y": top, "width": right - left, "height": bottom - top}, "query_tile": query_tile_source, "roi_832": {"x": mapped_left, "y": mapped_top, "width": mapped_right - mapped_left, "height": mapped_bottom - mapped_top}})
    vectors = extractor.embed_bgr_images(images)
    distances, indices = cosine_nearest(vectors, bank_vectors, config["query"]["k"])
    results = []
    for payload, query_vector, neighbor_distances, neighbor_indices in zip(payloads, vectors, distances, indices):
        record = payload["record"]
        nearest_index = int(neighbor_indices[0])
        results.append({
            "instance_id": record["instance_id"],
            "image_path": record["image_path"],
            "label": record["label"],
            "stage0_risk": record["risk"]["level"],
            "stage0_proto_iou": record["proto"]["iou_proto"],
            "roi": payload["roi"],
            "roi_832": payload["roi_832"],
            "query_tile": payload["query_tile"],
            "query_vector_norm": round(float(np.linalg.norm(query_vector)), 6),
            "nearest_distance": round(float(neighbor_distances[0]), 8),
            "knn_mean_distance": round(float(neighbor_distances.mean()), 8),
            "stage1_risk": classify_distance(float(neighbor_distances[0]), thresholds),
            "nearest_ok_tile": bank_metadata[nearest_index],
            "nearest_neighbor_distances": [round(float(value), 8) for value in neighbor_distances.tolist()],
            "nearest_neighbor_indices": [int(value) for value in neighbor_indices.tolist()],
        })
    return sorted(results, key=lambda item: item["nearest_distance"], reverse=True)
