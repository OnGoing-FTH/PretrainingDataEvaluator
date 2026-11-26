"""Stage 1：OK 整图自适应分簇、簇内同位置标定与缺陷查询。"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from src.geometry.letterbox import make_letterbox_transform

from .clustering import adaptive_ok_clustering, assign_to_ok_clusters
from .dinov2_extractor import DinoV2Extractor, DinoV2Settings, read_bgr
from .ok_bank import crop_centered_tile
from .ok_selection import discover_ok_images, split_ok_images
from .threshold_calibration import calibrate_thresholds, classify_distance


def run_stage1(input_dir: Path, stage0_records: list[dict[str, Any]], config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any], np.ndarray, list[dict[str, Any]]]:
    """以 OK 聚类定义视觉域，在簇内同位置进行 Audit 标定和实例评分。"""
    extractor = DinoV2Extractor(DinoV2Settings(**config["feature"]))
    ok_images = discover_ok_images(input_dir)
    ok_groups = _hash_groups(ok_images)
    unique_ok = [paths[0] for _, paths in sorted(ok_groups.items())]
    ok_global_vectors = extractor.embed_global_bgr_images(read_bgr(path) for path in unique_ok)
    clustering = config["clustering"]
    labels, centers, clustering_info = adaptive_ok_clustering(ok_global_vectors, clustering["min_cluster_size"], clustering["max_k"], config["ok_split"]["seed"])
    group_cluster = {hash_key: int(label) for (hash_key, _), label in zip(sorted(ok_groups.items()), labels)}

    bank_by_cluster: dict[int, list[Path]] = defaultdict(list)
    audit_by_cluster: dict[int, list[Path]] = defaultdict(list)
    split_details: dict[str, Any] = {}
    for cluster_id in sorted(set(labels.tolist())):
        cluster_images = [path for hash_key, paths in ok_groups.items() if group_cluster[hash_key] == cluster_id for path in paths]
        bank, audit, metadata = split_ok_images(cluster_images, config["ok_split"]["audit_fraction"], config["ok_split"]["seed"] + cluster_id)
        bank_by_cluster[cluster_id], audit_by_cluster[cluster_id] = bank, audit
        split_details[str(cluster_id)] = {**metadata, "bank_images": len(bank), "audit_images": len(audit)}

    thresholds, audit_scores, bank_metadata = _calibrate_clusters(extractor, bank_by_cluster, audit_by_cluster, config)
    image_assignments = _assign_non_ok_images(stage0_records, extractor, centers)
    queries = _build_instance_queries(stage0_records, image_assignments, extractor, bank_by_cluster, thresholds, config)
    all_bank_vectors = extractor.embed_global_bgr_images(read_bgr(path) for paths in bank_by_cluster.values() for path in paths)
    summary = {
        "stage": "stage1",
        "ok_selection_rule": "Images whose paired LabelMe JSON contains no valid polygon are treated as OK.",
        "ok_images": len(ok_images),
        "unique_ok_image_groups": len(ok_groups),
        "duplicate_ok_images_excluded_from_bank": len(ok_images) - len(ok_groups),
        "clustering": {**clustering_info, "method": "KMeans on Letterbox-448 DINOv2 CLS; K selected by cosine silhouette score", "cluster_centers": centers.tolist()},
        "cluster_splits": split_details,
        "cluster_thresholds": thresholds,
        "feature_extractor": extractor.describe(),
        "tile_strategy": {**config["tiles"], "matching": "same_cluster_same_normalized_position"},
        "instance_queries": len(queries),
        "query_risk_counts": {level: sum(query["stage1_risk"] == level for query in queries) for level in ("high", "warning", "normal")},
    }
    bank_metadata.extend({"kind": "audit_score", **score} for score in audit_scores)
    return queries, summary, all_bank_vectors, bank_metadata


def _hash_groups(images: list[Path]) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(images):
        groups[hashlib.sha256(path.read_bytes()).hexdigest()].append(path)
    return dict(groups)


def _calibrate_clusters(extractor: DinoV2Extractor, bank_by_cluster: dict[int, list[Path]], audit_by_cluster: dict[int, list[Path]], config: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    thresholds: dict[str, Any] = {}
    audit_scores: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    tile_size = config["tiles"]["size"]
    grid_size = config["calibration"]["grid_size"]
    centers = [{"u": round((index + 0.5) / grid_size, 8), "v": round((row + 0.5) / grid_size, 8)} for row in range(grid_size) for index in range(grid_size)]
    for cluster_id, bank_images in bank_by_cluster.items():
        bank_cache = {path: read_bgr(path) for path in bank_images}
        query_tiles: list[np.ndarray] = []
        pair_tiles: list[np.ndarray] = []
        pair_owner: list[tuple[int, dict[str, Any]]] = []
        score_payloads: list[dict[str, Any]] = []
        for audit_path in audit_by_cluster[cluster_id]:
            audit_image = read_bgr(audit_path)
            for center in centers:
                query_tile, _ = _crop_at_normalized_center(audit_image, center, tile_size)
                query_index = len(score_payloads)
                query_tiles.append(query_tile)
                score_payloads.append({"cluster_id": cluster_id, "audit_image_path": str(audit_path), "normalized_center": center})
                for bank_path, bank_image in bank_cache.items():
                    crop, source = _crop_at_normalized_center(bank_image, center, tile_size)
                    pair_tiles.append(crop)
                    pair_owner.append((query_index, {"image_path": str(bank_path), **source, "normalized_center": center}))
        scores, representatives = _score_paired_tiles(extractor, query_tiles, pair_tiles, pair_owner, score_payloads)
        threshold = calibrate_thresholds(np.asarray([item["distance_mean"] for item in scores]), **config["thresholds"])
        threshold.update({"cluster_id": cluster_id, "bank_images": len(bank_images), "audit_images": len(audit_by_cluster[cluster_id]), "score_metric": "same_cluster_same_position_mean_cosine_distance"})
        thresholds[str(cluster_id)] = threshold
        audit_scores.extend(scores)
        metadata.extend({"kind": "bank_image", "cluster_id": cluster_id, "image_path": str(path)} for path in bank_images)
    return thresholds, audit_scores, metadata


def _score_paired_tiles(extractor: DinoV2Extractor, query_tiles: list[np.ndarray], paired_tiles: list[np.ndarray], pair_owner: list[tuple[int, dict[str, Any]],], payloads: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    query_vectors = extractor.embed_bgr_images(query_tiles)
    paired_vectors = extractor.embed_bgr_images(paired_tiles)
    distances_by_query: dict[int, list[tuple[float, dict[str, Any]]]] = defaultdict(list)
    for (query_index, meta), vector in zip(pair_owner, paired_vectors):
        distance = 1.0 - float(np.clip(query_vectors[query_index] @ vector, -1.0, 1.0))
        distances_by_query[query_index].append((distance, meta))
    scores, representatives = [], []
    for index, payload in enumerate(payloads):
        ordered = sorted(distances_by_query[index], key=lambda item: item[0])
        values = np.asarray([item[0] for item in ordered])
        median_item = ordered[len(ordered) // 2]
        score = {**payload, "distance_mean": round(float(values.mean()), 8), "distance_median": round(float(np.median(values)), 8), "distance_std": round(float(values.std()), 8), "distance_min": round(float(values.min()), 8), "distance_max": round(float(values.max()), 8), "bank_support": len(values), "median_ok_tile": median_item[1]}
        scores.append(score)
        representatives.append(median_item[1])
    return scores, representatives


def _assign_non_ok_images(records: list[dict[str, Any]], extractor: DinoV2Extractor, centers: np.ndarray) -> dict[str, dict[str, Any]]:
    image_paths = sorted({record["image_path"] for record in records})
    vectors = extractor.embed_global_bgr_images(read_bgr(Path(path)) for path in image_paths)
    assigned, nearest, second = assign_to_ok_clusters(vectors, centers)
    return {path: {"assigned_ok_cluster_id": int(cluster), "distance_to_cluster_centroid": round(float(distance), 8), "second_cluster_distance": round(float(second_distance), 8), "cluster_assignment_ambiguous": bool(not np.isnan(second_distance) and second_distance - distance < 0.02)} for path, cluster, distance, second_distance in zip(image_paths, assigned, nearest, second)}


def _build_instance_queries(records: list[dict[str, Any]], assignments: dict[str, dict[str, Any]], extractor: DinoV2Extractor, bank_by_cluster: dict[int, list[Path]], thresholds: dict[str, Any], config: dict[str, Any]) -> list[dict[str, Any]]:
    tile_size = config["tiles"]["size"]
    bank_cache = {path: read_bgr(path) for paths in bank_by_cluster.values() for path in paths}
    query_tiles, payloads, paired_tiles, pair_owner = [], [], [], []
    for record in records:
        image = read_bgr(Path(record["image_path"]))
        height, width = image.shape[:2]
        x1, y1, x2, y2 = record["geometry"]["bbox_xyxy"]
        center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
        query_tile, query_source = crop_centered_tile(image, center_x, center_y, tile_size)
        center = _normalized_center(center_x, center_y, width, height)
        assignment = assignments[record["image_path"]]
        payload_index = len(payloads)
        payloads.append({"record": record, "query_tile": query_source, "normalized_center": center, **assignment})
        query_tiles.append(query_tile)
        for bank_path in bank_by_cluster[assignment["assigned_ok_cluster_id"]]:
            crop, source = _crop_at_normalized_center(bank_cache[bank_path], center, tile_size)
            paired_tiles.append(crop)
            pair_owner.append((payload_index, {"image_path": str(bank_path), **source, "normalized_center": center}))
    scores, _ = _score_paired_tiles(extractor, query_tiles, paired_tiles, pair_owner, payloads)
    results = []
    for score in scores:
        record = score.pop("record")
        threshold = thresholds[str(score["assigned_ok_cluster_id"])]
        transform = make_letterbox_transform(record["source_size"]["width"], record["source_size"]["height"], record["letterbox"]["target_size"])
        x1, y1, x2, y2 = record["geometry"]["bbox_xyxy"]
        results.append({"instance_id": record["instance_id"], "image_path": record["image_path"], "label": record["label"], "stage0_risk": record["risk"]["level"], "stage0_proto_iou": record["proto"]["iou_proto"], "roi_832": {"x": round(x1 * transform.scale + transform.pad_left), "y": round(y1 * transform.scale + transform.pad_top), "width": round((x2 - x1) * transform.scale), "height": round((y2 - y1) * transform.scale)}, "score_metric": "same_cluster_same_position_mean_cosine_distance", "low_threshold": threshold["warning_threshold"], "high_threshold": threshold["high_threshold"], "stage1_risk": classify_distance(score["distance_mean"], threshold), **score})
    return sorted(results, key=lambda item: item["distance_mean"], reverse=True)


def _normalized_center(x: float, y: float, width: int, height: int) -> dict[str, float]:
    return {"u": round(float(np.clip(x / width, 0, 1)), 8), "v": round(float(np.clip(y / height, 0, 1)), 8)}


def _crop_at_normalized_center(image: np.ndarray, center: dict[str, float], tile_size: int) -> tuple[np.ndarray, dict[str, int]]:
    height, width = image.shape[:2]
    return crop_centered_tile(image, center["u"] * width, center["v"] * height, tile_size)
