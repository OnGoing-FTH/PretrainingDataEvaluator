"""Stage 2：严格复用新版 Stage 1 的分簇、同位置多参考匹配。"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.geometry.polygons import rasterize_polygon

from .dinov2_extractor import DinoV2Extractor, DinoV2Settings, read_bgr
from .image_anomaly import find_missing_annotation_candidates, find_ok_contamination_candidates
from .instance_anomaly import _measure_mask
from .ok_bank import crop_centered_tile, make_fixed_tiles
from .paired_patch import build_mean_reference_patch, score_query_against_mean_reference
from .threshold_calibration import calibrate_thresholds


def run_stage2(input_dir: Path, stage0_records: list[dict[str, Any]], config: dict[str, Any], stage1_artifact_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    queries = json.loads((stage1_artifact_dir / "instance_queries.json").read_text(encoding="utf-8"))
    metadata = json.loads((stage1_artifact_dir / "ok_bank_metadata.json").read_text(encoding="utf-8"))
    stage1_summary = json.loads((stage1_artifact_dir / "stage1_summary.json").read_text(encoding="utf-8"))
    stage1_by_id = {item["instance_id"]: item for item in queries}
    bank_by_cluster = _bank_by_cluster(metadata)
    audit_by_cluster = _audit_by_cluster(metadata)
    extractor = DinoV2Extractor(DinoV2Settings(**config["feature"]))
    tile_size = config["tiles"]["size"]
    bank_images = {path: read_bgr(path) for paths in bank_by_cluster.values() for path in paths}

    scan_paths = sorted({Path(record["image_path"]) for record in stage0_records} | {path for paths in audit_by_cluster.values() for path in paths})
    scan_centers = _scan_centers(scan_paths, tile_size, config["tiles"]["stride"])
    reference_means = _build_reference_means(extractor, bank_by_cluster, bank_images, scan_centers, tile_size)
    thresholds = _calibrate_patch_thresholds(extractor, audit_by_cluster, reference_means, scan_centers, tile_size, config)
    instance_metrics, instance_maps = _analyze_instances(stage0_records, stage1_by_id, bank_by_cluster, bank_images, extractor, thresholds, tile_size, config)

    cluster_by_ng = {item["image_path"]: int(item["assigned_ok_cluster_id"]) for item in queries}
    ng_paths = sorted({Path(record["image_path"]) for record in stage0_records})
    ng_maps = _analyze_images(ng_paths, cluster_by_ng, reference_means, extractor, config)
    ng_high = {path: thresholds[str(cluster)]["high_threshold"] for path, cluster in cluster_by_ng.items()}
    missing = find_missing_annotation_candidates(ng_maps, stage0_records, ng_high, config)

    audit_paths = [path for paths in audit_by_cluster.values() for path in paths]
    audit_clusters = {str(path): cluster for cluster, paths in audit_by_cluster.items() for path in paths}
    ok_maps = _analyze_images(audit_paths, audit_clusters, reference_means, extractor, config)
    audit_high = {path: thresholds[str(cluster)]["high_threshold"] for path, cluster in audit_clusters.items()}
    contamination = _deduplicate_candidates(find_ok_contamination_candidates(ok_maps, audit_high, config))

    summary = {
        "stage": "stage2",
        "stage1_artifact_dir": str(stage1_artifact_dir),
        "reference_strategy": "Instance and image streams: exact same-cluster/same-normalized-position all-reference mean Patch distance; Stage1 median reference is display-only.",
        "feature_extractor": extractor.describe(),
        "stage1_clustering": stage1_summary["clustering"],
        "cluster_patch_thresholds": thresholds,
        "instance_count": len(instance_metrics),
        "instance_candidate_counts": {kind: sum(kind in item["candidate_types"] for item in instance_metrics) for kind in ("suspected_weak_annotation", "suspected_coarse_or_multiple", "suspected_boundary_misalignment", "strong_visual_anomaly")},
        "missing_annotation_candidates": len(missing),
        "ok_contamination_candidates": len(contamination),
    }
    artifacts = {
        "reference_metadata": {"bank_by_cluster": {str(key): [str(path) for path in value] for key, value in bank_by_cluster.items()}, "audit_by_cluster": {str(key): [str(path) for path in value] for key, value in audit_by_cluster.items()}, "image_stream_normalized_centers": scan_centers},
        "instance_metrics": instance_metrics, "instance_maps": instance_maps, "ng_maps": ng_maps, "ok_maps": ok_maps,
        "missing_candidates": missing, "contamination_candidates": contamination,
    }
    return summary, artifacts


def _bank_by_cluster(metadata: list[dict[str, Any]]) -> dict[int, list[Path]]:
    grouped: defaultdict[int, list[Path]] = defaultdict(list)
    for item in metadata:
        if item.get("kind") == "bank_image":
            grouped[int(item["cluster_id"])].append(Path(item["image_path"]))
    return {cluster: sorted(paths) for cluster, paths in grouped.items()}


def _audit_by_cluster(metadata: list[dict[str, Any]]) -> dict[int, list[Path]]:
    grouped: defaultdict[int, set[Path]] = defaultdict(set)
    for item in metadata:
        if item.get("kind") == "audit_score":
            grouped[int(item["cluster_id"])].add(Path(item["audit_image_path"]))
    return {cluster: sorted(paths) for cluster, paths in grouped.items()}


def _scan_centers(image_paths: list[Path], tile_size: int, stride: int) -> list[dict[str, float]]:
    centers = set()
    for path in image_paths:
        image = read_bgr(path)
        height, width = image.shape[:2]
        for tile in make_fixed_tiles(path, tile_size, stride):
            centers.add((round(float((tile.x + tile.width / 2) / width), 8), round(float((tile.y + tile.height / 2) / height), 8)))
    return [{"u": u, "v": v} for u, v in sorted(centers)]


def _center_key(cluster_id: int, center: dict[str, float]) -> tuple[int, float, float]:
    return cluster_id, center["u"], center["v"]


def _build_reference_means(extractor: DinoV2Extractor, bank_by_cluster: dict[int, list[Path]], bank_images: dict[Path, np.ndarray], centers: list[dict[str, float]], tile_size: int) -> dict[tuple[int, float, float], np.ndarray]:
    cache = {}
    for cluster_id, paths in bank_by_cluster.items():
        for center in centers:
            refs = [_crop_normalized(bank_images[path], center, tile_size)[0] for path in paths]
            cache[_center_key(cluster_id, center)] = build_mean_reference_patch(extractor, refs).astype(np.float16)
    return cache


def _calibrate_patch_thresholds(extractor: DinoV2Extractor, audit_by_cluster: dict[int, list[Path]], reference_means: dict[tuple[int, float, float], np.ndarray], centers: list[dict[str, float]], tile_size: int, config: dict[str, Any]) -> dict[str, Any]:
    values: defaultdict[int, list[np.ndarray]] = defaultdict(list)
    for cluster_id, paths in audit_by_cluster.items():
        for path in paths:
            image = read_bgr(path)
            for center in centers:
                tile, _ = _crop_normalized(image, center, tile_size)
                _, patches = extractor.embed_bgr_images_with_patches([tile])
                values[cluster_id].append(score_query_against_mean_reference(patches[0], reference_means[_center_key(cluster_id, center)]).reshape(-1))
    results = {}
    for cluster_id, score_values in values.items():
        threshold = calibrate_thresholds(np.concatenate(score_values), **config["patch_calibration"])
        threshold.update({"cluster_id": cluster_id, "audit_images": len(audit_by_cluster[cluster_id]), "calibration_grid_size": config["stage1_calibration_grid_size"], "score_metric": "same_cluster_same_position_mean_patch_cosine_distance"})
        results[str(cluster_id)] = threshold
    return results


def _analyze_instances(records: list[dict[str, Any]], stage1_by_id: dict[str, dict[str, Any]], bank_by_cluster: dict[int, list[Path]], bank_images: dict[Path, np.ndarray], extractor: DinoV2Extractor, thresholds: dict[str, Any], tile_size: int, config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    results, maps = [], {}
    for record in records:
        stage1 = stage1_by_id[record["instance_id"]]
        cluster = int(stage1["assigned_ok_cluster_id"])
        image = read_bgr(Path(record["image_path"]))
        query, source = _crop_normalized(image, stage1["normalized_center"], tile_size)
        refs = [_crop_normalized(bank_images[path], stage1["normalized_center"], tile_size)[0] for path in bank_by_cluster[cluster]]
        mean_reference = build_mean_reference_patch(extractor, refs)
        _, query_patches = extractor.embed_bgr_images_with_patches([query])
        anomaly_map = cv2.resize(score_query_against_mean_reference(query_patches[0], mean_reference), (tile_size, tile_size), interpolation=cv2.INTER_LINEAR)
        x1, y1, x2, y2 = record["geometry"]["bbox_xyxy"]
        center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
        points = np.asarray(record["polygon_points"], dtype=np.float32) - np.asarray([round(center_x - tile_size / 2), round(center_y - tile_size / 2)], dtype=np.float32)
        mask = rasterize_polygon(points, tile_size, tile_size)
        threshold = thresholds[str(cluster)]
        metrics = _measure_mask(anomaly_map, mask, threshold["warning_threshold"], threshold["high_threshold"], config["instance_rules"])
        kinds = []
        if metrics["mask_mean_anomaly"] < threshold["warning_threshold"] and metrics["mask_normal_pixel_ratio"] >= config["instance_rules"]["weak_normal_ratio"]: kinds.append("suspected_weak_annotation")
        if metrics["mask_normal_pixel_ratio"] >= config["instance_rules"]["coarse_normal_ratio"] or metrics["high_components"] >= 2: kinds.append("suspected_coarse_or_multiple")
        if metrics["outer_ring_anomaly"] >= metrics["inner_ring_anomaly"] + config["instance_rules"]["boundary_margin"]: kinds.append("suspected_boundary_misalignment")
        if metrics["mask_mean_anomaly"] > threshold["high_threshold"]: kinds.append("strong_visual_anomaly")
        results.append({"instance_id": record["instance_id"], "image_path": record["image_path"], "label": record["label"], "stage0_risk": record["risk"]["level"], "stage0_proto_iou": record["proto"]["iou_proto"], "assigned_ok_cluster_id": cluster, "normalized_center": stage1["normalized_center"], "query_tile": source, "median_ok_tile": stage1["median_ok_tile"], "bank_support": len(refs), "patch_low_threshold": threshold["warning_threshold"], "patch_high_threshold": threshold["high_threshold"], "patch_score_metric": threshold["score_metric"], "polygon_points": record["polygon_points"], "candidate_types": kinds, **metrics})
        maps[record["instance_id"]] = anomaly_map.astype(np.float16)
    return sorted(results, key=lambda item: item["mask_mean_anomaly"], reverse=True), maps


def _analyze_images(image_paths: list[Path], clusters: dict[str, int], means: dict[tuple[int, float, float], np.ndarray], extractor: DinoV2Extractor, config: dict[str, Any]) -> dict[str, np.ndarray]:
    results = {}
    tile_size, stride = config["tiles"]["size"], config["tiles"]["stride"]
    for path in image_paths:
        image = read_bgr(path)
        height, width = image.shape[:2]
        cluster = clusters[str(path)]
        tiles = make_fixed_tiles(path, tile_size, stride)
        accumulator = np.full((height, width), np.inf, dtype=np.float32)
        batch_tiles, batch_meta = [], []
        def flush() -> None:
            if not batch_tiles:
                return
            _, patch_batches = extractor.embed_bgr_images_with_patches(batch_tiles)
            for (tile, center), patch in zip(batch_meta, patch_batches):
                score = score_query_against_mean_reference(patch, means[_center_key(cluster, center)])
                tile_map = cv2.resize(score, (tile_size, tile_size), interpolation=cv2.INTER_LINEAR)[:tile.height, :tile.width]
                accumulator[tile.y:tile.y + tile.height, tile.x:tile.x + tile.width] = np.minimum(accumulator[tile.y:tile.y + tile.height, tile.x:tile.x + tile.width], tile_map)
            batch_tiles.clear(); batch_meta.clear()
        for tile in tiles:
            center = {"u": round(float(np.clip((tile.x + tile.width / 2) / width, 0, 1)), 8), "v": round(float(np.clip((tile.y + tile.height / 2) / height, 0, 1)), 8)}
            batch_tiles.append(tile.bgr); batch_meta.append((tile, center))
            if len(batch_tiles) >= extractor.settings.batch_size: flush()
        flush()
        results[str(path)] = accumulator.astype(np.float16)
    return results


def _crop_normalized(image: np.ndarray, center: dict[str, float], tile_size: int) -> tuple[np.ndarray, dict[str, int]]:
    height, width = image.shape[:2]
    return crop_centered_tile(image, center["u"] * width, center["v"] * height, tile_size)


def _deduplicate_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen, unique = set(), []
    for item in candidates:
        key = (str(Path(item["image_path"]).resolve()), tuple(item["bbox_xywh"]))
        if key not in seen:
            seen.add(key); unique.append(item)
    return unique
