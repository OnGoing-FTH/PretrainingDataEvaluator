"""Stage 2 编排：Patch Bank、独立标定、实例流 A 与整图流 B。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .dinov2_extractor import DinoV2Extractor, DinoV2Settings
from .image_anomaly import analyze_images, find_missing_annotation_candidates, find_ok_contamination_candidates
from .instance_anomaly import analyze_instances
from .ok_selection import discover_ok_images, split_ok_images
from .patch_bank import build_patch_prototypes, calibrate_patch_thresholds, collect_tiles


def run_stage2(input_dir: Path, stage0_records: list[dict[str, Any]], config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    ok_images = discover_ok_images(input_dir)
    bank_images, audit_images, split_metadata = split_ok_images(ok_images, config["ok_split"]["audit_fraction"], config["ok_split"]["seed"])
    extractor = DinoV2Extractor(DinoV2Settings(**config["feature"]))
    tile_size, stride = config["tiles"]["size"], config["tiles"]["stride"]
    bank_tiles = collect_tiles(bank_images, tile_size, stride)
    audit_tiles = collect_tiles(audit_images, tile_size, stride)
    prototypes, patch_metadata = build_patch_prototypes(bank_tiles, extractor)
    patch_thresholds = calibrate_patch_thresholds(audit_tiles, extractor, prototypes, config["patch_calibration"], config["position_constraint"]["patch_radius"])

    instance_metrics, instance_maps = analyze_instances(stage0_records, extractor, prototypes, patch_thresholds, config)
    ng_images = sorted({Path(record["image_path"]) for record in stage0_records})
    ng_maps = analyze_images(ng_images, extractor, prototypes, config)
    ok_maps = analyze_images(audit_images, extractor, prototypes, config)
    missing = find_missing_annotation_candidates(ng_maps, stage0_records, patch_thresholds["high_threshold"], config)
    contamination = find_ok_contamination_candidates(ok_maps, patch_thresholds["high_threshold"], config)

    summary = {
        "stage": "stage2",
        "feature_extractor": extractor.describe(),
        "ok_images": len(ok_images),
        "unique_ok_image_groups": split_metadata["unique_ok_image_groups"],
        "duplicate_ok_images_excluded_from_bank": split_metadata["duplicate_ok_images_excluded_from_bank"],
        "ok_bank_images": len(bank_images),
        "ok_audit_images": len(audit_images),
        "ok_bank_tiles": len(bank_tiles),
        "ok_audit_tiles": len(audit_tiles),
        "patch_bank_shape": list(prototypes.shape),
        "patch_thresholds": patch_thresholds,
        "instance_count": len(instance_metrics),
        "instance_candidate_counts": {kind: sum(kind in item["candidate_types"] for item in instance_metrics) for kind in ("suspected_weak_annotation", "suspected_coarse_or_multiple", "suspected_boundary_misalignment", "strong_visual_anomaly")},
        "missing_annotation_candidates": len(missing),
        "ok_contamination_candidates": len(contamination),
    }
    artifacts = {
        "prototypes": prototypes,
        "patch_metadata": patch_metadata,
        "instance_metrics": instance_metrics,
        "instance_maps": instance_maps,
        "ng_maps": ng_maps,
        "ok_maps": ok_maps,
        "missing_candidates": missing,
        "contamination_candidates": contamination,
    }
    return summary, artifacts
