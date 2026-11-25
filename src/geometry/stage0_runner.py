"""Stage 0 编排：标注解析、几何检查、832/208 Proto 模拟和结果汇总。"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2

from .annotations import discover_annotation_files, load_annotations
from .letterbox import make_letterbox_transform
from .polygons import measure_polygon, rasterize_polygon, validate_polygon
from .proto_simulator import simulate_proto
from .risk_rules import assess_risk


def run_stage0(input_dir: Path, target_size: int = 832, proto_stride: int = 4) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """执行 Stage 0，返回每实例审查记录和数据集汇总。"""
    proto_size = target_size // proto_stride
    records: list[dict[str, Any]] = []
    annotation_files = discover_annotation_files(input_dir)
    image_count = 0
    class_totals: Counter[str] = Counter()
    class_risks: defaultdict[str, Counter[str]] = defaultdict(Counter)
    files_with_annotations: set[Path] = set()

    for annotation_path in annotation_files:
        for annotation in load_annotations(annotation_path):
            if not annotation.image_path.exists():
                continue
            image = cv2.imread(str(annotation.image_path))
            if image is None:
                continue
            image_count += int(annotation.image_path not in files_with_annotations)
            files_with_annotations.add(annotation.image_path)
            height, width = image.shape[:2]
            validation_issues = validate_polygon(annotation.points, width, height)
            source_mask = rasterize_polygon(annotation.points, width, height)
            geometry = measure_polygon(annotation.points, source_mask, width, height)
            transform = make_letterbox_transform(width, height, target_size)
            mask_832 = transform.apply_to_mask(source_mask)
            proto = simulate_proto(mask_832, proto_size)
            risk = assess_risk(geometry, proto, validation_issues)
            class_totals[annotation.label] += 1
            class_risks[annotation.label][risk.level] += 1
            records.append(
                {
                    "instance_id": annotation.instance_id,
                    "image_path": str(annotation.image_path),
                    "annotation_path": str(annotation.annotation_path),
                    "label": annotation.label,
                    "source_size": {"width": width, "height": height},
                    "polygon_points": annotation.points.round(3).tolist(),
                    "letterbox": {
                        "target_size": target_size,
                        "scale": round(transform.scale, 8),
                        "resized_size": {"width": transform.resized_width, "height": transform.resized_height},
                        "padding": {"left": transform.pad_left, "top": transform.pad_top, "right": transform.pad_right, "bottom": transform.pad_bottom},
                    },
                    "geometry": asdict(geometry),
                    "proto": {
                        "proto_size": proto.proto_size,
                        "iou_proto": round(proto.iou_proto, 6),
                        "proto_nonzero_pixels": proto.proto_nonzero_pixels,
                        "original_components": proto.original_components,
                        "restored_components": proto.restored_components,
                        "fragmented": proto.fragmented,
                        "vanished": proto.vanished,
                    },
                    "risk": asdict(risk),
                }
            )

    records.sort(key=lambda record: record["risk"]["score"], reverse=True)
    high_risk_count = sum(record["risk"]["level"] in {"critical", "high"} for record in records)
    class_summary = []
    for label in sorted(class_totals):
        total = class_totals[label]
        risks = class_risks[label]
        high_risk = risks["critical"] + risks["high"]
        class_summary.append(
            {
                "label": label,
                "instances": total,
                "critical": risks["critical"],
                "high": risks["high"],
                "medium": risks["medium"],
                "pass": risks["pass"],
                "high_risk_ratio": round(high_risk / total if total else 0.0, 6),
                "strategy_warning": high_risk / total > 0.15 if total else False,
            }
        )
    summary = {
        "stage": "stage0",
        "target_size": target_size,
        "proto_stride": proto_stride,
        "proto_size": proto_size,
        "annotation_files": len(annotation_files),
        "images_with_instances": image_count,
        "instances": len(records),
        "high_risk_instances": high_risk_count,
        "class_summary": class_summary,
    }
    return records, summary
