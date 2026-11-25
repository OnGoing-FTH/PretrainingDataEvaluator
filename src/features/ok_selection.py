"""识别未标注缺陷的 OK 图片并进行防泄露划分。"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


def discover_ok_images(input_dir: Path) -> list[Path]:
    """将同名标注文件中没有有效 polygon 的图片视为 OK。"""
    images: list[Path] = []
    for image_path in sorted(path for path in input_dir.iterdir() if path.suffix.lower() in _IMAGE_EXTENSIONS):
        annotation_path = image_path.with_suffix(".json")
        if not annotation_path.exists():
            continue
        try:
            payload = json.loads(annotation_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        shapes = payload.get("shapes", [])
        has_polygon = any(shape.get("shape_type", "polygon") == "polygon" and len(shape.get("points", [])) >= 3 for shape in shapes)
        if not has_polygon:
            images.append(image_path)
    return images


def split_ok_images(images: list[Path], audit_fraction: float, seed: int) -> tuple[list[Path], list[Path], dict[str, int]]:
    """按像素级重复组隔离切分，且 Bank 每个重复组只保留一个代表图。"""
    if not 0 < audit_fraction < 1:
        raise ValueError("audit_fraction must be between 0 and 1")
    groups: defaultdict[str, list[Path]] = defaultdict(list)
    for image in sorted(images):
        groups[hashlib.sha256(image.read_bytes()).hexdigest()].append(image)
    grouped = sorted(groups.values(), key=lambda group: group[0].name)
    if len(grouped) < 2:
        raise ValueError("At least two unique OK image groups are required for bank/audit split.")
    rng = np.random.default_rng(seed)
    audit_count = max(1, min(len(grouped) - 1, int(round(len(grouped) * audit_fraction))))
    audit_group_indices = set(rng.permutation(len(grouped))[:audit_count].tolist())
    bank = [group[0] for index, group in enumerate(grouped) if index not in audit_group_indices]
    audit = [path for index, group in enumerate(grouped) if index in audit_group_indices for path in group]
    return bank, audit, {"unique_ok_image_groups": len(grouped), "duplicate_ok_images_excluded_from_bank": len(images) - len(grouped)}
