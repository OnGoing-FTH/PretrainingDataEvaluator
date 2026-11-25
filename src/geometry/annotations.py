"""读取 LabelMe JSON 与 YOLO-Seg TXT 标注。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class PolygonAnnotation:
    """单个实例的原图像素坐标多边形。"""

    image_path: Path
    annotation_path: Path
    label: str
    points: np.ndarray
    instance_id: str


def discover_annotation_files(input_dir: Path) -> list[Path]:
    """优先使用 LabelMe JSON；无 JSON 时回退到 YOLO-Seg TXT。"""
    json_files = sorted(input_dir.glob("*.json"))
    return json_files if json_files else sorted(input_dir.glob("*.txt"))


def load_annotations(annotation_path: Path, image_extensions: Iterable[str] = (".png", ".jpg", ".jpeg", ".bmp", ".webp")) -> list[PolygonAnnotation]:
    """从一个标注文件解析 polygon 实例。"""
    suffix = annotation_path.suffix.lower()
    if suffix == ".json":
        return _load_labelme(annotation_path)
    if suffix == ".txt":
        return _load_yolo_seg(annotation_path, image_extensions)
    raise ValueError(f"Unsupported annotation file: {annotation_path}")


def _load_labelme(annotation_path: Path) -> list[PolygonAnnotation]:
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    image_name = payload.get("imagePath") or f"{annotation_path.stem}.png"
    image_path = annotation_path.parent / image_name
    annotations: list[PolygonAnnotation] = []

    for index, shape in enumerate(payload.get("shapes", [])):
        if shape.get("shape_type", "polygon") != "polygon":
            continue
        points = np.asarray(shape.get("points", []), dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 2:
            continue
        label = str(shape.get("label") or "unlabeled")
        annotations.append(
            PolygonAnnotation(
                image_path=image_path,
                annotation_path=annotation_path,
                label=label,
                points=points,
                instance_id=f"{annotation_path.stem}:{index}",
            )
        )
    return annotations


def _load_yolo_seg(annotation_path: Path, image_extensions: Iterable[str]) -> list[PolygonAnnotation]:
    image_path = _find_image(annotation_path, image_extensions)
    if image_path is None:
        raise FileNotFoundError(f"No matching image for {annotation_path}")

    import cv2

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Unable to read image: {image_path}")
    height, width = image.shape[:2]
    annotations: list[PolygonAnnotation] = []

    for index, line in enumerate(annotation_path.read_text(encoding="utf-8").splitlines()):
        parts = line.split()
        if len(parts) < 7 or (len(parts) - 1) % 2:
            continue
        coords = np.asarray([float(value) for value in parts[1:]], dtype=np.float32).reshape(-1, 2)
        points = coords * np.asarray([width, height], dtype=np.float32)
        annotations.append(
            PolygonAnnotation(
                image_path=image_path,
                annotation_path=annotation_path,
                label=parts[0],
                points=points,
                instance_id=f"{annotation_path.stem}:{index}",
            )
        )
    return annotations


def _find_image(annotation_path: Path, image_extensions: Iterable[str]) -> Path | None:
    for extension in image_extensions:
        candidate = annotation_path.with_suffix(extension)
        if candidate.exists():
            return candidate
    return None
