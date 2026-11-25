"""模拟 YOLOv11-Seg 832 输入下的 1/4 Proto 掩膜分辨率。"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class ProtoMetrics:
    proto_size: int
    iou_proto: float
    proto_nonzero_pixels: int
    original_components: int
    restored_components: int
    fragmented: bool
    vanished: bool
    proto_mask: np.ndarray
    restored_mask: np.ndarray


def simulate_proto(mask_832: np.ndarray, proto_size: int = 208) -> ProtoMetrics:
    """以最近邻模拟二值 Mask 经 1/4 Proto 栅格后再上采样的留存情况。"""
    if mask_832.shape != (proto_size * 4, proto_size * 4):
        raise ValueError(f"Expected {(proto_size * 4, proto_size * 4)} mask, got {mask_832.shape}")
    source = (mask_832 > 0).astype(np.uint8)
    proto_mask = cv2.resize(source, (proto_size, proto_size), interpolation=cv2.INTER_AREA)
    proto_binary = (proto_mask >= 0.5).astype(np.uint8)
    restored = cv2.resize(proto_binary, (mask_832.shape[1], mask_832.shape[0]), interpolation=cv2.INTER_NEAREST)
    intersection = int(np.count_nonzero((source > 0) & (restored > 0)))
    union = int(np.count_nonzero((source > 0) | (restored > 0)))
    original_components = _component_count(source)
    restored_components = _component_count(restored)
    nonzero = int(np.count_nonzero(proto_binary))
    vanished = nonzero < 4
    fragmented = original_components == 1 and restored_components > 1
    return ProtoMetrics(
        proto_size=proto_size,
        iou_proto=intersection / union if union else 0.0,
        proto_nonzero_pixels=nonzero,
        original_components=original_components,
        restored_components=restored_components,
        fragmented=fragmented,
        vanished=vanished,
        proto_mask=proto_binary,
        restored_mask=restored,
    )


def _component_count(mask: np.ndarray) -> int:
    count, _, _, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    return max(0, count - 1)
