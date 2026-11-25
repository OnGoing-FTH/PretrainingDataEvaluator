"""与 YOLO 系列一致的等比例缩放与居中 padding 映射。"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class LetterboxTransform:
    source_width: int
    source_height: int
    target_size: int
    scale: float
    resized_width: int
    resized_height: int
    pad_left: int
    pad_top: int
    pad_right: int
    pad_bottom: int

    def map_points(self, points: np.ndarray) -> np.ndarray:
        mapped = points.astype(np.float32).copy()
        mapped *= self.scale
        mapped[:, 0] += self.pad_left
        mapped[:, 1] += self.pad_top
        return mapped

    def apply_to_mask(self, mask: np.ndarray) -> np.ndarray:
        resized = cv2.resize(mask, (self.resized_width, self.resized_height), interpolation=cv2.INTER_NEAREST)
        return cv2.copyMakeBorder(
            resized,
            self.pad_top,
            self.pad_bottom,
            self.pad_left,
            self.pad_right,
            cv2.BORDER_CONSTANT,
            value=0,
        )


def make_letterbox_transform(source_width: int, source_height: int, target_size: int = 832) -> LetterboxTransform:
    """构造 832 x 832 训练画布的等比例缩放参数。"""
    scale = min(target_size / source_width, target_size / source_height)
    resized_width = int(round(source_width * scale))
    resized_height = int(round(source_height * scale))
    pad_width = target_size - resized_width
    pad_height = target_size - resized_height
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    return LetterboxTransform(
        source_width=source_width,
        source_height=source_height,
        target_size=target_size,
        scale=scale,
        resized_width=resized_width,
        resized_height=resized_height,
        pad_left=pad_left,
        pad_top=pad_top,
        pad_right=pad_right,
        pad_bottom=pad_bottom,
    )
