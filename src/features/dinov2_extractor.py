"""DINOv2 特征提取，采用一次前向得到 Tile 与 ROI 的全局描述向量。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


@dataclass(frozen=True)
class DinoV2Settings:
    model_name: str = "dinov2_vitb14_reg"
    input_size: int = 448
    batch_size: int = 16
    device: str = "auto"


class DinoV2Extractor:
    """使用官方 torch.hub DINOv2 模型提取 L2 标准化 CLS embedding。"""

    def __init__(self, settings: DinoV2Settings) -> None:
        import torch

        self.settings = settings
        self.torch = torch
        self.device = self._resolve_device(settings.device)
        self.model = torch.hub.load("facebookresearch/dinov2", settings.model_name)
        self.model.eval().to(self.device)
        self._mean = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32, device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32, device=self.device).view(1, 3, 1, 1)

    def _resolve_device(self, requested: str) -> str:
        if requested == "auto":
            return "cuda" if self.torch.cuda.is_available() else "cpu"
        if requested.startswith("cuda") and not self.torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable.")
        return requested

    def embed_bgr_images(self, images: Iterable[np.ndarray]) -> np.ndarray:
        prepared = [self._prepare(image) for image in images]
        if not prepared:
            return np.empty((0, 0), dtype=np.float32)
        vectors: list[np.ndarray] = []
        with self.torch.inference_mode():
            for offset in range(0, len(prepared), self.settings.batch_size):
                batch = self.torch.stack(prepared[offset:offset + self.settings.batch_size]).to(self.device)
                batch = (batch - self._mean) / self._std
                output = self.model.forward_features(batch)
                embedding = output["x_norm_clstoken"].float()
                embedding = self.torch.nn.functional.normalize(embedding, p=2, dim=1)
                vectors.append(embedding.cpu().numpy().astype(np.float32))
        return np.vstack(vectors)

    def _prepare(self, image: np.ndarray) -> "object":
        if image is None or image.size == 0:
            raise ValueError("Cannot embed an empty image.")
        resized = cv2.resize(image, (self.settings.input_size, self.settings.input_size), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        return self.torch.from_numpy(rgb.transpose(2, 0, 1).copy()).float().div(255.0)

    def describe(self) -> dict[str, object]:
        return {
            "extractor": "torch_hub_dinov2",
            "model_name": self.settings.model_name,
            "input_size": self.settings.input_size,
            "batch_size": self.settings.batch_size,
            "device": self.device,
        }


def read_bgr(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Unable to read image: {path}")
    return image
