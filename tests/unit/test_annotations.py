import json

import cv2
import numpy as np

from src.geometry.annotations import load_annotations


def test_load_labelme_polygon(tmp_path):
    image_path = tmp_path / "sample.png"
    cv2.imwrite(str(image_path), np.zeros((20, 30, 3), dtype=np.uint8))
    annotation_path = tmp_path / "sample.json"
    annotation_path.write_text(json.dumps({"imagePath": "sample.png", "shapes": [{"label": "scratch", "shape_type": "polygon", "points": [[1, 2], [10, 2], [5, 8]]}]}), encoding="utf-8")
    annotations = load_annotations(annotation_path)
    assert len(annotations) == 1
    assert annotations[0].label == "scratch"
    assert annotations[0].points.shape == (3, 2)
