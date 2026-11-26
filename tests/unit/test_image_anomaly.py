import numpy as np

from src.features.image_anomaly import find_missing_annotation_candidates


def test_missing_scan_masks_existing_polygon_area():
    anomaly = np.zeros((40, 40), dtype=np.float32)
    anomaly[5:15, 5:15] = 0.9
    anomaly[25:35, 25:35] = 0.9
    records = [{"image_path": "part.png", "polygon_points": [[5, 5], [14, 5], [14, 14], [5, 14]]}]
    config = {"missing_annotation_scan": {"polygon_dilation_px": 0, "minimum_component_area_px": 20}}

    candidates = find_missing_annotation_candidates({"part.png": anomaly}, records, high_threshold=0.5, config=config)

    assert len(candidates) == 1
    assert candidates[0]["bbox_xywh"] == [25, 25, 10, 10]
