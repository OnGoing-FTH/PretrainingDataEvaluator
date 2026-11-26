import numpy as np

from src.features.instance_anomaly import _measure_mask


def test_mask_metrics_detect_normal_background_and_outer_boundary_anomaly():
    anomaly = np.full((20, 20), 0.02, dtype=np.float32)
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[7:13, 7:13] = 255
    anomaly[5:15, 5:15] = 0.2
    anomaly[7:13, 7:13] = 0.03

    metrics = _measure_mask(anomaly, mask, warning=0.05, high=0.15, rules={"ring_width_px": 2})

    assert metrics["mask_normal_pixel_ratio"] == 1.0
    assert metrics["outer_ring_anomaly"] > metrics["inner_ring_anomaly"]
