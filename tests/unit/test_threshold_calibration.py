import numpy as np

from src.features.threshold_calibration import calibrate_thresholds, classify_distance


def test_calibration_uses_audit_distribution_and_dynamic_classes():
    thresholds = calibrate_thresholds(np.asarray([0.01, 0.02, 0.03, 0.04, 0.05]), 80.0, 100.0, 1.2)

    assert thresholds["warning_threshold"] == 0.042
    assert thresholds["high_threshold"] == 0.06
    assert classify_distance(0.041, thresholds) == "normal"
    assert classify_distance(0.05, thresholds) == "warning"
    assert classify_distance(0.061, thresholds) == "high"
