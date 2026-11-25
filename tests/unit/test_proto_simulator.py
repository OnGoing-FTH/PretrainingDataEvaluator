import numpy as np

from src.geometry.proto_simulator import simulate_proto


def test_proto_marks_tiny_object_as_vanished():
    mask = np.zeros((832, 832), dtype=np.uint8)
    mask[100:102, 100:102] = 255
    result = simulate_proto(mask)
    assert result.proto_nonzero_pixels < 4
    assert result.vanished


def test_proto_keeps_large_object():
    mask = np.zeros((832, 832), dtype=np.uint8)
    mask[200:400, 300:550] = 255
    result = simulate_proto(mask)
    assert result.proto_nonzero_pixels > 4
    assert result.iou_proto > 0.9
    assert not result.fragmented
