from src.geometry.letterbox import make_letterbox_transform


def test_letterbox_1600x2200_to_832():
    transform = make_letterbox_transform(2200, 1600)
    assert transform.resized_width == 832
    assert transform.resized_height == 605
    assert transform.pad_left == transform.pad_right == 0
    assert transform.pad_top + transform.pad_bottom == 227
