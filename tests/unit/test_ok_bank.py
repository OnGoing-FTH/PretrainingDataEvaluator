import numpy as np

from src.features.ok_bank import crop_centered_tile


def test_centered_query_tile_preserves_fixed_physical_view_at_edges():
    image = np.full((100, 120, 3), 100, dtype=np.uint8)
    tile, source = crop_centered_tile(image, center_x=5, center_y=4, tile_size=64)

    assert tile.shape == (64, 64, 3)
    assert source == {"x": 0, "y": 0, "width": 37, "height": 36}
