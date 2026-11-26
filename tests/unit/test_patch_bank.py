import numpy as np

from src.features.patch_bank import score_patch_features


def test_patch_score_is_zero_for_matching_normalized_prototype():
    prototype = np.zeros((2, 2, 3), dtype=np.float32)
    prototype[..., 0] = 1.0
    queries = prototype[None, ...].copy()

    scores = score_patch_features(queries, prototype)

    assert scores.shape == (1, 2, 2)
    assert np.allclose(scores, 0.0)
