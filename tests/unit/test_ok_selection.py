import json

import cv2
import numpy as np

from src.features.ok_selection import discover_ok_images, split_ok_images


def test_empty_labelme_shapes_are_selected_as_ok_and_split_without_overlap(tmp_path):
    names = ["ok_a", "ok_b", "ok_c", "ng"]
    for index, name in enumerate(names):
        image_path = tmp_path / f"{name}.png"
        cv2.imwrite(str(image_path), np.full((32, 32, 3), 100 + index * 10, dtype=np.uint8))
        shapes = [] if name != "ng" else [{"label": "defect", "shape_type": "polygon", "points": [[1, 1], [5, 1], [5, 5]]}]
        (tmp_path / f"{name}.json").write_text(json.dumps({"imagePath": image_path.name, "shapes": shapes}), encoding="utf-8")

    ok_images = discover_ok_images(tmp_path)
    bank, audit, metadata = split_ok_images(ok_images, audit_fraction=1 / 3, seed=7)

    assert [path.stem for path in ok_images] == ["ok_a", "ok_b", "ok_c"]
    assert len(bank) == 2
    assert len(audit) == 1
    assert not set(bank) & set(audit)
    assert set(bank) | set(audit) == set(ok_images)
    assert metadata["unique_ok_image_groups"] == 3


def test_duplicate_ok_images_are_kept_within_one_split_group(tmp_path):
    for name, value in (("a", 100), ("a_copy", 100), ("b", 120), ("c", 140)):
        image_path = tmp_path / f"{name}.png"
        cv2.imwrite(str(image_path), np.full((32, 32, 3), value, dtype=np.uint8))
        (tmp_path / f"{name}.json").write_text(json.dumps({"imagePath": image_path.name, "shapes": []}), encoding="utf-8")

    bank, audit, metadata = split_ok_images(discover_ok_images(tmp_path), audit_fraction=0.5, seed=1)

    assert not ({"a.png", "a_copy.png"} <= {path.name for path in bank})
    assert ({"a.png", "a_copy.png"} <= {path.name for path in audit}) or not ({"a.png", "a_copy.png"} & {path.name for path in audit})
    assert metadata["unique_ok_image_groups"] == 3
    assert metadata["duplicate_ok_images_excluded_from_bank"] == 1
    assert {path.name for path in bank} | {path.name for path in audit} == {"a.png", "a_copy.png", "b.png", "c.png"}
