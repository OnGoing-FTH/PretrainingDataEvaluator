import json

import cv2
import numpy as np

from src.geometry.stage0_runner import run_stage0
from src.reports.stage0_report import write_stage0_reports


def test_stage0_generates_html_png_and_json_without_csv(tmp_path):
    image_path = tmp_path / "part.png"
    cv2.imwrite(str(image_path), np.full((1600, 2200, 3), 180, dtype=np.uint8))
    annotation = {
        "imagePath": image_path.name,
        "imageWidth": 2200,
        "imageHeight": 1600,
        "shapes": [
            {"label": "large", "shape_type": "polygon", "points": [[100, 100], [800, 100], [800, 800], [100, 800]]},
            {"label": "tiny", "shape_type": "polygon", "points": [[1200, 900], [1202, 900], [1202, 902], [1200, 902]]},
        ],
    }
    (tmp_path / "part.json").write_text(json.dumps(annotation), encoding="utf-8")
    records, summary = run_stage0(tmp_path)
    report_dir = tmp_path / "reports"
    artifact_dir = tmp_path / "artifacts"
    write_stage0_reports(records, summary, report_dir, artifact_dir, top_n=10)
    assert summary["target_size"] == 832
    assert summary["proto_size"] == 208
    assert len(records) == 2
    assert (report_dir / "index.html").exists()
    assert (report_dir / "overview.svg").exists()
    assert list((report_dir / "samples").glob("*_original.png"))
    assert list((report_dir / "samples").glob("*_proto.png"))
    assert (artifact_dir / "stage0_summary.json").exists()
    assert not list(tmp_path.rglob("*.csv"))
