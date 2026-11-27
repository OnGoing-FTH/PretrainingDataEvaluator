#!/usr/bin/env python3
"""执行 Stage 2：DINOv2 Patch 异常图与双流数据审查。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features.stage2_runner import run_stage2
from src.geometry.stage0_runner import run_stage0
from src.reports.stage2_report import write_stage2_reports


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO-Seg Stage 2: DINOv2 Patch anomaly dual-stream audit")
    parser.add_argument("--input-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "stage2.json")
    parser.add_argument("--report-dir", type=Path, default=ROOT / "outputs" / "reports" / "stage2")
    parser.add_argument("--artifact-dir", type=Path, default=ROOT / "outputs" / "artifacts" / "stage2")
    parser.add_argument("--stage1-artifact-dir", type=Path, default=ROOT / "outputs" / "artifacts" / "stage1", help="新版 Stage 1 的聚类、多参考匹配产物目录")
    parser.add_argument("--top-n", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    records, _ = run_stage0(args.input_dir, target_size=832, proto_stride=4)
    summary, artifacts = run_stage2(args.input_dir, records, config, args.stage1_artifact_dir)
    top_n = args.top_n if args.top_n is not None else config["reports"]["top_n"]
    write_stage2_reports(summary, artifacts, args.report_dir, args.artifact_dir, top_n)
    print(f"Stage 2 complete: {summary['instance_count']} instances, {summary['missing_annotation_candidates']} missing-label candidates, {summary['ok_contamination_candidates']} OK contamination candidates.")
    print(f"HTML report: {args.report_dir / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
