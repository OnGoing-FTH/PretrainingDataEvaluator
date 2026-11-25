#!/usr/bin/env python3
"""执行 Stage 1：OK 记忆库构建、Audit 标定与实例 ROI 查询。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features.stage1_runner import run_stage1
from src.geometry.stage0_runner import run_stage0
from src.reports.stage1_report import write_stage1_reports


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO-Seg Stage 1: OK memory bank and dynamic threshold calibration")
    parser.add_argument("--input-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "stage1.json")
    parser.add_argument("--report-dir", type=Path, default=ROOT / "outputs" / "reports" / "stage1")
    parser.add_argument("--artifact-dir", type=Path, default=ROOT / "outputs" / "artifacts" / "stage1")
    parser.add_argument("--top-n", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    stage0_records, _ = run_stage0(args.input_dir, target_size=832, proto_stride=4)
    queries, summary, bank_vectors, bank_metadata = run_stage1(args.input_dir, stage0_records, config)
    top_n = args.top_n if args.top_n is not None else config["reports"]["top_n"]
    write_stage1_reports(queries, summary, bank_vectors, bank_metadata, args.report_dir, args.artifact_dir, top_n)
    print(f"Stage 1 complete: {summary['ok_bank_images']} OK_Bank images, {summary['ok_audit_images']} OK_Audit images, {summary['instance_queries']} ROI queries.")
    print(f"HTML report: {args.report_dir / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
