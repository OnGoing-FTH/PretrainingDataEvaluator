#!/usr/bin/env python3
"""执行 Stage 0 几何与 Proto 1/4 数据审查。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.geometry.stage0_runner import run_stage0
from src.reports.stage0_report import write_stage0_reports


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO-Seg Stage 0: 832/208 几何和 Proto 可学习性预检")
    parser.add_argument("--input-dir", type=Path, default=ROOT / "data", help="包含 LabelMe JSON+图片或 YOLO-Seg TXT+图片的目录")
    parser.add_argument("--report-dir", type=Path, default=ROOT / "outputs" / "reports" / "stage0")
    parser.add_argument("--artifact-dir", type=Path, default=ROOT / "outputs" / "artifacts" / "stage0")
    parser.add_argument("--top-n", type=int, default=100, help="导出风险排序前 N 个实例图")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records, summary = run_stage0(args.input_dir, target_size=832, proto_stride=4)
    write_stage0_reports(records, summary, args.report_dir, args.artifact_dir, top_n=args.top_n)
    print(f"Stage 0 complete: {summary['instances']} instances, {summary['high_risk_instances']} high-risk.")
    print(f"HTML report: {args.report_dir / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
