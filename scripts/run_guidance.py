#!/usr/bin/env python3
"""使用 Stage 0–2 JSON 生成新版 Guidance 1、3、5。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.reports.guidance_report import write_guidance_135


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact-root', type=Path, default=ROOT / 'outputs/artifacts')
    parser.add_argument('--report-dir', type=Path, default=ROOT / 'outputs/reports/guidance')
    parser.add_argument('--stage2-report-dir', type=Path, default=ROOT / 'outputs/reports/stage2')
    parser.add_argument('--top-n', type=int, default=100)
    args = parser.parse_args()
    if not 1 <= args.top_n <= 300:
        parser.error('--top-n must be between 1 and 300')

    def load(relative):
        return json.loads((args.artifact_root / relative).read_text(encoding='utf-8'))

    try:
        summary = write_guidance_135(
            load('stage0/instances.json'), load('stage1/instance_queries.json'),
            load('stage2/instance_anomaly_metrics.json'), args.report_dir,
            args.artifact_root / 'guidance', args.top_n,
            image_root=ROOT, stage2_report_dir=args.stage2_report_dir,
        )
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f'Guidance: {exc}\n')
    print(json.dumps(summary, ensure_ascii=False))
    print(args.report_dir / 'index.html')


if __name__ == '__main__':
    main()
