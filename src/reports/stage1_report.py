"""渲染 Stage 1 OK 记忆库与动态阈值标定的 HTML/SVG 审查报告。"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import cv2


RISK_COLORS = {"high": "#b91c1c", "warning": "#ca8a04", "normal": "#15803d"}


def write_stage1_reports(queries: list[dict[str, Any]], summary: dict[str, Any], bank_vectors: object, bank_metadata: list[dict[str, Any]], report_dir: Path, artifact_dir: Path, top_n: int) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    import numpy as np

    np.savez_compressed(artifact_dir / "ok_bank_vectors.npz", vectors=bank_vectors)
    (artifact_dir / "ok_bank_metadata.json").write_text(json.dumps(bank_metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    (artifact_dir / "stage1_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (artifact_dir / "instance_queries.json").write_text(json.dumps(queries, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_overview_svg(summary, report_dir / "overview.svg")
    panels = [_write_query_panel(query, report_dir / "samples") for query in queries[:top_n]]
    _write_html(queries[:top_n], panels, summary, report_dir / "index.html")
    _write_html(queries[:top_n], panels, summary, report_dir / "nearest_neighbor_gallery.html", title="Stage 1 - 异常区域与最近 OK 区域对照")


def _write_overview_svg(summary: dict[str, Any], path: Path) -> None:
    thresholds = summary["thresholds"]
    counts = summary["query_risk_counts"]
    width, height = 1320, 520
    bars = [("正常", counts["normal"], "#15803d"), ("预警", counts["warning"], "#ca8a04"), ("高风险", counts["high"], "#b91c1c")]
    maximum = max(1, max(count for _, count, _ in bars))
    elements = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
        "<rect width='100%' height='100%' fill='#ffffff'/>",
        _text(660, 56, "Stage 1：OK 记忆库与动态阈值标定", 30, "middle", 700),
        _text(80, 118, f"OK 图片：{summary['ok_images']}  |  Bank：{summary['ok_bank_images']} 图 / {summary['ok_bank_tiles']} Tile  |  Audit：{summary['ok_audit_images']} 图 / {summary['ok_audit_tiles']} Tile", 20, "start", 500),
        _text(80, 154, f"1-NN 余弦距离阈值：正常上限 P{thresholds['warning_percentile']:.0f} = {thresholds['warning_threshold']:.4f}；显著异常 P{thresholds['high_percentile']:.1f} × {thresholds['high_multiplier']:.2f} = {thresholds['high_threshold']:.4f}", 20, "start", 500),
        _text(80, 210, "缺陷实例 ROI 查询结果", 24, "start", 700),
    ]
    for index, (label, count, color) in enumerate(bars):
        x, y = 120 + index * 370, 420
        bar_height = count / maximum * 160
        elements.extend([
            f"<rect x='{x}' y='{y - bar_height:.1f}' width='210' height='{bar_height:.1f}' fill='{color}'/>",
            _text(x + 105, 452, label, 20, "middle", 700),
            _text(x + 105, y - bar_height - 14, str(count), 24, "middle", 700),
        ])
    elements.append("</svg>")
    path.write_text("\n".join(elements), encoding="utf-8")


def _text(x: float, y: float, value: str, size: int, anchor: str, weight: int) -> str:
    return f"<text x='{x:.1f}' y='{y:.1f}' text-anchor='{anchor}' font-family='Arial, Microsoft YaHei, Noto Sans CJK SC, sans-serif' font-size='{size}' font-weight='{weight}' fill='#172033'>{html.escape(value)}</text>"


def _write_query_panel(query: dict[str, Any], sample_dir: Path) -> dict[str, str]:
    sample_dir.mkdir(parents=True, exist_ok=True)
    source = cv2.imread(query["image_path"])
    nearest = cv2.imread(query["nearest_ok_tile"]["image_path"])
    if source is None or nearest is None:
        return {}
    roi = query["query_tile"]
    source_roi = source[roi["y"]:roi["y"] + roi["height"], roi["x"]:roi["x"] + roi["width"]]
    tile = query["nearest_ok_tile"]
    ok_roi = nearest[tile["y"]:tile["y"] + tile["height"], tile["x"]:tile["x"] + tile["width"]]
    stem = query["instance_id"].replace(":", "_").replace("/", "_")
    query_path = sample_dir / f"{stem}_query.png"
    ok_path = sample_dir / f"{stem}_nearest_ok.png"
    cv2.imwrite(str(query_path), source_roi)
    cv2.imwrite(str(ok_path), ok_roi)
    return {"query": str(query_path.relative_to(sample_dir.parent)), "ok": str(ok_path.relative_to(sample_dir.parent))}


def _write_html(queries: list[dict[str, Any]], panels: list[dict[str, str]], summary: dict[str, Any], path: Path, title: str = "Stage 1 - OK 记忆库与动态阈值标定") -> None:
    threshold = summary["thresholds"]
    cards = []
    for query, panel in zip(queries, panels):
        if not panel:
            continue
        level = query["stage1_risk"]
        cards.append(f"""<article class='card'><header><span class='badge' style='background:{RISK_COLORS[level]}'>{html.escape(level.upper())}</span><strong>{html.escape(query['label'])}</strong><span>{html.escape(query['instance_id'])}</span></header><div class='visuals'><figure><figcaption>固定 448×448 缺陷上下文（Query）</figcaption><img src='{html.escape(panel['query'])}' loading='lazy'></figure><figure><figcaption>最近 OK Tile</figcaption><img src='{html.escape(panel['ok'])}' loading='lazy'></figure></div><section class='metrics'><div><small>1-NN 余弦距离</small><b>{query['nearest_distance']:.4f}</b></div><div><small>KNN 平均距离</small><b>{query['knn_mean_distance']:.4f}</b></div><div><small>动态阈值</small><b>{threshold['warning_threshold']:.4f} / {threshold['high_threshold']:.4f}</b></div><div><small>Stage 0 Proto</small><b>{html.escape(query['stage0_risk'])} / IoU {query['stage0_proto_iou']:.3f}</b></div></section><p class='source'>最近 OK 来源：{html.escape(Path(query['nearest_ok_tile']['image_path']).name)}，区域 ({query['nearest_ok_tile']['x']}, {query['nearest_ok_tile']['y']})</p></article>""")
    path.write_text(f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'><title>{html.escape(title)}</title><style>:root{{font-family:Arial,'Microsoft YaHei','Noto Sans CJK SC',sans-serif;color:#172033}}body{{margin:0;background:#f8fafc}}main{{max-width:1800px;margin:auto;padding:32px}}h1{{font-size:30px;margin:0 0 10px}}.note{{font-size:18px;line-height:1.5;margin:0 0 20px}}object{{width:100%;min-height:500px;border:1px solid #cbd5e1;background:#fff}}.grid{{display:grid;gap:24px;margin-top:28px}}.card{{background:#fff;border:1px solid #cbd5e1;padding:20px}}header{{display:flex;gap:12px;align-items:center;font-size:24px;margin-bottom:15px}}header span:last-child{{font-size:18px;color:#475569}}.badge{{font-size:13px;color:#fff;padding:5px 8px;border-radius:4px}}.visuals{{display:grid;grid-template-columns:1fr 1fr;gap:20px}}figure{{margin:0}}figcaption{{font-size:18px;font-weight:700;margin-bottom:8px}}img{{display:block;width:100%;max-height:560px;object-fit:contain;background:#000}}.metrics{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-top:18px}}.metrics div{{border-left:4px solid #e2e8f0;padding-left:10px}}small{{display:block;color:#64748b;font-size:14px}}b{{display:block;font-size:22px;margin-top:4px}}.source{{font-size:16px;color:#475569}}@media(max-width:900px){{main{{padding:16px}}.visuals,.metrics{{grid-template-columns:1fr}}}}</style></head><body><main><h1>{html.escape(title)}</h1><p class='note'>OK 图定义：同名 LabelMe JSON 中不存在有效 Polygon。Audit 图不进入 OK_Bank；阈值仅由 Audit 对 Bank 的独立 1-NN 余弦距离分布自动标定。</p><object data='overview.svg' type='image/svg+xml'>浏览器不支持 SVG 总览。</object><h2>最偏离 OK 记忆库的标注实例</h2><div class='grid'>{''.join(cards) or '<p>没有可渲染的查询结果。</p>'}</div></main></body></html>""", encoding="utf-8")
