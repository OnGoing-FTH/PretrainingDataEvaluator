"""渲染 Stage 0 的 HTML/SVG 审查报告与无文字图像面板。"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.geometry.letterbox import make_letterbox_transform
from src.geometry.polygons import rasterize_polygon
from src.geometry.proto_simulator import simulate_proto


RISK_COLORS = {"critical": "#b91c1c", "high": "#ea580c", "medium": "#ca8a04", "pass": "#15803d"}


def write_stage0_reports(records: list[dict[str, Any]], summary: dict[str, Any], report_dir: Path, artifact_dir: Path, top_n: int = 100) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "instances.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    (artifact_dir / "stage0_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    flagged = [record for record in records if record["risk"]["level"] != "pass"]
    (artifact_dir / "flagged_instances.json").write_text(json.dumps(flagged, ensure_ascii=False, indent=2), encoding="utf-8")

    _write_overview_svg(summary, report_dir / "overview.svg")
    panels = [_write_image_panels(record, report_dir / "samples") for record in records[:top_n]]
    contact_sheets = _write_contact_sheets(records[:top_n], panels, report_dir, per_sheet=12)
    _write_html(summary, records[:top_n], panels, contact_sheets, report_dir / "index.html")
    _write_html(summary, records[:top_n], panels, contact_sheets, report_dir / "proto_warning_gallery.html", title="Stage 0 - Proto 1/4 消失 / 断裂预警")


def _write_overview_svg(summary: dict[str, Any], path: Path) -> None:
    """生成文本保持矢量的 SVG 总览，不将文字烘焙进 PNG。"""
    classes = summary["class_summary"]
    width, height = 1760, 920
    left_x, chart_top, chart_height, chart_width = 100, 160, 600, 700
    right_x, right_width = 960, 690
    max_total = max((item["instances"] for item in classes), default=1)
    bar_width = 64
    gap = 20
    elements = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
        "<rect width='100%' height='100%' fill='#ffffff'/>",
        _svg_text(width / 2, 64, f"Stage 0 总览 | 832 x 832 输入，208 x 208 Proto | {summary['instances']} 个实例", 30, "middle", 700),
        _svg_text(left_x + chart_width / 2, 118, "各类别 Stage 0 风险分布", 25, "middle", 700),
        _svg_text(right_x + right_width / 2, 118, "高风险实例占比", 25, "middle", 700),
    ]
    for tick in range(0, max_total + 1, max(1, int(np.ceil(max_total / 5)))):
        y = chart_top + chart_height - (tick / max_total) * chart_height
        elements.extend([
            f"<line x1='{left_x}' y1='{y:.1f}' x2='{left_x + chart_width}' y2='{y:.1f}' stroke='#dbe3ee' stroke-width='1'/>",
            _svg_text(left_x - 15, y + 6, str(tick), 16, "end", 500),
        ])
    elements.append(_svg_text(left_x - 56, chart_top + chart_height / 2, "实例数", 19, "middle", 700, rotate=-90))
    for index, item in enumerate(classes):
        x = left_x + 28 + index * (bar_width + gap)
        y = chart_top + chart_height
        for key, label in (("pass", "pass"), ("medium", "medium"), ("high", "high"), ("critical", "critical")):
            value = item[key]
            bar_height = (value / max_total) * chart_height
            y -= bar_height
            if bar_height:
                elements.append(f"<rect x='{x}' y='{y:.1f}' width='{bar_width}' height='{bar_height:.1f}' fill='{RISK_COLORS[key]}'/>")
        label_y = chart_top + chart_height + 35
        elements.append(_svg_text(x + bar_width / 2, label_y, item["label"], 16, "end", 700, rotate=-30))
    legend_x, legend_y = left_x + chart_width - 160, chart_top + 20
    for index, (key, label) in enumerate((("pass", "通过"), ("medium", "中风险"), ("high", "高风险"), ("critical", "严重"))):
        y = legend_y + index * 32
        elements.append(f"<rect x='{legend_x}' y='{y - 16}' width='18' height='18' fill='{RISK_COLORS[key]}'/>")
        elements.append(_svg_text(legend_x + 28, y, label, 16, "start", 700))

    ratio_axis_y = chart_top + chart_height
    elements.append(f"<line x1='{right_x}' y1='{ratio_axis_y}' x2='{right_x + right_width}' y2='{ratio_axis_y}' stroke='#111827' stroke-width='2'/>")
    for tick in range(0, 26, 5):
        x = right_x + (tick / 25) * right_width
        elements.extend([
            f"<line x1='{x:.1f}' y1='{ratio_axis_y}' x2='{x:.1f}' y2='{ratio_axis_y + 8}' stroke='#111827' stroke-width='2'/>",
            _svg_text(x, ratio_axis_y + 34, str(tick), 16, "middle", 500),
        ])
    threshold_x = right_x + (15 / 25) * right_width
    elements.extend([
        f"<line x1='{threshold_x:.1f}' y1='{chart_top}' x2='{threshold_x:.1f}' y2='{ratio_axis_y}' stroke='#111827' stroke-width='2' stroke-dasharray='8 6'/>",
        _svg_text(threshold_x + 10, chart_top + 22, "策略预警线 15%", 16, "start", 700),
        _svg_text(right_x + right_width / 2, ratio_axis_y + 72, "占比 (%)", 19, "middle", 700),
    ])
    row_gap = chart_height / max(1, len(classes))
    for index, item in enumerate(classes):
        y = chart_top + 30 + index * row_gap
        ratio = item["high_risk_ratio"] * 100
        color = RISK_COLORS["critical"] if item["strategy_warning"] else RISK_COLORS["high"] if ratio else RISK_COLORS["pass"]
        elements.append(_svg_text(right_x - 15, y + 7, item["label"], 18, "end", 700))
        elements.append(f"<rect x='{right_x}' y='{y - 16:.1f}' width='{(ratio / 25) * right_width:.1f}' height='32' fill='{color}'/>")
        if ratio:
            elements.append(_svg_text(right_x + (ratio / 25) * right_width + 10, y + 7, f"{ratio:.1f}%", 16, "start", 700))
    elements.append("</svg>")
    path.write_text("\n".join(elements), encoding="utf-8")


def _svg_text(x: float, y: float, text: str, size: int, anchor: str, weight: int, rotate: int | None = None) -> str:
    transform = f" transform='rotate({rotate} {x} {y})'" if rotate is not None else ""
    return f"<text x='{x:.1f}' y='{y:.1f}' text-anchor='{anchor}' font-family='Arial, Microsoft YaHei, Noto Sans CJK SC, sans-serif' font-size='{size}' font-weight='{weight}' fill='#111827'{transform}>{html.escape(text)}</text>"


def _write_image_panels(record: dict[str, Any], sample_dir: Path) -> dict[str, str]:
    """只渲染图像内容；标题、指标和说明全部由 HTML 原生文字承担。"""
    sample_dir.mkdir(parents=True, exist_ok=True)
    image = cv2.imread(record["image_path"])
    if image is None:
        return {}
    height, width = image.shape[:2]
    points = np.asarray(record["polygon_points"], dtype=np.float32)
    mask = rasterize_polygon(points, width, height)
    x1, y1, x2, y2 = record["geometry"]["bbox_xyxy"]
    margin = max(30, int(max(x2 - x1, y2 - y1) * 0.35))
    left, top = max(0, x1 - margin), max(0, y1 - margin)
    right, bottom = min(width, x2 + margin + 1), min(height, y2 + margin + 1)
    annotated = image.copy()
    cv2.polylines(annotated, [np.rint(points).astype(np.int32)], True, (0, 255, 255), 3)
    cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 255, 0), 3)
    transform = make_letterbox_transform(width, height, record["letterbox"]["target_size"])
    source_832 = transform.apply_to_mask(mask)
    simulated = simulate_proto(source_832, record["proto"]["proto_size"])
    comparison = np.zeros((832, 832, 3), dtype=np.uint8)
    comparison[source_832 > 0] = (70, 210, 70)
    comparison[simulated.restored_mask > 0] = (30, 30, 230)
    stem = record["instance_id"].replace(":", "_").replace("/", "_")
    original_output = sample_dir / f"{stem}_original.png"
    proto_output = sample_dir / f"{stem}_proto.png"
    cv2.imwrite(str(original_output), annotated[top:bottom, left:right])
    cv2.imwrite(str(proto_output), comparison)
    return {"original": str(original_output.relative_to(sample_dir.parent)), "proto": str(proto_output.relative_to(sample_dir.parent))}


def _write_contact_sheets(records: list[dict[str, Any]], panels: list[dict[str, str]], report_dir: Path, per_sheet: int) -> list[str]:
    """生成文字清晰的 HTML 联系表，PNG 仅保留无文字视觉预览。"""
    for old_preview in report_dir.glob("proto_warning_contact_sheet_*.png"):
        old_preview.unlink()
    pages: list[str] = []
    for index in range(0, len(records), per_sheet):
        entries = list(zip(records[index:index + per_sheet], panels[index:index + per_sheet]))
        if not entries:
            continue
        page_number = index // per_sheet + 1
        cards = []
        for record, panel in entries:
            if not panel:
                continue
            risk = record["risk"]
            cards.append(f"""<article class='tile'><img src='{html.escape(panel['proto'])}' loading='lazy'><div class='tile-meta'><span class='tile-badge' style='background:{RISK_COLORS[risk['level']]}'>{html.escape(risk['level'].upper())}</span><strong>{html.escape(record['label'])}</strong><span>{html.escape(record['instance_id'])}</span><span>IoU {record['proto']['iou_proto']:.3f}</span><span>Proto {record['proto']['proto_nonzero_pixels']} px</span></div></article>""")
        name = f"proto_warning_contact_sheet_{page_number:02d}.html"
        (report_dir / name).write_text(f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'><title>Proto 风险联系表 {page_number}</title><style>:root{{font-family:Arial,'Microsoft YaHei','Noto Sans CJK SC',sans-serif;color:#172033}}body{{margin:0;background:#f8fafc}}main{{max-width:1800px;margin:auto;padding:28px}}h1{{font-size:28px;margin:0 0 20px}}.grid{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:18px}}.tile{{background:#fff;border:1px solid #cbd5e1;padding:12px}}.tile img{{display:block;width:100%;aspect-ratio:1;object-fit:contain;background:#000;image-rendering:pixelated}}.tile-meta{{display:grid;grid-template-columns:auto 1fr;gap:7px 10px;align-items:center;font-size:17px;padding-top:10px}}.tile-meta strong{{font-size:20px}}.tile-meta span:not(.tile-badge){{color:#475569}}.tile-badge{{justify-self:start;color:#fff;padding:4px 7px;border-radius:4px;font-size:12px;font-weight:700}}@media(max-width:900px){{main{{padding:16px}}.grid{{grid-template-columns:1fr}}}}</style></head><body><main><h1>Proto 1/4 风险联系表 {page_number} / {(len(records) + per_sheet - 1) // per_sheet}</h1><div class='grid'>{''.join(cards)}</div></main></body></html>""", encoding="utf-8")
        pages.append(name)
    return pages


def _write_html(summary: dict[str, Any], records: list[dict[str, Any]], panels: list[dict[str, str]], contact_sheets: list[str], path: Path, title: str = "Stage 0 - 几何与 Proto 预检") -> None:
    cards = []
    for record, panel in zip(records, panels):
        if not panel:
            continue
        risk = record["risk"]
        color = RISK_COLORS[risk["level"]]
        reason_list = "".join(f"<li>{html.escape(reason)}</li>" for reason in risk["reasons"] or ["未触发风险规则"])
        recommendation_list = "".join(f"<li>{html.escape(item)}</li>" for item in risk["recommendations"])
        cards.append(f"""
        <article class='card'>
          <header><span class='badge' style='background:{color}'>{html.escape(risk['level'].upper())}</span><strong>{html.escape(record['label'])}</strong><span class='instance'>{html.escape(record['instance_id'])}</span></header>
          <div class='visuals'>
            <figure><figcaption>原图局部 + Polygon / 紧致 BBox</figcaption><img src='{html.escape(panel['original'])}' loading='lazy'></figure>
            <figure><figcaption>832 Mask：绿=原始，红=Proto 还原</figcaption><img src='{html.escape(panel['proto'])}' loading='lazy'></figure>
          </div>
          <section class='details'>
            <dl><div><dt>风险分</dt><dd>{risk['score']}</dd></div><div><dt>IoU_proto</dt><dd>{record['proto']['iou_proto']:.3f}</dd></div><div><dt>Proto 像素</dt><dd>{record['proto']['proto_nonzero_pixels']}</dd></div><div><dt>连通域</dt><dd>{record['proto']['original_components']} → {record['proto']['restored_components']}</dd></div></dl>
            <div><h3>原因</h3><ul>{reason_list}</ul></div><div><h3>建议</h3><ul>{recommendation_list}</ul></div>
          </section>
        </article>""")
    table_rows = "".join(f"<tr><td>{html.escape(item['label'])}</td><td>{item['instances']}</td><td>{item['high_risk_ratio'] * 100:.1f}%</td><td>{'是' if item['strategy_warning'] else '否'}</td></tr>" for item in summary["class_summary"])
    contact_links = "".join(f"<a class='sheet-link' href='{html.escape(name)}'>联系表 {index + 1}</a>" for index, name in enumerate(contact_sheets))
    path.write_text(f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'><title>{html.escape(title)}</title><style>
    :root{{font-family:Arial,'Microsoft YaHei','Noto Sans CJK SC',sans-serif;color:#172033;background:#f8fafc}}body{{margin:0;background:#f8fafc}}main{{max-width:1800px;margin:auto;padding:32px}}h1{{font-size:30px;margin:0 0 20px}}h2{{font-size:25px;margin:34px 0 14px}}h3{{font-size:18px;margin:0 0 8px}}.summary,.sheet-links{{display:flex;gap:12px;flex-wrap:wrap}}.metric{{font-size:18px;background:#fff;border:1px solid #dbe3ee;padding:16px 20px;border-radius:6px}}.metric strong{{font-size:24px;margin-left:8px}}.sheet-link{{font-size:18px;color:#075985;background:#e0f2fe;border:1px solid #7dd3fc;padding:10px 14px;border-radius:5px;text-decoration:none}}object.overview{{width:100%;height:auto;min-height:700px;background:#fff;border:1px solid #dbe3ee;margin-top:24px}}table{{border-collapse:collapse;background:#fff;font-size:18px}}td,th{{border:1px solid #dbe3ee;padding:11px 16px;text-align:left}}.grid{{display:grid;gap:24px}}.card{{background:#fff;border:1px solid #cbd5e1;padding:20px}}.card header{{display:flex;align-items:center;gap:14px;font-size:24px;margin-bottom:16px}}.instance{{font-size:18px;color:#475569}}.badge{{color:#fff;border-radius:4px;padding:5px 9px;font-size:14px;letter-spacing:.4px}}.visuals{{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:18px}}figure{{margin:0}}figcaption{{font-size:19px;font-weight:700;margin:0 0 8px}}figure img{{width:100%;max-height:600px;object-fit:contain;background:#000;display:block}}.details{{display:grid;grid-template-columns:1.4fr 1fr 1.6fr;gap:24px;margin-top:20px;font-size:18px}}dl{{margin:0;display:grid;grid-template-columns:1fr 1fr;gap:12px}}dl div{{border-left:4px solid #e2e8f0;padding-left:10px}}dt{{color:#64748b;font-size:15px}}dd{{font-size:25px;font-weight:700;margin:3px 0 0}}ul{{margin:0;padding-left:22px;line-height:1.55}}@media(max-width:900px){{main{{padding:16px}}.visuals,.details{{grid-template-columns:1fr}}.card header{{font-size:20px}}object.overview{{min-height:540px}}}}
    </style></head><body><main><h1>{html.escape(title)}</h1><div class='summary'><div class='metric'>训练尺寸 <strong>832 x 832</strong></div><div class='metric'>Proto 网格 <strong>208 x 208</strong></div><div class='metric'>实例 <strong>{summary['instances']}</strong></div><div class='metric'>高风险 <strong>{summary['high_risk_instances']}</strong></div></div><object class='overview' data='overview.svg' type='image/svg+xml'>浏览器不支持 SVG 总览。</object><h2>Proto 风险联系表</h2><div class='sheet-links'>{contact_links}</div><h2>类别策略预警</h2><table><thead><tr><th>类别</th><th>实例</th><th>高风险占比</th><th>超过 15% 预警线</th></tr></thead><tbody>{table_rows}</tbody></table><h2>风险样本（按严重度排序）</h2><div class='grid'>{''.join(cards) or '<p>未发现有标注实例。</p>'}</div></main></body></html>""", encoding="utf-8")
