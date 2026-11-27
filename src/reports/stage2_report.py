"""渲染 Stage 2 Patch 异常图、标注质量与漏标候选审查报告。"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def write_stage2_reports(summary: dict[str, Any], artifacts: dict[str, Any], report_dir: Path, artifact_dir: Path, top_n: int) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "stage1_reference_metadata.json").write_text(json.dumps(artifacts["reference_metadata"], ensure_ascii=False, indent=2), encoding="utf-8")
    (artifact_dir / "stage2_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (artifact_dir / "instance_anomaly_metrics.json").write_text(json.dumps(artifacts["instance_metrics"], ensure_ascii=False, indent=2), encoding="utf-8")
    (artifact_dir / "missing_annotation_candidates.json").write_text(json.dumps(artifacts["missing_candidates"], ensure_ascii=False, indent=2), encoding="utf-8")
    (artifact_dir / "ok_bank_contamination_candidates.json").write_text(json.dumps(artifacts["contamination_candidates"], ensure_ascii=False, indent=2), encoding="utf-8")
    _write_overview_svg(summary, report_dir / "overview.svg")
    weak = sorted((item for item in artifacts["instance_metrics"] if "suspected_weak_annotation" in item["candidate_types"]), key=lambda item: item["mask_mean_anomaly"])[:top_n]
    coarse = sorted((item for item in artifacts["instance_metrics"] if any(kind in item["candidate_types"] for kind in ("suspected_coarse_or_multiple", "suspected_boundary_misalignment"))), key=lambda item: item["mask_normal_pixel_ratio"], reverse=True)[:top_n]
    weak_panels = [_write_instance_panel(item, artifacts["instance_maps"].get(item["instance_id"]), report_dir / "samples") for item in weak]
    coarse_panels = [_write_instance_panel(item, artifacts["instance_maps"].get(item["instance_id"]), report_dir / "samples") for item in coarse]
    missing_panels = [_write_candidate_panel(item, artifacts["ng_maps"].get(item["image_path"]), report_dir / "samples", "missing") for item in artifacts["missing_candidates"][:top_n]]
    contamination_panels = [_write_candidate_panel(item, artifacts["ok_maps"].get(item["image_path"]), report_dir / "samples", "ok") for item in artifacts["contamination_candidates"][:top_n]]
    _write_index(summary, report_dir / "index.html")
    _write_instance_gallery("Stage 2 - 疑似虚标 / 视觉弱异常", weak, weak_panels, summary, report_dir / "suspected_overlabel_gallery.html")
    _write_instance_gallery("Stage 2 - 粗标 / 一框包多点 / 边界错位", coarse, coarse_panels, summary, report_dir / "coarse_or_misaligned_mask_gallery.html")
    _write_candidate_gallery("Stage 2 - 疑似漏标候选", artifacts["missing_candidates"][:top_n], missing_panels, report_dir / "missing_annotation_gallery.html")
    _write_candidate_gallery("Stage 2 - OK Bank 污染候选", artifacts["contamination_candidates"][:top_n], contamination_panels, report_dir / "ok_bank_contamination_gallery.html")


def _write_overview_svg(summary: dict[str, Any], path: Path) -> None:
    counts = summary["instance_candidate_counts"]
    items = [("疑似虚标", counts["suspected_weak_annotation"], "#ca8a04"), ("粗标/错位", counts["suspected_coarse_or_multiple"] + counts["suspected_boundary_misalignment"], "#ea580c"), ("疑似漏标", summary["missing_annotation_candidates"], "#b91c1c"), ("OK 污染", summary["ok_contamination_candidates"], "#7c3aed")]
    maximum = max(1, max(item[1] for item in items))
    elements = [f"<svg xmlns='http://www.w3.org/2000/svg' width='1360' height='560' viewBox='0 0 1360 560'>", "<rect width='100%' height='100%' fill='#fff'/>", _text(680, 58, "Stage 2：DINOv2 Patch 异常图与标注审查", 30, "middle", 700), _text(80, 116, "Patch 分数：复用 Stage 1 同簇、同归一化位置的全部 OK 参考逐位置平均距离；中位数参考仅用于展示。", 19, "start", 500)]
    for index, (label, count, color) in enumerate(items):
        x, baseline = 110 + index * 305, 450
        height = count / maximum * 220
        elements.extend([f"<rect x='{x}' y='{baseline - height:.1f}' width='190' height='{height:.1f}' fill='{color}'/>", _text(x + 95, baseline + 34, label, 20, "middle", 700), _text(x + 95, baseline - height - 14, str(count), 25, "middle", 700)])
    elements.append("</svg>")
    path.write_text("\n".join(elements), encoding="utf-8")


def _text(x: float, y: float, value: str, size: int, anchor: str, weight: int) -> str:
    return f"<text x='{x:.1f}' y='{y:.1f}' text-anchor='{anchor}' font-family='Arial, Microsoft YaHei, Noto Sans CJK SC, sans-serif' font-size='{size}' font-weight='{weight}' fill='#172033'>{html.escape(value)}</text>"


def _write_instance_panel(item: dict[str, Any], anomaly_map: np.ndarray | None, sample_dir: Path) -> dict[str, str]:
    if anomaly_map is None:
        return {}
    image = cv2.imread(item["image_path"])
    if image is None:
        return {}
    tile = item["query_tile"]
    query = image[tile["y"]:tile["y"] + tile["height"], tile["x"]:tile["x"] + tile["width"]]
    if query.size == 0:
        return {}
    heat = _heat_overlay(query, anomaly_map[:query.shape[0], :query.shape[1]])
    origin_x = round(item["normalized_center"]["u"] * image.shape[1] - 224)
    origin_y = round(item["normalized_center"]["v"] * image.shape[0] - 224)
    local_points = np.asarray(item["polygon_points"], dtype=np.float32) - np.asarray([origin_x, origin_y], dtype=np.float32)
    cv2.polylines(heat, [np.rint(local_points).astype(np.int32)], True, (0, 255, 255), 2)
    median = item["median_ok_tile"]
    ok_image = cv2.imread(median["image_path"])
    if ok_image is None:
        return {}
    ok = ok_image[median["y"]:median["y"] + median["height"], median["x"]:median["x"] + median["width"]]
    stem = item["instance_id"].replace(":", "_").replace("/", "_")
    sample_dir.mkdir(parents=True, exist_ok=True)
    query_path, heat_path, ok_path = sample_dir / f"{stem}_query.png", sample_dir / f"{stem}_heat.png", sample_dir / f"{stem}_median_ok.png"
    cv2.imwrite(str(query_path), query)
    cv2.imwrite(str(heat_path), heat)
    cv2.imwrite(str(ok_path), ok)
    return {"query": str(query_path.relative_to(sample_dir.parent)), "heat": str(heat_path.relative_to(sample_dir.parent)), "ok": str(ok_path.relative_to(sample_dir.parent))}


def _write_candidate_panel(item: dict[str, Any], anomaly_map: np.ndarray | None, sample_dir: Path, prefix: str) -> dict[str, str]:
    if anomaly_map is None:
        return {}
    image = cv2.imread(item["image_path"])
    if image is None:
        return {}
    x, y, width, height = item["bbox_xywh"]
    margin = max(30, int(max(width, height) * 0.7))
    left, top, right, bottom = max(0, x - margin), max(0, y - margin), min(image.shape[1], x + width + margin), min(image.shape[0], y + height + margin)
    source = image[top:bottom, left:right]
    local_map = anomaly_map[top:bottom, left:right]
    heat = _heat_overlay(source, local_map)
    cv2.rectangle(heat, (x - left, y - top), (x - left + width, y - top + height), (0, 0, 255), 3)
    stem = f"{prefix}_{Path(item['image_path']).stem}_{x}_{y}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    source_path, heat_path = sample_dir / f"{stem}_source.png", sample_dir / f"{stem}_heat.png"
    cv2.imwrite(str(source_path), source)
    cv2.imwrite(str(heat_path), heat)
    return {"source": str(source_path.relative_to(sample_dir.parent)), "heat": str(heat_path.relative_to(sample_dir.parent))}


def _heat_overlay(image: np.ndarray, anomaly_map: np.ndarray) -> np.ndarray:
    scaled = cv2.normalize(anomaly_map.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heat = cv2.applyColorMap(scaled, cv2.COLORMAP_JET)
    return cv2.addWeighted(image, 0.48, heat, 0.52, 0)


def _write_index(summary: dict[str, Any], path: Path) -> None:
    links = [("疑似虚标 / 视觉弱异常", "suspected_overlabel_gallery.html"), ("粗标 / 一框包多点 / 边界错位", "coarse_or_misaligned_mask_gallery.html"), ("疑似漏标候选", "missing_annotation_gallery.html"), ("OK Bank 污染候选", "ok_bank_contamination_gallery.html")]
    path.write_text(f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><title>Stage 2 - Patch 异常审查</title><style>body{{margin:0;background:#f8fafc;color:#172033;font-family:Arial,'Microsoft YaHei','Noto Sans CJK SC',sans-serif}}main{{max-width:1500px;margin:auto;padding:32px}}h1{{font-size:30px}}p{{font-size:18px;line-height:1.6}}object{{width:100%;min-height:520px;background:#fff;border:1px solid #cbd5e1}}.links{{display:flex;gap:14px;flex-wrap:wrap}}a{{font-size:18px;padding:12px 16px;background:#fff;border:1px solid #94a3b8;border-radius:5px;color:#075985;text-decoration:none}}</style></head><body><main><h1>Stage 2 - DINOv2 Patch 异常审查</h1><p>Patch 阈值仅由与 Bank 隔离的 OK Audit 图自动标定。图集候选均为人工复核线索，不自动改写标注。</p><object data='overview.svg' type='image/svg+xml'>浏览器不支持 SVG。</object><h2>审查图集</h2><div class='links'>{''.join(f"<a href='{href}'>{label}</a>" for label, href in links)}</div></main></body></html>""", encoding="utf-8")


def _write_instance_gallery(title: str, items: list[dict[str, Any]], panels: list[dict[str, str]], summary: dict[str, Any], path: Path) -> None:
    cards = []
    for item, panel in zip(items, panels):
        if not panel:
            continue
        cards.append(f"""<article><h2>{html.escape(item['label'])} <small>{html.escape(item['instance_id'])}</small></h2><div class='visuals'><figure><figcaption>缺陷 Query Tile</figcaption><img src='{panel['query']}'></figure><figure><figcaption>多参考 Patch 异常热图 + Polygon</figcaption><img src='{panel['heat']}'></figure><figure><figcaption>Stage 1 中位数距离 OK 参考</figcaption><img src='{panel['ok']}'></figure></div><dl><div><dt>Mask 均值异常</dt><dd>{item['mask_mean_anomaly']:.4f}</dd></div><div><dt>Mask 正常像素占比</dt><dd>{item['mask_normal_pixel_ratio']:.1%}</dd></div><div><dt>内/外环</dt><dd>{item['inner_ring_anomaly']:.4f} / {item['outer_ring_anomaly']:.4f}</dd></div><div><dt>簇 / 参考数</dt><dd>{item['assigned_ok_cluster_id']} / {item['bank_support']}</dd></div><div><dt>候选规则</dt><dd>{html.escape(', '.join(item['candidate_types']) or 'none')}</dd></div></dl></article>""")
    _write_gallery_shell(title, ''.join(cards) or '<p>未发现候选。</p>', path)


def _write_candidate_gallery(title: str, items: list[dict[str, Any]], panels: list[dict[str, str]], path: Path) -> None:
    cards = []
    for item, panel in zip(items, panels):
        if not panel:
            continue
        cards.append(f"""<article><h2>{html.escape(Path(item['image_path']).name)}</h2><div class='visuals'><figure><figcaption>局部原图</figcaption><img src='{panel['source']}'></figure><figure><figcaption>Patch 异常热图 + 候选框</figcaption><img src='{panel['heat']}'></figure></div><dl><div><dt>Peak</dt><dd>{item['peak_anomaly']:.4f}</dd></div><div><dt>Mean</dt><dd>{item['mean_anomaly']:.4f}</dd></div><div><dt>面积</dt><dd>{item['area_px']} px</dd></div><div><dt>BBox</dt><dd>{item['bbox_xywh']}</dd></div></dl></article>""")
    _write_gallery_shell(title, ''.join(cards) or '<p>未发现候选。</p>', path)


def _write_gallery_shell(title: str, content: str, path: Path) -> None:
    path.write_text(f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>{html.escape(title)}</title><style>body{{margin:0;background:#f8fafc;color:#172033;font-family:Arial,'Microsoft YaHei','Noto Sans CJK SC',sans-serif}}main{{max-width:1800px;margin:auto;padding:32px}}h1{{font-size:30px}}article{{background:#fff;border:1px solid #cbd5e1;padding:20px;margin:22px 0}}h2{{margin:0 0 16px;font-size:23px}}small{{color:#64748b;font-size:17px}}.visuals{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:18px}}figure{{margin:0}}figcaption{{font-size:18px;font-weight:700;margin-bottom:8px}}img{{display:block;width:100%;max-height:600px;object-fit:contain;background:#000}}dl{{display:grid;grid-template-columns:repeat(5,1fr);gap:12px}}dt{{font-size:14px;color:#64748b}}dd{{font-size:20px;font-weight:700;margin:4px 0}}@media(max-width:900px){{main{{padding:16px}}.visuals,dl{{grid-template-columns:1fr}}}}</style></head><body><main><h1>{html.escape(title)}</h1>{content}</main></body></html>""", encoding="utf-8")
