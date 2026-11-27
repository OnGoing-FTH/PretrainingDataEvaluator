"""新版 Stage 1/2 证据驱动的 Guidance 图集 1、3、5。"""

from __future__ import annotations

import html
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.geometry.letterbox import make_letterbox_transform
from src.geometry.polygons import rasterize_polygon
from src.geometry.proto_simulator import simulate_proto


def write_guidance_135(records: list[dict[str, Any]], stage1_queries: list[dict[str, Any]], stage2_metrics: list[dict[str, Any]], report_dir: Path, artifact_dir: Path, top_n: int, *, image_root: Path = Path('.'), stage2_report_dir: Path | None = None) -> dict[str, Any]:
    """只消费已有阶段产物，不重新推理、聚类或搜索 OK Bank。"""
    record_by_id = _index(records, 'Stage 0')
    stage1_by_id = _index(stage1_queries, 'Stage 1')
    stage2_by_id = _index(stage2_metrics, 'Stage 2')
    for stage, index in [('Stage 1', stage1_by_id), ('Stage 2', stage2_by_id)]:
        if set(index) - set(record_by_id):
            raise ValueError(f'{stage} 包含不在 Stage 0 中的实例')
    for identifier, two in stage2_by_id.items():
        one = stage1_by_id.get(identifier)
        record = record_by_id[identifier]
        if two['patch_score_metric'] != 'same_cluster_same_position_mean_patch_cosine_distance':
            raise ValueError(f'{identifier}: 不支持旧版 Stage 2 Patch 度量')
        keys = ('image_path', 'label', 'assigned_ok_cluster_id', 'bank_support', 'normalized_center', 'median_ok_tile')
        if (one is None or any(one[key] != two[key] for key in keys)
                or any(record[key] != two[key] for key in ('image_path', 'label', 'polygon_points'))):
            raise ValueError(f'{identifier}: Stage 0/1/2 实例证据不一致')
    report_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    heats = _published_heats(stage2_report_dir)
    _write_json(artifact_dir / 'evidence_index.json', {
        identifier: {'stage0': record, 'stage1': stage1_by_id.get(identifier), 'stage2': stage2_by_id.get(identifier)}
        for identifier, record in record_by_id.items()
    })

    overlabel = _overlabel_candidates(records, stage1_by_id, stage2_by_id)
    mask_quality = _mask_quality_candidates(records, stage1_by_id, stage2_by_id)
    proto = _proto_candidates(records, stage1_by_id, stage2_by_id)
    payload = {"overlabel": overlabel, "mask_quality": mask_quality, "proto_visual_joint": proto}
    for name, value in payload.items():
        _write_json(artifact_dir / f'guidance_{name}.json', value)

    _write_overlabel(record_by_id, overlabel, report_dir / '01_overlabel.html', top_n, image_root, heats)
    _write_mask_quality(record_by_id, mask_quality, report_dir / '03_mask_quality.html', top_n, image_root, heats)
    _write_proto(record_by_id, proto, report_dir / '05_proto.html', top_n)
    summary = {
        "scope": "Guidance 1/3/5 rebuilt from Stage 0 + clustered Stage 1 + multi-reference Stage 2 artifacts",
        "instances": len(records),
        "overlabel_candidates": len(overlabel),
        "mask_quality_candidates": len(mask_quality),
        "proto_visual_joint_candidates": len(proto),
        "top_n": top_n,
        "without_stage1": len(record_by_id.keys() - stage1_by_id.keys()),
        "without_stage2": len(record_by_id.keys() - stage2_by_id.keys()),
    }
    _write_json(artifact_dir / 'summary_135.json', summary)
    _write_index(summary, report_dir / "index.html")
    return summary


def _write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def _index(items, stage):
    index = {item['instance_id']: item for item in items}
    if len(index) != len(items):
        raise ValueError(f'{stage} 存在重复 instance_id')
    return index


def _published_heats(directory):
    """仅复用当前图集明确引用、且文件名与卡片 ID 一致的图像。"""
    hits = {}
    if directory is None:
        return hits
    for name in ('suspected_overlabel_gallery.html', 'coarse_or_misaligned_mask_gallery.html'):
        page = directory / name
        if not page.is_file():
            continue
        for card in re.findall(r'<article>(.*?)</article>', page.read_text(encoding='utf-8'), re.S):
            label = re.search(r'<small>([^<]+)</small>', card)
            if not label:
                continue
            identifier = html.unescape(label[1])
            stem = identifier.replace(':', '_').replace('/', '_')
            relative = f'samples/{stem}_heat.png'
            if f"src='{_esc(relative)}'" in card and (directory / relative).is_file():
                hits[identifier] = directory / relative
    return hits


def _overlabel_candidates(records: list[dict[str, Any]], stage1: dict[str, dict[str, Any]], stage2: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for record in records:
        one, two = stage1.get(record["instance_id"]), stage2.get(record["instance_id"])
        if not one or not two:
            continue
        if record["risk"]["level"] not in {"pass", "medium"}:
            continue
        if one["stage1_risk"] not in {"normal", "warning"} or one.get("cluster_assignment_ambiguous", False):
            continue
        if "suspected_weak_annotation" not in two["candidate_types"]:
            continue
        output.append({"instance_id": record["instance_id"], "priority": round(two["mask_normal_pixel_ratio"] - two["mask_mean_anomaly"], 8), "reason": "Stage1 同簇整体接近 OK，Stage2 Polygon 内多参考 Patch 异常弱，且 Proto 可学习。", "stage1": one, "stage2": two})
    return sorted(output, key=lambda item: item["priority"], reverse=True)


def _mask_quality_candidates(records: list[dict[str, Any]], stage1: dict[str, dict[str, Any]], stage2: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for record in records:
        one, two = stage1.get(record["instance_id"]), stage2.get(record["instance_id"])
        if not one or not two:
            continue
        kinds = set(two["candidate_types"])
        relevant = kinds & {"suspected_coarse_or_multiple", "suspected_boundary_misalignment"}
        if not relevant:
            continue
        priority = max(two["mask_normal_pixel_ratio"], two["outer_ring_anomaly"] - two["inner_ring_anomaly"], min(two["high_components"] / 3, 1.0))
        output.append({"instance_id": record["instance_id"], "priority": round(priority, 8), "reason": "Stage2 同簇多参考 Patch 证据显示 Polygon 内背景比例高、存在多异常连通域或内外环异常反转。", "stage1": one, "stage2": two})
    return sorted(output, key=lambda item: item["priority"], reverse=True)


def _proto_candidates(records: list[dict[str, Any]], stage1: dict[str, dict[str, Any]], stage2: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for record in records:
        is_proto_risk = bool({'proto_iou_below_threshold', 'proto_insufficient_pixels', 'proto_fragmented'} & set(record['risk']['reasons']))
        if not is_proto_risk:
            continue
        one, two = stage1.get(record["instance_id"]), stage2.get(record["instance_id"])
        visual = "unknown" if not two else "strong" if "strong_visual_anomaly" in two["candidate_types"] else "weak" if "suspected_weak_annotation" in two["candidate_types"] else "mixed"
        output.append({"instance_id": record["instance_id"], "priority": record["risk"]["score"], "visual_evidence": visual, "stage1": one, "stage2": two})
    return sorted(output, key=lambda item: item["priority"], reverse=True)


def _shell(title: str, body: str, path: Path) -> None:
    path.write_text(f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>{_esc(title)}</title><style>body{{margin:0;background:#f1f5f9;color:#172033;font-family:Arial,'Microsoft YaHei','Noto Sans CJK SC',sans-serif}}main{{max-width:1800px;margin:auto;padding:28px}}article{{background:#fff;border:1px solid #cbd5e1;padding:20px;margin:20px 0}}.panels{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px}}figure{{margin:0}}figcaption{{font-weight:700;margin:8px 0;font-size:17px}}img{{width:100%;max-height:540px;object-fit:contain;background:#111}}p{{line-height:1.6;font-size:17px}}small{{color:#475569}}dl{{display:grid;grid-template-columns:repeat(5,1fr);gap:10px}}dt{{font-size:14px;color:#64748b}}dd{{font-size:19px;font-weight:700;margin:3px 0}}a{{color:#075985}}table{{border-collapse:collapse;width:100%;background:#fff}}td,th{{border:1px solid #cbd5e1;padding:10px;text-align:left}}@media(max-width:900px){{.panels,dl{{grid-template-columns:1fr}}}}</style></head><body><main><a href='index.html'>返回 Guidance 首页</a><h1>{_esc(title)}</h1><p>本页只编排 Stage 0、新版 Stage 1 和新版 Stage 2 已有证据；不会重建 OK Bank、重跑聚类或跨簇检索。所有候选均须人工复核。</p>{body}</main></body></html>""", encoding="utf-8")


def _save_image(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f'无法写入图像：{path}')
    return f'samples/{path.name}'


def _panel(caption, url=None, note='图像不可用。'):
    content = (f'<img loading="lazy" src="{_esc(url)}" alt="{_esc(caption)}">'
               if url else f'<p>{_esc(note)}</p>')
    return f'<figure><figcaption>{_esc(caption)}</figcaption>{content}</figure>'


def _instance_panels(record, evidence, sample_dir, image_root, heats):
    metric = evidence['stage2']
    identifier = record['instance_id']
    stem = identifier.replace(':', '_').replace('/', '_')
    image = cv2.imread(str(image_root / record['image_path']))
    source = normal = heat = None
    note = 'Stage 2 当前图集未发布该实例的热图；此处不重新推理。'
    if image is not None:
        bbox = metric['query_tile']
        x, y, w, h = (bbox[k] for k in ('x', 'y', 'width', 'height'))
        query = image[y:y+h, x:x+w].copy()
        if query.size:
            points = np.rint(np.asarray(record['polygon_points']) - [x, y]).astype(np.int32)
            cv2.polylines(query, [points], True, (0, 255, 255), 2)
            source = _save_image(sample_dir / f'{stem}_query.png', query)
            center = metric['normalized_center']
            origin = (round(center['u'] * image.shape[1] - 224), round(center['v'] * image.shape[0] - 224))
            if identifier in heats:
                if origin == (x, y):
                    heat = os.path.relpath(heats[identifier], sample_dir.parent)
                else:
                    note = '该热图涉及左侧或上侧裁切，现有 Stage 2 面板未校正填充偏移，不作对照。'
    median = evidence['stage1']['median_ok_tile']
    ok = cv2.imread(str(image_root / median['image_path']))
    if ok is not None:
        x, y, w, h = (median[k] for k in ('x', 'y', 'width', 'height'))
        crop = ok[y:y+h, x:x+w]
        if crop.size:
            normal = _save_image(sample_dir / f'{stem}_median_ok.png', crop)
    return '<div class="panels">' + ''.join([
        _panel('缺陷 Query Tile + Polygon', source),
        _panel('Stage 2 多参考 Patch 热图 + Polygon', heat, note),
        _panel('Stage 1 中位数距离 OK 参考（仅展示，不单独计分）', normal),
    ]) + '</div>'


def _threshold_note(record, one, two):
    return (f'<p>来源：{_esc(record["image_path"])}；'
            f'Stage 1 簇内阈值：{one["low_threshold"]:.4f} / {one["high_threshold"]:.4f}；'
            f'Stage 2 簇内阈值：{two["patch_low_threshold"]:.4f} / {two["patch_high_threshold"]:.4f}。</p>')


def _write_overlabel(records, items, path, top_n, image_root, heats):
    cards = []
    for item in items[:top_n]:
        record, one, two = records[item['instance_id']], item['stage1'], item['stage2']
        panels = _threshold_note(record, one, two) + _instance_panels(record, item, path.parent / 'samples', image_root, heats)
        cards.append(f"""<article><h2>{_esc(record['label'])} · <small>{_esc(record['instance_id'])}</small></h2><p>{_esc(item['reason'])}</p>{panels}<dl><div><dt>场景簇 / OK 参考数</dt><dd>{two['assigned_ok_cluster_id']} / {two['bank_support']}</dd></div><div><dt>Stage 1 状态 / 距离</dt><dd>{_esc(one['stage1_risk'])} / {one['distance_mean']:.4f}</dd></div><div><dt>Mask 异常均值</dt><dd>{two['mask_mean_anomaly']:.4f}</dd></div><div><dt>Mask 正常比例</dt><dd>{two['mask_normal_pixel_ratio']:.1%}</dd></div><div><dt>Proto</dt><dd>{_esc(record['risk']['level'])} / {record['proto']['iou_proto']:.3f}</dd></div></dl></article>""")
    _shell("图集 1：疑似过标 / 虚标（新版分簇证据）", f"<p>候选规则：Stage 1 为 normal/warning、簇归属不模糊、Stage 2 为弱异常、Stage 0 为 pass/medium。共 {len(items)} 个候选，展示 {len(cards)} 个。</p>{''.join(cards) or '<p>没有符合联合规则的候选。</p>'}", path)


def _write_mask_quality(records, items, path, top_n, image_root, heats):
    cards = []
    for item in items[:top_n]:
        record, two = records[item['instance_id']], item['stage2']
        panels = _threshold_note(record, item['stage1'], two) + _instance_panels(record, item, path.parent / 'samples', image_root, heats)
        cards.append(f"""<article><h2>{_esc(record['label'])} · <small>{_esc(record['instance_id'])}</small></h2><p>{_esc(item['reason'])}</p>{panels}<dl><div><dt>场景簇 / OK 参考数</dt><dd>{two['assigned_ok_cluster_id']} / {two['bank_support']}</dd></div><div><dt>Mask 正常比例</dt><dd>{two['mask_normal_pixel_ratio']:.1%}</dd></div><div><dt>高异常连通域</dt><dd>{two['high_components']}</dd></div><div><dt>内 / 外环异常</dt><dd>{two['inner_ring_anomaly']:.4f} / {two['outer_ring_anomaly']:.4f}</dd></div><div><dt>规则</dt><dd>{_esc(', '.join(two['candidate_types']))}</dd></div></dl></article>""")
    _shell("图集 3：粗标 / 一框包多点 / 边界错位（新版分簇证据）", f"<p>共 {len(items)} 个候选，展示 {len(cards)} 个。热图来自同簇、同位置全部 OK 参考的平均 Patch 距离；右图的中位数 OK 参考只供人工理解。</p>{''.join(cards) or '<p>没有候选。</p>'}", path)


def _write_proto(records, items, path, top_n):
    total = Counter(record["label"] for record in records.values())
    counts = Counter(records[item['instance_id']]['label'] for item in items)
    high = Counter(record['label'] for record in records.values() if record['risk']['level'] in {'high', 'critical'})
    table = ''.join(f"<tr><td>{_esc(label)}</td><td>{counts[label]}/{count} = {counts[label]/count:.1%}</td><td>{high[label]/count:.1%}</td><td>{'超过 15%，建议统一切图/最小尺寸策略' if high[label]/count > .15 else '逐例复核'}</td></tr>" for label, count in sorted(total.items()))
    cards = []
    for item in items[:top_n]:
        record = records[item['instance_id']]
        width, height = (record['source_size'][k] for k in ('width', 'height'))
        mask = rasterize_polygon(np.asarray(record["polygon_points"], dtype=np.float32), width, height)
        transform = make_letterbox_transform(width, height, 832)
        source = transform.apply_to_mask(mask)
        restored = simulate_proto(source).restored_mask
        points = transform.map_points(np.asarray(record["polygon_points"], dtype=np.float32))
        x, y, box_w, box_h = cv2.boundingRect(np.rint(points).astype(np.int32))
        left, top, right, bottom = max(0, x - 18), max(0, y - 18), min(832, x + box_w + 18), min(832, y + box_h + 18)
        sample_dir = path.parent / "samples"; sample_dir.mkdir(parents=True, exist_ok=True)
        stem = record["instance_id"].replace(":", "_").replace("/", "_")
        source_path, restored_path = sample_dir / f"{stem}_proto_source.png", sample_dir / f"{stem}_proto_restored.png"
        _save_image(source_path, (source[top:bottom, left:right] > 0).astype(np.uint8) * 255)
        _save_image(restored_path, restored[top:bottom, left:right] * 255)
        two = item.get("stage2")
        evidence = item["visual_evidence"]
        cards.append(f"""<article><h2>{_esc(record['label'])} · <small>{_esc(record['instance_id'])}</small></h2><div class='panels'><figure><figcaption>832 训练画布原始 Mask</figcaption><img src='samples/{_esc(source_path.name)}'></figure><figure><figcaption>208 Proto 后还原 Mask</figcaption><img src='samples/{_esc(restored_path.name)}'></figure><section><h3>视觉证据联动</h3><p>Stage 2：<strong>{_esc(evidence)}</strong></p><p>Stage 1：{_esc(item['stage1']['stage1_risk']) if item.get('stage1') else 'unknown'}</p><p>建议：视觉强但 Proto 不可学时优先讨论切图、提高有效尺寸或 SOP；视觉弱时不可仅据此删除标注。</p></section></div><dl><div><dt>IoU_proto</dt><dd>{record['proto']['iou_proto']:.3f}</dd></div><div><dt>Proto 像素</dt><dd>{record['proto']['proto_nonzero_pixels']}</dd></div><div><dt>断裂</dt><dd>{record['proto']['fragmented']}</dd></div><div><dt>Stage 2 Mask 均值</dt><dd>{format(two['mask_mean_anomaly'], '.4f') if two else '未评估'}</dd></div><div><dt>Stage 2 规则</dt><dd>{_esc(', '.join(two['candidate_types']) if two else 'n/a')}</dd></div></dl></article>""")
    body = f"<p>Stage 0 物理风险为主，Stage 1/2 只补充视觉证据，不覆盖 Proto 判断。共 {len(items)} 个候选，展示 {len(cards)} 个；类别占比基于完整候选。</p><table><thead><tr><th>类别</th><th>Proto 候选比例</th><th>Stage 0 高/严重风险比例</th><th>策略</th></tr></thead><tbody>{table}</tbody></table>{''.join(cards) or '<p>未发现 Proto 风险实例。</p>'}"
    _shell("图集 5：Proto 风险 × 视觉异常证据", body, path)


def _write_index(summary: dict[str, Any], path: Path) -> None:
    rows = [("01_overlabel.html", "图集 1：疑似过标 / 虚标", summary["overlabel_candidates"]), ("03_mask_quality.html", "图集 3：粗标 / 多点 / 错位", summary["mask_quality_candidates"]), ("05_proto.html", "图集 5：Proto 风险 × 视觉证据", summary["proto_visual_joint_candidates"])]
    body = "<p>本次升级仅更新图集 1、3、5；图集 2（灰区/拆类）与图集 4（漏标）将另行按新版分簇语义重构，旧版结果不再作为新入口。</p><table><thead><tr><th>图集</th><th>候选数</th></tr></thead><tbody>" + "".join(f"<tr><td><a href='{name}'>{title}</a></td><td>{count}</td></tr>" for name, title, count in rows) + "</tbody></table>"
    _shell("新版 Guidance 图集：1 / 3 / 5", body, path)


def _esc(value: object) -> str:
    return html.escape(str(value), quote=True)
