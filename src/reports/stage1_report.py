"""渲染 Stage 1 分簇、同位置 OK 对照报告。"""
from __future__ import annotations
import html, json
from pathlib import Path
from typing import Any
import cv2

RISK_COLORS={"high":"#b91c1c","warning":"#ca8a04","normal":"#15803d"}

def write_stage1_reports(queries:list[dict[str,Any]],summary:dict[str,Any],bank_vectors:object,bank_metadata:list[dict[str,Any]],report_dir:Path,artifact_dir:Path,top_n:int)->None:
 report_dir.mkdir(parents=True,exist_ok=True); artifact_dir.mkdir(parents=True,exist_ok=True)
 import numpy as np
 np.savez_compressed(artifact_dir/"ok_bank_vectors.npz",vectors=bank_vectors)
 (artifact_dir/"ok_bank_metadata.json").write_text(json.dumps(bank_metadata,ensure_ascii=False,indent=2),encoding="utf-8")
 (artifact_dir/"stage1_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
 (artifact_dir/"instance_queries.json").write_text(json.dumps(queries,ensure_ascii=False,indent=2),encoding="utf-8")
 _overview(summary,report_dir/"overview.svg")
 panels=[_panel(q,report_dir/"samples") for q in queries[:top_n]]
 _html(queries[:top_n],panels,summary,report_dir/"index.html")
 _html(queries[:top_n],panels,summary,report_dir/"nearest_neighbor_gallery.html","Stage 1 - 同簇同位置典型 OK 对照")

def _overview(s:dict[str,Any],path:Path)->None:
 counts=s["query_risk_counts"]; cluster=s["clustering"]; w,h=1320,520; maximum=max(1,max(counts.values()))
 e=[f"<svg xmlns='http://www.w3.org/2000/svg' width='{w}' height='{h}' viewBox='0 0 {w} {h}'>","<rect width='100%' height='100%' fill='#fff'/>",_text(660,56,"Stage 1：OK 分簇、簇内同位置动态标定",30,"middle",700),_text(80,112,f"OK 候选 {s['ok_images']} 张 | 唯一图像组 {s['unique_ok_image_groups']} | 自适应 K = {cluster['selected_k']} | silhouette = {max(cluster['silhouette_scores'].values(),default=0):.3f}",19,"start",500),_text(80,148,"主分数：同簇、同归一化位置的所有 OK-Bank Tile 余弦距离均值；展示图为距离中位数对应的典型 OK Tile。",19,"start",500)]
 for i,(label,key,color) in enumerate((("接近 OK / 疑似虚标候选","normal","#15803d"),("灰区 / 需对齐","warning","#ca8a04"),("显著偏离 OK","high","#b91c1c"))):
  x,y=120+i*370,420; bh=counts[key]/maximum*160; e += [f"<rect x='{x}' y='{y-bh:.1f}' width='250' height='{bh:.1f}' fill='{color}'/>",_text(x+125,452,label,18,"middle",700),_text(x+125,y-bh-14,str(counts[key]),24,"middle",700)]
 e.append("</svg>");path.write_text("\n".join(e),encoding="utf-8")
def _text(x:float,y:float,v:str,size:int,anchor:str,weight:int)->str:return f"<text x='{x:.1f}' y='{y:.1f}' text-anchor='{anchor}' font-family='Arial, Microsoft YaHei, Noto Sans CJK SC, sans-serif' font-size='{size}' font-weight='{weight}' fill='#172033'>{html.escape(v)}</text>"

def _panel(q:dict[str,Any],d:Path)->dict[str,str]:
 d.mkdir(parents=True,exist_ok=True); source=cv2.imread(q['image_path']); ok=cv2.imread(q['median_ok_tile']['image_path'])
 if source is None or ok is None:return {}
 r=q['query_tile']; t=q['median_ok_tile']; a=source[r['y']:r['y']+r['height'],r['x']:r['x']+r['width']]; b=ok[t['y']:t['y']+t['height'],t['x']:t['x']+t['width']]
 stem=q['instance_id'].replace(':','_').replace('/','_'); qp=d/f'{stem}_query.png'; op=d/f'{stem}_median_ok.png';cv2.imwrite(str(qp),a);cv2.imwrite(str(op),b)
 return {'query':str(qp.relative_to(d.parent)),'ok':str(op.relative_to(d.parent))}

def _html(qs:list[dict[str,Any]],panels:list[dict[str,str]],s:dict[str,Any],path:Path,title:str='Stage 1 - OK 分簇与动态阈值标定')->None:
 cards=[]
 for q,p in zip(qs,panels):
  if not p:continue
  cards.append(f"""<article class='card'><header><span class='badge' style='background:{RISK_COLORS[q['stage1_risk']]}'>{q['stage1_risk'].upper()}</span><strong>{html.escape(q['label'])}</strong><span>{html.escape(q['instance_id'])}</span></header><div class='visuals'><figure><figcaption>固定 448×448 缺陷上下文（Query）</figcaption><img src='{html.escape(p['query'])}'></figure><figure><figcaption>同簇同位置：距离中位数的典型 OK Tile</figcaption><img src='{html.escape(p['ok'])}'></figure></div><section class='metrics'><div><small>主分：距离均值</small><b>{q['distance_mean']:.4f}</b></div><div><small>距离中位数 / 标准差</small><b>{q['distance_median']:.4f} / {q['distance_std']:.4f}</b></div><div><small>本簇 low / high</small><b>{q['low_threshold']:.4f} / {q['high_threshold']:.4f}</b></div><div><small>OK 簇 / Bank 支持</small><b>{q['assigned_ok_cluster_id']} / {q['bank_support']}</b></div></section><p class='source'>典型 OK：{html.escape(Path(q['median_ok_tile']['image_path']).name)}；同位置 ({q['normalized_center']['u']:.3f}, {q['normalized_center']['v']:.3f})；Stage 0：{q['stage0_risk']} / IoU {q['stage0_proto_iou']:.3f}</p></article>""")
 path.write_text(f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>{html.escape(title)}</title><style>:root{{font-family:Arial,'Microsoft YaHei','Noto Sans CJK SC',sans-serif;color:#172033}}body{{margin:0;background:#f8fafc}}main{{max-width:1800px;margin:auto;padding:32px}}h1{{font-size:30px}}.note{{font-size:18px;line-height:1.55}}object{{width:100%;min-height:500px;background:#fff;border:1px solid #cbd5e1}}.grid{{display:grid;gap:24px;margin-top:28px}}.card{{padding:20px;background:#fff;border:1px solid #cbd5e1}}header{{display:flex;gap:12px;align-items:center;font-size:24px;margin-bottom:15px}}header span:last-child{{font-size:18px;color:#475569}}.badge{{color:#fff;font-size:13px;padding:5px 8px;border-radius:4px}}.visuals,.metrics{{display:grid;grid-template-columns:1fr 1fr;gap:20px}}figure{{margin:0}}figcaption{{font-size:18px;font-weight:700;margin-bottom:8px}}img{{width:100%;max-height:560px;object-fit:contain;background:#000;display:block}}.metrics{{grid-template-columns:repeat(4,1fr);margin-top:18px}}.metrics div{{border-left:4px solid #e2e8f0;padding-left:10px}}small{{display:block;color:#64748b;font-size:14px}}b{{font-size:21px;display:block;margin-top:4px}}.source{{color:#475569;font-size:16px}}@media(max-width:900px){{main{{padding:16px}}.visuals,.metrics{{grid-template-columns:1fr}}}}</style></head><body><main><h1>{html.escape(title)}</h1><p class='note'>OK 图先按整图 Letterbox-448 DINOv2 CLS 自适应分簇；含缺陷图只跟随最近 OK 簇。每个实例仅与该簇、同归一化位置的所有 OK-Bank 图对照，以距离均值动态分级；右图为距离中位数对应的典型 OK 对照。</p><object data='overview.svg' type='image/svg+xml'></object><h2>按同簇同位置平均偏离度排序</h2><div class='grid'>{''.join(cards)}</div></main></body></html>""",encoding='utf-8')
