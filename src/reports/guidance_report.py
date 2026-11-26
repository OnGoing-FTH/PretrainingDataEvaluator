"""五大指导图集：原生 HTML/SVG 文字、PNG 图像面板和人工复核 SOP。"""
from __future__ import annotations
import hashlib
import html
import json
from collections import Counter, defaultdict
from pathlib import Path
import cv2
import numpy as np
from src.features.dinov2_extractor import read_bgr
from src.features.ok_bank import crop_centered_tile
from src.geometry.polygons import rasterize_polygon
from src.geometry.letterbox import make_letterbox_transform
from src.geometry.proto_simulator import simulate_proto


def esc(value):
    return html.escape(str(value), quote=True)


def shell(title,content,path):
    path.write_text(f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{esc(title)}</title><style>
body{{margin:0;background:#f1f5f9;color:#172033;font-family:Arial,'Microsoft YaHei',sans-serif}}main{{max-width:1600px;margin:auto;padding:28px}}article,section{{background:white;border:1px solid #cbd5e1;padding:20px;margin:20px 0}}.panels{{display:flex;gap:16px;flex-wrap:wrap}}figure{{flex:1;min-width:240px;margin:0}}img{{width:100%;max-height:540px;object-fit:contain;background:#111}}figcaption{{font-weight:bold;margin:8px 0}}p{{line-height:1.7}}.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}}a{{color:#075985}}.muted{{color:#64748b}}svg{{max-width:100%;height:auto}}details{{margin:12px 0}}table{{border-collapse:collapse;width:100%}}td,th{{padding:10px;border:1px solid #cbd5e1;text-align:left}}@media(max-width:700px){{.grid{{grid-template-columns:1fr}}}}@keyframes pulse{{50%{{opacity:.35}}}}.peak{{animation:pulse 1.4s infinite}}@media(prefers-reduced-motion:reduce){{.peak{{animation:none}}}}
</style></head><body><main><a href="index.html">返回五大图集</a><h1>{esc(title)}</h1><p class="muted">所有候选均需人工复核。阈值、聚类与探针结果不直接决定标签变更；图像颜色不代表确认缺陷。</p>{content}</main></body></html>''',encoding='utf-8')


def save(image,name,directory):
    path = directory/'samples'/f'{name}.png'
    path.parent.mkdir(parents=True,exist_ok=True)
    if not cv2.imwrite(str(path),image):
        raise OSError(path)
    return 'samples/'+path.name


def figure(path,caption):
    return f'<figure><figcaption>{esc(caption)}</figcaption><img loading="lazy" src="{esc(path)}" alt="{esc(caption)}"></figure>'


def tile_for(record):
    x1,y1,x2,y2 = record['geometry']['bbox_xyxy']
    cx,cy=(x1+x2)/2,(y1+y2)/2
    origin=np.asarray([round(cx-224),round(cy-224)])
    tile,_=crop_centered_tile(read_bgr(Path(record['image_path'])),cx,cy,448)
    points=np.rint(np.asarray(record['polygon_points'])-origin).astype(np.int32)
    return tile,points


def overlay(tile,anomaly,thresholds):
    # 全部面板共享标尺：0 到高风险阈值，避免逐图自动拉伸造成误导。
    scale=np.clip(anomaly.astype(np.float32)/max(thresholds['high_threshold'],1e-8),0,1)
    heat=cv2.applyColorMap((scale*255).astype(np.uint8),cv2.COLORMAP_JET)
    return cv2.addWeighted(tile,.45,heat,.55,0)


def panel_set(record,metric,anomaly,thresholds,directory,mode):
    tile,points=tile_for(record)
    source=tile.copy()
    cv2.polylines(source,[points],True,(0,255,255),2)
    if mode=='over':
        x,y,w,h=cv2.boundingRect(points)
        for offset in range(0,max(w,h),12):
            if offset<w:
                for yy in (y,y+h): cv2.line(source,(x+offset,yy),(min(x+w,x+offset+6),yy),(255,255,255),1)
            if offset<h:
                for xx in (x,x+w): cv2.line(source,(xx,y+offset),(xx,min(y+h,y+offset+6)),(255,255,255),1)
    heat=overlay(tile,anomaly,thresholds)
    cv2.polylines(heat,[points],True,(255,255,255),1)
    key=hashlib.sha256(record['instance_id'].encode()).hexdigest()[:16]+mode
    panels=[figure(save(source,key+'source',directory),'原图＋Polygon（黄色）'),figure(save(heat,key+'heat',directory),'异常图＋Polygon；各图共用分数标尺')]
    if mode=='over':
        meta=metric['nearest_ok_tile']
        normal=read_bgr(Path(meta['image_path']))[meta['y']:meta['y']+meta['height'],meta['x']:meta['x']+meta['width']]
        panels.append(figure(save(normal,key+'normal',directory),'最近真实 OK Tile：'+Path(meta['image_path']).name))
    else:
        mask=rasterize_polygon(points,448,448)>0
        pollution=tile.copy()
        pollution[mask & (anomaly<thresholds['warning_threshold'])]=(0,255,0)
        cv2.polylines(pollution,[points],True,(0,255,255),1)
        panels.append(figure(save(pollution,key+'pollution',directory),'亮绿色：Mask 内低异常背景候选'))
    return '<div class="panels">'+''.join(panels)+'</div>'


def write_guidance(records,metrics,maps,clusters,missing,contamination,thresholds,directory,top_n,candidate_maps):
    directory.mkdir(parents=True,exist_ok=True)
    lookup={r['instance_id']:r for r in records}
    over=sorted([m for m in metrics if m['overlabel']],key=lambda m: (m['ok_neighbor_ratio'], -m['mask_mean_anomaly']),reverse=True)
    coarse=sorted([m for m in metrics if m['coarse'] or m['misaligned']],key=lambda m:max(m['mask_normal_pixel_ratio'],m['outer_ring_anomaly']-m['inner_ring_anomaly']),reverse=True)
    for items,mode,title,filename in [(over,'over','图集 1：疑似过标 / 虚标','01_overlabel.html'),(coarse,'coarse','图集 3：粗标 / 多点 / 错位','03_mask_quality.html')]:
        cards=[]
        for m in items[:top_n]:
            r=lookup[m['instance_id']]
            cards.append(f'<article><h2>{esc(r["label"])} · {esc(r["instance_id"])}</h2><p>来源：{esc(r["image_path"])}；Mask 异常 {m["mask_mean_anomaly"]:.4f}；Mask/背景比 {m["mask_background_ratio"]:.3f}；OK 邻居 {m["ok_neighbor_ratio"]:.1%}；正常背景占比 {m["mask_normal_pixel_ratio"]:.1%}；内/外环 {m["inner_ring_anomaly"]:.4f}/{m["outer_ring_anomaly"]:.4f}；高异常连通域 {m["high_components"]}</p>'+panel_set(r,m,maps[r['instance_id']],thresholds,directory,mode)+'<p>复核建议：核对缺陷定义及实际轮廓；确认正常纹理、多点分布或边界错位后再决定豁免、拆分或重画。</p></article>')
        shell(title,f'<p>共 {len(items)} 个候选，展示前 {min(len(items),top_n)} 个。弱标需同时满足局部弱异常、Mask/背景比 &lt;1.15、混合 KNN 的 OK 占比 &gt;80%。背景不可比较时不触发弱标结论。</p>'+(''.join(cards) or '<p>无满足全部条件的候选。</p>'),directory/filename)
    write_gray(records,metrics,clusters,directory,top_n)
    missing_summary=write_missing(records,missing,contamination,directory,top_n,candidate_maps,thresholds)
    proto_count=write_proto(records,directory,top_n)
    gray=sum(m['gray_zone'] for m in metrics)
    summary={'instances':len(records),'overlabel_candidates':len(over),'gray_zone_instances':gray,'coarse_or_misaligned_instances':len(coarse),'missing_review_regions':missing_summary,'proto_candidates':proto_count,'top_n':top_n,'warning_threshold':thresholds['warning_threshold'],'high_threshold':thresholds['high_threshold']}
    links=[('01_overlabel.html','图集 1：疑似过标 / 虚标',len(over)),('02_gray_and_clusters.html','图集 2：灰区对齐与形态拆类',gray),('03_mask_quality.html','图集 3：粗标 / 多点 / 错位',len(coarse)),('04_missing.html','图集 4：疑似漏标扫雷',missing_summary),('05_proto.html','图集 5：Proto 消失 / 断裂',proto_count)]
    content='<p>输入 832×832，Proto 208×208；DINOv2 视野 448×448。不同图集可包含同一实例，计数不可相加。默认展示 Top 100，完整指标见 JSON 产物。</p><table><tr><th>图集入口</th><th>候选数量</th></tr>'+''.join(f'<tr><td><a href="{name}">{title}</a></td><td>{count}</td></tr>' for name,title,count in links)+'</table><p><a href="sop.html">标注复核与修改 SOP</a></p><p>Patch 低阈值 '+str(thresholds['warning_threshold'])+'；高阈值 '+str(thresholds['high_threshold'])+'。全图候选复用 Stage 2 输出；OK 来源属于独立 Audit，不能代表全体 Bank 已排除污染。</p>'
    shell('五大可视化指导图集',content,directory/'index.html')
    shell('标注复核与修改 SOP','''<ol><li>先检查图集 5：对 Proto 严重消失或断裂的类别讨论切图训练或最小可标尺寸；超过 15% 的类别先统一训练与标注策略。</li><li>图集 1：对照真实 OK 参考核对定义，禁止仅凭低异常分删除真实小缺陷。</li><li>图集 2：专家用概率梯度页统一 OK/NG 边界；子簇只作为形态讨论依据，噪声簇不代表错误标签，也不自动拆类。</li><li>图集 3：亮绿色区域需核对正常背景；多个独立缺陷按实例规则拆分，偏移轮廓应重画。</li><li>图集 4：先排除边缘、照明及结构差异，再在原图位置补标。OK Audit 的重复文件属于同一审查事件。</li><li>争议缺陷若决定豁免，不应仅删除多边形而让可见缺陷作为普通背景继续训练。按项目流程隔离整图，或经人工批准制作可追溯的图像修订。</li><li>记录实例 ID、原文件、结论、审核人及原因；本工具只产生候选与建议，所有修订由人工确认。</li></ol>''',directory/'sop.html')
    return summary


def write_gray(records,metrics,clusters,directory,top_n):
    lookup={r['instance_id']:r for r in records}
    candidates=sorted([m for m in metrics if .35<=m['probe_probability']<=.65],key=lambda m:abs(m['probe_probability']-.5))[:top_n]
    candidates.sort(key=lambda m:m['probe_probability'])
    content='<p>五折逻辑回归按原图 SHA-256 分组；同图实例和重复文件不跨训练/验证折。0.40–0.60 为核心灰区；0.35–0.65 为上下文展示范围。概率仅为模型分数，未作概率校准。</p><div class="grid">'
    for m in candidates:
        r=lookup[m['instance_id']]; tile,points=tile_for(r)
        cv2.polylines(tile,[points],True,(0,255,255),2)
        name=hashlib.sha256(r['instance_id'].encode()).hexdigest()[:16]+'gray'
        content+='<section>'+figure(save(tile,name,directory),f"P(NG)={m['probe_probability']:.3f} · {r['label']} · {r['instance_id']}")+f'<p>{"核心灰区" if m["gray_zone"] else "边界对照"}；验证折 {m["fold"]}</p></section>'
    content+='</div><h2>逐类别形态子簇</h2><p>组合特征为 CLS＋0.3×标准化几何特征。UMAP 用于展示；HDBSCAN 在组合特征空间聚类。近 OK 子簇按成员到正常库的平均最近余弦距离确定，不把 UMAP 图上的距离当作量化依据。</p>'
    for result in clusters:
        content+=f'<article><h2>{esc(result["label"])}</h2>'
        if result['status']!='ok':
            content+=f'<p>样本不足（{result["sample_count"]}），跳过聚类。</p></article>';continue
        xy=np.asarray(result['coordinates']); xy=(xy-xy.min(axis=0))/np.maximum(np.ptp(xy,axis=0),1e-8)
        colors=['#2563eb','#dc2626','#16a34a','#9333ea','#ea580c','#0891b2']
        svg='<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 440" role="img" aria-label="UMAP 类别子簇"><rect width="800" height="440" fill="#fff"/>'
        for point,cluster,index in zip(xy,result['clusters'],result['indices']):
            color='#94a3b8' if cluster==-1 else colors[cluster%len(colors)]
            svg+=f'<circle cx="{40+point[0]*720:.2f}" cy="{30+point[1]*370:.2f}" r="5" fill="{color}"><title>{esc(records[index]["instance_id"])} 簇 {cluster}</title></circle>'
        svg+='<text x="30" y="430" font-size="16">UMAP 展示坐标；灰色为噪声点</text></svg>'
        content+=svg
        for group in result['groups']:
            close=' · 最接近 OK 的有效子簇' if group['cluster']==result['nearest_ok_cluster'] else ''
            content+=f'<h3>{"噪声组" if group["noise"] else "子簇 "+str(group["cluster"])}{close}</h3><p>样本 {group["sample_count"]}；平均填充率 {group["mean_fill_ratio"]:.3f}；平均面积 {group["mean_area_px"]:.1f}px；最近 OK 平均距离 {group["mean_nearest_ok_distance"]:.4f}</p><div class="grid">'
            for identifier in group['representative_ids']:
                r=lookup[identifier]; tile,points=tile_for(r); cv2.polylines(tile,[points],True,(0,255,255),2)
                name=hashlib.sha256(identifier.encode()).hexdigest()[:16]+'cluster'
                content+=figure(save(tile,name,directory),identifier)
            content+='</div>'
        content+='</article>'
    shell('图集 2：灰区对齐与形态拆类',content,directory/'02_gray_and_clusters.html')


def write_missing(records,missing,contamination,directory,top_n,candidate_maps,thresholds):
    by_image=defaultdict(list)
    for record in records: by_image[record['image_path']].append(record)
    seen=set();items=[];hashes={}
    for item in sorted(missing+contamination,key=lambda m:m['peak_anomaly'],reverse=True):
        path=item['image_path']
        if path not in hashes: hashes[path]=hashlib.sha256(Path(path).read_bytes()).hexdigest()
        key=(hashes[path],tuple(item['bbox_xywh']))
        if key in seen: continue
        seen.add(key);items.append(item)
    content='<p>按峰值分数降序；精确重复图像的同位置候选合并。绿色轮廓为已有标注，红框为待审查位置。候选复用 Stage 2，OK 扫描范围为 Audit；边缘候选可能来自照明或上下文不足。</p>'
    for i,item in enumerate(items[:top_n]):
        image=read_bgr(Path(item['image_path']));h,w=image.shape[:2]
        scale=min(900/w,650/h); thumb=cv2.resize(image,(round(w*scale),round(h*scale)))
        for r in by_image.get(item['image_path'],[]): cv2.polylines(thumb,[np.rint(np.asarray(r['polygon_points'])*scale).astype(np.int32)],True,(0,255,0),2)
        x,y,bw,bh=item['bbox_xywh']
        tile,_=crop_centered_tile(image,x+bw/2,y+bh/2,448)
        origin=np.asarray([round(x+bw/2-224),round(y+bh/2-224)])
        heat=overlay(tile,candidate_maps[(item['image_path'],tuple(item['bbox_xywh']))],thresholds)
        for panel in (tile,heat):
            cv2.rectangle(panel,tuple((np.asarray([x,y])-origin).astype(int)),tuple((np.asarray([x+bw,y+bh])-origin).astype(int)),(0,0,255),2)
        source=save(thumb,f'missing_{i}_full',directory)
        local=save(np.hstack([tile,heat]),f'missing_{i}_local',directory)
        svg=f'<svg viewBox="0 0 {thumb.shape[1]} {thumb.shape[0]}" xmlns="http://www.w3.org/2000/svg"><image href="{source}" width="100%" height="100%"/><rect class="peak" x="{x*scale}" y="{y*scale}" width="{bw*scale}" height="{bh*scale}" fill="none" stroke="red" stroke-width="3"/></svg>'
        content+=f'<article><h2>{esc(Path(item["image_path"]).name)}</h2><p>坐标 {item["bbox_xywh"]}；峰值 {item["peak_anomaly"]:.4f}；面积 {item["area_px"]}px。来源：{esc(item["image_path"])}</p><div class="panels"><figure><figcaption>全图定位</figcaption>{svg}</figure>{figure(local,"448×448 高清局部；红框为候选")}</div><p>复核建议：在原图确认真实缺陷后补标；边缘亮带、产品结构和照明变化应先排除。</p></article>'
    shell('图集 4：疑似漏标扫雷',content,directory/'04_missing.html')
    return len(items)


def write_proto(records,directory,top_n):
    flagged=[r for r in records if r['proto']['iou_proto']<.35 or r['proto']['proto_nonzero_pixels']<4 or r['proto']['fragmented']]
    flagged.sort(key=lambda r:r['risk']['score'],reverse=True)
    total=Counter(r['label'] for r in records);count=Counter(r['label'] for r in flagged)
    content='<p>固定训练输入 832×832，Proto 208×208。展示原始标注在训练画布上的局部视图，以及 208 网格后还原的相同区域。</p><table><tr><th>类别</th><th>风险占比</th><th>策略提示</th></tr>'
    for label,n in total.items():
        ratio=count[label]/n
        content+=f'<tr><td>{esc(label)}</td><td>{count[label]}/{n} = {ratio:.1%}</td><td>{"超过 15%，需统一切图/最小尺寸策略" if ratio>.15 else "逐例复核"}</td></tr>'
    content+='</table>'
    for r in flagged[:top_n]:
        image=read_bgr(Path(r['image_path']));h,w=image.shape[:2]
        mask=rasterize_polygon(np.asarray(r['polygon_points'],np.float32),w,h)
        transform=make_letterbox_transform(w,h,832)
        canvas=transform.apply_to_mask(mask)
        proto=simulate_proto(canvas)
        points=transform.map_points(np.asarray(r['polygon_points'],np.float32));x,y,bw,bh=cv2.boundingRect(np.rint(points).astype(np.int32))
        left,top,right,bottom=max(0,x-15),max(0,y-15),min(832,x+bw+15),min(832,y+bh+15)
        key=hashlib.sha256(r['instance_id'].encode()).hexdigest()[:16]
        original=save(((canvas[top:bottom,left:right]>0)*255).astype(np.uint8),key+'proto_source',directory)
        restored=save(proto.restored_mask[top:bottom,left:right]*255,key+'proto_restored',directory)
        content+=f'<article><h2>{esc(r["label"])} · {esc(r["instance_id"])}</h2><p>IoU {r["proto"]["iou_proto"]:.3f}；Proto 像素 {r["proto"]["proto_nonzero_pixels"]}；断裂 {r["proto"]["fragmented"]}</p><div class="panels">{figure(original,"原始 Mask 在 832 训练画布的局部")}{figure(restored,"208 Proto 后还原的相同局部")}<section><h3>人工复核建议</h3><p>{esc("；".join(r["risk"]["recommendations"]))}</p><p>先评估保留原始分辨率的切图方案与最小有效标注尺寸；外扩宽度应由业务 SOP 确认，不自动改变 Polygon。</p></section></div></article>'
    shell('图集 5：Proto 消失 / 断裂预警',content,directory/'05_proto.html')
    return len(flagged)
