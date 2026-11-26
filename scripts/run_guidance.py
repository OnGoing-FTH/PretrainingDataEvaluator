#!/usr/bin/env python3
"""关联三个阶段产物，补提取实例特征，生成五大指导图集。"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.features.dinov2_extractor import DinoV2Extractor, DinoV2Settings, read_bgr
from src.features.ok_bank import crop_centered_tile
from src.features.guidance_analysis import grouped_probe, mixed_ok_ratio, image_groups, cluster_classes
from src.features.patch_bank import score_patch_features
from src.features.instance_anomaly import _measure_mask
from src.geometry.polygons import rasterize_polygon
from src.reports.guidance_report import write_guidance


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--artifact-root', type=Path, default=ROOT/'outputs/artifacts')
    parser.add_argument('--report-dir', type=Path, default=ROOT/'outputs/reports/guidance')
    parser.add_argument('--top-n', type=int, default=100)
    args = parser.parse_args()
    if not 1 <= args.top_n <= 300:
        parser.error('--top-n must be between 1 and 300')
    root = args.artifact_root
    destination = root/'guidance'
    destination.mkdir(parents=True, exist_ok=True)
    records = load(root/'stage0/instances.json')
    bank = np.load(root/'stage1/ok_bank_vectors.npz')['vectors']
    metadata = load(root/'stage1/ok_bank_metadata.json')
    config = load(ROOT/'configs/stage2.json')
    thresholds = load(root/'stage2/stage2_summary.json')['patch_thresholds']
    prototypes = np.load(root/'stage2/patch_bank_prototypes.npz')['prototypes'].astype(np.float32)
    prototypes /= np.maximum(np.linalg.norm(prototypes, axis=-1, keepdims=True), 1e-12)
    extractor = DinoV2Extractor(DinoV2Settings(**config['feature']))
    vectors, maps, metrics = [], {}, []
    print('Extracting instance CLS/Patch features', flush=True)
    for start in range(0, len(records), config['feature']['batch_size']):
        batch = records[start:start+config['feature']['batch_size']]
        tiles, masks = [], []
        for record in batch:
            points = np.asarray(record['polygon_points'], dtype=np.float32)
            x1,y1,x2,y2 = record['geometry']['bbox_xyxy']
            cx,cy = (x1+x2)/2,(y1+y2)/2
            tile,_ = crop_centered_tile(read_bgr(Path(record['image_path'])),cx,cy,448)
            origin = np.asarray([round(cx-224), round(cy-224)])
            tiles.append(tile)
            masks.append(rasterize_polygon(points-origin,448,448))
        cls,patches = extractor.embed_bgr_images_with_patches(tiles)
        vectors.append(cls)
        scores = score_patch_features(patches, prototypes, config['position_constraint']['patch_radius'])
        import cv2
        for record, score, mask in zip(batch,scores,masks):
            anomaly = cv2.resize(score,(448,448),interpolation=cv2.INTER_LINEAR)
            measurement = _measure_mask(anomaly,mask,thresholds['warning_threshold'],thresholds['high_threshold'],{'ring_width_px':3})
            measurement.update(instance_id=record['instance_id'], label=record['label'])
            measurement['mask_background_ratio'] = measurement['mask_mean_anomaly']/max(measurement['bbox_background_mean_anomaly'],1e-8)
            measurement['background_available'] = bool(record['geometry']['fill_ratio'] < 0.999)
            maps[record['instance_id']] = anomaly.astype(np.float16)
            metrics.append(measurement)
        print(f'Instances {min(start+len(batch),len(records))}/{len(records)}',flush=True)
    vectors = np.vstack(vectors)
    targets = np.r_[np.ones(len(vectors),dtype=int),np.zeros(len(bank),dtype=int)]
    pool = np.vstack([vectors,bank])
    paths = [r['image_path'] for r in records]+[m['image_path'] for m in metadata]
    groups = image_groups(paths)
    print('Running grouped 5-fold probe and mixed KNN',flush=True)
    probabilities,folds,evidence = grouped_probe(pool,targets,groups)
    ratios = mixed_ok_ratio(vectors,pool,targets,groups[:len(vectors)],groups)
    nearest = np.argmax(vectors@bank.T,axis=1)
    for i,item in enumerate(metrics):
        item.update(probe_probability=float(probabilities[i]),fold=int(folds[i]),ok_neighbor_ratio=float(ratios[i]),nearest_ok_tile=metadata[nearest[i]])
        item['overlabel'] = bool(item['background_available'] and item['mask_mean_anomaly'] < thresholds['warning_threshold'] and item['mask_background_ratio'] < 1.15 and ratios[i] > .8)
        item['gray_zone'] = bool(.4 <= probabilities[i] <= .6)
        item['coarse'] = bool(item['mask_normal_pixel_ratio'] > .4 or item['high_components'] > 1)
        item['misaligned'] = bool(item['outer_ring_anomaly'] >= item['inner_ring_anomaly'])
    print('Running class UMAP and HDBSCAN',flush=True)
    clusters = cluster_classes(records,vectors,bank)
    probe_samples = [{'image_path':path,'group':str(group),'target':int(target),'probability':float(prob),'fold':int(fold)} for path,group,target,prob,fold in zip(paths,groups,targets,probabilities,folds)]
    for name,value in [('instance_metrics',metrics),('class_clusters',clusters),('probe_folds',evidence),('probe_samples',probe_samples)]:
        (destination/f'{name}.json').write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    np.savez_compressed(destination/'instance_vectors.npz',vectors=vectors,instance_ids=np.asarray([r['instance_id'] for r in records]))
    missing = load(root/'stage2/missing_annotation_candidates.json')
    contamination = load(root/'stage2/ok_bank_contamination_candidates.json')
    candidate_maps = {}
    candidates = sorted(missing+contamination,key=lambda item:item['peak_anomaly'],reverse=True)
    print('Extracting missing-candidate context heatmaps',flush=True)
    for start in range(0,len(candidates),config['feature']['batch_size']):
        batch = candidates[start:start+config['feature']['batch_size']]
        tiles = []
        for item in batch:
            x,y,w,h = item['bbox_xywh']
            tile,_ = crop_centered_tile(read_bgr(Path(item['image_path'])),x+w/2,y+h/2,448)
            tiles.append(tile)
        _,patches = extractor.embed_bgr_images_with_patches(tiles)
        scores = score_patch_features(patches,prototypes,config['position_constraint']['patch_radius'])
        for item,score in zip(batch,scores):
            candidate_maps[(item['image_path'],tuple(item['bbox_xywh']))] = cv2.resize(score,(448,448))
    summary = write_guidance(records,metrics,maps,clusters,missing,contamination,thresholds,args.report_dir,args.top_n,candidate_maps)
    (destination/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False),flush=True)
    print(args.report_dir/'index.html',flush=True)

if __name__ == '__main__':
    main()
