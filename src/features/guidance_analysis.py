"""五大图集的混合 KNN、按重复图分组的折外探针与类别子聚类。"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


def image_groups(paths):
    cache = {}
    for path in paths:
        if path not in cache:
            cache[path] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return np.asarray([cache[path] for path in paths])


def grouped_probe(vectors, targets, groups, folds=5):
    """每张原图及其精确重复文件只进入一个验证折。"""
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=20261125)
    probabilities = np.full(len(targets), np.nan)
    assignments = np.full(len(targets), -1, dtype=int)
    evidence = []
    for fold, (train, valid) in enumerate(splitter.split(vectors, targets, groups)):
        if set(groups[train]) & set(groups[valid]):
            raise ValueError("Image group leakage in probe")
        if len(np.unique(targets[train])) != 2:
            raise ValueError("Probe training fold requires both OK and NG groups")
        model = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, class_weight='balanced', max_iter=2000, random_state=42))
        model.fit(vectors[train], targets[train])
        probabilities[valid] = model.predict_proba(vectors[valid])[:, 1]
        assignments[valid] = fold
        evidence.append({'fold': fold, 'train_groups': sorted(set(groups[train])), 'validation_groups': sorted(set(groups[valid]))})
    if not np.isfinite(probabilities).all():
        raise ValueError('Incomplete out-of-fold predictions')
    return probabilities, assignments, evidence


def mixed_ok_ratio(queries, pool, targets, query_groups, pool_groups, k=15):
    ratios = []
    for vector, group in zip(queries, query_groups):
        scores = pool @ vector
        scores[pool_groups == group] = -np.inf
        eligible = np.flatnonzero(np.isfinite(scores))
        nearest = eligible[np.argsort(scores[eligible])[-k:]]
        ratios.append(float((targets[nearest] == 0).mean()))
    return np.asarray(ratios)


def cluster_classes(records, vectors, ok_vectors, alpha=0.3):
    import umap
    import hdbscan

    output = []
    for label in sorted({item['label'] for item in records}):
        indices = [i for i, item in enumerate(records) if item['label'] == label]
        if len(indices) < 10:
            output.append({'label': label, 'status': 'insufficient_samples', 'sample_count': len(indices)})
            continue
        geometry = np.asarray([[np.log1p(records[i]['geometry']['mask_area_px']), records[i]['geometry']['fill_ratio'], np.log1p(records[i]['geometry']['aspect_ratio']), records[i]['geometry']['convexity']] for i in indices])
        features = np.column_stack([vectors[indices], alpha * StandardScaler().fit_transform(geometry)])
        # UMAP 只用于显示，HDBSCAN 在原组合特征空间中聚类。
        coordinates = umap.UMAP(n_neighbors=min(15, len(indices)-1), min_dist=0.1, random_state=42, n_jobs=1).fit_transform(features)
        clusters = hdbscan.HDBSCAN(min_cluster_size=max(5, len(indices)//15), min_samples=3).fit_predict(features)
        groups = []
        for cluster in sorted(set(clusters)):
            local = np.flatnonzero(clusters == cluster)
            center = features[local].mean(axis=0)
            representatives = local[np.argsort(np.linalg.norm(features[local]-center, axis=1))[:9]]
            normal_distance = 1 - np.max(vectors[np.asarray(indices)[local]] @ ok_vectors.T, axis=1)
            groups.append({'cluster': int(cluster), 'noise': bool(cluster == -1), 'sample_count': len(local), 'mean_fill_ratio': float(geometry[local, 1].mean()), 'mean_area_px': float(np.expm1(geometry[local, 0]).mean()), 'mean_nearest_ok_distance': float(normal_distance.mean()), 'representative_ids': [records[indices[j]]['instance_id'] for j in representatives]})
        closest = min((g for g in groups if not g['noise']), key=lambda g:g['mean_nearest_ok_distance'], default=None)
        output.append({'label': label, 'status': 'ok', 'indices': indices, 'coordinates': coordinates.tolist(), 'clusters': clusters.tolist(), 'groups': groups, 'nearest_ok_cluster': closest['cluster'] if closest else None})
    return output
