import json

import cv2
import numpy as np
import pytest

from src.reports.guidance_report import write_guidance_135


def test_empty_guidance_publishes_only_current_pages(tmp_path):
    result = write_guidance_135([], [], [], tmp_path, tmp_path, 100)
    assert result['instances'] == 0
    assert {p.name for p in tmp_path.glob('*.html')} == {
        'index.html', '01_overlabel.html', '03_mask_quality.html', '05_proto.html',
    }
    assert not list(tmp_path.rglob('*.csv'))


def test_joint_rules_and_gallery_limits(tmp_path):
    cv2.imwrite(str(tmp_path / 'source.png'), np.zeros((500, 500, 3), dtype=np.uint8))
    records, queries, metrics = [], [], []
    for i in range(3):
        shared = dict(instance_id=f'fixed:{i}', image_path='source.png', label='样本')
        points = [[240, 240], [260, 240], [250, 260]]
        records.append(dict(shared, polygon_points=points, source_size={'width': 500, 'height': 500},
                            proto={'iou_proto': .3 if i == 2 else .9, 'proto_nonzero_pixels': 4, 'fragmented': False},
                            risk={'level': 'high' if i == 2 else 'pass', 'score': i,
                                  'reasons': ['proto_iou_below_threshold'] if i == 2 else []}))
        tile = {'x': 26, 'y': 26, 'width': 448, 'height': 448}
        reference = dict(shared, assigned_ok_cluster_id=0, normalized_center={'u': .5, 'v': .5}, bank_support=2,
                         median_ok_tile=dict(tile, image_path='source.png'))
        queries.append(dict(reference, stage1_risk='normal', distance_mean=.1, low_threshold=.2,
                            high_threshold=.4, cluster_assignment_ambiguous=i == 1))
        metrics.append(dict(reference, polygon_points=points, query_tile=tile,
                            patch_score_metric='same_cluster_same_position_mean_patch_cosine_distance',
                            patch_low_threshold=.25, patch_high_threshold=.5,
                            candidate_types=['suspected_weak_annotation', 'suspected_coarse_or_multiple', 'suspected_boundary_misalignment'],
                            mask_mean_anomaly=.1, mask_normal_pixel_ratio=.9, high_components=1,
                            outer_ring_anomaly=.3, inner_ring_anomaly=.2))
    source_report = tmp_path / 'stage2'
    (source_report / 'samples').mkdir(parents=True)
    cv2.imwrite(str(source_report / 'samples/fixed_0_heat.png'), np.zeros((448, 448, 3), dtype=np.uint8))
    (source_report / 'suspected_overlabel_gallery.html').write_text(
        "<article><small>fixed:0</small><img src='samples/fixed_0_heat.png'></article>", encoding='utf-8')
    report, artifacts = tmp_path / 'report', tmp_path / 'artifacts'
    summary = write_guidance_135(records, queries, metrics[::-1], report, artifacts, 1,
                                 image_root=tmp_path, stage2_report_dir=source_report)
    assert summary['overlabel_candidates'] == 1
    assert summary['mask_quality_candidates'] == 3  # overlapping rules count once per instance
    assert summary['proto_visual_joint_candidates'] == 1
    assert '../stage2/samples/fixed_0_heat.png' in (report / '01_overlabel.html').read_text()
    assert '共 3 个候选，展示 1 个' in (report / '03_mask_quality.html').read_text()
    assert '100.0%' not in (report / '05_proto.html').read_text()
    evidence = json.loads((artifacts / 'evidence_index.json').read_text())
    assert evidence['fixed:2']['stage0']['risk']['level'] == 'high'
    with pytest.raises(ValueError, match='证据不一致'):
        write_guidance_135(records, queries, [dict(metrics[0], bank_support=99)], report, artifacts, 1)


def test_proto_can_be_reported_without_visual_evidence(tmp_path):
    record = dict(instance_id='proto:1', image_path='not-needed.png', label='样本',
                  polygon_points=[[1, 1], [2, 1], [2, 2]], source_size={'width': 832, 'height': 832},
                  risk={'level': 'high', 'score': 80, 'reasons': ['proto_insufficient_pixels']},
                  proto={'iou_proto': .1, 'proto_nonzero_pixels': 1, 'fragmented': False})
    result = write_guidance_135([record], [], [], tmp_path, tmp_path, 10)
    assert result['without_stage2'] == 1
    assert result['proto_visual_joint_candidates'] == 1
    assert '未评估' in (tmp_path / '05_proto.html').read_text()
