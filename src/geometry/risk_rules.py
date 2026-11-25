"""Stage 0 的几何与 Proto 风险判定。"""

from __future__ import annotations

from dataclasses import dataclass

from .polygons import GeometryMetrics
from .proto_simulator import ProtoMetrics


@dataclass(frozen=True)
class RiskAssessment:
    level: str
    score: float
    reasons: list[str]
    recommendations: list[str]


def assess_risk(
    geometry: GeometryMetrics,
    proto: ProtoMetrics,
    validation_issues: list[str],
    iou_threshold: float = 0.35,
    min_proto_pixels: int = 4,
) -> RiskAssessment:
    reasons = list(validation_issues)
    score = 0.0
    if validation_issues:
        score += 100.0
    if proto.iou_proto < iou_threshold:
        score += 55.0 + (iou_threshold - proto.iou_proto) * 100.0
        reasons.append("proto_iou_below_threshold")
    if proto.proto_nonzero_pixels < min_proto_pixels:
        score += 45.0 + (min_proto_pixels - proto.proto_nonzero_pixels) * 5.0
        reasons.append("proto_insufficient_pixels")
    if proto.fragmented:
        score += 40.0
        reasons.append("proto_fragmented")
    if geometry.mask_area_px < 9:
        score += 20.0
        reasons.append("extremely_small_mask")
    if geometry.aspect_ratio > 30:
        score += 10.0
        reasons.append("extreme_aspect_ratio")
    if geometry.fill_ratio < 0.03:
        score += 8.0
        reasons.append("very_low_fill_ratio")

    reasons = list(dict.fromkeys(reasons))
    if score >= 100:
        level = "critical"
    elif score >= 55:
        level = "high"
    elif score > 0:
        level = "medium"
    else:
        level = "pass"

    recommendations: list[str] = []
    if any(reason in reasons for reason in ("invalid_vertex_count", "self_intersection", "out_of_bounds", "non_finite_coordinate")):
        recommendations.append("修复多边形几何：确保顶点合法、边界闭合且不自交。")
    if proto.proto_nonzero_pixels < min_proto_pixels:
        recommendations.append("该实例在 208 x 208 Proto 网格几乎消失，建议切图训练或按 SOP 外扩标注。")
    elif proto.fragmented:
        recommendations.append("该实例在 Proto 网格发生断裂，建议采用切图训练或提高目标的最小有效宽度。")
    elif proto.iou_proto < iou_threshold:
        recommendations.append("Proto 留存率偏低，建议复核细线、小目标标注并评估切图策略。")
    if not recommendations:
        recommendations.append("几何与 Proto 留存未触发 Stage 0 风险规则。")

    return RiskAssessment(level=level, score=round(score, 3), reasons=reasons, recommendations=recommendations)
