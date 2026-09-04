"""根据评估指标组合生成确定性排查提示。"""

from __future__ import annotations

import math

from assess.models import DiagnosticFinding


def diagnose(
    *,
    hit_at_k: float,
    rrf_mrr: float,
    final_mrr: float,
    final_ndcg: float,
    violation_rate: float,
    degradation_success_rate: float,
    hit_threshold: float = 0.95,
    mrr_threshold: float = 0.75,
    ndcg_threshold: float = 0.80,
    rerank_delta_threshold: float = 0.05,
) -> tuple[DiagnosticFinding, ...]:
    """根据指标关系返回稳定、非排他的可能原因提示。"""

    hit_value = _rate(hit_at_k, field_name="hit_at_k")
    rrf_value = _rate(rrf_mrr, field_name="rrf_mrr")
    final_mrr_value = _rate(final_mrr, field_name="final_mrr")
    final_ndcg_value = _rate(final_ndcg, field_name="final_ndcg")
    violation_value = _rate(violation_rate, field_name="violation_rate")
    degradation_value = _rate(
        degradation_success_rate,
        field_name="degradation_success_rate",
    )
    hit_limit = _rate(hit_threshold, field_name="hit_threshold")
    mrr_limit = _rate(mrr_threshold, field_name="mrr_threshold")
    ndcg_limit = _rate(ndcg_threshold, field_name="ndcg_threshold")
    delta_limit = _rate(
        rerank_delta_threshold,
        field_name="rerank_delta_threshold",
    )

    findings: list[DiagnosticFinding] = []
    if violation_value > 0.0:
        findings.append(
            DiagnosticFinding(
                code="hard_constraint_violation",
                severity="high",
                message=(
                    "存在硬条件违规结果，优先检查共享过滤、聚合二次保护和禁止项标注。"
                ),
            )
        )
    if degradation_value < 1.0:
        findings.append(
            DiagnosticFinding(
                code="fallback_failure",
                severity="high",
                message=(
                    "部分故障用例未按预期降级，优先检查超时捕获、非法响应保护和确定性回退。"
                ),
            )
        )

    if hit_value < hit_limit:
        findings.append(
            DiagnosticFinding(
                code="low_hit",
                severity="high",
                message=(
                    "检索覆盖不足，优先检查意图主题、查询构造、Embedding、分词、阈值、索引覆盖和过滤误杀。"
                ),
            )
        )
        return tuple(findings)

    if rrf_value < mrr_limit:
        findings.append(
            DiagnosticFinding(
                code="retrieval_ranking",
                severity="warning",
                message=(
                    "正确结果已被召回但位置偏后，优先检查 BM25/Vector 原始评分、RRF 参数和单路噪声。"
                ),
            )
        )

    rerank_delta = final_mrr_value - rrf_value
    if rerank_delta < -delta_limit:
        findings.append(
            DiagnosticFinding(
                code="rerank_regression",
                severity="high",
                message=(
                    "最终 MRR 明显低于 RRF MRR，优先检查 LLM 重排、画像影响和最终混合分。"
                ),
            )
        )
    elif rerank_delta > delta_limit:
        findings.append(
            DiagnosticFinding(
                code="rerank_improvement",
                severity="info",
                message="最终 MRR 明显提升，重排有效；仍需结合 NDCG 检查多结果整体顺序。",
            )
        )

    if final_mrr_value >= mrr_limit and final_ndcg_value < ndcg_limit:
        findings.append(
            DiagnosticFinding(
                code="low_ndcg",
                severity="warning",
                message=(
                    "首个正确结果位置正常，但后续结果质量或整体顺序较差，需检查多结果重排与多样性。"
                ),
            )
        )
    return tuple(findings)


def _rate(value: object, *, field_name: str) -> float:
    """校验诊断输入为闭区间 0..1 内的有限比例。"""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} 必须是 0..1 内的有限数值")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{field_name} 必须是 0..1 内的有限数值")
    return number
