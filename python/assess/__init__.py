"""推荐系统离线评估工具。"""

from assess.diagnostics import diagnose
from assess.metrics import (
    degradation_success_rate,
    hit_rate_at_k,
    latency_summary,
    metric_lift,
    mrr_at_k,
    ndcg_at_k,
    parallel_summary,
    precision_at_k,
    recall_at_k,
    violation_rate,
)
from assess.models import (
    DiagnosticFinding,
    EvaluationCase,
    LatencySummary,
    LiftSummary,
    MetricResult,
    ParallelSummary,
    SuccessSummary,
    ViolationSummary,
)
from assess.pipeline_evaluation import (
    PipelineEvaluationCase,
    PipelineEvaluationReport,
    PipelineVariantReport,
    evaluate_pipeline,
    load_pipeline_cases,
)

__all__ = [
    "DiagnosticFinding",
    "EvaluationCase",
    "LatencySummary",
    "LiftSummary",
    "MetricResult",
    "ParallelSummary",
    "PipelineEvaluationCase",
    "PipelineEvaluationReport",
    "PipelineVariantReport",
    "SuccessSummary",
    "ViolationSummary",
    "degradation_success_rate",
    "diagnose",
    "evaluate_pipeline",
    "hit_rate_at_k",
    "latency_summary",
    "load_pipeline_cases",
    "metric_lift",
    "mrr_at_k",
    "ndcg_at_k",
    "parallel_summary",
    "precision_at_k",
    "recall_at_k",
    "violation_rate",
]
