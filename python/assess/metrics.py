"""推荐系统离线评估指标。"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from assess.models import (
    EvaluationCase,
    LatencySummary,
    LiftSummary,
    MetricResult,
    ParallelSummary,
    SuccessSummary,
    ViolationSummary,
)


def _prepare_rankings(
    cases: Sequence[EvaluationCase],
    rankings: Mapping[str, Sequence[str]],
    *,
    k: int,
) -> tuple[tuple[EvaluationCase, ...], dict[str, tuple[str, ...]]]:
    """校验查询集合和排名，并返回与外部输入隔离的快照。"""

    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k 必须是正整数")
    if not isinstance(rankings, Mapping):
        raise ValueError("rankings 必须是查询 ID 到排名序列的映射")

    case_snapshot = tuple(cases)
    by_query: dict[str, EvaluationCase] = {}
    for case in case_snapshot:
        if not isinstance(case, EvaluationCase):
            raise ValueError("cases 只能包含 EvaluationCase")
        if case.query_id in by_query:
            raise ValueError("评估查询 ID 不能重复")
        by_query[case.query_id] = case

    unknown_queries = set(rankings).difference(by_query)
    if unknown_queries:
        raise ValueError("rankings 包含未知查询 ID")

    ranking_snapshot: dict[str, tuple[str, ...]] = {}
    for query_id, raw_items in rankings.items():
        if isinstance(raw_items, (str, bytes)) or not isinstance(
            raw_items,
            Sequence,
        ):
            raise ValueError("单条排名必须是结果 ID 序列")
        items: list[str] = []
        seen: set[str] = set()
        for item_id in raw_items:
            if not isinstance(item_id, str) or not item_id.strip():
                raise ValueError("排名结果 ID 必须是非空字符串")
            if item_id in seen:
                raise ValueError("单条排名不能包含重复结果 ID")
            seen.add(item_id)
            items.append(item_id)
        ranking_snapshot[query_id] = tuple(items)
    return case_snapshot, ranking_snapshot


def _metric_result(values: Sequence[float], *, skipped_queries: int) -> MetricResult:
    """对逐查询指标做宏平均，并保留有效和跳过数量。"""

    if not values:
        return MetricResult(
            value=0.0,
            evaluated_queries=0,
            skipped_queries=skipped_queries,
        )
    return MetricResult(
        value=sum(values) / len(values),
        evaluated_queries=len(values),
        skipped_queries=skipped_queries,
    )


def hit_rate_at_k(
    cases: Sequence[EvaluationCase],
    rankings: Mapping[str, Sequence[str]],
    *,
    k: int,
    relevance_threshold: float = 2.0,
) -> MetricResult:
    case_snapshot, ranking_snapshot = _prepare_rankings(cases, rankings, k=k)
    values: list[float] = []
    skipped = 0
    for case in case_snapshot:
        relevant = case.relevant_ids(relevance_threshold=relevance_threshold)
        if not relevant:
            skipped += 1
            continue
        top_k = ranking_snapshot.get(case.query_id, ())[:k]
        values.append(float(any(item_id in relevant for item_id in top_k)))
    return _metric_result(values, skipped_queries=skipped)


def recall_at_k(
    cases: Sequence[EvaluationCase],
    rankings: Mapping[str, Sequence[str]],
    *,
    k: int,
    relevance_threshold: float = 2.0,
) -> MetricResult:
    case_snapshot, ranking_snapshot = _prepare_rankings(cases, rankings, k=k)
    values: list[float] = []
    skipped = 0
    for case in case_snapshot:
        relevant = case.relevant_ids(relevance_threshold=relevance_threshold)
        if not relevant:
            skipped += 1
            continue
        top_k = ranking_snapshot.get(case.query_id, ())[:k]
        hits = sum(item_id in relevant for item_id in top_k)
        values.append(hits / len(relevant))
    return _metric_result(values, skipped_queries=skipped)


def precision_at_k(
    cases: Sequence[EvaluationCase],
    rankings: Mapping[str, Sequence[str]],
    *,
    k: int,
    relevance_threshold: float = 2.0,
) -> MetricResult:
    case_snapshot, ranking_snapshot = _prepare_rankings(cases, rankings, k=k)
    values: list[float] = []
    for case in case_snapshot:
        relevant = case.relevant_ids(relevance_threshold=relevance_threshold)
        top_k = ranking_snapshot.get(case.query_id, ())[:k]
        hits = sum(item_id in relevant for item_id in top_k)
        values.append(hits / k)
    return _metric_result(values, skipped_queries=0)


def mrr_at_k(
    cases: Sequence[EvaluationCase],
    rankings: Mapping[str, Sequence[str]],
    *,
    k: int,
    relevance_threshold: float = 2.0,
) -> MetricResult:
    case_snapshot, ranking_snapshot = _prepare_rankings(cases, rankings, k=k)
    values: list[float] = []
    skipped = 0
    for case in case_snapshot:
        relevant = case.relevant_ids(relevance_threshold=relevance_threshold)
        if not relevant:
            skipped += 1
            continue
        reciprocal_rank = 0.0
        for rank, item_id in enumerate(
            ranking_snapshot.get(case.query_id, ())[:k],
            start=1,
        ):
            if item_id in relevant:
                reciprocal_rank = 1.0 / rank
                break
        values.append(reciprocal_rank)
    return _metric_result(values, skipped_queries=skipped)


def ndcg_at_k(
    cases: Sequence[EvaluationCase],
    rankings: Mapping[str, Sequence[str]],
    *,
    k: int,
) -> MetricResult:
    case_snapshot, ranking_snapshot = _prepare_rankings(cases, rankings, k=k)
    values: list[float] = []
    skipped = 0
    for case in case_snapshot:
        ideal_grades = sorted(
            (
                grade
                for item_id, grade in case.relevance.items()
                if item_id not in case.forbidden_ids and grade > 0.0
            ),
            reverse=True,
        )[:k]
        ideal_dcg = _dcg(ideal_grades)
        if ideal_dcg == 0.0:
            skipped += 1
            continue
        actual_grades = [
            (
                case.relevance.get(item_id, 0.0)
                if item_id not in case.forbidden_ids
                else 0.0
            )
            for item_id in ranking_snapshot.get(case.query_id, ())[:k]
        ]
        values.append(_dcg(actual_grades) / ideal_dcg)
    return _metric_result(values, skipped_queries=skipped)


def _dcg(grades: Sequence[float]) -> float:
    """计算使用指数增益和对数折扣的 DCG。"""

    return sum(
        ((2.0**grade) - 1.0) / math.log2(rank + 1.0)
        for rank, grade in enumerate(grades, start=1)
    )


def metric_lift(*, baseline: float, candidate: float) -> LiftSummary:
    baseline_value = _finite_metric_number(baseline, field_name="baseline")
    candidate_value = _finite_metric_number(candidate, field_name="candidate")
    absolute_lift = candidate_value - baseline_value
    relative_lift = (
        None if baseline_value == 0.0 else absolute_lift / abs(baseline_value)
    )
    return LiftSummary(
        baseline=baseline_value,
        candidate=candidate_value,
        absolute_lift=absolute_lift,
        relative_lift=relative_lift,
    )


def latency_summary(samples: Sequence[float]) -> LatencySummary:
    if isinstance(samples, (str, bytes)) or not isinstance(samples, Sequence):
        raise ValueError("延迟样本必须是数值序列")
    values = sorted(
        _nonnegative_metric_number(value, field_name="延迟样本") for value in samples
    )
    if not values:
        raise ValueError("延迟样本不能为空")
    return LatencySummary(
        sample_count=len(values),
        mean_ms=sum(values) / len(values),
        p50_ms=_percentile(values, 0.50),
        p95_ms=_percentile(values, 0.95),
        p99_ms=_percentile(values, 0.99),
        max_ms=values[-1],
    )


def parallel_summary(
    *,
    bm25_ms: float,
    vector_ms: float,
    hybrid_ms: float,
) -> ParallelSummary:
    bm25_value = _positive_metric_number(bm25_ms, field_name="bm25_ms")
    vector_value = _positive_metric_number(vector_ms, field_name="vector_ms")
    hybrid_value = _positive_metric_number(hybrid_ms, field_name="hybrid_ms")
    sequential_ms = bm25_value + vector_value
    saving_ms = sequential_ms - hybrid_value
    return ParallelSummary(
        bm25_ms=bm25_value,
        vector_ms=vector_value,
        hybrid_ms=hybrid_value,
        sequential_ms=sequential_ms,
        saving_ms=saving_ms,
        saving_rate=saving_ms / sequential_ms,
        speedup=sequential_ms / hybrid_value,
        overlap_efficiency=saving_ms / min(bm25_value, vector_value),
    )


def violation_rate(
    cases: Sequence[EvaluationCase],
    rankings: Mapping[str, Sequence[str]],
    *,
    k: int | None = None,
) -> ViolationSummary:
    validation_k = 1 if k is None else k
    case_snapshot, ranking_snapshot = _prepare_rankings(
        cases,
        rankings,
        k=validation_k,
    )
    violation_count = 0
    result_count = 0
    for case in case_snapshot:
        ranked = ranking_snapshot.get(case.query_id, ())
        considered = ranked if k is None else ranked[:k]
        result_count += len(considered)
        violation_count += sum(item_id in case.forbidden_ids for item_id in considered)
    rate = 0.0 if result_count == 0 else violation_count / result_count
    return ViolationSummary(
        violation_rate=rate,
        violation_count=violation_count,
        result_count=result_count,
    )


def degradation_success_rate(outcomes: Sequence[bool]) -> SuccessSummary:
    if isinstance(outcomes, (str, bytes)) or not isinstance(outcomes, Sequence):
        raise ValueError("降级结果必须是布尔序列")
    values = tuple(outcomes)
    if not values:
        raise ValueError("降级结果不能为空")
    if any(not isinstance(value, bool) for value in values):
        raise ValueError("降级结果只能包含布尔值")
    success_count = sum(values)
    return SuccessSummary(
        success_rate=success_count / len(values),
        success_count=success_count,
        total_count=len(values),
    )


def required_fact_coverage(
    matches: Sequence[Sequence[bool]],
) -> MetricResult:
    """按样本宏平均必需事实覆盖率，并跳过没有事实标注的样本。"""

    if isinstance(matches, (str, bytes)) or not isinstance(matches, Sequence):
        raise ValueError("事实覆盖结果必须是布尔序列的序列")
    values: list[float] = []
    skipped = 0
    for raw_group in matches:
        if isinstance(raw_group, (str, bytes)) or not isinstance(
            raw_group,
            Sequence,
        ):
            raise ValueError("事实覆盖结果必须是布尔序列的序列")
        group = tuple(raw_group)
        if any(not isinstance(value, bool) for value in group):
            raise ValueError("事实覆盖结果只能包含布尔值")
        if not group:
            skipped += 1
            continue
        values.append(sum(group) / len(group))
    return _metric_result(values, skipped_queries=skipped)


def answerability_accuracy(
    expected: Sequence[bool],
    predicted: Sequence[bool],
) -> SuccessSummary:
    """计算可回答性预测准确率，并保留显式总样本数。"""

    expected_values = tuple(expected)
    predicted_values = tuple(predicted)
    if not expected_values or len(expected_values) != len(predicted_values):
        raise ValueError("可回答性期望与预测必须是等长非空序列")
    if any(
        not isinstance(value, bool)
        for value in (*expected_values, *predicted_values)
    ):
        raise ValueError("可回答性结果只能包含布尔值")
    success_count = sum(
        expected_value == predicted_value
        for expected_value, predicted_value in zip(
            expected_values,
            predicted_values,
            strict=True,
        )
    )
    return SuccessSummary(
        success_rate=success_count / len(expected_values),
        success_count=success_count,
        total_count=len(expected_values),
    )


def false_support_rate(
    expected: Sequence[bool],
    predicted: Sequence[bool],
) -> ViolationSummary:
    """统计不可回答样本被错误判断为可回答的比例。"""

    expected_values = tuple(expected)
    predicted_values = tuple(predicted)
    if not expected_values or len(expected_values) != len(predicted_values):
        raise ValueError("可回答性期望与预测必须是等长非空序列")
    if any(
        not isinstance(value, bool)
        for value in (*expected_values, *predicted_values)
    ):
        raise ValueError("可回答性结果只能包含布尔值")
    negative_pairs = tuple(
        predicted_value
        for expected_value, predicted_value in zip(
            expected_values,
            predicted_values,
            strict=True,
        )
        if not expected_value
    )
    violation_count = sum(negative_pairs)
    total_count = len(negative_pairs)
    return ViolationSummary(
        violation_rate=(0.0 if total_count == 0 else violation_count / total_count),
        violation_count=violation_count,
        result_count=total_count,
    )


def _finite_metric_number(value: object, *, field_name: str) -> float:
    """校验指标计算使用的真实有限数值。"""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} 必须是有限数值")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field_name} 必须是有限数值")
    return number


def _nonnegative_metric_number(value: object, *, field_name: str) -> float:
    """校验非负有限数值。"""

    number = _finite_metric_number(value, field_name=field_name)
    if number < 0.0:
        raise ValueError(f"{field_name} 不能为负数")
    return number


def _positive_metric_number(value: object, *, field_name: str) -> float:
    """校验正有限数值。"""

    number = _finite_metric_number(value, field_name=field_name)
    if number <= 0.0:
        raise ValueError(f"{field_name} 必须大于 0")
    return number


def _percentile(sorted_values: Sequence[float], quantile: float) -> float:
    """对已经升序排列的样本执行线性插值分位数。"""

    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * quantile
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return sorted_values[lower_index]
    fraction = position - lower_index
    lower = sorted_values[lower_index]
    upper = sorted_values[upper_index]
    return lower + (upper - lower) * fraction
