"""推荐系统评估输入与结果模型。"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal, Mapping


def _finite_number(value: object, *, field_name: str) -> float:
    """把真实有限数值转换为浮点数，并拒绝布尔伪数值。"""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} 必须是有限数值")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field_name} 必须是有限数值")
    return number


def _nonnegative_count(value: object, *, field_name: str) -> int:
    """校验不可为负的整数计数。"""

    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} 必须是非负整数")
    return value


def _nonempty_text(value: object, *, field_name: str) -> str:
    """校验不改变原值的非空文本。"""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} 必须是非空字符串")
    return value


def _rate(value: object, *, field_name: str) -> float:
    """校验闭区间 0..1 内的有限比例。"""

    number = _finite_number(value, field_name=field_name)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{field_name} 必须位于 0..1")
    return number


@dataclass(frozen=True)
class EvaluationCase:
    """保存单条查询的分级相关性与业务禁止项。"""

    query_id: str
    relevance: Mapping[str, float]
    forbidden_ids: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        """复制并校验外部标注，避免调用方后续修改污染评估。"""

        _nonempty_text(self.query_id, field_name="query_id")
        if not isinstance(self.relevance, Mapping):
            raise ValueError("relevance 必须是映射")
        normalized_relevance: dict[str, float] = {}
        for item_id, raw_score in self.relevance.items():
            _nonempty_text(item_id, field_name="相关结果 ID")
            score = _finite_number(raw_score, field_name="相关性")
            if not 0.0 <= score <= 3.0:
                raise ValueError("相关性必须位于 0..3")
            normalized_relevance[item_id] = score

        try:
            normalized_forbidden = frozenset(self.forbidden_ids)
        except TypeError as exc:
            raise ValueError("forbidden_ids 必须是可迭代的结果 ID") from exc
        for item_id in normalized_forbidden:
            _nonempty_text(item_id, field_name="禁止结果 ID")

        object.__setattr__(
            self,
            "relevance",
            MappingProxyType(normalized_relevance),
        )
        object.__setattr__(self, "forbidden_ids", normalized_forbidden)

    def relevant_ids(
        self,
        *,
        relevance_threshold: float = 2.0,
    ) -> frozenset[str]:
        """返回达到相关性阈值的结果 ID。"""

        threshold = _finite_number(
            relevance_threshold,
            field_name="relevance_threshold",
        )
        if not 0.0 <= threshold <= 3.0:
            raise ValueError("relevance_threshold 必须位于 0..3")
        return frozenset(
            item_id
            for item_id, score in self.relevance.items()
            if score >= threshold and item_id not in self.forbidden_ids
        )


@dataclass(frozen=True)
class MetricResult:
    value: float
    evaluated_queries: int
    skipped_queries: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", _rate(self.value, field_name="value"))
        object.__setattr__(
            self,
            "evaluated_queries",
            _nonnegative_count(
                self.evaluated_queries,
                field_name="evaluated_queries",
            ),
        )
        object.__setattr__(
            self,
            "skipped_queries",
            _nonnegative_count(self.skipped_queries, field_name="skipped_queries"),
        )


@dataclass(frozen=True)
class LatencySummary:
    sample_count: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float

    def __post_init__(self) -> None:
        count = _nonnegative_count(self.sample_count, field_name="sample_count")
        if count == 0:
            raise ValueError("sample_count 必须大于 0")
        object.__setattr__(self, "sample_count", count)
        for field_name in ("mean_ms", "p50_ms", "p95_ms", "p99_ms", "max_ms"):
            value = _finite_number(getattr(self, field_name), field_name=field_name)
            if value < 0.0:
                raise ValueError(f"{field_name} 不能为负数")
            object.__setattr__(self, field_name, value)


@dataclass(frozen=True)
class ParallelSummary:
    bm25_ms: float
    vector_ms: float
    hybrid_ms: float
    sequential_ms: float
    saving_ms: float
    saving_rate: float
    speedup: float
    overlap_efficiency: float

    def __post_init__(self) -> None:
        for field_name in (
            "bm25_ms",
            "vector_ms",
            "hybrid_ms",
            "sequential_ms",
            "saving_ms",
            "saving_rate",
            "speedup",
            "overlap_efficiency",
        ):
            object.__setattr__(
                self,
                field_name,
                _finite_number(getattr(self, field_name), field_name=field_name),
            )
        if self.bm25_ms <= 0.0 or self.vector_ms <= 0.0 or self.hybrid_ms <= 0.0:
            raise ValueError("BM25、Vector 和 Hybrid 耗时必须大于 0")
        if self.sequential_ms <= 0.0 or self.speedup <= 0.0:
            raise ValueError("串行估计和加速比必须大于 0")


@dataclass(frozen=True)
class LiftSummary:
    baseline: float
    candidate: float
    absolute_lift: float
    relative_lift: float | None

    def __post_init__(self) -> None:
        for field_name in ("baseline", "candidate", "absolute_lift"):
            object.__setattr__(
                self,
                field_name,
                _finite_number(getattr(self, field_name), field_name=field_name),
            )
        if self.relative_lift is not None:
            object.__setattr__(
                self,
                "relative_lift",
                _finite_number(self.relative_lift, field_name="relative_lift"),
            )


@dataclass(frozen=True)
class ViolationSummary:
    violation_rate: float
    violation_count: int
    result_count: int

    def __post_init__(self) -> None:
        rate = _rate(self.violation_rate, field_name="violation_rate")
        violation_count = _nonnegative_count(
            self.violation_count,
            field_name="violation_count",
        )
        result_count = _nonnegative_count(
            self.result_count,
            field_name="result_count",
        )
        if violation_count > result_count:
            raise ValueError("violation_count 不能大于 result_count")
        if result_count == 0 and rate != 0.0:
            raise ValueError("没有结果时 violation_rate 必须为 0")
        object.__setattr__(self, "violation_rate", rate)
        object.__setattr__(self, "violation_count", violation_count)
        object.__setattr__(self, "result_count", result_count)


@dataclass(frozen=True)
class SuccessSummary:
    success_rate: float
    success_count: int
    total_count: int

    def __post_init__(self) -> None:
        rate = _rate(self.success_rate, field_name="success_rate")
        success_count = _nonnegative_count(
            self.success_count,
            field_name="success_count",
        )
        total_count = _nonnegative_count(self.total_count, field_name="total_count")
        if total_count == 0:
            raise ValueError("total_count 必须大于 0")
        if success_count > total_count:
            raise ValueError("success_count 不能大于 total_count")
        object.__setattr__(self, "success_rate", rate)
        object.__setattr__(self, "success_count", success_count)
        object.__setattr__(self, "total_count", total_count)


@dataclass(frozen=True)
class DiagnosticFinding:
    code: str
    severity: str
    message: str

    def __post_init__(self) -> None:
        _nonempty_text(self.code, field_name="code")
        _nonempty_text(self.severity, field_name="severity")
        _nonempty_text(self.message, field_name="message")
        if self.severity not in {"info", "warning", "high"}:
            raise ValueError("severity 必须是 info、warning 或 high")


EvaluationExecutionStatus = Literal["executed", "failed", "not_run"]
KnowledgeEvidenceAction = Literal["answer", "rewrite", "ask", "select", "refuse"]
KnowledgeEvidenceReasonCode = Literal[
    "enough_evidence",
    "low_relevance_retry_available",
    "missing_information",
    "unresolved_reference",
    "multiple_document_candidates",
    "multiple_skill_candidates",
    "unsafe_request",
    "out_of_scope",
    "skill_scope_conflict",
    "no_relevant_evidence",
    "insufficient_answerability",
    "invalid_gate_input",
    "rewrite_exhausted",
]

_ACTION_REASON_CODES: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        "answer": frozenset(("enough_evidence",)),
        "rewrite": frozenset(("low_relevance_retry_available",)),
        "ask": frozenset(("missing_information", "unresolved_reference")),
        "select": frozenset(
            ("multiple_document_candidates", "multiple_skill_candidates")
        ),
        "refuse": frozenset(
            (
                "unsafe_request",
                "out_of_scope",
                "skill_scope_conflict",
                "no_relevant_evidence",
                "insufficient_answerability",
                "invalid_gate_input",
                "rewrite_exhausted",
            )
        ),
    }
)


@dataclass(frozen=True)
class EvidenceRequirement:
    """描述一个事实及任一可接受的证据正文锚点。"""

    fact_id: str
    any_of: tuple[str, ...]

    def __post_init__(self) -> None:
        """校验事实 ID，并复制、去重证据锚点。"""

        _nonempty_text(self.fact_id, field_name="fact_id")
        if isinstance(self.any_of, (str, bytes)):
            raise ValueError("any_of 必须是证据锚点序列")
        try:
            normalized = tuple(dict.fromkeys(self.any_of))
        except (TypeError, ValueError) as exc:
            raise ValueError("any_of 必须是证据锚点序列") from exc
        if not normalized:
            raise ValueError("any_of 不能为空")
        for anchor in normalized:
            _nonempty_text(anchor, field_name="证据锚点")
        object.__setattr__(self, "any_of", normalized)


@dataclass(frozen=True)
class KnowledgeQaEvaluationCase:
    """保存单条知识问答的人工标注与评估分组。"""

    case_id: str
    question: str
    answerable: bool
    expected_action: KnowledgeEvidenceAction
    expected_reason_code: KnowledgeEvidenceReasonCode
    expected_document_ids: tuple[str, ...] = ()
    required_evidence: tuple[EvidenceRequirement, ...] = ()
    document_scope: tuple[str, ...] = ()
    expected_rewrite_terms: tuple[str, ...] = ()
    expected_option_ids: tuple[str, ...] = ()
    tags: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        """复制外部标注，并校验正负样本约束。"""

        _nonempty_text(self.case_id, field_name="case_id")
        _nonempty_text(self.question, field_name="question")
        if not isinstance(self.answerable, bool):
            raise ValueError("answerable 必须是布尔值")
        try:
            expected = tuple(dict.fromkeys(self.expected_document_ids))
            scope = tuple(dict.fromkeys(self.document_scope))
            facts = tuple(self.required_evidence)
            rewrite_terms = tuple(dict.fromkeys(self.expected_rewrite_terms))
            option_ids = tuple(dict.fromkeys(self.expected_option_ids))
            tags = frozenset(self.tags)
        except (TypeError, ValueError) as exc:
            raise ValueError("知识问答评估标注必须是可迭代序列") from exc
        if self.expected_action not in _ACTION_REASON_CODES:
            raise ValueError("expected_action 无效")
        if self.expected_reason_code not in _ACTION_REASON_CODES[self.expected_action]:
            raise ValueError("expected_action 与 expected_reason_code 不一致")
        for value in (*expected, *scope, *rewrite_terms, *option_ids, *tags):
            _nonempty_text(value, field_name="知识问答评估 ID 或标签")
        if any(not isinstance(fact, EvidenceRequirement) for fact in facts):
            raise ValueError("required_evidence 只能包含 EvidenceRequirement")
        if len({fact.fact_id for fact in facts}) != len(facts):
            raise ValueError("单条样本的 fact_id 不能重复")
        if self.answerable and (not expected or not facts):
            raise ValueError("可回答样本必须提供期望文档和关键事实")
        if not self.answerable and (expected or facts):
            raise ValueError("不可回答样本不能提供期望文档或关键事实")
        if self.expected_action in {"answer", "rewrite"} and not self.answerable:
            raise ValueError("answer/rewrite 动作样本必须在当前数据下可回答")
        if self.expected_action in {"ask", "select", "refuse"} and self.answerable:
            raise ValueError("ask/select/refuse 动作样本当前轮不得直接回答")
        if self.expected_action == "rewrite":
            if not rewrite_terms or option_ids:
                raise ValueError("rewrite 必须提供改写术语且不能提供选项")
        elif rewrite_terms:
            raise ValueError("非 rewrite 动作不能提供改写术语")
        if self.expected_action == "select":
            if not 2 <= len(option_ids) <= 5:
                raise ValueError("select 必须提供 2 到 5 个期望选项")
        elif option_ids:
            raise ValueError("非 select 动作不能提供期望选项")
        object.__setattr__(self, "expected_document_ids", expected)
        object.__setattr__(self, "required_evidence", facts)
        object.__setattr__(self, "document_scope", scope)
        object.__setattr__(self, "expected_rewrite_terms", rewrite_terms)
        object.__setattr__(self, "expected_option_ids", option_ids)
        object.__setattr__(self, "tags", tags)


@dataclass(frozen=True)
class KnowledgeQaCaseReport:
    """保存单条问答的执行状态、命中结果与事实保护结果。"""

    case_id: str
    execution_status: EvaluationExecutionStatus
    error_code: str | None
    ranked_document_ids: tuple[str, ...]
    selected_chunk_ids: tuple[str, ...]
    matched_fact_ids: tuple[str, ...]
    missing_fact_ids: tuple[str, ...]
    predicted_answerable: bool | None
    answer_status: str | None
    citation_integrity: bool | None
    predicted_action: KnowledgeEvidenceAction | None
    predicted_reason_code: KnowledgeEvidenceReasonCode | None
    rewrite_count: int
    rewritten_query: str | None
    option_ids: tuple[str, ...]
    non_answer_evidence_leak: bool | None
    vector_status: str
    latency_ms: float | None

    def __post_init__(self) -> None:
        """复制逐题结果序列，并校验稳定状态字段。"""

        _nonempty_text(self.case_id, field_name="case_id")
        if self.execution_status not in {"executed", "failed", "not_run"}:
            raise ValueError("execution_status 无效")
        if self.error_code is not None:
            _nonempty_text(self.error_code, field_name="error_code")
        for field_name in (
            "ranked_document_ids",
            "selected_chunk_ids",
            "matched_fact_ids",
            "missing_fact_ids",
            "option_ids",
        ):
            values = tuple(getattr(self, field_name))
            for value in values:
                _nonempty_text(value, field_name=field_name)
            object.__setattr__(self, field_name, values)
        if self.predicted_answerable is not None and not isinstance(
            self.predicted_answerable,
            bool,
        ):
            raise ValueError("predicted_answerable 必须是布尔值或 None")
        if self.answer_status is not None:
            _nonempty_text(self.answer_status, field_name="answer_status")
        if self.citation_integrity is not None and not isinstance(
            self.citation_integrity,
            bool,
        ):
            raise ValueError("citation_integrity 必须是布尔值或 None")
        if self.predicted_action is None:
            if self.predicted_reason_code is not None:
                raise ValueError("缺少 predicted_action 时不能有原因码")
        elif (
            self.predicted_action not in _ACTION_REASON_CODES
            or self.predicted_reason_code
            not in _ACTION_REASON_CODES[self.predicted_action]
        ):
            raise ValueError("预测动作与原因码不一致")
        object.__setattr__(
            self,
            "rewrite_count",
            _nonnegative_count(self.rewrite_count, field_name="rewrite_count"),
        )
        if self.rewritten_query is not None:
            _nonempty_text(self.rewritten_query, field_name="rewritten_query")
        if self.predicted_action == "rewrite" and not self.rewritten_query:
            raise ValueError("rewrite 预测必须记录 rewritten_query")
        if self.predicted_action != "rewrite" and self.rewritten_query is not None:
            raise ValueError("非 rewrite 预测不能记录 rewritten_query")
        if self.non_answer_evidence_leak is not None and not isinstance(
            self.non_answer_evidence_leak,
            bool,
        ):
            raise ValueError("non_answer_evidence_leak 必须是布尔值或 None")
        if self.execution_status == "executed" and (
            self.predicted_action is None
            or self.predicted_reason_code is None
            or self.non_answer_evidence_leak is None
        ):
            raise ValueError("已执行样本必须包含完整动作结果")
        _nonempty_text(self.vector_status, field_name="vector_status")
        if self.latency_ms is not None:
            latency = _finite_number(self.latency_ms, field_name="latency_ms")
            if latency < 0.0:
                raise ValueError("latency_ms 不能为负数")
            object.__setattr__(self, "latency_ms", latency)


@dataclass(frozen=True)
class KnowledgeQaMetricSnapshot:
    """保存知识问答检索、证据与安全层的聚合指标。"""

    document_hit_at_k: MetricResult
    document_recall_at_k: MetricResult
    document_mrr_at_k: MetricResult
    fact_coverage: MetricResult
    answerability_accuracy: SuccessSummary
    false_support_rate: ViolationSummary
    citation_integrity: SuccessSummary


@dataclass(frozen=True)
class KnowledgeQaActionMetrics:
    """保存五类动作分类与非回答安全指标。"""

    precision_by_action: Mapping[str, float | None]
    recall_by_action: Mapping[str, float | None]
    f1_by_action: Mapping[str, float | None]
    macro_f1: float | None
    false_answer_rate: ViolationSummary | None
    rewrite_recovery_rate: SuccessSummary | None
    selection_option_accuracy: SuccessSummary | None
    non_answer_evidence_leak_count: int

    def __post_init__(self) -> None:
        expected_actions = frozenset(_ACTION_REASON_CODES)
        for field_name in (
            "precision_by_action",
            "recall_by_action",
            "f1_by_action",
        ):
            values = dict(getattr(self, field_name))
            if frozenset(values) != expected_actions:
                raise ValueError(f"{field_name} 必须覆盖全部五类动作")
            normalized = {
                action: (
                    None
                    if value is None
                    else _rate(value, field_name=f"{field_name}.{action}")
                )
                for action, value in values.items()
            }
            object.__setattr__(self, field_name, MappingProxyType(normalized))
        if self.macro_f1 is not None:
            object.__setattr__(
                self,
                "macro_f1",
                _rate(self.macro_f1, field_name="macro_f1"),
            )
        object.__setattr__(
            self,
            "non_answer_evidence_leak_count",
            _nonnegative_count(
                self.non_answer_evidence_leak_count,
                field_name="non_answer_evidence_leak_count",
            ),
        )


@dataclass(frozen=True)
class KnowledgeQaEvaluationReport:
    """保存一次知识问答离线评估的完整确定性报告。"""

    total_cases: int
    executed_cases: int
    failed_cases: int
    not_run_cases: int
    overall: KnowledgeQaMetricSnapshot | None
    action_metrics: KnowledgeQaActionMetrics | None
    by_tag: Mapping[str, KnowledgeQaMetricSnapshot]
    latency: LatencySummary | None
    cases: tuple[KnowledgeQaCaseReport, ...]
    findings: tuple[DiagnosticFinding, ...]
    used_external_services: bool = False

    def __post_init__(self) -> None:
        """复制报告集合，并校验执行计数之间的一致性。"""

        counts = {
            field_name: _nonnegative_count(
                getattr(self, field_name),
                field_name=field_name,
            )
            for field_name in (
                "total_cases",
                "executed_cases",
                "failed_cases",
                "not_run_cases",
            )
        }
        if (
            counts["executed_cases"]
            + counts["failed_cases"]
            + counts["not_run_cases"]
            != counts["total_cases"]
        ):
            raise ValueError("知识问答评估执行计数必须等于总样本数")
        if not isinstance(self.by_tag, Mapping):
            raise ValueError("by_tag 必须是指标映射")
        if self.action_metrics is not None and not isinstance(
            self.action_metrics,
            KnowledgeQaActionMetrics,
        ):
            raise ValueError("action_metrics 类型无效")
        by_tag = dict(self.by_tag)
        for tag, snapshot in by_tag.items():
            _nonempty_text(tag, field_name="评估标签")
            if not isinstance(snapshot, KnowledgeQaMetricSnapshot):
                raise ValueError("by_tag 只能包含 KnowledgeQaMetricSnapshot")
        cases = tuple(self.cases)
        findings = tuple(self.findings)
        if len(cases) != counts["total_cases"]:
            raise ValueError("逐题报告数量必须等于总样本数")
        if any(not isinstance(case, KnowledgeQaCaseReport) for case in cases):
            raise ValueError("cases 只能包含 KnowledgeQaCaseReport")
        if any(not isinstance(item, DiagnosticFinding) for item in findings):
            raise ValueError("findings 只能包含 DiagnosticFinding")
        if not isinstance(self.used_external_services, bool):
            raise ValueError("used_external_services 必须是布尔值")
        object.__setattr__(self, "by_tag", MappingProxyType(by_tag))
        object.__setattr__(self, "cases", cases)
        object.__setattr__(self, "findings", findings)
