"""复用真实知识检索组件运行完全离线的知识问答评估。"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.agents.knowledge_answer_agent import KnowledgeAnswerAgent
from app.agents.knowledge_query_analysis_agent import KnowledgeQueryAnalysisAgent
from app.application.knowledge_qa import KnowledgeQaService, KnowledgeSearch
from app.domain.services.knowledge_evidence_gate import KnowledgeEvidenceGate
from app.domain.services.knowledge_evidence_selector import (
    KnowledgeEvidenceSelector,
)
from app.domain.services.runtime_skill_matcher import RuntimeSkillMatcher
from app.infrastructure.database.sqlite.knowledge_repository import (
    SQLiteKnowledgeRepository,
)
from app.infrastructure.retrieval.knowledge_search import InMemoryKnowledgeSearch
from app.models.knowledge_qa import (
    KnowledgeAnswerResult,
    KnowledgeChunkRecord,
    KnowledgeQueryAnalysis,
    KnowledgeSearchResult,
)
from app.models.runtime_skill import (
    CompiledRuntimeSkill,
    RuntimeSkillResponsePolicy,
    RuntimeSkillSnapshot,
)
from assess.metrics import (
    answerability_accuracy,
    false_support_rate,
    hit_rate_at_k,
    latency_summary,
    mrr_at_k,
    recall_at_k,
    required_fact_coverage,
)
from assess.models import (
    DiagnosticFinding,
    EvaluationCase,
    EvidenceRequirement,
    KnowledgeQaCaseReport,
    KnowledgeQaActionMetrics,
    KnowledgeQaEvaluationCase,
    KnowledgeQaEvaluationReport,
    KnowledgeQaMetricSnapshot,
    SuccessSummary,
    ViolationSummary,
)


DEFAULT_KNOWLEDGE_QA_CASES_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "knowledge_qa_evaluation_cases.json"
)


class _StoredEvidenceRequirement(BaseModel):
    """磁盘中的单个必需事实及其可接受正文锚点。"""

    model_config = ConfigDict(extra="forbid")

    fact_id: str = Field(min_length=1, max_length=100)
    any_of: list[str] = Field(min_length=1, max_length=10)


class _StoredKnowledgeQaCase(BaseModel):
    """磁盘中的严格知识问答人工标注契约。"""

    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(min_length=1, max_length=100)
    question: str = Field(min_length=1, max_length=500)
    answerable: bool
    expected_action: str = Field(min_length=1, max_length=20)
    expected_reason_code: str = Field(min_length=1, max_length=100)
    expected_document_ids: list[str] = Field(default_factory=list, max_length=10)
    required_evidence: list[_StoredEvidenceRequirement] = Field(
        default_factory=list,
        max_length=10,
    )
    document_scope: list[str] = Field(default_factory=list, max_length=10)
    expected_rewrite_terms: list[str] = Field(default_factory=list, max_length=10)
    expected_option_ids: list[str] = Field(default_factory=list, max_length=5)
    tags: list[str] = Field(default_factory=list, max_length=10)


class _RecordingEvidenceGate:
    """复用生产 Gate，并记录单题实际动作序列。"""

    def __init__(self) -> None:
        self._delegate = KnowledgeEvidenceGate()
        self.decisions: list[Any] = []

    def reset(self) -> None:
        self.decisions.clear()

    def precheck(self, **kwargs: Any) -> Any:
        decision = self._delegate.precheck(**kwargs)
        if decision is not None:
            self.decisions.append(decision)
        return decision

    def decide_after_retrieval(self, *args: Any, **kwargs: Any) -> Any:
        decision = self._delegate.decide_after_retrieval(*args, **kwargs)
        self.decisions.append(decision)
        return decision


class _EvaluationQueryAnalysisAgent:
    """为动作覆盖提供固定离线分析，其余请求复用生产规则路径。"""

    _REWRITE_QUESTION = "What workloads best fit Project Loom?"
    _RETRY_QUERY = "Java 虚拟线程适合哪类任务？"
    _ASK_QUESTION = "请分析部署方案"

    def __init__(self) -> None:
        self._delegate = KnowledgeQueryAnalysisAgent(llm=None)

    async def analyze(self, question: str, **kwargs: Any) -> KnowledgeQueryAnalysis:
        if question == self._REWRITE_QUESTION:
            return KnowledgeQueryAnalysis(
                standalone_query=question,
                retry_query=self._RETRY_QUERY,
                question_type="factual",
                strategy="direct",
                confidence=1.0,
            )
        if question == self._ASK_QUESTION:
            return KnowledgeQueryAnalysis(
                standalone_query=question,
                missing_information=("目标版本",),
                clarification_question="请补充目标版本。",
                question_type="analytical",
                strategy="direct",
                confidence=1.0,
            )
        return await self._delegate.analyze(question, **kwargs)

    async def aclose(self) -> None:
        await self._delegate.aclose()


class _EvaluationRuntimeSkillRegistry:
    """提供两个同分 Fake Skill，只覆盖固定 select 样本。"""

    def __init__(self) -> None:
        policy = RuntimeSkillResponsePolicy()
        skills = tuple(
            CompiledRuntimeSkill(
                skill_id=skill_id,
                version="1.0.0",
                enabled=True,
                applies_to=("knowledge_qa",),
                activation_keywords=("多领域并发诊断",),
                gate_profile="default_evidence",
                response_policy=policy,
                priority=100,
                content_hash=character * 64,
            )
            for skill_id, character in (
                ("java-domain", "a"),
                ("python-domain", "b"),
            )
        )
        self._snapshot = RuntimeSkillSnapshot.build(
            generation=1,
            loaded_at=datetime(2026, 8, 14, tzinfo=timezone.utc),
            skills=skills,
        )

    def capture_snapshot(self) -> RuntimeSkillSnapshot:
        return self._snapshot


def load_knowledge_qa_cases(
    path: str | Path = DEFAULT_KNOWLEDGE_QA_CASES_PATH,
) -> tuple[KnowledgeQaEvaluationCase, ...]:
    """读取并严格校验固定知识问答人工标注。"""

    case_path = Path(path)
    try:
        payload = json.loads(case_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ValueError("知识问答评估标注文件不存在") from None
    except (json.JSONDecodeError, OSError, UnicodeError):
        raise ValueError("知识问答评估标注文件无法安全读取") from None
    if not isinstance(payload, list) or not payload:
        raise ValueError("知识问答评估标注必须是非空数组")

    cases: list[KnowledgeQaEvaluationCase] = []
    seen_ids: set[str] = set()
    for index, raw_case in enumerate(payload, start=1):
        try:
            stored = _StoredKnowledgeQaCase.model_validate(raw_case)
            case = KnowledgeQaEvaluationCase(
                case_id=" ".join(stored.case_id.split()),
                question=" ".join(stored.question.split()),
                answerable=stored.answerable,
                expected_action=stored.expected_action,
                expected_reason_code=stored.expected_reason_code,
                expected_document_ids=tuple(stored.expected_document_ids),
                required_evidence=tuple(
                    EvidenceRequirement(item.fact_id, tuple(item.any_of))
                    for item in stored.required_evidence
                ),
                document_scope=tuple(stored.document_scope),
                expected_rewrite_terms=tuple(stored.expected_rewrite_terms),
                expected_option_ids=tuple(stored.expected_option_ids),
                tags=frozenset(stored.tags),
            )
        except (TypeError, ValueError, ValidationError):
            raise ValueError(f"知识问答评估标注第 {index} 条无效") from None
        if case.case_id in seen_ids:
            raise ValueError("知识问答评估 case_id 不能重复")
        seen_ids.add(case.case_id)
        cases.append(case)
    return tuple(cases)


class KnowledgeQaEvaluator:
    """复用生产检索与事实保护，对固定标注执行只读分层评估。"""

    def __init__(
        self,
        *,
        repository: SQLiteKnowledgeRepository,
        search: KnowledgeSearch,
        evidence_selector: KnowledgeEvidenceSelector | None = None,
    ) -> None:
        """装配禁用真实模型的知识问答评估运行时。"""

        self._repository = repository
        self._search = search
        self._selector = evidence_selector or KnowledgeEvidenceSelector()
        self._gate = _RecordingEvidenceGate()
        self._service = KnowledgeQaService(
            repository=repository,
            search=search,
            answer_agent=KnowledgeAnswerAgent(llm=None),
            query_analysis_agent=_EvaluationQueryAnalysisAgent(),
            evidence_selector=self._selector,
            runtime_skill_registry=_EvaluationRuntimeSkillRegistry(),
            runtime_skill_matcher=RuntimeSkillMatcher(),
            evidence_gate=self._gate,
        )

    async def evaluate(
        self,
        cases: Sequence[KnowledgeQaEvaluationCase],
        *,
        k: int = 5,
    ) -> KnowledgeQaEvaluationReport:
        """执行全部标注，并把单题异常隔离为稳定失败结果。"""

        if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= 20:
            raise ValueError("k 必须是 1 到 20 的整数")
        case_snapshot = tuple(cases)
        if not case_snapshot:
            raise ValueError("知识问答评估样本不能为空")
        if any(
            not isinstance(case, KnowledgeQaEvaluationCase)
            for case in case_snapshot
        ):
            raise ValueError("知识问答评估样本类型无效")
        if len({case.case_id for case in case_snapshot}) != len(case_snapshot):
            raise ValueError("知识问答评估 case_id 不能重复")

        chunks = await asyncio.to_thread(self._repository.list_ready_chunks)
        self._validate_documents(case_snapshot, chunks)
        await self._search.refresh(chunks)
        reports = tuple(
            [await self._evaluate_case(case, k=k) for case in case_snapshot]
        )
        return self._build_report(case_snapshot, reports, k=k)

    async def aclose(self) -> None:
        """关闭评估器拥有的模型占位和检索资源。"""

        await self._service.aclose()

    @staticmethod
    def _validate_documents(
        cases: Sequence[KnowledgeQaEvaluationCase],
        chunks: Sequence[KnowledgeChunkRecord],
    ) -> None:
        """在计分前拒绝引用不存在文档的无效标注。"""

        available_ids = {record.document_id for record in chunks}
        annotated_ids = {
            document_id
            for case in cases
            for document_id in (
                *case.expected_document_ids,
                *case.document_scope,
            )
        }
        if annotated_ids.difference(available_ids):
            raise ValueError("知识问答评估标注包含不存在或未就绪的文档")

    async def _evaluate_case(
        self,
        case: KnowledgeQaEvaluationCase,
        *,
        k: int,
    ) -> KnowledgeQaCaseReport:
        """执行单条评估，并隐藏内部异常正文与路径。"""

        started = time.perf_counter()
        try:
            return await self._execute_case(case, k=k, started=started)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            return KnowledgeQaCaseReport(
                case_id=case.case_id,
                execution_status="failed",
                error_code=self._error_code(exc),
                ranked_document_ids=(),
                selected_chunk_ids=(),
                matched_fact_ids=(),
                missing_fact_ids=tuple(
                    requirement.fact_id for requirement in case.required_evidence
                ),
                predicted_answerable=None,
                answer_status=None,
                citation_integrity=None,
                predicted_action=None,
                predicted_reason_code=None,
                rewrite_count=0,
                rewritten_query=None,
                option_ids=(),
                non_answer_evidence_leak=None,
                vector_status="not_run",
                latency_ms=(time.perf_counter() - started) * 1000,
            )

    async def _execute_case(
        self,
        case: KnowledgeQaEvaluationCase,
        *,
        k: int,
        started: float,
    ) -> KnowledgeQaCaseReport:
        """依次执行检索、回查、证据选择与服务级引用保护。"""

        self._gate.reset()
        retrieval = await self._search.search(
            case.question,
            limit=k,
            document_ids=case.document_scope,
        )
        if retrieval.diagnostics.vector_status != "skipped":
            raise RuntimeError("离线知识问答评估禁止执行向量阶段")
        records = await asyncio.to_thread(
            self._repository.get_chunks_by_ids,
            tuple(hit.chunk_id for hit in retrieval.hits),
        )
        records_by_id = {record.chunk_id: record for record in records}
        retrieval_integrity = all(
            (record := records_by_id.get(hit.chunk_id)) is not None
            and record.content_hash == hit.content_hash
            for hit in retrieval.hits
        )
        evidence = self._selector.select(
            retrieval,
            records,
            scope_document_ids=case.document_scope,
        )
        evidence = self._selector.select_direct_support(case.question, evidence)
        answer = await self._service.ask(
            case.question,
            limit=k,
            document_ids=case.document_scope,
            prepared_query=case.question,
        )
        if answer.diagnostics.vector_status != "skipped":
            raise RuntimeError("离线知识问答评估禁止执行向量阶段")
        decisions = tuple(self._gate.decisions)
        if not decisions:
            raise RuntimeError("知识问答未形成证据门控动作")
        first_decision = decisions[0]
        final_decision = decisions[-1]
        rewrite_count = sum(
            decision.action == "rewrite" for decision in decisions
        )
        if rewrite_count:
            evidence = await asyncio.to_thread(
                self._repository.get_chunks_by_ids,
                tuple(citation.chunk_id for citation in answer.citations),
            )

        ranked_document_ids = self._ranked_document_ids(
            retrieval,
            records_by_id,
        )
        matched_fact_ids, missing_fact_ids = self._match_facts(case, evidence)
        predicted_answerable = answer.status in {"success", "degraded"}
        citation_integrity = (
            self._citations_match_repository(answer)
            if rewrite_count
            else retrieval_integrity
            and self._citations_are_valid(
                answer,
                retrieval,
                evidence,
                records_by_id,
            )
        )
        non_answer_evidence_leak = (
            final_decision.action in {"ask", "select", "refuse"}
            and bool(answer.citations or answer.images)
        )
        return KnowledgeQaCaseReport(
            case_id=case.case_id,
            execution_status="executed",
            error_code=None,
            ranked_document_ids=ranked_document_ids,
            selected_chunk_ids=tuple(record.chunk_id for record in evidence),
            matched_fact_ids=matched_fact_ids,
            missing_fact_ids=missing_fact_ids,
            predicted_answerable=predicted_answerable,
            answer_status=answer.status,
            citation_integrity=citation_integrity,
            predicted_action=first_decision.action,
            predicted_reason_code=first_decision.reason_code,
            rewrite_count=rewrite_count,
            rewritten_query=first_decision.rewritten_query,
            option_ids=tuple(
                option.option_id for option in first_decision.options
            ),
            non_answer_evidence_leak=non_answer_evidence_leak,
            vector_status=retrieval.diagnostics.vector_status,
            latency_ms=(time.perf_counter() - started) * 1000,
        )

    @staticmethod
    def _ranked_document_ids(
        retrieval: KnowledgeSearchResult,
        records_by_id: Mapping[str, KnowledgeChunkRecord],
    ) -> tuple[str, ...]:
        """按 Chunk 排名生成顺序去重的文档排名。"""

        ranked: list[str] = []
        seen: set[str] = set()
        for hit in retrieval.hits:
            record = records_by_id.get(hit.chunk_id)
            if record is not None and record.document_id not in seen:
                ranked.append(record.document_id)
                seen.add(record.document_id)
        return tuple(ranked)

    @staticmethod
    def _match_facts(
        case: KnowledgeQaEvaluationCase,
        evidence: Sequence[KnowledgeChunkRecord],
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """用稳定正文锚点判断当前证据是否覆盖必需事实。"""

        combined_content = "\n".join(record.content for record in evidence)
        matched: list[str] = []
        missing: list[str] = []
        for requirement in case.required_evidence:
            target = (
                matched
                if any(anchor in combined_content for anchor in requirement.any_of)
                else missing
            )
            target.append(requirement.fact_id)
        return tuple(matched), tuple(missing)

    @staticmethod
    def _citations_are_valid(
        answer: KnowledgeAnswerResult,
        retrieval: KnowledgeSearchResult,
        evidence: Sequence[KnowledgeChunkRecord],
        records_by_id: Mapping[str, KnowledgeChunkRecord],
    ) -> bool:
        """验证公开引用能够回到同一检索快照和已选证据。"""

        evidence_ids = {record.chunk_id for record in evidence}
        hit_hashes = {hit.chunk_id: hit.content_hash for hit in retrieval.hits}
        for citation in answer.citations:
            record = records_by_id.get(citation.chunk_id)
            if (
                record is None
                or citation.chunk_id not in evidence_ids
                or citation.document_id != record.document_id
                or citation.title != record.title
                or citation.heading_path != record.heading_path
                or hit_hashes.get(citation.chunk_id) != record.content_hash
            ):
                return False
        return True

    def _citations_match_repository(self, answer: KnowledgeAnswerResult) -> bool:
        """重写后直接按 SQLite 事实回查公开引用。"""

        records = self._repository.get_chunks_by_ids(
            tuple(citation.chunk_id for citation in answer.citations)
        )
        records_by_id = {record.chunk_id: record for record in records}
        return all(
            (record := records_by_id.get(citation.chunk_id)) is not None
            and citation.document_id == record.document_id
            and citation.title == record.title
            and citation.heading_path == record.heading_path
            for citation in answer.citations
        )

    @staticmethod
    def _error_code(exc: Exception) -> str:
        """把内部异常类型映射为不含正文的稳定错误码。"""

        if isinstance(exc, RuntimeError):
            return "runtime_error"
        if isinstance(exc, ValueError):
            return "validation_error"
        if isinstance(exc, OSError):
            return "io_error"
        return "unexpected_error"

    def _build_report(
        self,
        cases: Sequence[KnowledgeQaEvaluationCase],
        case_reports: Sequence[KnowledgeQaCaseReport],
        *,
        k: int,
    ) -> KnowledgeQaEvaluationReport:
        """只用成功执行样本构造总体、标签与工程指标。"""

        reports = tuple(case_reports)
        executed_ids = {
            report.case_id
            for report in reports
            if report.execution_status == "executed"
        }
        executed_cases = tuple(
            case for case in cases if case.case_id in executed_ids
        )
        reports_by_id = {report.case_id: report for report in reports}
        overall = (
            self._metric_snapshot(executed_cases, reports_by_id, k=k)
            if executed_cases
            else None
        )
        action_metrics = (
            self._action_metrics(executed_cases, reports_by_id)
            if executed_cases
            else None
        )
        tags = sorted({tag for case in executed_cases for tag in case.tags})
        by_tag = {
            tag: self._metric_snapshot(
                tuple(case for case in executed_cases if tag in case.tags),
                reports_by_id,
                k=k,
            )
            for tag in tags
        }
        latency_samples = tuple(
            report.latency_ms
            for report in reports
            if report.execution_status == "executed"
            and report.latency_ms is not None
        )
        executed_count = len(executed_cases)
        failed_count = sum(
            report.execution_status == "failed" for report in reports
        )
        not_run_count = sum(
            report.execution_status == "not_run" for report in reports
        )
        findings = self._diagnose(
            failed_cases=failed_count,
            overall=overall,
        )
        return KnowledgeQaEvaluationReport(
            total_cases=len(cases),
            executed_cases=executed_count,
            failed_cases=failed_count,
            not_run_cases=not_run_count,
            overall=overall,
            action_metrics=action_metrics,
            by_tag=by_tag,
            latency=(latency_summary(latency_samples) if latency_samples else None),
            cases=reports,
            findings=findings,
            used_external_services=False,
        )

    @staticmethod
    def _action_metrics(
        cases: Sequence[KnowledgeQaEvaluationCase],
        reports_by_id: Mapping[str, KnowledgeQaCaseReport],
    ) -> KnowledgeQaActionMetrics:
        """计算五类动作分类、恢复率和非回答泄漏。"""

        actions = ("answer", "rewrite", "ask", "select", "refuse")
        precision: dict[str, float | None] = {}
        recall: dict[str, float | None] = {}
        f1: dict[str, float | None] = {}
        for action in actions:
            true_positive = sum(
                case.expected_action == action
                and reports_by_id[case.case_id].predicted_action == action
                for case in cases
            )
            predicted_count = sum(
                reports_by_id[case.case_id].predicted_action == action
                for case in cases
            )
            expected_count = sum(
                case.expected_action == action for case in cases
            )
            precision[action] = (
                true_positive / predicted_count if predicted_count else None
            )
            recall[action] = (
                true_positive / expected_count if expected_count else None
            )
            if precision[action] is None or recall[action] is None:
                f1[action] = None
            elif precision[action] + recall[action] == 0.0:
                f1[action] = 0.0
            else:
                f1[action] = (
                    2.0
                    * precision[action]
                    * recall[action]
                    / (precision[action] + recall[action])
                )
        scored_f1 = tuple(value for value in f1.values() if value is not None)
        non_answer_cases = tuple(
            case
            for case in cases
            if case.expected_action in {"ask", "select", "refuse"}
        )
        false_answers = sum(
            reports_by_id[case.case_id].answer_status in {"success", "degraded"}
            for case in non_answer_cases
        )
        rewrite_cases = tuple(
            case for case in cases if case.expected_action == "rewrite"
        )
        rewrite_recoveries = sum(
            reports_by_id[case.case_id].rewrite_count == 1
            and reports_by_id[case.case_id].answer_status in {"success", "degraded"}
            and reports_by_id[case.case_id].rewritten_query is not None
            and all(
                term.casefold()
                in reports_by_id[case.case_id].rewritten_query.casefold()
                for term in case.expected_rewrite_terms
            )
            for case in rewrite_cases
        )
        select_cases = tuple(
            case for case in cases if case.expected_action == "select"
        )
        correct_options = sum(
            reports_by_id[case.case_id].option_ids == case.expected_option_ids
            for case in select_cases
        )
        return KnowledgeQaActionMetrics(
            precision_by_action=precision,
            recall_by_action=recall,
            f1_by_action=f1,
            macro_f1=(sum(scored_f1) / len(scored_f1) if scored_f1 else None),
            false_answer_rate=(
                ViolationSummary(
                    violation_rate=false_answers / len(non_answer_cases),
                    violation_count=false_answers,
                    result_count=len(non_answer_cases),
                )
                if non_answer_cases
                else None
            ),
            rewrite_recovery_rate=(
                SuccessSummary(
                    success_rate=rewrite_recoveries / len(rewrite_cases),
                    success_count=rewrite_recoveries,
                    total_count=len(rewrite_cases),
                )
                if rewrite_cases
                else None
            ),
            selection_option_accuracy=(
                SuccessSummary(
                    success_rate=correct_options / len(select_cases),
                    success_count=correct_options,
                    total_count=len(select_cases),
                )
                if select_cases
                else None
            ),
            non_answer_evidence_leak_count=sum(
                bool(reports_by_id[case.case_id].non_answer_evidence_leak)
                for case in cases
            ),
        )

    @staticmethod
    def _metric_snapshot(
        cases: Sequence[KnowledgeQaEvaluationCase],
        reports_by_id: Mapping[str, KnowledgeQaCaseReport],
        *,
        k: int,
    ) -> KnowledgeQaMetricSnapshot:
        """复用同一套纯函数计算任意样本分组的指标。"""

        metric_cases = tuple(
            EvaluationCase(
                query_id=case.case_id,
                relevance={
                    document_id: 3.0
                    for document_id in case.expected_document_ids
                },
            )
            for case in cases
        )
        rankings = {
            case.case_id: reports_by_id[case.case_id].ranked_document_ids
            for case in cases
        }
        fact_matches = tuple(
            tuple(
                requirement.fact_id
                in reports_by_id[case.case_id].matched_fact_ids
                for requirement in case.required_evidence
            )
            for case in cases
        )
        expected_answerability = tuple(case.answerable for case in cases)
        predicted_answerability = tuple(
            bool(reports_by_id[case.case_id].predicted_answerable)
            for case in cases
        )
        citation_outcomes = tuple(
            bool(reports_by_id[case.case_id].citation_integrity)
            for case in cases
        )
        citation_success_count = sum(citation_outcomes)
        return KnowledgeQaMetricSnapshot(
            document_hit_at_k=hit_rate_at_k(metric_cases, rankings, k=k),
            document_recall_at_k=recall_at_k(metric_cases, rankings, k=k),
            document_mrr_at_k=mrr_at_k(metric_cases, rankings, k=k),
            fact_coverage=required_fact_coverage(fact_matches),
            answerability_accuracy=answerability_accuracy(
                expected_answerability,
                predicted_answerability,
            ),
            false_support_rate=false_support_rate(
                expected_answerability,
                predicted_answerability,
            ),
            citation_integrity=SuccessSummary(
                success_rate=citation_success_count / len(citation_outcomes),
                success_count=citation_success_count,
                total_count=len(citation_outcomes),
            ),
        )

    @staticmethod
    def _diagnose(
        *,
        failed_cases: int,
        overall: KnowledgeQaMetricSnapshot | None,
    ) -> tuple[DiagnosticFinding, ...]:
        """根据分层指标生成确定性排查提示，不充当通过阈值。"""

        findings: list[DiagnosticFinding] = []
        if failed_cases:
            findings.append(
                DiagnosticFinding("case_failure", "high", "存在执行失败样本。")
            )
        if overall is None:
            return tuple(findings)
        if overall.citation_integrity.success_rate < 1.0:
            findings.append(
                DiagnosticFinding(
                    "citation_integrity",
                    "high",
                    "引用事实完整性未达到 100%。",
                )
            )
        if overall.false_support_rate.violation_rate > 0.0:
            findings.append(
                DiagnosticFinding(
                    "false_support",
                    "high",
                    "不可回答问题存在错误支持。",
                )
            )
        if (
            overall.document_hit_at_k.evaluated_queries
            and overall.document_hit_at_k.value < 0.8
        ):
            findings.append(
                DiagnosticFinding(
                    "low_retrieval_hit",
                    "warning",
                    "优先检查召回覆盖。",
                )
            )
        elif (
            overall.document_mrr_at_k.evaluated_queries
            and overall.document_mrr_at_k.value < 0.65
        ):
            findings.append(
                DiagnosticFinding(
                    "low_mrr",
                    "warning",
                    "正确文档存在但排名靠后。",
                )
            )
        if (
            overall.fact_coverage.evaluated_queries
            and overall.fact_coverage.value < 0.75
        ):
            findings.append(
                DiagnosticFinding(
                    "low_fact_coverage",
                    "warning",
                    "优先检查切片和证据预算。",
                )
            )
        return tuple(findings)


async def evaluate_knowledge_qa(
    path: str | Path = DEFAULT_KNOWLEDGE_QA_CASES_PATH,
    *,
    repository_path: str | Path | None = None,
    k: int = 5,
) -> KnowledgeQaEvaluationReport:
    """装配无外部服务的真实 SQLite 与 BM25 并运行固定评估。"""

    cases = load_knowledge_qa_cases(path)
    evaluator = KnowledgeQaEvaluator(
        repository=SQLiteKnowledgeRepository(repository_path),
        search=InMemoryKnowledgeSearch(),
    )
    try:
        return await evaluator.evaluate(cases, k=k)
    finally:
        await evaluator.aclose()


def report_to_dict(report: KnowledgeQaEvaluationReport) -> dict[str, Any]:
    """把不可变报告显式转换为可安全输出的 JSON 数据。"""

    converted = _to_json_value(report)
    if not isinstance(converted, dict):
        raise ValueError("知识问答评估报告无法转换为 JSON 对象")
    return converted


def _to_json_value(value: Any) -> Any:
    """递归转换 dataclass、映射和只读集合，不展开生产证据正文。"""

    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: _to_json_value(getattr(value, item.name))
            for item in fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): _to_json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_to_json_value(item) for item in value]
    return value


async def _main() -> None:
    """运行默认固定基线并向标准输出打印安全 JSON。"""

    report = await evaluate_knowledge_qa()
    print(json.dumps(report_to_dict(report), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(_main())
