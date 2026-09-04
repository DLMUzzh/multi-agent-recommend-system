"""使用 SQLite Chunk 统一召回运行固定推荐评估。"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.agents.document_recall_agent import DocumentRecallAgent
from app.agents.document_rerank_agent import DocumentRerankAgent
from app.config import Settings
from app.domain.services.document_result_aggregator import DocumentResultAggregator
from app.infrastructure.database.sqlite.knowledge_repository import (
    SQLiteKnowledgeRepository,
)
from app.infrastructure.retrieval.knowledge_search import InMemoryKnowledgeSearch
from assess.metrics import (
    hit_rate_at_k,
    latency_summary,
    mrr_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    violation_rate,
)
from assess.models import (
    EvaluationCase,
    LatencySummary,
    MetricResult,
    ViolationSummary,
)


DEFAULT_EVALUATION_CASES_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "evaluation_cases.json"
)


class _StoredPipelineCase(BaseModel):
    """磁盘中的最小统一召回评估契约。"""

    model_config = ConfigDict(extra="forbid")

    query_id: str = Field(min_length=1)
    query: str = Field(min_length=1, max_length=500)
    size: int = Field(default=5, ge=1, le=10)
    relevance: dict[str, float]
    seen_document_ids: list[str] = Field(default_factory=list)


class _EvaluationSettings(Settings):
    """只接受评估代码显式参数，不读取环境或本地密钥。"""

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[Settings],
        init_settings: object,
        env_settings: object,
        dotenv_settings: object,
        file_secret_settings: object,
    ) -> tuple[object, ...]:
        del cls, settings_cls, env_settings, dotenv_settings, file_secret_settings
        return (init_settings,)


@dataclass(frozen=True)
class PipelineEvaluationCase:
    """单条 SQLite 文档推荐评估输入。"""

    query_id: str
    query: str
    size: int
    relevance: Mapping[str, float]
    seen_document_ids: tuple[str, ...]


@dataclass(frozen=True)
class PipelineVariantReport:
    """单个统一召回变体的排名质量、边界、延迟和降级统计。"""

    rankings: Mapping[str, tuple[str, ...]]
    hit_at_k: MetricResult
    recall_at_k: MetricResult
    precision_at_k: MetricResult
    mrr_at_k: MetricResult
    ndcg_at_k: MetricResult
    violation_rate: ViolationSummary
    latency: LatencySummary
    llm_call_count: int
    degraded_query_count: int


@dataclass(frozen=True)
class PipelineEvaluationReport:
    """统一 Chunk 召回和可选文档重排的离线评估结果。"""

    evaluated_queries: int
    recall: PipelineVariantReport
    rerank: PipelineVariantReport
    recall_rankings: Mapping[str, tuple[str, ...]]
    rerank_rankings: Mapping[str, tuple[str, ...]]
    retrieval_modes: Mapping[str, str]
    vector_statuses: Mapping[str, str]
    used_real_pipeline_components: bool = True
    used_external_services: bool = False


@dataclass(frozen=True)
class _EvaluationRuntime:
    """保存一次评估共享的真实组件。"""

    cases: tuple[PipelineEvaluationCase, ...]
    recall_agent: DocumentRecallAgent
    rerank_agent: DocumentRerankAgent
    aggregator: DocumentResultAggregator


@dataclass
class _EvaluationState:
    """累计统一召回评估的请求级结果。"""

    recall_rankings: dict[str, tuple[str, ...]]
    rerank_rankings: dict[str, tuple[str, ...]]
    retrieval_modes: dict[str, str]
    vector_statuses: dict[str, str]
    recall_latencies: list[float]
    rerank_latencies: list[float]
    llm_call_count: int = 0
    degraded_query_count: int = 0

    @classmethod
    def empty(cls) -> _EvaluationState:
        """创建不共享可变容器的一次性累计状态。"""

        return cls({}, {}, {}, {}, [], [])


def load_pipeline_cases(
    path: str | Path = DEFAULT_EVALUATION_CASES_PATH,
) -> tuple[PipelineEvaluationCase, ...]:
    """读取并严格校验 SQLite 文档推荐固定标注。"""

    case_path = Path(path)
    try:
        payload = json.loads(case_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ValueError("推荐链评估标注文件不存在") from None
    except (json.JSONDecodeError, OSError, UnicodeError):
        raise ValueError("推荐链评估标注文件无法安全读取") from None
    if not isinstance(payload, list) or not payload:
        raise ValueError("推荐链评估标注必须是非空数组")

    cases: list[PipelineEvaluationCase] = []
    seen_query_ids: set[str] = set()
    for index, raw_case in enumerate(payload, start=1):
        try:
            stored = _StoredPipelineCase.model_validate(raw_case)
            metric_case = EvaluationCase(
                query_id=stored.query_id,
                relevance=stored.relevance,
                forbidden_ids=frozenset(stored.seen_document_ids),
            )
            seen_ids = _normalized_ids(stored.seen_document_ids)
        except (TypeError, ValueError, ValidationError):
            raise ValueError(f"推荐链评估标注第 {index} 条无效") from None
        if stored.query_id in seen_query_ids:
            raise ValueError("推荐链评估 query_id 不能重复")
        seen_query_ids.add(stored.query_id)
        cases.append(
            PipelineEvaluationCase(
                query_id=stored.query_id,
                query=" ".join(stored.query.split()),
                size=stored.size,
                relevance=metric_case.relevance,
                seen_document_ids=seen_ids,
            )
        )
    return tuple(cases)


async def evaluate_pipeline(
    path: str | Path = DEFAULT_EVALUATION_CASES_PATH,
    *,
    k: int = 5,
    rerank_llm: Any | None = None,
) -> PipelineEvaluationReport:
    """运行本地 SQLite BM25、文档聚合和可选 Fake LLM 重排。"""

    if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= 10:
        raise ValueError("k 必须是 1 到 10 的整数")
    cases = load_pipeline_cases(path)
    runtime, search = await _build_runtime(cases, rerank_llm=rerank_llm)
    try:
        state = await _evaluate_cases(runtime, k=k)
    finally:
        await search.aclose()
    return _build_report(runtime.cases, state, k=k)


async def _build_runtime(
    cases: tuple[PipelineEvaluationCase, ...],
    *,
    rerank_llm: Any | None,
) -> tuple[_EvaluationRuntime, InMemoryKnowledgeSearch]:
    """构造只读 SQLite、BM25 和文档重排评估运行时。"""

    repository = SQLiteKnowledgeRepository()
    search = InMemoryKnowledgeSearch()
    await search.refresh(repository.list_ready_chunks())
    return (
        _EvaluationRuntime(
            cases=cases,
            recall_agent=DocumentRecallAgent(
                repository=repository,
                search=search,
            ),
            rerank_agent=DocumentRerankAgent(
                llm=rerank_llm,
                enable_llm=False,
                settings=_EvaluationSettings(),
            ),
            aggregator=DocumentResultAggregator(),
        ),
        search,
    )


async def _evaluate_cases(
    runtime: _EvaluationRuntime,
    *,
    k: int,
) -> _EvaluationState:
    """逐查询执行召回、证据重排和确定性聚合。"""

    state = _EvaluationState.empty()
    for case in runtime.cases:
        result_size = min(max(case.size, k), 10)
        recall_started = time.perf_counter()
        recall = await runtime.recall_agent.run(
            query=case.query,
            size=result_size,
            seen_document_ids=case.seen_document_ids,
        )
        if not recall.success:
            raise ValueError(f"评估查询 {case.query_id} 的文档召回失败")
        state.recall_rankings[case.query_id] = tuple(
            candidate.document_id for candidate in recall.candidates
        )
        state.recall_latencies.append(
            (time.perf_counter() - recall_started) * 1000
        )
        state.retrieval_modes[case.query_id] = recall.retrieval_mode
        state.vector_statuses[case.query_id] = (
            recall.retrieval_diagnostics.vector_status
        )

        rerank_started = time.perf_counter()
        reranked = await runtime.rerank_agent.run(
            query=case.query,
            candidates=recall.candidates,
        )
        if not reranked.success:
            raise ValueError(f"评估查询 {case.query_id} 的文档重排失败")
        aggregated = runtime.aggregator.aggregate(
            candidates=recall.candidates,
            ranked_documents=reranked.ranked_documents,
            seen_document_ids=case.seen_document_ids,
            size=result_size,
        )
        state.rerank_rankings[case.query_id] = tuple(
            document.document_id for document in aggregated
        )
        state.rerank_latencies.append(
            (time.perf_counter() - rerank_started) * 1000
        )
        state.llm_call_count += int(reranked.data.get("llm_call_count", 0))
        state.degraded_query_count += int(reranked.degraded_reason is not None)
    return state


def _build_report(
    cases: tuple[PipelineEvaluationCase, ...],
    state: _EvaluationState,
    *,
    k: int,
) -> PipelineEvaluationReport:
    """冻结一次评估结果并计算统一排名指标。"""

    metric_cases = [
        EvaluationCase(
            query_id=case.query_id,
            relevance=case.relevance,
            forbidden_ids=frozenset(case.seen_document_ids),
        )
        for case in cases
    ]
    frozen_recall = MappingProxyType(dict(state.recall_rankings))
    frozen_rerank = MappingProxyType(dict(state.rerank_rankings))
    return PipelineEvaluationReport(
        evaluated_queries=len(cases),
        recall=_variant_report(
            metric_cases,
            state.recall_rankings,
            state.recall_latencies,
            llm_call_count=0,
            degraded_query_count=0,
            k=k,
        ),
        rerank=_variant_report(
            metric_cases,
            state.rerank_rankings,
            state.rerank_latencies,
            llm_call_count=state.llm_call_count,
            degraded_query_count=state.degraded_query_count,
            k=k,
        ),
        recall_rankings=frozen_recall,
        rerank_rankings=frozen_rerank,
        retrieval_modes=MappingProxyType(dict(state.retrieval_modes)),
        vector_statuses=MappingProxyType(dict(state.vector_statuses)),
    )


def _variant_report(
    cases: list[EvaluationCase],
    rankings: dict[str, tuple[str, ...]],
    latencies: list[float],
    *,
    llm_call_count: int,
    degraded_query_count: int,
    k: int,
) -> PipelineVariantReport:
    frozen = MappingProxyType(dict(rankings))
    return PipelineVariantReport(
        rankings=frozen,
        hit_at_k=hit_rate_at_k(cases, frozen, k=k),
        recall_at_k=recall_at_k(cases, frozen, k=k),
        precision_at_k=precision_at_k(cases, frozen, k=k),
        mrr_at_k=mrr_at_k(cases, frozen, k=k),
        ndcg_at_k=ndcg_at_k(cases, frozen, k=k),
        violation_rate=violation_rate(cases, frozen, k=k),
        latency=latency_summary(latencies),
        llm_call_count=llm_call_count,
        degraded_query_count=degraded_query_count,
    )


def _normalized_ids(values: list[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("已展示文档 ID 不能为空")
        cleaned = value.strip()
        if cleaned in seen:
            raise ValueError("已展示文档 ID 不能重复")
        normalized.append(cleaned)
        seen.add(cleaned)
    return tuple(normalized)


__all__ = [
    "DEFAULT_EVALUATION_CASES_PATH",
    "PipelineEvaluationCase",
    "PipelineEvaluationReport",
    "PipelineVariantReport",
    "evaluate_pipeline",
    "load_pipeline_cases",
]
