"""知识问答链和无会话 HTTP 入口的严格数据契约。"""

from __future__ import annotations

import math
import re
from typing import Literal

from pydantic import Field, field_validator, model_validator

from app.models.common import _StrictModel
from app.models.document import (
    DocumentContentType,
    DocumentDifficulty,
    ImageMimeType,
    ImageStatus,
)


class KnowledgeChunkRecord(_StrictModel):
    """从 SQLite 一致读取、可用于检索和证据回查的知识 Chunk。"""

    chunk_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    topics: list[str] = Field(min_length=1, max_length=20)
    content_type: DocumentContentType
    difficulty: DocumentDifficulty
    author_id: str = Field(min_length=1)
    position: int = Field(ge=0)
    heading_path: tuple[str, ...] = ()
    content: str = Field(min_length=1)
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    token_count: int = Field(ge=1)


RetrievalChannelStatus = Literal["executed", "degraded", "skipped"]
KnowledgeRetrievalMode = Literal["bm25", "hybrid"]
KnowledgeAnswerStatus = Literal[
    "success",
    "degraded",
    "insufficient_evidence",
    "needs_clarification",
]
KnowledgeAnswerAbstainReason = Literal[
    "insufficient_evidence",
    "conflicting_evidence",
    "unsupported_scope",
]
KnowledgeQuestionType = Literal[
    "factual",
    "comparative",
    "procedural",
    "analytical",
    "exploratory",
    "verification",
    "summarization",
]
KnowledgeRetrievalStrategy = Literal["direct", "decomposed"]
KnowledgeReasoningStrategy = Literal[
    "comparison_matrix",
    "facet_analysis",
    "coverage_synthesis",
]
KnowledgePlanFacet = Literal[
    "subject",
    "definition",
    "mechanism",
    "cause",
    "impact",
    "constraint",
    "tradeoff",
    "scenario",
    "alternative",
    "comparison",
    "example",
]
KnowledgePlanStepStatus = Literal["covered", "weak", "uncovered", "failed"]
KnowledgePlanReasonCode = Literal[
    "enough_evidence",
    "no_hits",
    "scope_filtered",
    "stale_candidates",
    "insufficient_subject_coverage",
    "search_failed",
]
KnowledgePlanSupportLevel = Literal["direct", "partial", "none"]
KnowledgePlanDecision = Literal["answer", "replan", "insufficient_evidence"]

_REASONING_STRATEGY_BY_QUESTION_TYPE: dict[str, KnowledgeReasoningStrategy] = {
    "comparative": "comparison_matrix",
    "analytical": "facet_analysis",
    "exploratory": "coverage_synthesis",
}
_STEP_ID_PATTERN = re.compile(r"^step-[1-9]\d{0,2}$")


def _normalize_bounded_text(
    value: object,
    *,
    field_name: str,
    max_length: int,
) -> str:
    """统一受控计划文本的空白并拒绝危险尖括号。"""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name}不能为空")
    normalized = " ".join(value.split())
    if len(normalized) > max_length:
        raise ValueError(f"{field_name}长度不能超过 {max_length} 个字符")
    if "<" in normalized or ">" in normalized:
        raise ValueError(f"{field_name}不能包含危险尖括号")
    return normalized


def _normalize_unique_text_tuple(
    value: object,
    *,
    field_name: str,
    min_length: int,
    max_length: int,
    item_max_length: int,
) -> tuple[str, ...]:
    """清理有界文本序列并按大小写不敏感规则拒绝重复。"""

    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field_name}必须是列表或元组")
    normalized: list[str] = []
    seen: set[str] = set()
    for item in value:
        cleaned = _normalize_bounded_text(
            item,
            field_name=field_name,
            max_length=item_max_length,
        )
        key = cleaned.casefold()
        if key in seen:
            raise ValueError(f"{field_name}不能重复")
        normalized.append(cleaned)
        seen.add(key)
    if not min_length <= len(normalized) <= max_length:
        raise ValueError(
            f"{field_name}数量必须在 {min_length} 到 {max_length} 之间"
        )
    return tuple(normalized)


def _normalize_step_id_tuple(value: object, *, field_name: str) -> tuple[str, ...]:
    """校验计划修订关系引用的步骤 ID。"""

    normalized = _normalize_unique_text_tuple(
        value,
        field_name=field_name,
        min_length=0,
        max_length=5,
        item_max_length=8,
    )
    if any(_STEP_ID_PATTERN.fullmatch(step_id) is None for step_id in normalized):
        raise ValueError(f"{field_name}包含非法步骤 ID")
    return normalized


class KnowledgePlanStep(_StrictModel):
    """Planner 可生成且 Executor 可执行的单个受控检索步骤。"""

    step_id: str = Field(pattern=r"^step-[1-9]\d{0,2}$")
    facet: KnowledgePlanFacet
    query: str = Field(min_length=1, max_length=500)
    target_subjects: tuple[str, ...] = Field(min_length=1, max_length=3)
    required: bool

    @field_validator("query", mode="before")
    @classmethod
    def normalize_query(cls, value: object) -> str:
        """统一检索查询并拒绝危险文本。"""

        return _normalize_bounded_text(
            value,
            field_name="计划查询",
            max_length=500,
        )

    @field_validator("target_subjects", mode="before")
    @classmethod
    def normalize_target_subjects(cls, value: object) -> tuple[str, ...]:
        """目标对象必须非空、有界且互不重复。"""

        return _normalize_unique_text_tuple(
            value,
            field_name="计划目标对象",
            min_length=1,
            max_length=3,
            item_max_length=200,
        )


class KnowledgeReasoningPlan(_StrictModel):
    """复杂知识问题的首版或唯一一次修订计划。"""

    revision: Literal[1, 2]
    question_type: Literal["comparative", "analytical", "exploratory"]
    strategy: KnowledgeReasoningStrategy
    steps: tuple[KnowledgePlanStep, ...] = Field(min_length=2, max_length=5)
    kept_step_ids: tuple[str, ...] = Field(default=(), max_length=5)
    replaced_step_ids: tuple[str, ...] = Field(default=(), max_length=5)
    confidence: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)

    @field_validator("kept_step_ids", mode="before")
    @classmethod
    def normalize_kept_step_ids(cls, value: object) -> tuple[str, ...]:
        """校验修订版原样保留的步骤 ID。"""

        return _normalize_step_id_tuple(value, field_name="保留步骤 ID")

    @field_validator("replaced_step_ids", mode="before")
    @classmethod
    def normalize_replaced_step_ids(cls, value: object) -> tuple[str, ...]:
        """校验修订版声明替换的前版步骤 ID。"""

        return _normalize_step_id_tuple(value, field_name="替换步骤 ID")

    @model_validator(mode="after")
    def validate_plan_contract(self) -> KnowledgeReasoningPlan:
        """保证策略、步骤和修订关系不会形成矛盾计划。"""

        expected_strategy = _REASONING_STRATEGY_BY_QUESTION_TYPE[
            self.question_type
        ]
        if self.strategy != expected_strategy:
            raise ValueError("知识推理策略与问题类型不匹配")
        step_ids = tuple(step.step_id for step in self.steps)
        if len(step_ids) != len(set(step_ids)):
            raise ValueError("知识推理计划的步骤 ID 不能重复")
        for step in self.steps:
            _normalize_bounded_text(
                step.query,
                field_name="计划查询",
                max_length=500,
            )
            _normalize_unique_text_tuple(
                step.target_subjects,
                field_name="计划目标对象",
                min_length=1,
                max_length=3,
                item_max_length=200,
            )
        if self.revision == 1:
            if self.kept_step_ids or self.replaced_step_ids:
                raise ValueError("首版计划不能声明保留或替换步骤")
        else:
            kept_ids = set(self.kept_step_ids)
            replaced_ids = set(self.replaced_step_ids)
            if kept_ids & replaced_ids:
                raise ValueError("保留步骤与替换步骤不能重叠")
            if not kept_ids.issubset(step_ids):
                raise ValueError("保留步骤必须出现在当前修订计划中")
        if self.question_type == "comparative":
            self._validate_comparison_matrix()
        return self

    def _validate_comparison_matrix(self) -> None:
        """比较计划必须分别覆盖两个对象及一个共同维度。"""

        subject_steps = [
            step
            for step in self.steps
            if step.required and len(step.target_subjects) == 1
        ]
        distinct_subjects = {
            step.target_subjects[0].casefold() for step in subject_steps
        }
        if len(distinct_subjects) < 2:
            raise ValueError("比较计划必须包含两个对象的独立必选取证步骤")
        has_common_dimension = any(
            step.required
            and step.facet == "comparison"
            and len(
                {
                    subject.casefold()
                    for subject in step.target_subjects
                }
                & distinct_subjects
            )
            >= 2
            for step in self.steps
        )
        if not has_common_dimension:
            raise ValueError("比较计划必须包含同时覆盖两个对象的共同维度步骤")


class KnowledgePlanCandidateRelation(_StrictModel):
    """计划重排前的步骤与候选 Chunk 确定性关系。"""

    step_id: str = Field(pattern=r"^step-[1-9]\d{0,2}$")
    chunk_id: str = Field(min_length=1)
    deterministic_score: float = Field(
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
    )


class KnowledgePlanEvidenceRelation(_StrictModel):
    """计划重排后供 Coverage Checker 使用的受控证据关系。"""

    step_id: str = Field(pattern=r"^step-[1-9]\d{0,2}$")
    chunk_id: str = Field(min_length=1)
    support_level: KnowledgePlanSupportLevel
    score: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)


class KnowledgePlanStepResult(_StrictModel):
    """单个计划步骤经覆盖检查后的安全结果。"""

    step_id: str = Field(pattern=r"^step-[1-9]\d{0,2}$")
    status: KnowledgePlanStepStatus
    search_query: str = Field(min_length=1, max_length=500)
    selected_chunk_ids: tuple[str, ...] = Field(default=(), max_length=6)
    selected_document_ids: tuple[str, ...] = Field(default=(), max_length=6)
    reason_code: KnowledgePlanReasonCode

    @field_validator("search_query", mode="before")
    @classmethod
    def normalize_search_query(cls, value: object) -> str:
        """公开步骤查询沿用计划查询的文本保护。"""

        return _normalize_bounded_text(
            value,
            field_name="步骤检索查询",
            max_length=500,
        )

    @field_validator("selected_chunk_ids", "selected_document_ids", mode="before")
    @classmethod
    def normalize_selected_ids(cls, value: object) -> tuple[str, ...]:
        """最终选中 ID 必须非空、去重并保持有界。"""

        return _normalize_unique_text_tuple(
            value,
            field_name="步骤选中证据 ID",
            min_length=0,
            max_length=6,
            item_max_length=200,
        )

    @model_validator(mode="after")
    def validate_status_reason(self) -> KnowledgePlanStepResult:
        """步骤状态只能搭配预定义的确定性原因码。"""

        expected_reason_by_status: dict[
            KnowledgePlanStepStatus,
            set[KnowledgePlanReasonCode],
        ] = {
            "covered": {"enough_evidence"},
            "weak": {"insufficient_subject_coverage"},
            "failed": {"search_failed"},
            "uncovered": {"no_hits", "scope_filtered", "stale_candidates"},
        }
        if self.reason_code not in expected_reason_by_status[self.status]:
            raise ValueError("计划步骤状态与原因码不一致")
        return self


class KnowledgePlanCoverage(_StrictModel):
    """确定性 Coverage Checker 的有界汇总结果。"""

    step_results: tuple[KnowledgePlanStepResult, ...] = Field(
        min_length=2,
        max_length=5,
    )
    required_steps: int = Field(ge=0, le=5)
    covered_required_steps: int = Field(ge=0, le=5)
    covered_steps: int = Field(ge=0, le=5)
    coverage_ratio: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    replanned: bool = False
    decision: KnowledgePlanDecision

    @model_validator(mode="after")
    def validate_coverage_counts(self) -> KnowledgePlanCoverage:
        """覆盖率只计算完整 covered 步骤并与计数保持一致。"""

        step_ids = tuple(result.step_id for result in self.step_results)
        if len(step_ids) != len(set(step_ids)):
            raise ValueError("覆盖结果的步骤 ID 不能重复")
        actual_covered = sum(
            result.status == "covered" for result in self.step_results
        )
        if self.covered_steps != actual_covered:
            raise ValueError("已覆盖步骤计数与步骤结果不一致")
        expected_ratio = actual_covered / len(self.step_results)
        if not math.isclose(
            self.coverage_ratio,
            expected_ratio,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError("覆盖率必须只按 covered 步骤计算")
        if self.required_steps > len(self.step_results):
            raise ValueError("必选步骤数不能超过计划步骤总数")
        if self.covered_required_steps > self.required_steps:
            raise ValueError("已覆盖必选步骤数不能超过必选步骤数")
        if self.covered_required_steps > self.covered_steps:
            raise ValueError("已覆盖必选步骤不能超过全部已覆盖步骤")
        if (
            self.decision == "answer"
            and self.covered_required_steps != self.required_steps
        ):
            raise ValueError("仍有必选步骤未覆盖时不能生成答案")
        return self


class KnowledgePlanTraceStep(_StrictModel):
    """公开执行轨迹中的单版计划步骤白名单。"""

    revision: Literal[1, 2]
    step_id: str = Field(pattern=r"^step-[1-9]\d{0,2}$")
    facet: KnowledgePlanFacet
    query: str = Field(min_length=1, max_length=500)
    required: bool
    status: KnowledgePlanStepStatus
    reason_code: KnowledgePlanReasonCode
    selected_chunk_ids: tuple[str, ...] = Field(default=(), max_length=6)

    @field_validator("query", mode="before")
    @classmethod
    def normalize_query(cls, value: object) -> str:
        """公开轨迹只保留清理后的安全查询。"""

        return _normalize_bounded_text(
            value,
            field_name="轨迹步骤查询",
            max_length=500,
        )

    @field_validator("selected_chunk_ids", mode="before")
    @classmethod
    def normalize_selected_chunk_ids(cls, value: object) -> tuple[str, ...]:
        """轨迹中的 Chunk ID 必须去重且最多六个。"""

        return _normalize_unique_text_tuple(
            value,
            field_name="轨迹选中 Chunk ID",
            min_length=0,
            max_length=6,
            item_max_length=200,
        )

    @model_validator(mode="after")
    def validate_status_reason(self) -> KnowledgePlanTraceStep:
        """轨迹状态与原因码复用步骤结果的严格组合。"""

        KnowledgePlanStepResult(
            step_id=self.step_id,
            status=self.status,
            search_query=self.query,
            selected_chunk_ids=self.selected_chunk_ids,
            reason_code=self.reason_code,
        )
        return self


class KnowledgeSearchHit(_StrictModel):
    """派生索引返回的 Chunk 排名，不携带可直接生成答案的正文。"""

    chunk_id: str = Field(min_length=1)
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    score: float = Field(ge=0.0)
    bm25_rank: int | None = Field(default=None, ge=1)
    vector_rank: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_rank(self) -> KnowledgeSearchHit:
        """至少一个检索通道必须实际命中当前 Chunk。"""

        if self.bm25_rank is None and self.vector_rank is None:
            raise ValueError("知识检索命中必须包含至少一个通道排名")
        return self


class KnowledgeRetrievalDiagnostics(_StrictModel):
    """公开检索通道状态，不包含查询向量或内部异常。"""

    bm25_status: RetrievalChannelStatus = "executed"
    vector_status: RetrievalChannelStatus = "skipped"


class KnowledgeSearchResult(_StrictModel):
    """一次知识检索的融合命中和安全诊断。"""

    hits: tuple[KnowledgeSearchHit, ...] = ()
    mode: KnowledgeRetrievalMode = "bm25"
    diagnostics: KnowledgeRetrievalDiagnostics = Field(
        default_factory=KnowledgeRetrievalDiagnostics
    )


class KnowledgeGeneratedAnswer(_StrictModel):
    """回答 Agent 经程序校验后的内部输出。"""

    outcome: Literal["answer", "abstain"] = "answer"
    answer: str = Field(min_length=1)
    cited_chunk_ids: tuple[str, ...] = ()
    cited_image_ids: tuple[str, ...] = ()
    abstain_reason: KnowledgeAnswerAbstainReason | None = None
    degraded: bool = False

    @model_validator(mode="after")
    def validate_outcome(self) -> KnowledgeGeneratedAnswer:
        """正式答案与主动拒答必须保持引用、原因和降级语义互斥。"""

        has_citations = bool(self.cited_chunk_ids or self.cited_image_ids)
        if self.outcome == "answer":
            if self.abstain_reason is not None:
                raise ValueError("正式知识答案不能携带主动拒答原因")
            if not has_citations:
                raise ValueError("知识回答必须引用文本或图片证据")
            return self
        if self.abstain_reason is None:
            raise ValueError("知识主动拒答必须提供受控原因")
        if has_citations:
            raise ValueError("知识主动拒答不能携带引用")
        if self.degraded:
            raise ValueError("知识主动拒答不能标记为模型故障降级")
        return self


class KnowledgeQueryAnalysis(_StrictModel):
    """知识问答内部的一次性改写、分类与有界检索计划。"""

    standalone_query: str = Field(min_length=1, max_length=500)
    uses_history: bool = False
    question_type: KnowledgeQuestionType
    strategy: KnowledgeRetrievalStrategy
    sub_queries: tuple[str, ...] = Field(default=(), max_length=3)
    retry_query: str | None = Field(default=None, max_length=500)
    missing_information: tuple[str, ...] = Field(default=(), max_length=3)
    clarification_question: str | None = Field(default=None, max_length=500)
    confidence: float = Field(ge=0.0, le=1.0)
    degraded: bool = False

    @field_validator("standalone_query", mode="before")
    @classmethod
    def normalize_standalone_query(cls, value: object) -> str:
        """拒绝空查询，并统一规划阶段使用的空白。"""

        if not isinstance(value, str) or not value.strip():
            raise ValueError("知识规划查询不能为空")
        return " ".join(value.split())

    @field_validator("sub_queries", mode="before")
    @classmethod
    def normalize_sub_queries(cls, value: object) -> tuple[str, ...]:
        """清理并保护最多三个去重子查询。"""

        if value is None:
            return ()
        if not isinstance(value, (list, tuple)):
            raise ValueError("知识规划子查询必须是列表或元组")
        normalized: list[str] = []
        seen: set[str] = set()
        for item in value:
            if not isinstance(item, str) or not item.strip():
                raise ValueError("知识规划子查询不能为空")
            cleaned = " ".join(item.split())
            if len(cleaned) > 500:
                raise ValueError("知识规划子查询长度不能超过 500 个字符")
            key = cleaned.casefold()
            if key in seen:
                raise ValueError("知识规划子查询不能重复")
            normalized.append(cleaned)
            seen.add(key)
        return tuple(normalized)

    @field_validator("retry_query", mode="before")
    @classmethod
    def normalize_retry_query(cls, value: object) -> str | None:
        """校验同一次分析预生成的唯一安全重试查询。"""

        if value is None:
            return None
        return _normalize_bounded_text(
            value,
            field_name="知识重试查询",
            max_length=500,
        )

    @field_validator("missing_information", mode="before")
    @classmethod
    def normalize_missing_information(cls, value: object) -> tuple[str, ...]:
        """缺少的必要条件最多三个且不能重复。"""

        return _normalize_unique_text_tuple(
            value,
            field_name="知识查询必要信息",
            min_length=0,
            max_length=3,
            item_max_length=100,
        )

    @field_validator("clarification_question", mode="before")
    @classmethod
    def normalize_clarification_question(cls, value: object) -> str | None:
        """只接受有必要信息缺口时使用的单个安全问题。"""

        if value is None:
            return None
        return _normalize_bounded_text(
            value,
            field_name="知识澄清问题",
            max_length=500,
        )

    @model_validator(mode="after")
    def validate_strategy_payload(self) -> KnowledgeQueryAnalysis:
        """保证直接与分解策略不会携带矛盾负载。"""

        if (
            self.retry_query is not None
            and self.retry_query.casefold() == self.standalone_query.casefold()
        ):
            raise ValueError("知识重试查询必须不同于独立查询")
        if self.missing_information:
            if self.clarification_question is None:
                raise ValueError("缺少必要信息时必须提供澄清问题")
            if self.retry_query is not None:
                raise ValueError("缺少必要信息时不能同时提供重试查询")
        elif self.clarification_question is not None:
            raise ValueError("没有必要信息缺口时不能提供澄清问题")

        if self.question_type == "comparative" and self.strategy != "decomposed":
            raise ValueError("比较问题必须使用分解检索计划")
        if (
            self.question_type
            in {"factual", "procedural", "verification", "summarization"}
            and self.strategy != "direct"
        ):
            raise ValueError("简单知识问题必须使用直接检索计划")
        if self.strategy == "direct":
            if self.sub_queries:
                raise ValueError("直接检索计划不能携带子查询")
            return self
        if not 2 <= len(self.sub_queries) <= 3:
            raise ValueError("分解检索计划必须包含 2 到 3 个子查询")
        return self


class KnowledgeDocumentEvidence(_StrictModel):
    """回答前按文档聚合的有界 Chunk 证据包。"""

    document_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    score: float = Field(ge=0.0)
    chunks: tuple[KnowledgeChunkRecord, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_document_membership(self) -> KnowledgeDocumentEvidence:
        """证据包内的 Chunk 必须来自同一篇文档并按位置排列。"""

        if any(
            chunk.document_id != self.document_id or chunk.title != self.title
            for chunk in self.chunks
        ):
            raise ValueError("文档证据包包含其他文档的 Chunk")
        positions = tuple(chunk.position for chunk in self.chunks)
        if positions != tuple(sorted(positions)):
            raise ValueError("文档证据包中的 Chunk 必须按原文位置排列")
        return self


class KnowledgeScopeResolution(_StrictModel):
    """当前轮知识检索范围，不进入持久会话。"""

    document_ids: tuple[str, ...] = ()
    needs_clarification: bool = False
    clarification_question: str | None = None
    candidate_document_ids: tuple[str, ...] = Field(default=(), max_length=5)

    @field_validator("candidate_document_ids", mode="before")
    @classmethod
    def normalize_candidate_document_ids(cls, value: object) -> tuple[str, ...]:
        """候选文档必须是来自当前快照的 2 到 5 个唯一 ID。"""

        return _normalize_unique_text_tuple(
            value,
            field_name="候选文档 ID",
            min_length=0,
            max_length=5,
            item_max_length=200,
        )

    @model_validator(mode="after")
    def validate_clarification(self) -> KnowledgeScopeResolution:
        """澄清结果不能同时携带范围，且必须提供安全问题。"""

        if self.needs_clarification:
            if self.document_ids or not self.clarification_question:
                raise ValueError("知识范围澄清结果无效")
            if self.candidate_document_ids and not 2 <= len(
                self.candidate_document_ids
            ) <= 5:
                raise ValueError("知识范围候选必须包含 2 到 5 篇文档")
        elif (
            self.clarification_question is not None
            or self.candidate_document_ids
        ):
            raise ValueError("非澄清结果不能携带澄清问题或候选文档")
        return self


class KnowledgeCitation(_StrictModel):
    """可公开展示且能够回到 SQLite 原文的知识引用。"""

    citation_id: str = Field(pattern=r"^[1-9]\d*$")
    document_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    chunk_id: str = Field(min_length=1)
    heading_path: tuple[str, ...] = ()
    excerpt: str = Field(min_length=1)


class KnowledgeImageReference(_StrictModel):
    """文档导入后公开返回的安全图片声明。"""

    image_id: str = Field(pattern=r"^img-[0-9a-f]{32}$")
    image_key: str = Field(
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$"
    )
    caption: str = Field(min_length=1, max_length=500)
    status: ImageStatus


class KnowledgeImageEvidence(_StrictModel):
    """只提供给回答链的图片白名单，不包含存储位置。"""

    image_id: str = Field(pattern=r"^img-[0-9a-f]{32}$")
    document_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    image_key: str = Field(
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$"
    )
    heading_path: tuple[str, ...] = ()
    caption: str = Field(min_length=1, max_length=500)
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    linked_chunk_ids: tuple[str, ...] = Field(min_length=1)


class KnowledgeImageCitation(_StrictModel):
    """知识答案公开返回的可读取图片引用。"""

    citation_id: str = Field(pattern=r"^图[1-9]\d*$")
    image_id: str = Field(pattern=r"^img-[0-9a-f]{32}$")
    document_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    heading_path: tuple[str, ...] = ()
    caption: str = Field(min_length=1, max_length=500)
    url: str = Field(
        pattern=(
            r"^/api/v1/knowledge/images/img-[0-9a-f]{32}"
            r"\?v=[0-9a-f]{12}$"
        )
    )


class KnowledgeImageUploadResult(_StrictModel):
    """图片上传成功后的安全元数据。"""

    image_id: str = Field(pattern=r"^img-[0-9a-f]{32}$")
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    mime_type: ImageMimeType
    byte_size: int = Field(ge=1, le=8 * 1024 * 1024)


class KnowledgeExecutionInput(_StrictModel):
    """测试诊断公开的有界请求输入摘要。"""

    history_message_count: int = Field(default=0, ge=0, le=12)
    has_conversation_summary: bool = False
    prepared_query: bool = False
    requested_document_ids: tuple[str, ...] = Field(default=(), max_length=20)


class KnowledgeExecutionChunk(_StrictModel):
    """测试诊断公开的单个 Chunk 召回摘要。"""

    rank: int = Field(ge=1, le=20)
    chunk_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    title: str = Field(min_length=1, max_length=500)
    heading_path: tuple[str, ...] = ()
    score: float = Field(ge=0.0)
    bm25_rank: int | None = Field(default=None, ge=1)
    vector_rank: int | None = Field(default=None, ge=1)
    selected: bool = False
    excerpt: str = Field(min_length=1, max_length=220)


class KnowledgeExecutionDocument(_StrictModel):
    """测试诊断公开的文档级回答证据摘要。"""

    document_id: str = Field(min_length=1)
    title: str = Field(min_length=1, max_length=500)
    score: float = Field(ge=0.0)
    retrieved_chunk_ids: tuple[str, ...] = Field(default=(), max_length=20)
    selected_chunk_ids: tuple[str, ...] = Field(default=(), max_length=20)


class KnowledgeExecutionResult(_StrictModel):
    """测试诊断公开的最终结果统计。"""

    status: KnowledgeAnswerStatus
    citation_count: int = Field(default=0, ge=0)
    image_count: int = Field(default=0, ge=0)
    elapsed_ms: float = Field(ge=0.0)
    degraded_components: tuple[str, ...] = ()


class KnowledgeExecutionTrace(_StrictModel):
    """面向测试人员的安全执行摘要，不包含模型私有推理。"""

    trace_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    route: Literal["knowledge_qa"] = "knowledge_qa"
    request_route: Literal[
        "/api/v1/knowledge/ask",
        "/api/v1/chat",
    ]
    question: str = Field(min_length=1, max_length=4000)
    input: KnowledgeExecutionInput = Field(default_factory=KnowledgeExecutionInput)
    standalone_query: str = Field(min_length=1, max_length=500)
    uses_history: bool = False
    question_type: KnowledgeQuestionType = "factual"
    strategy: KnowledgeRetrievalStrategy = "direct"
    sub_queries: tuple[str, ...] = Field(default=(), max_length=3)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    search_queries: tuple[str, ...] = Field(default=(), max_length=10)
    reasoning_strategy: KnowledgeReasoningStrategy | None = None
    plan_revision_count: int = Field(default=0, ge=0, le=2)
    plan_steps: tuple[KnowledgePlanTraceStep, ...] = Field(
        default=(),
        max_length=10,
    )
    coverage: KnowledgePlanCoverage | None = None
    retrieved_chunks: tuple[KnowledgeExecutionChunk, ...] = Field(
        default=(),
        max_length=20,
    )
    documents: tuple[KnowledgeExecutionDocument, ...] = Field(
        default=(),
        max_length=20,
    )
    retrieval_mode: KnowledgeRetrievalMode = "bm25"
    diagnostics: KnowledgeRetrievalDiagnostics = Field(
        default_factory=KnowledgeRetrievalDiagnostics
    )
    result: KnowledgeExecutionResult


class KnowledgeAnswerResult(_StrictModel):
    """无会话单轮问答服务的公开结果。"""

    status: KnowledgeAnswerStatus
    answer: str = Field(min_length=1)
    citations: tuple[KnowledgeCitation, ...] = ()
    images: tuple[KnowledgeImageCitation, ...] = ()
    retrieval_mode: KnowledgeRetrievalMode = "bm25"
    diagnostics: KnowledgeRetrievalDiagnostics = Field(
        default_factory=KnowledgeRetrievalDiagnostics
    )
    degraded_components: tuple[str, ...] = ()
    execution_trace: KnowledgeExecutionTrace | None = None
    resolved_document_ids: tuple[str, ...] = Field(default=(), exclude=True)
    resolved_document_titles: tuple[str, ...] = Field(default=(), exclude=True)


class KnowledgeDocumentIngestResult(_StrictModel):
    """独立文档导入接口的安全结果。"""

    document_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    chunk_count: int = Field(ge=1)
    images: tuple[KnowledgeImageReference, ...] = ()


class KnowledgeDocumentIngestRequest(_StrictModel):
    """无会话 HTTP 接口接收的完整 Markdown 文档。"""

    document_id: str = Field(min_length=1, max_length=200)
    title: str = Field(min_length=1, max_length=500)
    content_markdown: str = Field(min_length=1)
    topics: list[str] = Field(min_length=1, max_length=20)
    content_type: DocumentContentType
    difficulty: DocumentDifficulty
    author_id: str = Field(min_length=1, max_length=200)

    @field_validator("topics")
    @classmethod
    def normalize_topics(cls, values: list[str]) -> list[str]:
        """按展示顺序保留主题并执行大小写无关去重。"""

        normalized: list[str] = []
        seen: set[str] = set()
        for value in values:
            if not isinstance(value, str) or not value.strip():
                raise ValueError("文档主题不能为空")
            cleaned = " ".join(value.split())
            key = cleaned.casefold()
            if key not in seen:
                normalized.append(cleaned)
                seen.add(key)
        if not normalized:
            raise ValueError("文档主题不能为空")
        return normalized


class KnowledgeAskRequest(_StrictModel):
    """无会话单轮知识问答请求。"""

    question: str = Field(min_length=1, max_length=4000)
    limit: int = Field(default=5, ge=1, le=20)


class KnowledgeHealthResponse(_StrictModel):
    """兼容原知识入口且不暴露敏感配置的健康状态。"""

    status: Literal["healthy"] = "healthy"
    version: str = Field(min_length=1)
    llm_configured: bool
    embedding_configured: bool


__all__ = [
    "KnowledgeChunkRecord",
    "KnowledgeAnswerResult",
    "KnowledgeAnswerAbstainReason",
    "KnowledgeAnswerStatus",
    "KnowledgeCitation",
    "KnowledgeAskRequest",
    "KnowledgeDocumentIngestRequest",
    "KnowledgeDocumentIngestResult",
    "KnowledgeDocumentEvidence",
    "KnowledgeExecutionChunk",
    "KnowledgeExecutionDocument",
    "KnowledgeExecutionInput",
    "KnowledgeExecutionResult",
    "KnowledgeExecutionTrace",
    "KnowledgeGeneratedAnswer",
    "KnowledgeImageCitation",
    "KnowledgeImageEvidence",
    "KnowledgeImageReference",
    "KnowledgeImageUploadResult",
    "KnowledgePlanCoverage",
    "KnowledgePlanDecision",
    "KnowledgePlanFacet",
    "KnowledgePlanReasonCode",
    "KnowledgePlanStep",
    "KnowledgePlanStepResult",
    "KnowledgePlanStepStatus",
    "KnowledgePlanSupportLevel",
    "KnowledgePlanTraceStep",
    "KnowledgeQueryAnalysis",
    "KnowledgeReasoningPlan",
    "KnowledgeReasoningStrategy",
    "KnowledgeHealthResponse",
    "KnowledgeRetrievalDiagnostics",
    "KnowledgeRetrievalMode",
    "KnowledgeSearchHit",
    "KnowledgeSearchResult",
    "KnowledgeScopeResolution",
    "RetrievalChannelStatus",
]
