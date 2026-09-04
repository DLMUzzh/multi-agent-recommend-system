"""个人自然语言反馈、结果快照和受控补救的严格内部契约。"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field, field_validator, model_validator

from app.models.common import _StrictModel
from app.models.intent import IntentState
from app.models.knowledge_qa import KnowledgeAnswerStatus


FeedbackType = Literal[
    "recommendation_irrelevant",
    "answer_incorrect",
    "answer_incomplete",
    "article_not_found",
    "answer_style",
    "route_correction",
    "no_feedback",
]
FeedbackNextAction = Literal[
    "normal",
    "clarify",
    "retry_recommendation",
    "retry_retrieval",
    "retry_answer_from_evidence",
]
FeedbackPersistence = Literal[
    "current_recovery_only",
    "long_term_candidate",
    "explicit_long_term",
]
FeedbackMemoryRoute = Literal[
    "interaction_memory",
    "intent_memory",
    "recommendation_profile",
]
FeedbackEventStatus = Literal[
    "classifying",
    "awaiting_detail",
    "recovering",
    "recovered",
    "recovery_failed",
    "closed",
]


def _normalize_optional_text(
    value: object,
    *,
    field_name: str,
    max_length: int,
) -> str | None:
    """清理可空原文，避免空字符串和无界文本进入内部契约。"""

    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field_name}必须是字符串")
    normalized = " ".join(value.split())
    if not normalized:
        return None
    if len(normalized) > max_length:
        raise ValueError(f"{field_name}长度不能超过 {max_length} 个字符")
    return normalized


def _normalize_unique_ids(
    value: object,
    *,
    field_name: str,
    max_length: int,
) -> tuple[str, ...]:
    """清理有界身份序列并拒绝重复或空身份。"""

    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field_name}必须是列表或元组")
    normalized: list[str] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{field_name}不能包含空身份")
        cleaned = item.strip()
        if len(cleaned) > 200:
            raise ValueError(f"{field_name}中的身份长度不能超过 200 个字符")
        if cleaned in seen:
            raise ValueError(f"{field_name}不能重复")
        normalized.append(cleaned)
        seen.add(cleaned)
    if len(normalized) > max_length:
        raise ValueError(f"{field_name}数量不能超过 {max_length}")
    return tuple(normalized)


def _validate_aware_time(value: datetime | None, *, field_name: str) -> None:
    """要求持久化时间显式包含时区。"""

    if value is not None and (value.tzinfo is None or value.utcoffset() is None):
        raise ValueError(f"{field_name}必须包含时区")


class ConversationResultSnapshot(_StrictModel):
    """保存上一轮结果身份，不保存答案或推荐理由正文。"""

    result_id: str = Field(min_length=1, max_length=200)
    user_id: str = Field(min_length=1, max_length=200)
    session_id: str = Field(min_length=1, max_length=200)
    assistant_sequence_no: int = Field(ge=0)
    result_type: Literal["recommendation", "knowledge_answer"]
    query: str | None = Field(default=None, max_length=500)
    recommendation_document_ids: tuple[str, ...] = Field(
        default=(), max_length=10
    )
    citation_document_ids: tuple[str, ...] = Field(default=(), max_length=20)
    citation_chunk_ids: tuple[str, ...] = Field(default=(), max_length=20)
    knowledge_status: KnowledgeAnswerStatus | None = None
    resolved_document_ids: tuple[str, ...] = Field(default=(), max_length=20)
    created_at: datetime
    raw_purged_at: datetime | None = None

    @field_validator("query", mode="before")
    @classmethod
    def normalize_query(cls, value: object) -> str | None:
        """保存补救需要的有界原查询。"""

        return _normalize_optional_text(value, field_name="结果查询", max_length=500)

    @field_validator(
        "recommendation_document_ids",
        "citation_document_ids",
        "citation_chunk_ids",
        "resolved_document_ids",
        mode="before",
    )
    @classmethod
    def normalize_identity_lists(
        cls,
        value: object,
        info: object,
    ) -> tuple[str, ...]:
        """统一结果快照中的文档与 Chunk 身份。"""

        field_name = getattr(info, "field_name", "结果身份")
        limit = 10 if field_name == "recommendation_document_ids" else 20
        return _normalize_unique_ids(
            value,
            field_name=field_name,
            max_length=limit,
        )

    @model_validator(mode="after")
    def validate_snapshot_payload(self) -> ConversationResultSnapshot:
        """保证推荐与知识快照不会混写不相容负载。"""

        _validate_aware_time(self.created_at, field_name="结果创建时间")
        _validate_aware_time(self.raw_purged_at, field_name="结果清理时间")
        if self.result_type == "recommendation":
            if not self.recommendation_document_ids:
                raise ValueError("推荐结果必须包含实际返回的文档身份")
            if (
                self.citation_document_ids
                or self.citation_chunk_ids
                or self.knowledge_status is not None
                or self.resolved_document_ids
            ):
                raise ValueError("推荐结果不能携带知识回答负载")
        else:
            if self.recommendation_document_ids:
                raise ValueError("知识回答结果不能携带推荐文档身份")
            if self.knowledge_status is None:
                raise ValueError("知识回答结果必须包含回答状态")
            if self.knowledge_status in {"success", "degraded"} and not (
                self.citation_chunk_ids
            ):
                raise ValueError("成功知识回答必须包含可信 Chunk 身份")
        if self.raw_purged_at is not None and self.query is not None:
            raise ValueError("结果原文已清理时不能继续保留查询")
        return self


class RecommendationMemorySignal(_StrictModel):
    """质量反馈允许提交给推荐画像的受控负向候选。"""

    target_type: Literal["article", "topic", "difficulty", "author"]
    target_value: str = Field(min_length=1, max_length=200)
    direction: Literal["avoid"] = "avoid"
    source_document_ids: tuple[str, ...] = Field(default=(), max_length=10)
    specific: bool = False
    persistence: FeedbackPersistence = "current_recovery_only"

    @field_validator("target_value", mode="before")
    @classmethod
    def normalize_target_value(cls, value: object) -> str:
        """拒绝空或超长画像目标值。"""

        normalized = _normalize_optional_text(
            value,
            field_name="推荐记忆目标",
            max_length=200,
        )
        if normalized is None:
            raise ValueError("推荐记忆目标不能为空")
        return normalized

    @field_validator("source_document_ids", mode="before")
    @classmethod
    def normalize_source_document_ids(cls, value: object) -> tuple[str, ...]:
        """限制推荐记忆信号只能引用少量文档身份。"""

        return _normalize_unique_ids(
            value,
            field_name="推荐记忆来源文档",
            max_length=10,
        )

    @model_validator(mode="after")
    def validate_specific_signal(self) -> RecommendationMemorySignal:
        """文章信号必须有可审计文档来源。"""

        if self.target_type == "article" and not self.source_document_ids:
            raise ValueError("文章负反馈必须关联来源文档")
        if self.persistence == "explicit_long_term" and not self.specific:
            raise ValueError("明确长期信号必须指向可验证目标")
        return self


class FeedbackAnalysis(_StrictModel):
    """质量反馈 Agent 只能生成的结构化候选。"""

    is_feedback: bool
    feedback_type: FeedbackType
    completeness: Literal["complete", "incomplete"]
    corrected_query: str | None = Field(default=None, max_length=500)
    target_document_ids: tuple[str, ...] = Field(default=(), max_length=10)
    missing_information: tuple[
        Literal["reason", "topic", "article_identity", "correct_fact", "scope"],
        ...,
    ] = Field(default=(), max_length=3)
    suggested_action: FeedbackNextAction
    recommendation_signals: tuple[RecommendationMemorySignal, ...] = Field(
        default=(), max_length=4
    )
    route_target: IntentState | None = None
    reason_code: str = Field(min_length=1, max_length=80)
    confidence: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)

    @field_validator("corrected_query", mode="before")
    @classmethod
    def normalize_corrected_query(cls, value: object) -> str | None:
        """清理候选纠正查询。"""

        return _normalize_optional_text(
            value,
            field_name="纠正查询",
            max_length=500,
        )

    @field_validator("target_document_ids", mode="before")
    @classmethod
    def normalize_target_document_ids(cls, value: object) -> tuple[str, ...]:
        """限制 LLM 只能提出有界文档身份候选。"""

        return _normalize_unique_ids(
            value,
            field_name="反馈目标文档",
            max_length=10,
        )

    @model_validator(mode="after")
    def validate_analysis_shape(self) -> FeedbackAnalysis:
        """拒绝明显矛盾的 Agent 候选。"""

        if not self.is_feedback or self.feedback_type == "no_feedback":
            if self.is_feedback or self.feedback_type != "no_feedback":
                raise ValueError("无反馈判断的标志与类型必须一致")
            if (
                self.suggested_action != "normal"
                or self.corrected_query is not None
                or self.target_document_ids
                or self.recommendation_signals
                or self.route_target is not None
            ):
                raise ValueError("无反馈判断不能携带业务动作或记忆信号")
        if self.completeness == "incomplete" and not self.missing_information:
            raise ValueError("信息不完整时必须说明缺失信息")
        if self.suggested_action == "clarify" and self.completeness != "incomplete":
            raise ValueError("只有信息不完整时才能建议追问")
        return self


class FeedbackDecision(_StrictModel):
    """确定性策略保护后供编排层执行的有限决策。"""

    is_feedback: bool
    feedback_type: FeedbackType
    completeness: Literal["complete", "incomplete"]
    next_action: FeedbackNextAction
    protected_query: str | None = Field(default=None, max_length=500)
    target_document_ids: tuple[str, ...] = Field(default=(), max_length=10)
    excluded_document_ids: tuple[str, ...] = Field(default=(), max_length=10)
    clarification_question: str | None = Field(default=None, max_length=300)
    recommendation_signals: tuple[RecommendationMemorySignal, ...] = Field(
        default=(), max_length=4
    )
    route_target: IntentState | None = None
    memory_routes: tuple[FeedbackMemoryRoute, ...] = Field(
        default=(), max_length=3
    )
    reason_code: str = Field(min_length=1, max_length=80)

    @field_validator("protected_query", mode="before")
    @classmethod
    def normalize_protected_query(cls, value: object) -> str | None:
        """清理最终执行查询。"""

        return _normalize_optional_text(
            value,
            field_name="受保护查询",
            max_length=500,
        )

    @field_validator("clarification_question", mode="before")
    @classmethod
    def normalize_clarification_question(cls, value: object) -> str | None:
        """限制追问文案长度。"""

        return _normalize_optional_text(
            value,
            field_name="反馈追问",
            max_length=300,
        )

    @field_validator("target_document_ids", "excluded_document_ids", mode="before")
    @classmethod
    def normalize_decision_document_ids(
        cls,
        value: object,
        info: object,
    ) -> tuple[str, ...]:
        """限制确定性决策中的文档身份。"""

        return _normalize_unique_ids(
            value,
            field_name=getattr(info, "field_name", "决策文档"),
            max_length=10,
        )

    @field_validator("memory_routes", mode="before")
    @classmethod
    def normalize_memory_routes(cls, value: object) -> tuple[str, ...]:
        """拒绝重复记忆路由。"""

        if not isinstance(value, (list, tuple)):
            raise ValueError("记忆路由必须是列表或元组")
        routes = tuple(value)
        if len(routes) != len(set(routes)):
            raise ValueError("记忆路由不能重复")
        return routes

    @model_validator(mode="after")
    def validate_decision_shape(self) -> FeedbackDecision:
        """保证追问、正常流程和补救动作互不矛盾。"""

        if not self.is_feedback:
            if self.feedback_type != "no_feedback" or self.next_action != "normal":
                raise ValueError("普通消息必须使用无反馈正常决策")
        if self.next_action == "clarify":
            if self.completeness != "incomplete" or self.clarification_question is None:
                raise ValueError("追问决策必须包含缺失信息和追问文案")
        elif self.clarification_question is not None:
            raise ValueError("非追问决策不能携带追问文案")
        if self.next_action in {"retry_recommendation", "retry_retrieval"} and (
            self.protected_query is None
        ):
            raise ValueError("重新推荐或检索必须包含受保护查询")
        return self


class ConversationResultSnapshotDraft(_StrictModel):
    """Graph 生成、由会话服务补全身份和序号的快照草稿。"""

    result_type: Literal["recommendation", "knowledge_answer"]
    query: str | None = Field(default=None, max_length=500)
    recommendation_document_ids: tuple[str, ...] = Field(
        default=(), max_length=10
    )
    citation_document_ids: tuple[str, ...] = Field(default=(), max_length=20)
    citation_chunk_ids: tuple[str, ...] = Field(default=(), max_length=20)
    knowledge_status: KnowledgeAnswerStatus | None = None
    resolved_document_ids: tuple[str, ...] = Field(default=(), max_length=20)


class PersonalFeedbackEvent(_StrictModel):
    """一条个人反馈从分类到补救终态的持久状态。"""

    feedback_id: str = Field(min_length=1, max_length=200)
    user_id: str = Field(min_length=1, max_length=200)
    session_id: str = Field(min_length=1, max_length=200)
    source_result_id: str = Field(min_length=1, max_length=200)
    feedback_message: str | None = Field(default=None, max_length=4000)
    feedback_type: FeedbackType | None = None
    completeness: Literal["complete", "incomplete"] | None = None
    corrected_query: str | None = Field(default=None, max_length=500)
    target_document_ids: tuple[str, ...] = Field(default=(), max_length=10)
    next_action: FeedbackNextAction = "normal"
    status: FeedbackEventStatus
    clarification_count: int = Field(default=0, ge=0, le=1)
    recovery_count: int = Field(default=0, ge=0, le=1)
    recovery_result_id: str | None = Field(default=None, min_length=1, max_length=200)
    recommendation_signals: tuple[RecommendationMemorySignal, ...] = Field(
        default=(), max_length=4
    )
    memory_routes: tuple[FeedbackMemoryRoute, ...] = Field(default=(), max_length=3)
    memory_statuses: dict[
        str,
        Literal["pending", "applied", "degraded", "skipped"],
    ] = Field(default_factory=dict)
    reason_code: str = Field(min_length=1, max_length=80)
    created_at: datetime
    updated_at: datetime
    raw_purged_at: datetime | None = None

    @field_validator("feedback_message", mode="before")
    @classmethod
    def normalize_feedback_message(cls, value: object) -> str | None:
        """限制反馈原文。"""

        return _normalize_optional_text(
            value,
            field_name="反馈原文",
            max_length=4000,
        )

    @field_validator("corrected_query", mode="before")
    @classmethod
    def normalize_event_query(cls, value: object) -> str | None:
        """限制事件中的纠正查询。"""

        return _normalize_optional_text(
            value,
            field_name="事件纠正查询",
            max_length=500,
        )

    @field_validator("target_document_ids", mode="before")
    @classmethod
    def normalize_event_document_ids(cls, value: object) -> tuple[str, ...]:
        """限制事件目标文档身份。"""

        return _normalize_unique_ids(
            value,
            field_name="事件目标文档",
            max_length=10,
        )

    @model_validator(mode="after")
    def validate_event_state(self) -> PersonalFeedbackEvent:
        """保护状态、计数、结果身份和原文清理不变量。"""

        _validate_aware_time(self.created_at, field_name="反馈创建时间")
        _validate_aware_time(self.updated_at, field_name="反馈更新时间")
        _validate_aware_time(self.raw_purged_at, field_name="反馈清理时间")
        if self.updated_at < self.created_at:
            raise ValueError("反馈更新时间不能早于创建时间")
        if self.status == "classifying":
            if (
                self.feedback_type is not None
                or self.completeness is not None
                or self.next_action != "normal"
                or self.clarification_count
                or self.recovery_count
                or self.recommendation_signals
                or self.memory_routes
            ):
                raise ValueError("待分类事件不能提前携带分类、动作或记忆信号")
        elif self.feedback_type is None or self.completeness is None:
            raise ValueError("离开待分类状态后必须保存完整分类")
        if self.status == "awaiting_detail" and self.clarification_count != 1:
            raise ValueError("待补充事件必须且只能追问一次")
        if self.status in {"recovering", "recovered", "recovery_failed"} and (
            self.recovery_count != 1
        ):
            raise ValueError("补救事件必须且只能执行一次")
        if self.status == "recovered" and self.recovery_result_id is None:
            raise ValueError("补救成功事件必须关联修正结果")
        if self.status != "recovered" and self.recovery_result_id is not None:
            raise ValueError("只有补救成功事件能关联修正结果")
        if self.raw_purged_at is not None and (
            self.feedback_message is not None or self.corrected_query is not None
        ):
            raise ValueError("反馈原文已清理时不能继续保留原文")
        return self


class ConversationFeedbackContext(_StrictModel):
    """会话服务传给 Graph 的最小反馈上下文。"""

    latest_result: ConversationResultSnapshot | None = None
    pending_feedback: PersonalFeedbackEvent | None = None

    @model_validator(mode="after")
    def validate_context_identity(self) -> ConversationFeedbackContext:
        """pending 反馈只能引用同一快照、用户和 Session。"""

        if self.pending_feedback is None:
            return self
        if self.latest_result is None:
            raise ValueError("待补充反馈必须关联结果快照")
        if (
            self.pending_feedback.user_id != self.latest_result.user_id
            or self.pending_feedback.session_id != self.latest_result.session_id
            or self.pending_feedback.source_result_id != self.latest_result.result_id
        ):
            raise ValueError("反馈上下文中的用户、Session 或结果身份不一致")
        return self


__all__ = [
    "ConversationFeedbackContext",
    "ConversationResultSnapshot",
    "ConversationResultSnapshotDraft",
    "FeedbackAnalysis",
    "FeedbackDecision",
    "FeedbackEventStatus",
    "FeedbackMemoryRoute",
    "FeedbackNextAction",
    "FeedbackPersistence",
    "FeedbackType",
    "PersonalFeedbackEvent",
    "RecommendationMemorySignal",
]
