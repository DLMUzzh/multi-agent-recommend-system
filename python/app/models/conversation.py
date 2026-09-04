"""会话持久化、压缩和内部轮次回复契约。"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field, field_validator, model_validator

from app.models.article import DocumentRecommendation
from app.models.common import AgentResult, _StrictModel
from app.models.intent import (
    ArbitrationAction,
    IntentState,
    RecognitionSource,
    RecommendationContext,
)
from app.models.knowledge_qa import (
    KnowledgeCitation,
    KnowledgeExecutionTrace,
    KnowledgeImageCitation,
)


ConversationSessionType = Literal["main", "article_qa"]
ConversationSessionStatus = Literal["active", "suspended", "closed"]
ConversationMessageType = Literal["chat", "child_handoff"]


class ConversationTurn(_StrictModel):
    """持久会话中的单条用户或助手消息。"""

    role: Literal["user", "assistant"]
    content: str
    message_id: int | None = Field(default=None, ge=1)
    sequence_no: int | None = Field(default=None, ge=0)
    created_at: datetime | None = None
    message_type: ConversationMessageType = "chat"
    related_session_id: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def validate_message_relation(self) -> ConversationTurn:
        """交接消息必须指向子会话，普通消息不得携带关联会话。"""

        if self.message_type == "child_handoff":
            if self.role != "assistant" or self.related_session_id is None:
                raise ValueError("子会话交接消息必须由助手发出并关联子会话")
        elif self.related_session_id is not None:
            raise ValueError("普通会话消息不能关联子会话")
        if self.created_at is not None and (
            self.created_at.tzinfo is None
            or self.created_at.utcoffset() is None
        ):
            raise ValueError("会话消息时间必须包含时区")
        return self


class LlmConversationSummaryOutput(_StrictModel):
    """会话摘要 LLM 允许返回的来源消息选择结果。"""

    selected_turn_indexes: list[int] = Field(max_length=24)
    user_constraint_indexes: list[int] = Field(default_factory=list, max_length=24)
    unresolved_question_indexes: list[int] = Field(
        default_factory=list,
        max_length=24,
    )

    @field_validator(
        "selected_turn_indexes",
        "user_constraint_indexes",
        "unresolved_question_indexes",
    )
    @classmethod
    def validate_selected_turn_indexes(cls, value: list[int]) -> list[int]:
        """索引必须非负且唯一，具体上界由 Agent 根据输入消息校验。"""

        if any(index < 0 for index in value):
            raise ValueError("会话摘要来源索引不能为负数")
        if len(value) != len(set(value)):
            raise ValueError("会话摘要来源索引不能重复")
        return value

    @model_validator(mode="after")
    def validate_disjoint_index_groups(self) -> LlmConversationSummaryOutput:
        """普通上下文、用户约束和未解决问题不能重复消费同一轮。"""

        all_indexes = [
            *self.selected_turn_indexes,
            *self.user_constraint_indexes,
            *self.unresolved_question_indexes,
        ]
        if len(all_indexes) != len(set(all_indexes)):
            raise ValueError("会话摘要三组来源索引必须互斥")
        return self


class ConversationSession(_StrictModel):
    """按用户与会话 ID 隔离保存的持久状态。"""

    session_id: str
    user_id: str
    session_type: ConversationSessionType = "main"
    parent_session_id: str | None = Field(default=None, min_length=1)
    active_child_session_id: str | None = Field(default=None, min_length=1)
    focus_document_id: str | None = Field(default=None, min_length=1)
    focus_document_title: str | None = Field(default=None, min_length=1)
    session_status: ConversationSessionStatus = "active"
    intent_state: IntentState = IntentState.RECOMMENDATION
    active_context: RecommendationContext | None = None
    history: list[ConversationTurn] = Field(default_factory=list)
    turn_count: int = Field(default=0, ge=0)
    summary: str | None = Field(default=None, max_length=2000)
    summary_watermark: int = Field(default=-1, ge=-1)
    summarized_turn_count: int = Field(default=0, ge=0)
    dropped_turn_count: int = Field(default=0, ge=0)
    unresolved_questions: list[str] = Field(default_factory=list, max_length=20)
    cited_document_ids: list[str] = Field(default_factory=list, max_length=50)
    handoff_summary: str | None = Field(default=None, max_length=2000)
    pending_feedback_id: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def validate_session_relationship(self) -> ConversationSession:
        """保护主会话和文章问答子会话的身份边界。"""

        if self.session_type == "article_qa":
            if (
                self.parent_session_id is None
                or self.focus_document_id is None
                or self.focus_document_title is None
            ):
                raise ValueError("文章问答子会话必须关联父会话和聚焦文档")
            if self.parent_session_id == self.session_id:
                raise ValueError("文章问答子会话不能把自己作为父会话")
            if self.active_child_session_id is not None:
                raise ValueError("文章问答子会话不能再关联活动子会话")
        elif any(
            value is not None
            for value in (
                self.parent_session_id,
                self.focus_document_id,
                self.focus_document_title,
                self.handoff_summary,
            )
        ):
            raise ValueError("主会话不能携带文章子会话专属字段")
        return self


class ConversationSummaryResult(AgentResult):
    """会话摘要 Agent 的结构化结果。"""

    agent_name: str = "conversation_summary"
    summary: str | None = Field(default=None, max_length=2000)


class ConversationCompressionInfo(_StrictModel):
    """允许通过 HTTP 返回的会话压缩状态。"""

    status: Literal["not_needed", "compressed", "pending"] = "not_needed"
    summary: str | None = Field(default=None, max_length=2000)
    summarized_turn_count: int = Field(default=0, ge=0)
    retained_turn_count: int = Field(default=0, ge=0, le=12)
    dropped_turn_count: int = Field(default=0, ge=0)


class ConversationReply(_StrictModel):
    """会话图返回给应用服务的完整轮次回复。"""

    session_id: str
    session_type: ConversationSessionType = "main"
    parent_session_id: str | None = None
    active_child_session_id: str | None = None
    focus_document_id: str | None = None
    focus_document_title: str | None = None
    session_status: ConversationSessionStatus = "active"
    message: str
    intent_source: RecognitionSource
    action: ArbitrationAction
    intent_state: IntentState = IntentState.RECOMMENDATION
    active_context: RecommendationContext | None = None
    recommendations: list[DocumentRecommendation] = Field(default_factory=list)
    citations: list[KnowledgeCitation] = Field(default_factory=list)
    images: list[KnowledgeImageCitation] = Field(default_factory=list)
    execution_trace: KnowledgeExecutionTrace | None = None
    needs_clarification: bool = False
    agent_statuses: dict[str, str] = Field(default_factory=dict)
    compression: ConversationCompressionInfo = Field(
        default_factory=ConversationCompressionInfo
    )


__all__ = [
    "ConversationMessageType",
    "ConversationCompressionInfo",
    "ConversationReply",
    "ConversationSession",
    "ConversationSessionStatus",
    "ConversationSessionType",
    "ConversationSummaryResult",
    "ConversationTurn",
    "LlmConversationSummaryOutput",
]
