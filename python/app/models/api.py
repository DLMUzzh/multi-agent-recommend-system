"""FastAPI 请求、响应、健康检查和安全错误契约。"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, field_validator

from app.models.common import _StrictModel
from app.models.conversation import (
    ConversationCompressionInfo,
    ConversationSessionStatus,
    ConversationSessionType,
    ConversationTurn,
)
from app.models.intent import ArbitrationAction, IntentState
from app.models.knowledge_qa import (
    KnowledgeCitation,
    KnowledgeExecutionTrace,
    KnowledgeImageCitation,
)


class ChatRequest(_StrictModel):
    """聊天 HTTP 接口的请求契约。"""

    user_id: str = Field(min_length=1)
    session_id: str | None = Field(default=None, min_length=1)
    message: str = Field(min_length=1, max_length=2000)

    @field_validator("user_id", "session_id", "message")
    @classmethod
    def validate_non_blank_text(cls, value: str | None) -> str | None:
        """拒绝只包含空白字符的公开请求字段。"""

        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("字段不能为空")
        return normalized


class PublicRecommendationContext(_StrictModel):
    """允许通过 HTTP 返回的当前统一检索查询和数量。"""

    query: str | None = Field(default=None, max_length=500)
    size: int = Field(ge=1, le=10)


class SessionHistoryResponse(_StrictModel):
    """只读会话接口返回的近期历史、摘要和公开推荐条件。"""

    session_id: str
    user_id: str
    session_type: ConversationSessionType = "main"
    parent_session_id: str | None = None
    active_child_session_id: str | None = None
    focus_document_id: str | None = None
    focus_document_title: str | None = None
    session_status: ConversationSessionStatus = "active"
    intent_state: IntentState = IntentState.RECOMMENDATION
    history: list[ConversationTurn] = Field(default_factory=list)
    active_context: PublicRecommendationContext | None = None
    turn_count: int = Field(ge=0)
    compression: ConversationCompressionInfo = Field(
        default_factory=ConversationCompressionInfo
    )


class PublicArticleRecommendation(_StrictModel):
    """允许通过 HTTP 返回的最小真实文档推荐。"""

    document_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    excerpt: str = Field(min_length=1, max_length=1200)
    score: float = Field(ge=0.0, le=1.0)
    reason: str = Field(min_length=1, max_length=200)


DegradedComponent = Literal[
    "intent_recognition",
    "user_profile",
    "document_recall",
    "document_rerank",
    "recommendation_query_rewrite",
    "knowledge_query_analysis",
    "knowledge_planner",
    "knowledge_plan_execution",
    "knowledge_plan_coverage",
    "knowledge_vector",
    "knowledge_answer",
]


class SimilarDocumentResponse(_StrictModel):
    """独立相似文章推荐接口的安全响应。"""

    source_document_id: str = Field(min_length=1)
    recommendations: list[PublicArticleRecommendation] = Field(
        default_factory=list,
        max_length=5,
    )
    degraded: bool = False
    degraded_components: list[DegradedComponent] = Field(default_factory=list)


class ChatResponse(_StrictModel):
    """聊天 HTTP 接口的安全响应契约。"""

    session_id: str
    session_type: ConversationSessionType = "main"
    parent_session_id: str | None = None
    active_child_session_id: str | None = None
    focus_document_id: str | None = None
    focus_document_title: str | None = None
    session_status: ConversationSessionStatus = "active"
    message: str
    action: ArbitrationAction
    intent_state: IntentState = IntentState.RECOMMENDATION
    active_context: PublicRecommendationContext | None = None
    recommendations: list[PublicArticleRecommendation] = Field(default_factory=list)
    citations: list[KnowledgeCitation] = Field(default_factory=list)
    images: list[KnowledgeImageCitation] = Field(default_factory=list)
    execution_trace: KnowledgeExecutionTrace | None = None
    degraded: bool = False
    degraded_components: list[DegradedComponent] = Field(default_factory=list)
    compression: ConversationCompressionInfo = Field(
        default_factory=ConversationCompressionInfo
    )


class HealthResponse(_StrictModel):
    """不暴露模型地址或密钥的统一健康检查响应。"""

    status: Literal["healthy"] = "healthy"
    version: str
    llm_configured: bool
    embedding_configured: bool


class ErrorDetail(_StrictModel):
    """公开错误码和中文安全说明。"""

    code: Literal[
        "USER_NOT_FOUND",
        "DOCUMENT_NOT_FOUND",
        "VALIDATION_ERROR",
        "SERVICE_UNAVAILABLE",
    ]
    message: str


class ErrorResponse(_StrictModel):
    """HTTP 接口统一错误响应。"""

    error: ErrorDetail


class ChatStreamProcessEvent(_StrictModel):
    """聊天流中逐阶段返回的安全业务执行事件。"""

    event: Literal["process"] = "process"
    trace_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    sequence: int = Field(ge=1)
    elapsed_ms: float = Field(ge=0.0)
    stage: str = Field(min_length=1, max_length=80)
    component: str = Field(min_length=1, max_length=100)
    status: Literal["started", "success", "degraded", "failed", "skipped"]
    title: str = Field(min_length=1, max_length=120)
    summary: str = Field(default="", max_length=500)
    details: dict[str, Any] = Field(default_factory=dict)


class ChatStreamResultEvent(_StrictModel):
    """聊天流完成时返回的唯一最终业务响应。"""

    event: Literal["result"] = "result"
    trace_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    sequence: int = Field(ge=1)
    elapsed_ms: float = Field(ge=0.0)
    response: ChatResponse


class ChatStreamErrorEvent(_StrictModel):
    """聊天流启动后发生失败时返回的安全终止事件。"""

    event: Literal["error"] = "error"
    trace_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    sequence: int = Field(ge=1)
    elapsed_ms: float = Field(ge=0.0)
    error: ErrorDetail


__all__ = [
    "ChatRequest",
    "ChatResponse",
    "ChatStreamErrorEvent",
    "ChatStreamProcessEvent",
    "ChatStreamResultEvent",
    "DegradedComponent",
    "ErrorDetail",
    "ErrorResponse",
    "HealthResponse",
    "PublicArticleRecommendation",
    "PublicRecommendationContext",
    "SessionHistoryResponse",
    "SimilarDocumentResponse",
]
