"""单轮会话图的状态、合并规则和提交结果。"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from pydantic import BaseModel, ConfigDict, Field

from app.models.schemas import (
    ArbitrationDecision,
    ConversationReply,
    ConversationTurn,
    DocumentRecallResult,
    DocumentRerankResult,
    IntentState,
    IntentRecognition,
    RankedDocument,
    RecommendationContext,
    UserProfile,
)
from app.models.intent_memory import UserIntentMemoryProjection
from app.models.interaction_memory import UserInteractionMemoryProjection
from app.models.knowledge_qa import KnowledgeAnswerResult
from app.models.personal_feedback import (
    ConversationFeedbackContext,
    ConversationResultSnapshotDraft,
    FeedbackAnalysis,
    FeedbackDecision,
)


def _merge_agent_statuses(
    current: dict[str, str] | None,
    update: dict[str, str] | None,
) -> dict[str, str]:
    """合并各节点产生的 Agent 状态增量。"""

    merged = dict(current or {})
    merged.update(update or {})
    return merged


class ConversationGraphResult(BaseModel):
    """供 ``ConversationService`` 一次性提交的无副作用结果。"""

    model_config = ConfigDict(extra="forbid")

    reply: ConversationReply
    history_message: str
    pending_context: RecommendationContext | None = None
    commit_context: bool = False
    pending_intent_state: IntentState | None = None
    commit_intent_state: bool = False
    knowledge_document_ids: tuple[str, ...] = ()
    knowledge_document_titles: tuple[str, ...] = ()
    error_stage: str | None = None
    feedback_analysis: FeedbackAnalysis | None = None
    feedback_decision: FeedbackDecision | None = None
    feedback_recovery_succeeded: bool | None = None
    result_snapshot_draft: ConversationResultSnapshotDraft | None = None
    trace: list[str] = Field(default_factory=list)


class ConversationGraphState(TypedDict, total=False):
    """单轮 LangGraph 节点之间传递的增量状态。"""

    user_id: str
    session_id: str
    message: str
    history: list[ConversationTurn]
    conversation_summary: str | None
    previous_context: RecommendationContext | None
    current_intent_state: IntentState
    intent_memory: UserIntentMemoryProjection | None
    interaction_memory: UserInteractionMemoryProjection | None
    knowledge_document_ids: tuple[str, ...]
    feedback_context: ConversationFeedbackContext | None

    recognition: IntentRecognition
    decision: ArbitrationDecision
    profile_result: Any
    user_profile: UserProfile | None
    document_recall_result: DocumentRecallResult
    effective_context: RecommendationContext
    retrieval_query: str
    document_rerank_result: DocumentRerankResult
    final_documents: list[RankedDocument]
    knowledge_result: KnowledgeAnswerResult
    knowledge_document_titles: tuple[str, ...]
    feedback_analysis: FeedbackAnalysis | None
    feedback_decision: FeedbackDecision | None
    feedback_recovery_succeeded: bool | None
    result_snapshot_draft: ConversationResultSnapshotDraft | None
    agent_statuses: Annotated[dict[str, str], _merge_agent_statuses]

    error_stage: str
    error_type: str
    reply: ConversationReply
    history_message: str
    pending_context: RecommendationContext | None
    commit_context: bool
    pending_intent_state: IntentState | None
    commit_intent_state: bool
    trace: Annotated[list[str], operator.add]


__all__ = [
    "ConversationGraphResult",
    "ConversationGraphState",
]
