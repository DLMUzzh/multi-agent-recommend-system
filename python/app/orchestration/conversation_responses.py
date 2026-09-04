"""会话图的短路、成功、失败响应与展示消息组装。"""

from __future__ import annotations

from typing import Any, Protocol

from app.models.schemas import (
    ArbitrationAction,
    ConversationReply,
    DocumentRecommendation,
    IntentState,
    RankedDocument,
    RecognitionSource,
    RecommendationContext,
)
from app.models.personal_feedback import ConversationResultSnapshotDraft
from app.orchestration.conversation_state import ConversationGraphState


class _ResponseRuntime(Protocol):
    """成功响应组装所需的会话图方法。"""

    def _to_document_recommendations(
        self,
        documents: list[RankedDocument],
    ) -> list[DocumentRecommendation]: ...

    def _unique(self, values: list[str]) -> list[str]: ...

    def _build_document_message(
        self,
        action: ArbitrationAction,
        context: RecommendationContext,
        *,
        item_count: int,
    ) -> str: ...

async def _respond_no_action(
    state: ConversationGraphState,
) -> dict[str, Any]:
    recognition = state["recognition"]
    message = (
        "你好！我可以处理文章推荐和知识问答。你可以说“推荐 Java 入门文章”，"
        "也可以直接询问知识库内容。"
    )
    return {
        "reply": ConversationReply(
            session_id=state["session_id"],
            message=message,
            intent_source=recognition.source,
            action=ArbitrationAction.UNSUPPORTED,
            active_context=state.get("previous_context"),
            agent_statuses=state.get("agent_statuses", {}),
        ),
        "history_message": message,
        "pending_context": None,
        "commit_context": False,
        "trace": ["respond_no_action"],
    }


async def _respond_unknown(
    state: ConversationGraphState,
) -> dict[str, Any]:
    recognition = state["recognition"]
    message = (
        "我还不能唯一判断你是要推荐文章还是询问知识。请直接说明想看的内容，"
        "或把问题补充成可以独立理解的一句话。"
    )
    return {
        "reply": ConversationReply(
            session_id=state["session_id"],
            message=message,
            intent_source=recognition.source,
            action=ArbitrationAction.CLARIFY,
            active_context=state.get("previous_context"),
            needs_clarification=True,
            agent_statuses=state.get("agent_statuses", {}),
        ),
        "history_message": message,
        "pending_context": None,
        "commit_context": False,
        "trace": ["respond_unknown"],
    }


async def _respond_knowledge(
    state: ConversationGraphState,
) -> dict[str, Any]:
    """把知识结果转换为唯一历史、公开引用和独立意图提交。"""

    recognition = state["recognition"]
    result = state["knowledge_result"]
    message = result.answer
    history_message = _knowledge_history_message(message, list(result.citations))
    needs_clarification = result.status == "needs_clarification"
    return {
        "reply": ConversationReply(
            session_id=state["session_id"],
            message=message,
            intent_source=recognition.source,
            action=(
                ArbitrationAction.CLARIFY
                if needs_clarification
                else ArbitrationAction.KNOWLEDGE_ANSWER
            ),
            intent_state=(
                state["current_intent_state"]
                if needs_clarification
                else IntentState.KNOWLEDGE_QA
            ),
            active_context=state.get("previous_context"),
            citations=list(result.citations),
            images=list(result.images),
            execution_trace=result.execution_trace,
            needs_clarification=needs_clarification,
            agent_statuses=state.get("agent_statuses", {}),
        ),
        "history_message": history_message,
        "pending_context": None,
        "commit_context": False,
        "pending_intent_state": (
            None if needs_clarification else IntentState.KNOWLEDGE_QA
        ),
        "commit_intent_state": not needs_clarification,
        "trace": ["respond_knowledge"],
    }


def _knowledge_history_message(message: str, citations: list[Any]) -> str:
    """把 Chunk 引用按文档聚合，使刷新后的历史只显示唯一来源。"""

    if not citations:
        return message
    lines = [message, "", "参考资料："]
    references: dict[str, dict[str, Any]] = {}
    for citation in citations:
        reference = references.setdefault(
            citation.document_id,
            {
                "title": citation.title,
                "citation_ids": [],
                "headings": [],
            },
        )
        if citation.citation_id not in reference["citation_ids"]:
            reference["citation_ids"].append(citation.citation_id)
        heading_parts = list(citation.heading_path)
        if heading_parts and heading_parts[0] == citation.title:
            heading_parts = heading_parts[1:]
        heading = " > ".join(heading_parts)
        if heading and heading not in reference["headings"]:
            reference["headings"].append(heading)
    for reference in references.values():
        marker = ", ".join(reference["citation_ids"])
        headings = "；".join(reference["headings"])
        location = f"（{headings}）" if headings else ""
        lines.append(f"[{marker}] {reference['title']}{location}")
    return "\n".join(lines)


async def _respond_decision(
    state: ConversationGraphState,
) -> dict[str, Any]:
    recognition = state["recognition"]
    decision = state["decision"]
    message = decision.clarification_question or "请再说明一下想看的文章主题。"
    return {
        "reply": ConversationReply(
            session_id=state["session_id"],
            message=message,
            intent_source=recognition.source,
            action=decision.action,
            active_context=state.get("previous_context"),
            needs_clarification=decision.action == ArbitrationAction.CLARIFY,
            agent_statuses=state.get("agent_statuses", {}),
        ),
        "history_message": message,
        "pending_context": None,
        "commit_context": False,
        "trace": ["respond_decision"],
    }


async def _respond_success(
    runtime: _ResponseRuntime,
    state: ConversationGraphState,
) -> dict[str, Any]:
    recognition = state["recognition"]
    decision = state["decision"]
    context = state["effective_context"]
    if decision.context is None:
        return {
            "error_stage": "transition",
            "error_type": "MissingDecisionContext",
            "trace": ["respond_success"],
        }

    recommendations = runtime._to_document_recommendations(
        state["final_documents"]
    )
    persisted_context = context.model_copy(deep=True)
    persisted_context.avoid_seen = False
    persisted_context.seen_article_ids = runtime._unique(
        persisted_context.seen_article_ids
        + [item.document_id for item in recommendations]
    )
    message = runtime._build_document_message(
        decision.action,
        context,
        item_count=len(recommendations),
    )
    history_message = message
    if recommendations:
        history_message += " 推荐结果：" + "；".join(
            item.title for item in recommendations
        )

    return {
        "reply": ConversationReply(
            session_id=state["session_id"],
            message=message,
            intent_source=recognition.source,
            action=decision.action,
            active_context=persisted_context,
            recommendations=recommendations,
            agent_statuses=state.get("agent_statuses", {}),
        ),
        "history_message": history_message,
        "pending_context": persisted_context,
        "commit_context": True,
        "pending_intent_state": IntentState.RECOMMENDATION,
        "commit_intent_state": True,
        "trace": ["respond_success"],
    }


async def _respond_failure(
    state: ConversationGraphState,
) -> dict[str, Any]:
    recognition = state.get("recognition")
    decision = state.get("decision")
    error_stage = state.get("error_stage", "workflow")
    recommendation_stages = {
        "document_recall",
        "document_rerank",
        "aggregation",
    }
    message = (
        "文章推荐暂时失败，请稍后重试。"
        if error_stage in recommendation_stages
        else "对话处理暂时失败，请稍后重试。"
    )
    return {
        "reply": ConversationReply(
            session_id=state["session_id"],
            message=message,
            intent_source=(
                recognition.source if recognition else RecognitionSource.FALLBACK
            ),
            action=(decision.action if decision else ArbitrationAction.CLARIFY),
            active_context=state.get("previous_context"),
            needs_clarification=decision is None,
            agent_statuses=state.get("agent_statuses", {}),
        ),
        "history_message": message,
        "pending_context": None,
        "commit_context": False,
        "trace": ["respond_failure"],
    }


async def _prepare_transition(_: ConversationGraphState) -> dict[str, Any]:
    return {"trace": ["prepare_transition"]}


def _result_status(result: Any) -> str:
    if not getattr(result, "success", False):
        return "failed"
    if getattr(result, "degraded_reason", None):
        return "degraded"
    data = getattr(result, "data", None) or {}
    if data.get("degraded_reason"):
        return "degraded"
    diagnostics = getattr(result, "retrieval_diagnostics", None)
    if getattr(diagnostics, "vector_status", None) in {"failed", "degraded"}:
        return "degraded"
    return "success"


def _to_document_recommendations(
    documents: list[RankedDocument],
) -> list[DocumentRecommendation]:
    """把受保护的文档排序结果转换为内部推荐结果。"""

    return [
        DocumentRecommendation(
            document_id=item.document_id,
            title=item.title,
            excerpt=item.excerpt,
            score=item.final_score,
            reason=item.rerank_reason,
            recall_score=item.recall_score,
            llm_score=item.llm_score,
            profile_score=item.profile_score,
        )
        for item in documents
    ]


def _build_document_message(
    action: ArbitrationAction,
    context: RecommendationContext,
    *,
    item_count: int,
) -> str:
    """使用实际检索查询构造不依赖旧元数据的推荐消息。"""

    subject = " ".join(context.query.split())[:80]
    if item_count == 0:
        if action == ArbitrationAction.REPEAT:
            return f"当前文档库里没有更多未展示的“{subject}”内容了。"
        return f"当前文档库里没有找到与“{subject}”匹配的内容。"
    if action == ArbitrationAction.NEW:
        prefix = f"已按“{subject}”开启一轮新推荐"
    elif action == ArbitrationAction.REPEAT:
        prefix = f"沿用“{subject}”为你换了一批"
    else:
        prefix = f"已按新的“{subject}”检索条件调整推荐"
    return f"{prefix}。找到 {item_count} 篇："


def _unique(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value not in seen:
            result.append(value)
            seen.add(value)
    return result


def _feedback_reply(
    *,
    session_id: str,
    message: str,
    intent_state: IntentState,
    previous_context: RecommendationContext | None,
    action: ArbitrationAction,
    needs_clarification: bool = False,
    citations: list[Any] | None = None,
    images: list[Any] | None = None,
    execution_trace: Any | None = None,
    succeeded: bool = True,
) -> ConversationReply:
    """组装不改变 HTTP 字段的反馈补救回复。"""

    return ConversationReply(
        session_id=session_id,
        message=message,
        intent_source=RecognitionSource.RULE,
        action=action,
        intent_state=intent_state,
        active_context=previous_context,
        citations=citations or [],
        images=images or [],
        execution_trace=execution_trace,
        needs_clarification=needs_clarification,
        agent_statuses={
            "feedback_recovery": "success" if succeeded else "failed"
        },
    )


def _snapshot_draft_from_output(
    output: dict[str, Any],
) -> ConversationResultSnapshotDraft | None:
    """把普通推荐或知识结果投影为无身份快照草稿。"""

    reply = output.get("reply")
    if reply is None:
        return None
    recommendations = tuple(
        item.document_id for item in getattr(reply, "recommendations", ())
    )
    if recommendations:
        context = output.get("pending_context") or getattr(
            reply,
            "active_context",
            None,
        )
        return ConversationResultSnapshotDraft(
            result_type="recommendation",
            query=context.query if context is not None else None,
            recommendation_document_ids=recommendations,
        )
    result = output.get("knowledge_result")
    if result is not None and result.status in {
        "success",
        "degraded",
        "insufficient_evidence",
    }:
        recognition = output.get("recognition")
        return ConversationResultSnapshotDraft(
            result_type="knowledge_answer",
            query=(recognition.rewritten_query if recognition is not None else None),
            citation_document_ids=tuple(
                dict.fromkeys(item.document_id for item in result.citations)
            ),
            citation_chunk_ids=tuple(item.chunk_id for item in result.citations),
            knowledge_status=result.status,
            resolved_document_ids=result.resolved_document_ids,
        )
    return None


__all__ = []
