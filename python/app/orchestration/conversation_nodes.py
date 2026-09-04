"""会话图的意图、仲裁和推荐链节点。"""

from __future__ import annotations

import asyncio
from typing import Any, Literal, Protocol

from app.infrastructure.observability.conversation_trace import record_trace_event
from app.models.schemas import (
    ArbitrationAction,
    ArbitrationDecision,
    ConversationTurn,
    IntentName,
    IntentRecognition,
    IntentState,
    RecognitionSource,
    RecommendationContext,
    RecommendationIntent,
    RelationHint,
)
from app.models.interaction_memory import UserInteractionMemoryProjection
from app.models.personal_feedback import (
    ConversationFeedbackContext,
    ConversationResultSnapshotDraft,
    FeedbackAnalysis,
    FeedbackDecision,
)
from app.orchestration.conversation_responses import (
    _feedback_reply,
    _knowledge_history_message,
)
from app.orchestration.conversation_state import ConversationGraphResult
from app.orchestration.conversation_state import ConversationGraphState


class _NodeRuntime(Protocol):
    """节点执行所需的会话图运行时属性。"""

    intent_agent: Any
    profile_agent: Any
    arbitrator: Any
    recall_agent: Any
    rerank_agent: Any
    aggregator: Any
    knowledge_qa_service: Any
    feedback_agent: Any
    feedback_policy: Any

    def _result_status(self, result: Any) -> str: ...

    def _unique(self, values: list[str]) -> list[str]: ...

    async def _run_user_profile_agent(
        self,
        state: ConversationGraphState,
    ) -> dict[str, Any]: ...

    async def _run_recommendation_retrieval_branch(
        self,
        state: ConversationGraphState,
    ) -> dict[str, Any]: ...

    async def _run_document_rerank_agent(
        self,
        state: ConversationGraphState,
    ) -> dict[str, Any]: ...

    async def _aggregate_document_results(
        self,
        state: ConversationGraphState,
    ) -> dict[str, Any]: ...

    async def _respond_success(
        self,
        state: ConversationGraphState,
    ) -> dict[str, Any]: ...


async def _classify_feedback(
    runtime: _NodeRuntime,
    *,
    message: str,
    history: list[ConversationTurn],
    feedback_context: ConversationFeedbackContext | dict[str, Any],
) -> tuple[FeedbackAnalysis | None, FeedbackDecision]:
    """只分类并保护反馈，不执行推荐、检索或回答动作。"""

    context = ConversationFeedbackContext.model_validate(feedback_context)
    snapshot = context.latest_result
    pending = context.pending_feedback
    if not runtime.feedback_policy.is_candidate(
        message,
        snapshot=snapshot,
        pending_event=pending,
    ):
        return None, runtime.feedback_policy.fallback_decision(
            "",
            snapshot=None,
            pending_event=None,
        )
    previous_user_message, previous_assistant_message = _latest_exchange(history)
    analysis = None
    if runtime.feedback_agent is not None and snapshot is not None:
        try:
            analysis = await runtime.feedback_agent.analyze(
                message=message,
                snapshot=snapshot,
                pending_event=pending,
                previous_user_message=previous_user_message,
                previous_assistant_message=previous_assistant_message,
            )
        except Exception as exc:
            record_trace_event(
                "component.degraded",
                "feedback_recovery_agent",
                status="degraded",
                error={"type": type(exc).__name__},
            )
    if analysis is None:
        return None, runtime.feedback_policy.fallback_decision(
            message,
            snapshot=snapshot,
            pending_event=pending,
        )
    try:
        decision = runtime.feedback_policy.protect(
            message,
            snapshot=snapshot,
            pending_event=pending,
            analysis=analysis,
        )
    except Exception as exc:
        record_trace_event(
            "component.degraded",
            "feedback_recovery_policy",
            status="degraded",
            error={"type": type(exc).__name__},
        )
        return None, runtime.feedback_policy.fallback_decision(
            message,
            snapshot=snapshot,
            pending_event=pending,
        )
    return analysis, decision


async def _run_feedback_action(
    runtime: _NodeRuntime,
    *,
    user_id: str,
    session_id: str,
    message: str,
    history: list[ConversationTurn],
    previous_context: RecommendationContext | None,
    conversation_summary: str | None,
    intent_state: IntentState,
    interaction_memory: UserInteractionMemoryProjection | None,
    feedback_context: ConversationFeedbackContext | None,
    feedback_analysis: FeedbackAnalysis | None,
    feedback_decision: FeedbackDecision,
) -> ConversationGraphResult:
    """执行一次确定性允许的反馈动作，不形成循环。"""

    if feedback_context is None or feedback_context.latest_result is None:
        raise ValueError("反馈补救缺少结果快照")
    if feedback_analysis is None:
        raise ValueError("反馈补救缺少结构化分析")
    if feedback_decision.next_action == "clarify":
        question = feedback_decision.clarification_question or (
            "请补充你不满意的具体原因。"
        )
        reply = _feedback_reply(
            session_id=session_id,
            message=question,
            intent_state=intent_state,
            previous_context=previous_context,
            action=ArbitrationAction.CLARIFY,
            needs_clarification=True,
        )
        return ConversationGraphResult(
            reply=reply,
            history_message=question,
            feedback_analysis=feedback_analysis,
            feedback_decision=feedback_decision,
            trace=["classify_feedback", "respond_feedback_clarification"],
        )
    if feedback_decision.next_action == "retry_recommendation":
        return await _run_feedback_recommendation(
            runtime,
            user_id=user_id,
            session_id=session_id,
            previous_context=previous_context,
            feedback_context=feedback_context,
            feedback_analysis=feedback_analysis,
            feedback_decision=feedback_decision,
        )
    if feedback_decision.next_action in {
        "retry_retrieval",
        "retry_answer_from_evidence",
    }:
        return await _run_feedback_knowledge(
            runtime,
            session_id=session_id,
            message=message,
            history=history,
            previous_context=previous_context,
            conversation_summary=conversation_summary,
            interaction_memory=interaction_memory,
            feedback_context=feedback_context,
            feedback_analysis=feedback_analysis,
            feedback_decision=feedback_decision,
        )
    raise ValueError("反馈决策没有可执行的补救动作")


async def _run_feedback_knowledge(
    runtime: _NodeRuntime,
    *,
    session_id: str,
    message: str,
    history: list[ConversationTurn],
    previous_context: RecommendationContext | None,
    conversation_summary: str | None,
    interaction_memory: UserInteractionMemoryProjection | None,
    feedback_context: ConversationFeedbackContext,
    feedback_analysis: FeedbackAnalysis,
    feedback_decision: FeedbackDecision,
) -> ConversationGraphResult:
    """复用知识服务执行重新检索或可信证据再回答。"""

    if runtime.knowledge_qa_service is None:
        raise RuntimeError("知识问答服务未装配")
    snapshot = feedback_context.latest_result
    assert snapshot is not None
    if feedback_decision.next_action == "retry_answer_from_evidence":
        result = await runtime.knowledge_qa_service.regenerate_from_evidence(
            feedback_decision.protected_query or message,
            chunk_ids=snapshot.citation_chunk_ids,
            interaction_memory=interaction_memory,
            request_route="/api/v1/chat",
        )
        trace = ["classify_feedback", "retry_answer_from_evidence"]
    else:
        result = await runtime.knowledge_qa_service.ask(
            message,
            history=history,
            conversation_summary=conversation_summary,
            prepared_query=feedback_decision.protected_query,
            interaction_memory=interaction_memory,
            request_route="/api/v1/chat",
        )
        trace = ["classify_feedback", "retry_retrieval"]
    succeeded = result.status in {"success", "degraded"}
    prefix = "已根据你的反馈修正：" if succeeded else "本次补救未完成："
    public_message = prefix + result.answer
    history_message = _knowledge_history_message(
        public_message,
        list(result.citations),
    )
    reply = _feedback_reply(
        session_id=session_id,
        message=public_message,
        intent_state=IntentState.KNOWLEDGE_QA,
        previous_context=previous_context,
        action=ArbitrationAction.KNOWLEDGE_ANSWER,
        citations=list(result.citations),
        images=list(result.images),
        execution_trace=result.execution_trace,
        succeeded=succeeded,
    )
    draft = (
        ConversationResultSnapshotDraft(
            result_type="knowledge_answer",
            query=feedback_decision.protected_query,
            citation_document_ids=tuple(
                dict.fromkeys(item.document_id for item in result.citations)
            ),
            citation_chunk_ids=tuple(item.chunk_id for item in result.citations),
            knowledge_status=result.status,
            resolved_document_ids=result.resolved_document_ids,
        )
        if succeeded
        else None
    )
    return ConversationGraphResult(
        reply=reply,
        history_message=history_message,
        pending_intent_state=IntentState.KNOWLEDGE_QA if succeeded else None,
        commit_intent_state=succeeded,
        knowledge_document_ids=result.resolved_document_ids,
        knowledge_document_titles=result.resolved_document_titles,
        feedback_analysis=feedback_analysis,
        feedback_decision=feedback_decision,
        feedback_recovery_succeeded=succeeded,
        result_snapshot_draft=draft,
        trace=trace + ["respond_feedback_knowledge"],
    )


async def _run_feedback_recommendation(
    runtime: _NodeRuntime,
    *,
    user_id: str,
    session_id: str,
    previous_context: RecommendationContext | None,
    feedback_context: ConversationFeedbackContext,
    feedback_analysis: FeedbackAnalysis,
    feedback_decision: FeedbackDecision,
) -> ConversationGraphResult:
    """使用现有推荐节点执行一次受控补救。"""

    snapshot = feedback_context.latest_result
    assert snapshot is not None
    excluded = list(feedback_decision.excluded_document_ids)
    seen = list(previous_context.seen_article_ids) if previous_context else []
    context = RecommendationContext(
        query=feedback_decision.protected_query or snapshot.query or "",
        size=(
            previous_context.size
            if previous_context is not None
            else max(1, min(10, len(snapshot.recommendation_document_ids)))
        ),
        seen_article_ids=runtime._unique(seen + excluded),
    )
    recognition = IntentRecognition(
        intent=IntentName.RECOMMEND_ARTICLES,
        source=RecognitionSource.RULE,
        relation=RelationHint.REFINE,
        confidence=1.0,
        rewritten_query=context.query,
        resolved_intent=RecommendationIntent(size=context.size),
    )
    decision = ArbitrationDecision(
        action=ArbitrationAction.REFINE,
        context=context,
        reason="个人反馈补救使用受保护查询。",
    )
    state: ConversationGraphState = {
        "user_id": user_id,
        "session_id": session_id,
        "message": context.query,
        "history": [],
        "previous_context": previous_context,
        "current_intent_state": IntentState.RECOMMENDATION,
        "recognition": recognition,
        "decision": decision,
        "agent_statuses": {},
        "trace": [],
    }
    profile_update, retrieval_update = await asyncio.gather(
        runtime._run_user_profile_agent(state),
        runtime._run_recommendation_retrieval_branch(state),
    )
    state.update(profile_update)
    state.update(retrieval_update)
    if state.get("error_stage"):
        raise RuntimeError("推荐补救召回失败")
    state.update(await runtime._run_document_rerank_agent(state))
    if state.get("error_stage"):
        raise RuntimeError("推荐补救重排失败")
    state.update(await runtime._aggregate_document_results(state))
    if state.get("error_stage"):
        raise RuntimeError("推荐补救聚合失败")
    response = await runtime._respond_success(state)
    reply = response["reply"].model_copy(
        update={
            "message": "已根据你的反馈修正：" + response["reply"].message,
            "agent_statuses": {
                **response["reply"].agent_statuses,
                "feedback_recovery": "success",
            },
        },
        deep=True,
    )
    return ConversationGraphResult(
        reply=reply,
        history_message="已根据你的反馈修正：" + response["history_message"],
        pending_context=response.get("pending_context"),
        commit_context=bool(response.get("commit_context")),
        pending_intent_state=IntentState.RECOMMENDATION,
        commit_intent_state=True,
        feedback_analysis=feedback_analysis,
        feedback_decision=feedback_decision,
        feedback_recovery_succeeded=True,
        result_snapshot_draft=ConversationResultSnapshotDraft(
            result_type="recommendation",
            query=context.query,
            recommendation_document_ids=tuple(
                item.document_id for item in reply.recommendations
            ),
        ),
        trace=[
            "classify_feedback",
            "prepare_feedback_recommendation",
            *profile_update.get("trace", []),
            *retrieval_update.get("trace", []),
            "document_rerank_agent",
            "aggregate_document_results",
            "respond_feedback_recommendation",
        ],
    )


def _latest_exchange(
    history: list[ConversationTurn],
) -> tuple[str | None, str | None]:
    """提取最近一组普通问答供反馈语义分类。"""

    for index in range(len(history) - 1, 0, -1):
        if history[index].role == "assistant" and history[index - 1].role == "user":
            return history[index - 1].content, history[index].content
    return None, None

async def _recognize_intent(
    runtime: _NodeRuntime,
    state: ConversationGraphState,
) -> dict[str, Any]:
    component = "intent_recognition_agent"
    record_trace_event(
        "agent.started",
        component,
        input_data={
            "message": state["message"],
            "history": state["history"],
            "conversation_summary": state.get("conversation_summary"),
            "active_context": state.get("previous_context"),
            "intent_state": state["current_intent_state"],
        },
    )
    try:
        recognition = await runtime.intent_agent.run(
            state["message"],
            history=state["history"],
            active_context=state.get("previous_context"),
            conversation_summary=state.get("conversation_summary"),
            intent_state=state["current_intent_state"],
            intent_memory=state.get("intent_memory"),
        )
        record_trace_event(
            "agent.completed",
            component,
            output_data=recognition,
            status=(
                "degraded"
                if recognition.source is RecognitionSource.FALLBACK
                else "success"
            ),
        )
        return {"recognition": recognition, "trace": ["recognize_intent"]}
    except Exception as exc:
        record_trace_event(
            "agent.failed",
            component,
            status="error",
            error=exc,
        )
        return {
            "error_stage": "intent",
            "error_type": type(exc).__name__,
            "trace": ["recognize_intent"],
        }


async def _route_intent(
    state: ConversationGraphState,
) -> Literal[
    "no_action",
    "unknown",
    "recommend",
    "knowledge_qa",
    "failure",
]:
    if state.get("error_stage"):
        return "failure"
    intent = state["recognition"].intent
    if intent is IntentName.NO_ACTION:
        return "no_action"
    if intent is IntentName.UNKNOWN:
        return "unknown"
    if intent is IntentName.KNOWLEDGE_QA:
        return "knowledge_qa"
    return "recommend"


async def _run_knowledge_qa(
    runtime: _NodeRuntime,
    state: ConversationGraphState,
) -> dict[str, Any]:
    """调用知识问答应用服务，不在 Graph 中展开检索内部阶段。"""

    component = "knowledge_qa"
    record_trace_event(
        "component.started",
        component,
        input_data={
            "question": state["message"],
            "history": state["history"],
            "conversation_summary": state.get("conversation_summary"),
            "knowledge_document_ids": state.get("knowledge_document_ids", ()),
        },
    )
    try:
        if runtime.knowledge_qa_service is None:
            raise RuntimeError("知识问答服务未装配")
        ask_kwargs = {
            "history": state["history"],
            "conversation_summary": state.get("conversation_summary"),
            "document_ids": state.get("knowledge_document_ids", ()),
            "prepared_query": state["recognition"].rewritten_query,
            "request_route": "/api/v1/chat",
        }
        interaction_memory = state.get("interaction_memory")
        if interaction_memory is not None:
            ask_kwargs["interaction_memory"] = interaction_memory
        result = await runtime.knowledge_qa_service.ask(
            state["message"],
            **ask_kwargs,
        )
    except Exception as exc:
        record_trace_event(
            "component.failed",
            component,
            status="error",
            error=exc,
        )
        return {
            "error_stage": "knowledge_qa",
            "error_type": type(exc).__name__,
            "agent_statuses": {"knowledge_answer": "failed"},
            "trace": ["run_knowledge_qa"],
        }
    degraded = set(result.degraded_components)
    statuses = {
        "knowledge_query_analysis": (
            "degraded" if "query_analysis" in degraded else "success"
        ),
        "knowledge_planner": (
            "degraded" if "planner" in degraded else "success"
        ),
        "knowledge_plan_execution": (
            "degraded" if "plan_execution" in degraded else "success"
        ),
        "knowledge_plan_coverage": (
            "degraded" if "coverage" in degraded else "success"
        ),
        "knowledge_vector": (
            "degraded" if "vector" in degraded else "success"
        ),
        "knowledge_answer": (
            "degraded" if "answer" in degraded else "success"
        ),
    }
    record_trace_event(
        "component.completed",
        component,
        output_data=result,
        status="degraded" if degraded else "success",
    )
    return {
        "knowledge_result": result,
        "knowledge_document_ids": result.resolved_document_ids,
        "knowledge_document_titles": result.resolved_document_titles,
        "agent_statuses": statuses,
        "trace": ["run_knowledge_qa"],
    }


async def _arbitrate(
    runtime: _NodeRuntime,
    state: ConversationGraphState,
) -> dict[str, Any]:
    component = "conversation_arbitrator"
    record_trace_event(
        "component.started",
        component,
        input_data={
            "recognition": state["recognition"],
            "previous_context": state.get("previous_context"),
        },
    )
    try:
        decision = runtime.arbitrator.decide(
            state["recognition"],
            state.get("previous_context"),
        )
        record_trace_event(
            "component.completed",
            component,
            output_data=decision,
            status="success",
        )
        return {"decision": decision, "trace": ["arbitrate"]}
    except Exception as exc:
        record_trace_event(
            "component.failed",
            component,
            status="error",
            error=exc,
        )
        return {
            "error_stage": "arbitration",
            "error_type": type(exc).__name__,
            "trace": ["arbitrate"],
        }


async def _route_arbitration(
    state: ConversationGraphState,
) -> Literal["respond", "recommend", "failure"]:
    if state.get("error_stage"):
        return "failure"
    decision = state["decision"]
    if (
        decision.action in {ArbitrationAction.CLARIFY, ArbitrationAction.UNSUPPORTED}
        or decision.context is None
    ):
        return "respond"
    return "recommend"


async def _run_document_recall_agent(
    runtime: _NodeRuntime,
    state: ConversationGraphState,
) -> dict[str, Any]:
    """使用共享 Chunk 索引召回 SQLite 文档候选。"""

    if state.get("error_stage"):
        return {"trace": ["document_recall_agent"]}
    component = "document_recall_agent"
    context = state["effective_context"]
    record_trace_event(
        "agent.started",
        component,
        input_data={
            "query": state["retrieval_query"],
            "size": context.size,
            "seen_document_count": len(context.seen_article_ids),
        },
    )
    try:
        result = await runtime.recall_agent.run(
            query=state["retrieval_query"],
            size=context.size,
            seen_document_ids=tuple(context.seen_article_ids),
        )
    except Exception as exc:
        record_trace_event(
            "agent.failed",
            component,
            status="error",
            error=exc,
        )
        return {
            "agent_statuses": {"document_recall": "failed"},
            "error_stage": "document_recall",
            "error_type": type(exc).__name__,
            "trace": ["document_recall_agent"],
        }
    status = runtime._result_status(result)
    record_trace_event(
        "agent.completed",
        component,
        output_data=result,
        status=status,
    )
    if not getattr(result, "success", False):
        return {
            "document_recall_result": result,
            "agent_statuses": {"document_recall": "failed"},
            "error_stage": "document_recall",
            "error_type": "AgentFailure",
            "trace": ["document_recall_agent"],
        }
    return {
        "document_recall_result": result,
        "agent_statuses": {"document_recall": status},
        "trace": ["document_recall_agent"],
    }


async def _run_document_rerank_agent(
    runtime: _NodeRuntime,
    state: ConversationGraphState,
) -> dict[str, Any]:
    """融合查询相关性和并行画像结果执行文档重排。"""

    component = "document_rerank_agent"
    record_trace_event(
        "agent.started",
        component,
        input_data={
            "query": state["retrieval_query"],
            "candidate_count": len(
                state["document_recall_result"].candidates
            ),
            "profile_available": state.get("user_profile") is not None,
        },
    )
    try:
        result = await runtime.rerank_agent.run(
            query=state["retrieval_query"],
            candidates=state["document_recall_result"].candidates,
            user_profile=state.get("user_profile"),
            current_topics=(),
        )
    except Exception as exc:
        record_trace_event(
            "agent.failed",
            component,
            status="error",
            error=exc,
        )
        return {
            "agent_statuses": {"document_rerank": "failed"},
            "error_stage": "document_rerank",
            "error_type": type(exc).__name__,
            "trace": ["document_rerank_agent"],
        }
    status = runtime._result_status(result)
    record_trace_event(
        "agent.completed",
        component,
        output_data={
            "success": bool(getattr(result, "success", False)),
            "llm_status": result.data.get("llm_status"),
            "profile_status": result.data.get("profile_status"),
            "profile_confidence": result.data.get("profile_confidence", 0.0),
            "blend_weights": result.data.get("blend_weights", {}),
            "document_scores": [
                {
                    "document_id": item.document_id,
                    "relevance_score": item.relevance_score,
                    "profile_score": item.profile_score,
                    "llm_score": item.llm_score,
                    "final_score": item.final_score,
                    "rerank_reason": item.rerank_reason,
                }
                for item in result.ranked_documents
            ],
        },
        status=status,
    )
    if not getattr(result, "success", False):
        return {
            "document_rerank_result": result,
            "agent_statuses": {"document_rerank": "failed"},
            "error_stage": "document_rerank",
            "error_type": "AgentFailure",
            "trace": ["document_rerank_agent"],
        }
    return {
        "document_rerank_result": result,
        "agent_statuses": {"document_rerank": status},
        "trace": ["document_rerank_agent"],
    }


async def _aggregate_document_results(
    runtime: _NodeRuntime,
    state: ConversationGraphState,
) -> dict[str, Any]:
    """对白名单文档执行去重、已展示保护和数量限制。"""

    component = "document_result_aggregator"
    context = state["effective_context"]
    record_trace_event(
        "component.started",
        component,
        input_data={
            "candidate_count": len(
                state["document_recall_result"].candidates
            ),
            "ranked_document_count": len(
                state["document_rerank_result"].ranked_documents
            ),
            "requested_size": context.size,
            "seen_document_count": len(context.seen_article_ids),
        },
    )
    try:
        documents = runtime.aggregator.aggregate(
            candidates=state["document_recall_result"].candidates,
            ranked_documents=state["document_rerank_result"].ranked_documents,
            seen_document_ids=tuple(context.seen_article_ids),
            size=context.size,
        )
    except Exception as exc:
        record_trace_event(
            "component.failed",
            component,
            status="error",
            error=exc,
        )
        return {
            "error_stage": "aggregation",
            "error_type": type(exc).__name__,
            "trace": ["aggregate_document_results"],
        }
    record_trace_event(
        "component.completed",
        component,
        output_data={
            "documents": documents,
            "kept_document_ids": [item.document_id for item in documents],
            "discarded_document_ids": [
                item.document_id
                for item in state["document_rerank_result"].ranked_documents
                if item.document_id
                not in {document.document_id for document in documents}
            ],
        },
        status="success",
    )
    return {
        "final_documents": documents,
        "trace": ["aggregate_document_results"],
    }


async def _run_user_profile_agent(
    runtime: _NodeRuntime,
    state: ConversationGraphState,
) -> dict[str, Any]:
    component = "user_profile_agent"
    record_trace_event(
        "agent.started",
        component,
        input_data={"user_id": state["user_id"]},
    )
    try:
        result = await runtime.profile_agent.run(user_id=state["user_id"])
    except Exception as exc:
        record_trace_event(
            "agent.failed",
            component,
            output_data={
                "user_profile": None,
                "agent_status": "failed",
            },
            status="degraded",
            error=exc,
        )
        return {
            "user_profile": None,
            "agent_statuses": {"user_profile": "failed"},
            "trace": ["user_profile_agent"],
        }

    profile = getattr(result, "profile", None)
    status = runtime._result_status(result)
    semantic = getattr(profile, "semantic_profile", None)
    interest = getattr(semantic, "interest_analysis", None)
    reading = getattr(semantic, "reading_preferences", None)
    reader = getattr(semantic, "reader_profile", None)
    record_trace_event(
        "agent.completed",
        component,
        output_data={
            "success": bool(getattr(result, "success", False)),
            "profile_available": profile is not None,
            "profile_status": getattr(profile, "profile_status", None),
            "profile_confidence": getattr(profile, "profile_confidence", 0.0),
            "core_topics": [
                item.topic
                for item in getattr(interest, "core_interests", ())[:5]
            ],
            "negative_topics": [
                item.topic
                for item in getattr(interest, "negative_interests", ())[:5]
            ],
            "preferred_difficulty": getattr(
                reading,
                "recommended_difficulty",
                None,
            ),
            "preferred_reading_length": getattr(
                reading,
                "preferred_reading_length",
                None,
            ),
            "preferred_content_types": list(
                getattr(reading, "preferred_content_types", ())[:5]
            ),
            "activity_level": getattr(reader, "activity_level", None),
        },
        status=status,
    )
    return {
        "profile_result": result,
        "user_profile": profile,
        "agent_statuses": {"user_profile": status},
        "trace": ["user_profile_agent"],
    }


async def _prepare_recommendation_branches(
    runtime: _NodeRuntime,
    state: ConversationGraphState,
) -> dict[str, Any]:
    """建立推荐并行分支，不复制或改写业务状态。"""

    del runtime, state
    return {"trace": ["prepare_recommendation_branches"]}


async def _run_recommendation_retrieval_branch(
    runtime: _NodeRuntime,
    state: ConversationGraphState,
) -> dict[str, Any]:
    """直接使用仲裁后的受保护查询执行 Chunk 召回。"""

    context = state["decision"].context
    if context is None:
        return {
            "error_stage": "document_recall",
            "error_type": "MissingRecommendationContext",
            "trace": ["document_recall_agent"],
        }
    prepare_update = {
        "retrieval_query": context.query,
        "effective_context": context.model_copy(deep=True),
    }
    branch_state = dict(state)
    branch_state.update(prepare_update)
    recall_update = await _run_document_recall_agent(runtime, branch_state)
    statuses = dict(recall_update.get("agent_statuses", {}))
    result = {
        key: value
        for key, value in {**prepare_update, **recall_update}.items()
        if key not in {"trace", "agent_statuses"}
    }
    result["agent_statuses"] = statuses
    result["trace"] = list(recall_update.get("trace", []))
    return result


async def _join_recommendation_branches(
    runtime: _NodeRuntime,
    state: ConversationGraphState,
) -> dict[str, Any]:
    """等待画像与检索分支汇合，再交给统一失败路由。"""

    del runtime, state
    return {"trace": ["join_recommendation_branches"]}


async def _route_agent_stage(
    state: ConversationGraphState,
) -> Literal["success", "failure"]:
    return "failure" if state.get("error_stage") else "success"


__all__ = []
