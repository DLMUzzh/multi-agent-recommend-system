"""文章推荐单轮对话的 LangGraph 编排入口。"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from app.agents.document_recall_agent import DocumentRecallAgent
from app.agents.document_rerank_agent import DocumentRerankAgent
from app.agents.intent_recognition_agent import IntentRecognitionAgent
from app.agents.user_profile_agent import UserProfileAgent
from app.application.knowledge_qa import KnowledgeQaService
from app.domain.services.conversation_arbitrator import ConversationArbitrator
from app.domain.services.document_result_aggregator import DocumentResultAggregator
from app.domain.services.feedback_recovery_policy import FeedbackRecoveryPolicy
from app.infrastructure.observability.conversation_trace import record_trace_event
from app.models.schemas import (
    ConversationTurn,
    IntentState,
    RecommendationContext,
)
from app.models.intent_memory import UserIntentMemoryProjection
from app.models.interaction_memory import UserInteractionMemoryProjection
from app.models.personal_feedback import (
    ConversationFeedbackContext,
    FeedbackAnalysis,
    FeedbackDecision,
)
from app.orchestration.conversation_nodes import (
    _aggregate_document_results as aggregate_document_results_node,
    _arbitrate as arbitrate_node,
    _classify_feedback as classify_feedback_node,
    _recognize_intent as recognize_intent_node,
    _route_agent_stage as route_agent_stage,
    _route_arbitration as route_arbitration,
    _route_intent as route_intent,
    _run_document_recall_agent as run_document_recall_agent_node,
    _run_document_rerank_agent as run_document_rerank_agent_node,
    _run_feedback_action as run_feedback_action_node,
    _run_knowledge_qa as run_knowledge_qa_node,
    _run_user_profile_agent as run_user_profile_agent_node,
    _prepare_recommendation_branches as prepare_recommendation_branches_node,
    _run_recommendation_retrieval_branch as run_recommendation_retrieval_branch_node,
    _join_recommendation_branches as join_recommendation_branches_node,
)
from app.orchestration.conversation_responses import (
    _prepare_transition as prepare_transition,
    _respond_decision as respond_decision,
    _respond_failure as respond_failure,
    _respond_knowledge as respond_knowledge,
    _respond_no_action as respond_no_action,
    _respond_success as respond_success,
    _respond_unknown as respond_unknown,
    _result_status as result_status,
    _snapshot_draft_from_output as snapshot_draft_from_output,
    _to_document_recommendations as to_document_recommendations,
    _build_document_message as build_document_message,
    _unique as unique_values,
)
from app.orchestration.conversation_state import (
    ConversationGraphResult,
    ConversationGraphState,
)


class ConversationGraph:
    """按当前意图调度文章推荐链或知识问答链。

    知识问答在推荐仲裁前分流；推荐意图继续由仲裁器决定如何合并上下文。
    用户画像与受保护查询的 Chunk 召回并行执行，只在重排前汇合；画像不参与召回或过滤，
    重排 Agent 再把画像作为软排序证据。确定性聚合器保护候选事实和数量边界。
    """

    _recognize_intent = recognize_intent_node
    _route_intent = staticmethod(route_intent)
    _arbitrate = arbitrate_node
    _route_arbitration = staticmethod(route_arbitration)
    _run_user_profile_agent = run_user_profile_agent_node
    _prepare_recommendation_branches = prepare_recommendation_branches_node
    _run_recommendation_retrieval_branch = (
        run_recommendation_retrieval_branch_node
    )
    _join_recommendation_branches = join_recommendation_branches_node
    _run_document_recall_agent = run_document_recall_agent_node
    _run_document_rerank_agent = run_document_rerank_agent_node
    _aggregate_document_results = aggregate_document_results_node
    _route_agent_stage = staticmethod(route_agent_stage)

    _respond_no_action = staticmethod(respond_no_action)
    _respond_unknown = staticmethod(respond_unknown)
    _run_knowledge_qa = run_knowledge_qa_node
    _respond_knowledge = staticmethod(respond_knowledge)
    _respond_decision = staticmethod(respond_decision)
    _respond_success = respond_success
    _respond_failure = staticmethod(respond_failure)
    _prepare_transition = staticmethod(prepare_transition)
    _result_status = staticmethod(result_status)
    _to_document_recommendations = staticmethod(to_document_recommendations)
    _build_document_message = staticmethod(build_document_message)
    _unique = staticmethod(unique_values)
    classify_feedback = classify_feedback_node
    _run_feedback_action = run_feedback_action_node
    _snapshot_draft_from_output = staticmethod(snapshot_draft_from_output)

    def __init__(
        self,
        *,
        intent_agent: IntentRecognitionAgent,
        arbitrator: ConversationArbitrator,
        recall_agent: DocumentRecallAgent,
        rerank_agent: DocumentRerankAgent,
        aggregator: DocumentResultAggregator,
        profile_agent: UserProfileAgent | None = None,
        knowledge_qa_service: KnowledgeQaService | None = None,
        feedback_agent: Any | None = None,
        feedback_policy: FeedbackRecoveryPolicy | None = None,
    ) -> None:
        self.intent_agent = intent_agent
        self.profile_agent = profile_agent
        self.arbitrator = arbitrator
        self.recall_agent = recall_agent
        self.rerank_agent = rerank_agent
        self.aggregator = aggregator
        self.knowledge_qa_service = knowledge_qa_service
        self.feedback_agent = feedback_agent
        self.feedback_policy = feedback_policy or FeedbackRecoveryPolicy()
        self.graph = self._build_graph()

    async def run(
        self,
        *,
        user_id: str,
        session_id: str,
        message: str,
        history: list[ConversationTurn],
        previous_context: RecommendationContext | None,
        conversation_summary: str | None = None,
        intent_state: IntentState | str = IntentState.RECOMMENDATION,
        intent_memory: UserIntentMemoryProjection | None = None,
        interaction_memory: UserInteractionMemoryProjection | None = None,
        knowledge_document_ids: tuple[str, ...] = (),
        feedback_context: ConversationFeedbackContext | dict[str, Any] | None = None,
        protected_feedback_analysis: FeedbackAnalysis | None = None,
        protected_feedback_decision: FeedbackDecision | None = None,
    ) -> ConversationGraphResult:
        current_intent_state = IntentState(intent_state)
        protected_intent_memory = (
            UserIntentMemoryProjection.model_validate(intent_memory).model_copy(
                deep=True
            )
            if intent_memory is not None
            else None
        )
        protected_interaction_memory = (
            UserInteractionMemoryProjection.model_validate(
                interaction_memory
            ).model_copy(deep=True)
            if interaction_memory is not None
            else None
        )
        protected_feedback_context = (
            ConversationFeedbackContext.model_validate(feedback_context).model_copy(
                deep=True
            )
            if feedback_context is not None
            else None
        )
        feedback_analysis = (
            FeedbackAnalysis.model_validate(protected_feedback_analysis).model_copy(
                deep=True
            )
            if protected_feedback_analysis is not None
            else None
        )
        feedback_decision = (
            FeedbackDecision.model_validate(protected_feedback_decision).model_copy(
                deep=True
            )
            if protected_feedback_decision is not None
            else None
        )
        if (feedback_analysis is None) != (feedback_decision is None):
            raise ValueError("反馈分析与受保护决策必须同时提供")
        if feedback_decision is not None and feedback_decision.is_feedback:
            return await self._run_feedback_action(
                user_id=str(user_id),
                session_id=session_id,
                message=message,
                history=history,
                previous_context=previous_context,
                conversation_summary=conversation_summary,
                intent_state=current_intent_state,
                interaction_memory=protected_interaction_memory,
                feedback_context=protected_feedback_context,
                feedback_analysis=feedback_analysis,
                feedback_decision=feedback_decision,
            )
        graph_input = {
            "user_id": str(user_id),
            "session_id": session_id,
            "message": message,
            "history": history,
            "conversation_summary": conversation_summary,
            "previous_context": previous_context,
            "intent_state": current_intent_state,
            "intent_memory": protected_intent_memory,
            "interaction_memory": protected_interaction_memory,
            "knowledge_document_ids": knowledge_document_ids,
        }
        record_trace_event(
            "component.started",
            "conversation_graph",
            input_data=graph_input,
        )
        state: ConversationGraphState = {
            "user_id": str(user_id),
            "session_id": session_id,
            "message": message,
            "history": [turn.model_copy(deep=True) for turn in history],
            "conversation_summary": conversation_summary,
            "previous_context": (
                previous_context.model_copy(deep=True) if previous_context else None
            ),
            "current_intent_state": current_intent_state,
            "intent_memory": protected_intent_memory,
            "interaction_memory": protected_interaction_memory,
            "knowledge_document_ids": tuple(knowledge_document_ids),
            "agent_statuses": {},
            "trace": [],
        }
        try:
            output = await self.graph.ainvoke(state)
        except BaseException as exc:
            record_trace_event(
                "component.failed",
                "conversation_graph",
                status="error",
                error=exc,
            )
            raise
        trace = list(output.get("trace", []))
        result = ConversationGraphResult(
            reply=output["reply"],
            history_message=output["history_message"],
            pending_context=output.get("pending_context"),
            commit_context=bool(output.get("commit_context")),
            pending_intent_state=output.get("pending_intent_state"),
            commit_intent_state=bool(output.get("commit_intent_state")),
            knowledge_document_ids=tuple(
                output.get("knowledge_document_ids", ())
            ),
            knowledge_document_titles=tuple(
                output.get("knowledge_document_titles", ())
            ),
            error_stage=output.get("error_stage"),
            result_snapshot_draft=self._snapshot_draft_from_output(output),
            trace=trace,
        )
        record_trace_event(
            "component.completed",
            "conversation_graph",
            output_data=result,
            status="failed" if result.error_stage else "success",
        )
        return result

    def _build_graph(self):
        builder = StateGraph(ConversationGraphState)
        builder.add_node("recognize_intent", self._recognize_intent)
        builder.add_node("arbitrate", self._arbitrate)
        builder.add_node(
            "prepare_recommendation_branches",
            self._prepare_recommendation_branches,
        )
        builder.add_node("run_knowledge_qa", self._run_knowledge_qa)
        builder.add_node(
            "recommendation_retrieval_branch",
            self._run_recommendation_retrieval_branch,
        )
        builder.add_node("user_profile_agent", self._run_user_profile_agent)
        builder.add_node(
            "join_recommendation_branches",
            self._join_recommendation_branches,
        )
        builder.add_node("document_rerank_agent", self._run_document_rerank_agent)
        builder.add_node(
            "aggregate_document_results",
            self._aggregate_document_results,
        )
        builder.add_node("respond_no_action", self._respond_no_action)
        builder.add_node("respond_unknown", self._respond_unknown)
        builder.add_node(
            "respond_knowledge",
            self._respond_knowledge,
        )
        builder.add_node("respond_decision", self._respond_decision)
        builder.add_node("respond_success", self._respond_success)
        builder.add_node("respond_failure", self._respond_failure)
        builder.add_node("prepare_transition", self._prepare_transition)

        builder.add_edge(START, "recognize_intent")
        builder.add_conditional_edges(
            "recognize_intent",
            self._route_intent,
            {
                "no_action": "respond_no_action",
                "unknown": "respond_unknown",
                "recommend": "arbitrate",
                "knowledge_qa": "run_knowledge_qa",
                "failure": "respond_failure",
            },
        )
        builder.add_conditional_edges(
            "run_knowledge_qa",
            self._route_agent_stage,
            {
                "success": "respond_knowledge",
                "failure": "respond_failure",
            },
        )
        builder.add_conditional_edges(
            "arbitrate",
            self._route_arbitration,
            {
                "respond": "respond_decision",
                "recommend": "prepare_recommendation_branches",
                "failure": "respond_failure",
            },
        )
        builder.add_edge(
            "prepare_recommendation_branches",
            "user_profile_agent",
        )
        builder.add_edge(
            "prepare_recommendation_branches",
            "recommendation_retrieval_branch",
        )
        builder.add_edge(
            ["user_profile_agent", "recommendation_retrieval_branch"],
            "join_recommendation_branches",
        )
        builder.add_conditional_edges(
            "join_recommendation_branches",
            self._route_agent_stage,
            {
                "success": "document_rerank_agent",
                "failure": "respond_failure",
            },
        )
        builder.add_conditional_edges(
            "document_rerank_agent",
            self._route_agent_stage,
            {
                "success": "aggregate_document_results",
                "failure": "respond_failure",
            },
        )
        builder.add_conditional_edges(
            "aggregate_document_results",
            self._route_agent_stage,
            {
                "success": "respond_success",
                "failure": "respond_failure",
            },
        )
        for response_node in (
            "respond_no_action",
            "respond_unknown",
            "respond_knowledge",
            "respond_decision",
            "respond_success",
            "respond_failure",
        ):
            builder.add_edge(response_node, "prepare_transition")
        builder.add_edge("prepare_transition", END)
        return builder.compile()


__all__ = [
    "ConversationGraph",
    "ConversationGraphResult",
    "ConversationGraphState",
]
