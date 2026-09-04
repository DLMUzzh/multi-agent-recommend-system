"""围绕单轮会话编排提供 SQLite 会话边界和一次性状态提交。"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from uuid import uuid4

import structlog

from app.agents.conversation_summary_agent import ConversationSummaryAgent
from app.agents.conversation_feedback_agent import ConversationFeedbackAgent
from app.agents.document_recall_agent import DocumentRecallAgent
from app.agents.document_rerank_agent import DocumentRerankAgent
from app.agents.feedback_recovery_agent import FeedbackRecoveryAgent
from app.agents.intent_recognition_agent import IntentRecognitionAgent
from app.agents.user_profile_agent import UserProfileAgent
from app.application.knowledge_qa import KnowledgeQaService
from app.application.personal_feedback_learning import (
    PersonalFeedbackLearningService,
)
from app.application.user_interaction_memory import (
    UserInteractionMemoryRepository,
    UserInteractionMemoryWorker,
)
from app.config import get_settings
from app.models.schemas import (
    ArbitrationAction,
    ConversationCompressionInfo,
    ConversationReply,
    ConversationSession,
    ConversationSummaryResult,
    ConversationTurn,
    IntentState,
    RecognitionSource,
)
from app.orchestration.conversation_graph import ConversationGraph
from app.domain.services.conversation_arbitrator import ConversationArbitrator
from app.domain.services.document_result_aggregator import DocumentResultAggregator
from app.domain.services.feedback_recovery_policy import FeedbackRecoveryPolicy
from app.domain.services.user_intent_memory import (
    UserIntentMemoryRepository,
    UserIntentMemoryService,
)
from app.domain.services.user_interaction_memory import (
    UserInteractionMemoryService,
)
from app.infrastructure.observability.conversation_trace import (
    current_conversation_trace,
    emit_stream_event,
    record_trace_event,
)
from app.infrastructure.llm.client import llm_upgrade_scope
from app.infrastructure.database.json.feature_store import FeatureStore, UserNotFoundError
from app.infrastructure.database.conversation_store import (
    ConversationStore,
    ConversationStoreError,
)
from app.models.interaction_memory import (
    ConversationFeedbackAnalysis,
    ConversationFeedbackEvent,
    UserInteractionMemoryProjection,
)
from app.models.personal_feedback import (
    ConversationFeedbackContext,
    ConversationResultSnapshot,
    ConversationResultSnapshotDraft,
    FeedbackAnalysis,
    FeedbackDecision,
    PersonalFeedbackEvent,
)


logger = structlog.get_logger()


class ServiceUnavailableError(RuntimeError):
    """表示本轮核心推荐链路不可用，且不暴露内部异常细节。"""


@dataclass
class _SessionLockEntry:
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    users: int = 0


@dataclass(frozen=True)
class _PreparedFeedback:
    """保存本轮已经保护且可交给 Graph 执行的反馈状态。"""

    context: ConversationFeedbackContext
    analysis: FeedbackAnalysis
    decision: FeedbackDecision
    event: PersonalFeedbackEvent


class ConversationService:
    """隔离不同会话，并由编排图处理每一轮对话。"""

    _DEFAULT_RECENT_HISTORY_MESSAGES = 12

    def __init__(
        self,
        *,
        user_store: FeatureStore,
        recall_agent: DocumentRecallAgent,
        rerank_agent: DocumentRerankAgent,
        intent_agent: IntentRecognitionAgent | None = None,
        profile_agent: UserProfileAgent | None = None,
        arbitrator: ConversationArbitrator | None = None,
        aggregator: DocumentResultAggregator | None = None,
        conversation_store: ConversationStore,
        enable_llm: bool | None = None,
        history_limit: int | None = None,
        knowledge_qa_service: KnowledgeQaService | None = None,
        intent_memory_repository: UserIntentMemoryRepository | None = None,
        intent_memory_service: UserIntentMemoryService | None = None,
        interaction_memory_repository: UserInteractionMemoryRepository
        | None = None,
        interaction_memory_service: UserInteractionMemoryService | None = None,
        interaction_memory_worker: UserInteractionMemoryWorker | None = None,
        interaction_feedback_agent: ConversationFeedbackAgent | None = None,
        feedback_agent: FeedbackRecoveryAgent | None = None,
        feedback_policy: FeedbackRecoveryPolicy | None = None,
        personal_feedback_learning: PersonalFeedbackLearningService | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if history_limit is not None and history_limit < 2:
            raise ValueError("history_limit 不能小于 2")
        self.user_store = user_store
        self.intent_agent = intent_agent or IntentRecognitionAgent(
            enable_llm=enable_llm
        )
        self.profile_agent = profile_agent
        self.arbitrator = arbitrator or ConversationArbitrator()
        self.recall_agent = recall_agent
        self.rerank_agent = rerank_agent
        self.aggregator = aggregator or DocumentResultAggregator()
        self.conversation_store = conversation_store
        self.summary_agent = ConversationSummaryAgent(enable_llm=enable_llm)
        self.knowledge_qa_service = knowledge_qa_service
        self.intent_memory_repository = intent_memory_repository
        self.intent_memory_service = (
            intent_memory_service or UserIntentMemoryService()
        )
        self.interaction_memory_repository = interaction_memory_repository
        self.interaction_memory_service = (
            interaction_memory_service or UserInteractionMemoryService()
        )
        self.interaction_memory_worker = interaction_memory_worker
        self.interaction_feedback_agent = interaction_feedback_agent
        self.feedback_agent = feedback_agent
        self.feedback_policy = feedback_policy or FeedbackRecoveryPolicy()
        self.personal_feedback_learning = personal_feedback_learning
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self.workflow = ConversationGraph(
            intent_agent=self.intent_agent,
            arbitrator=self.arbitrator,
            recall_agent=self.recall_agent,
            rerank_agent=self.rerank_agent,
            aggregator=self.aggregator,
            profile_agent=self.profile_agent,
            knowledge_qa_service=self.knowledge_qa_service,
            feedback_agent=self.feedback_agent,
            feedback_policy=self.feedback_policy,
        )
        self.history_limit = min(
            history_limit or self._DEFAULT_RECENT_HISTORY_MESSAGES,
            self._DEFAULT_RECENT_HISTORY_MESSAGES,
        )
        self.chat_request_timeout_seconds = getattr(
            get_settings(),
            "chat_request_timeout_seconds",
            45.0,
        )
        self._session_locks: dict[tuple[str, str], _SessionLockEntry] = {}
        self._intent_memory_locks: dict[str, _SessionLockEntry] = {}
        self._last_feedback_raw_purge_date: date | None = None

    async def get_session(
        self,
        user_id: str,
        session_id: str = "default",
    ) -> ConversationSession:
        """读取会话；不存在时返回尚未写盘的空会话。"""

        normalized_user_id = str(user_id)
        try:
            session = await self.conversation_store.load(
                normalized_user_id,
                session_id,
            )
        except ConversationStoreError:
            raise ServiceUnavailableError("文章推荐服务暂时不可用") from None
        return session or ConversationSession(
            session_id=session_id,
            user_id=normalized_user_id,
        )

    async def read_session(
        self,
        user_id: str,
        session_id: str = "default",
    ) -> ConversationSession:
        """校验用户后读取会话，不存在时返回未写盘的空会话。"""

        normalized_user_id = str(user_id)
        await self._ensure_user_exists(normalized_user_id)
        return await self.get_session(normalized_user_id, session_id)

    async def reset_session(self, user_id: str, session_id: str = "default") -> None:
        normalized_user_id = str(user_id)
        await self._ensure_user_exists(normalized_user_id)
        key = (normalized_user_id, session_id)
        async with self._session_guard(key):
            try:
                await self.conversation_store.delete(normalized_user_id, session_id)
            except ConversationStoreError:
                raise ServiceUnavailableError("文章推荐服务暂时不可用") from None

    async def chat(
        self,
        user_id: str,
        message: str,
        *,
        session_id: str | None = None,
    ) -> ConversationReply:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.chat_request_timeout_seconds
        normalized_user_id = str(user_id)
        await self._ensure_user_exists(normalized_user_id)
        resolved_session_id = session_id or create_session_id()
        trace = current_conversation_trace()
        if trace is not None:
            trace.set_resolved_session_id(resolved_session_id)
        record_trace_event(
            "session.resolved",
            "conversation_service",
            output_data={"session_id": resolved_session_id},
        )
        preliminary = await self.get_session(
            normalized_user_id,
            resolved_session_id,
        )
        session_ids = (
            [preliminary.parent_session_id, resolved_session_id]
            if preliminary.session_type == "article_qa"
            and preliminary.parent_session_id is not None
            else [resolved_session_id]
        )
        keys = [
            (normalized_user_id, current_session_id)
            for current_session_id in session_ids
        ]
        async with self._session_guards(keys):
            session = await self.get_session(normalized_user_id, resolved_session_id)
            parent_session = (
                await self.get_session(
                    normalized_user_id,
                    session.parent_session_id,
                )
                if session.session_type == "article_qa"
                and session.parent_session_id is not None
                else None
            )
            record_trace_event(
                "session.loaded",
                "conversation_service",
                input_data={
                    "user_id": normalized_user_id,
                    "session_id": resolved_session_id,
                },
                output_data={"session": session},
            )
            previous_intent_state = session.intent_state
            previous_feedback_exchange = self._latest_feedback_exchange(
                session.history
            )
            prepared_feedback = None
            preanalyzed_interaction = None
            if (
                session.session_type == "article_qa"
                and parent_session is not None
                and self._is_explicit_child_close(message)
            ):
                return await self._close_article_child(
                    child=session,
                    parent=parent_session,
                    message=message,
                    deadline=deadline,
                )
            try:
                feedback_context = await self._load_feedback_context(
                    normalized_user_id,
                    resolved_session_id,
                )
                duplicate_feedback_reply = await self._processed_feedback_reply(
                    session=session,
                    message=message,
                    feedback_context=feedback_context,
                )
                if duplicate_feedback_reply is not None:
                    record_trace_event(
                        "service.completed",
                        "conversation_service",
                        output_data={"reply": duplicate_feedback_reply},
                        status="success",
                    )
                    return duplicate_feedback_reply
                prepared_feedback = await self._prepare_feedback(
                    session=session,
                    message=message,
                    feedback_context=feedback_context,
                    deadline=deadline,
                )
                if (
                    prepared_feedback is not None
                    and prepared_feedback.decision.feedback_type == "answer_style"
                ):
                    preanalyzed_interaction = (
                        await self._analyze_interaction_feedback(
                            user_id=normalized_user_id,
                            session_id=resolved_session_id,
                            feedback_id=prepared_feedback.event.feedback_id,
                            feedback_message=message,
                            previous_exchange=previous_feedback_exchange,
                            deadline=deadline,
                        )
                    )
                workflow_kwargs = {
                    "user_id": normalized_user_id,
                    "session_id": resolved_session_id,
                    "message": message,
                    "history": session.history[-self.history_limit :],
                    "previous_context": (
                        parent_session.active_context
                        if parent_session is not None
                        else session.active_context
                    ),
                    "conversation_summary": session.summary,
                    "intent_state": session.intent_state,
                }
                if prepared_feedback is not None:
                    workflow_kwargs.update(
                        {
                            "feedback_context": prepared_feedback.context,
                            "protected_feedback_analysis": (
                                prepared_feedback.analysis
                            ),
                            "protected_feedback_decision": (
                                prepared_feedback.decision
                            ),
                        }
                    )
                intent_memory = await self._load_intent_memory_projection(
                    normalized_user_id
                )
                if intent_memory is not None:
                    workflow_kwargs["intent_memory"] = intent_memory
                interaction_memory = (
                    await self._load_interaction_memory_projection(
                        normalized_user_id
                    )
                )
                if preanalyzed_interaction is not None:
                    interaction_memory = self._merge_interaction_memory(
                        self.interaction_memory_service.project_analysis(
                            preanalyzed_interaction
                        ),
                        interaction_memory,
                    )
                if interaction_memory is not None:
                    workflow_kwargs["interaction_memory"] = interaction_memory
                if session.session_type == "article_qa":
                    workflow_kwargs["knowledge_document_ids"] = (
                        session.focus_document_id,
                    )
                with llm_upgrade_scope(deadline=deadline):
                    transition = await asyncio.wait_for(
                        self.workflow.run(**workflow_kwargs),
                        timeout=self._remaining_seconds(deadline),
                    )
            except Exception as exc:
                if (
                    prepared_feedback is not None
                    and prepared_feedback.decision.next_action != "clarify"
                ):
                    logger.warning(
                        "个人反馈补救执行失败，提交失败终态",
                        error_type=type(exc).__name__,
                    )
                    reply = await self._commit_feedback_failure(
                        session=session,
                        message=message,
                        prepared=prepared_feedback,
                        deadline=deadline,
                    )
                    record_trace_event(
                        "service.completed",
                        "conversation_service",
                        output_data={"reply": reply},
                        status="degraded",
                    )
                    return reply
                record_trace_event(
                    "service.failed",
                    "conversation_service",
                    status="error",
                    error=exc,
                )
                logger.error(
                    "会话编排执行失败",
                    error_type=type(exc).__name__,
                )
                raise ServiceUnavailableError(
                    "文章推荐服务暂时不可用"
                ) from None

            if transition.error_stage:
                record_trace_event(
                    "service.failed",
                    "conversation_service",
                    status="error",
                    error={"type": "WorkflowFailure"},
                    output_data={"error_stage": transition.error_stage},
                )
                logger.error(
                    "会话编排返回失败状态",
                    error_stage=transition.error_stage,
                )
                raise ServiceUnavailableError(
                    "文章推荐服务暂时不可用"
                ) from None

            if prepared_feedback is not None:
                reply = await self._commit_feedback_session(
                    session=session,
                    message=message,
                    transition=transition,
                    prepared=prepared_feedback,
                    deadline=deadline,
                )
            elif (
                session.session_type == "article_qa"
                and parent_session is not None
                and transition.commit_intent_state
                and transition.pending_intent_state is IntentState.RECOMMENDATION
            ):
                reply = await self._return_to_parent(
                    child=session,
                    parent=parent_session,
                    message=message,
                    transition=transition,
                    deadline=deadline,
                )
            elif self._should_create_article_child(session, transition):
                reply = await self._enter_article_child(
                    parent=session,
                    message=message,
                    transition=transition,
                    deadline=deadline,
                )
            else:
                reply = await self._commit_current_session(
                    session=session,
                    message=message,
                    transition=transition,
                    deadline=deadline,
                )
            await self._record_intent_memory(
                user_id=normalized_user_id,
                message=message,
                action=reply.action,
                previous_intent_state=previous_intent_state,
                current_intent_state=reply.intent_state,
            )
            await self._record_interaction_feedback(
                user_id=normalized_user_id,
                session_id=session.session_id,
                feedback_message=message,
                previous_exchange=previous_feedback_exchange,
                preanalyzed=preanalyzed_interaction,
                event_id=(
                    f"personal-{prepared_feedback.event.feedback_id}"
                    if prepared_feedback is not None
                    and preanalyzed_interaction is not None
                    else None
                ),
            )
            if (
                prepared_feedback is not None
                and getattr(transition, "feedback_recovery_succeeded", None) is True
            ):
                await self._apply_personal_feedback_learning(
                    session=session,
                    feedback_id=prepared_feedback.event.feedback_id,
                )
            record_trace_event(
                "service.completed",
                "conversation_service",
                output_data={"reply": reply},
                status="success",
            )
            return reply

    async def _apply_personal_feedback_learning(
        self,
        *,
        session: ConversationSession,
        feedback_id: str,
    ) -> None:
        """在修正结果提交后尽力学习，并只推进终态事件的记忆状态。"""

        if self.personal_feedback_learning is None:
            return
        try:
            event = next(
                (
                    item
                    for item in await self.conversation_store.list_feedback_events(
                        session.user_id,
                        limit=100,
                    )
                    if item.feedback_id == feedback_id
                ),
                None,
            )
            if event is None or event.status != "recovered":
                return
            statuses = await self.personal_feedback_learning.apply(event=event)
            if statuses == event.memory_statuses:
                return
            updated = event.model_copy(
                update={
                    "memory_statuses": statuses,
                    "updated_at": self._now(),
                },
                deep=True,
            )
            await self._save_sessions([session], feedback_events=(updated,))
            emit_stream_event(
                stage="个人记忆更新",
                component="personal_feedback_learning",
                status=(
                    "degraded"
                    if "degraded" in statuses.values()
                    else "success"
                ),
                title="个人反馈记忆状态已更新",
                summary="仅更新个人回答方式、意图或推荐画像状态",
                details={"memory_statuses": statuses},
            )
        except Exception as exc:
            logger.warning(
                "个人反馈学习失败，不撤销修正结果",
                exception_type=type(exc).__name__,
            )
            emit_stream_event(
                stage="个人记忆更新",
                component="personal_feedback_learning",
                status="degraded",
                title="个人反馈学习暂未完成",
                summary="修正结果已保留，后续可重新处理记忆状态",
                details={"error_type": type(exc).__name__},
            )

    async def _load_feedback_context(
        self,
        user_id: str,
        session_id: str,
    ) -> ConversationFeedbackContext:
        """读取最小反馈上下文；旧 Store 未实现新契约时按无反馈继续。"""

        loader = getattr(self.conversation_store, "load_feedback_context", None)
        if not callable(loader):
            return ConversationFeedbackContext()
        try:
            return ConversationFeedbackContext.model_validate(
                await loader(user_id, session_id)
            )
        except ConversationStoreError:
            raise ServiceUnavailableError("文章推荐服务暂时不可用") from None

    async def _processed_feedback_reply(
        self,
        *,
        session: ConversationSession,
        message: str,
        feedback_context: ConversationFeedbackContext,
    ) -> ConversationReply | None:
        """相邻重复反馈直接返回已提交结果，不再次追加消息或执行动作。"""

        if len(session.history) < 2 or feedback_context.pending_feedback is not None:
            return None
        previous_user = session.history[-2]
        previous_assistant = session.history[-1]
        if (
            previous_user.role != "user"
            or previous_user.content != message
            or previous_assistant.role != "assistant"
            or not previous_assistant.content.startswith(
                ("已根据你的反馈修正", "本次补救未完成")
            )
        ):
            return None
        try:
            events = await self.conversation_store.list_feedback_events(
                session.user_id,
                limit=100,
            )
        except (AttributeError, ConversationStoreError):
            return None
        matching = next(
            (
                event
                for event in events
                if event.session_id == session.session_id
                and event.status in {"recovered", "recovery_failed"}
                and event.feedback_message is not None
                and (
                    event.feedback_message == message
                    or event.feedback_message.endswith(f"补充：{message}")
                )
            ),
            None,
        )
        if matching is None:
            return None
        public_message = previous_assistant.content
        for separator in ("\n参考资料：", " 推荐结果："):
            public_message = public_message.split(separator, maxsplit=1)[0]
        succeeded = matching.status == "recovered"
        reply = ConversationReply(
            session_id=session.session_id,
            message=public_message,
            intent_source=RecognitionSource.RULE,
            action=(
                ArbitrationAction.REFINE
                if feedback_context.latest_result is not None
                and feedback_context.latest_result.result_type == "recommendation"
                else ArbitrationAction.KNOWLEDGE_ANSWER
            ),
            intent_state=session.intent_state,
            active_context=session.active_context,
            agent_statuses={
                "feedback_recovery": "success" if succeeded else "failed"
            },
        )
        return self._reply_for_session(
            reply,
            session,
            compression=self._compression_info(session),
        )

    async def _prepare_feedback(
        self,
        *,
        session: ConversationSession,
        message: str,
        feedback_context: ConversationFeedbackContext,
        deadline: float,
    ) -> _PreparedFeedback | None:
        """先持久化分类状态，再返回最多一个可执行反馈动作。"""

        context = feedback_context.model_copy(deep=True)
        pending = context.pending_feedback
        if pending is not None and pending.status == "recovering":
            failed = pending.model_copy(
                update={
                    "status": "recovery_failed",
                    "reason_code": "interrupted_recovery_not_replayed",
                    "updated_at": self._now(),
                },
                deep=True,
            )
            session.pending_feedback_id = None
            await self._save_sessions([session], feedback_events=(failed,))
            return None
        snapshot = context.latest_result
        if not self.feedback_policy.is_candidate(
            message,
            snapshot=snapshot,
            pending_event=pending,
        ):
            return None
        if snapshot is None:
            return None

        emit_stream_event(
            stage="反馈检测",
            component="personal_feedback",
            status="success",
            title="检测到可能的结果反馈",
            summary=f"目标结果类型：{snapshot.result_type}",
            details={
                "result_type": snapshot.result_type,
                "has_pending_feedback": pending is not None,
            },
        )

        await self._purge_feedback_raw_if_due()

        event = pending
        if event is None:
            now = self._now()
            event = PersonalFeedbackEvent(
                feedback_id=uuid4().hex,
                user_id=session.user_id,
                session_id=session.session_id,
                source_result_id=snapshot.result_id,
                feedback_message=message,
                status="classifying",
                reason_code="classifying_pending",
                created_at=now,
                updated_at=now,
            )
            session.pending_feedback_id = event.feedback_id
            await self._save_sessions([session], feedback_events=(event,))
            context = context.model_copy(
                update={"pending_feedback": event},
                deep=True,
            )

        analysis, decision = await asyncio.wait_for(
            self.workflow.classify_feedback(
                message=message,
                history=session.history[-self.history_limit :],
                feedback_context=context,
            ),
            timeout=self._remaining_seconds(deadline),
        )
        protected_decision = FeedbackDecision.model_validate(decision).model_copy(
            deep=True
        )
        emit_stream_event(
            stage="反馈分类",
            component="personal_feedback",
            status="success",
            title="反馈分类与动作保护完成",
            summary=protected_decision.feedback_type,
            details={
                "feedback_type": protected_decision.feedback_type,
                "next_action": protected_decision.next_action,
                "reason_code": protected_decision.reason_code,
            },
        )
        emit_stream_event(
            stage="信息完整度",
            component="personal_feedback",
            status="success",
            title=(
                "反馈信息完整"
                if protected_decision.completeness == "complete"
                else "反馈信息需要补充"
            ),
            summary=protected_decision.completeness,
            details={
                "completeness": protected_decision.completeness,
                "clarification_required": (
                    protected_decision.next_action == "clarify"
                ),
            },
        )
        if not protected_decision.is_feedback:
            closed = self._feedback_event_from_decision(
                event,
                message=message,
                decision=protected_decision,
                status="closed",
            )
            session.pending_feedback_id = None
            await self._save_sessions([session], feedback_events=(closed,))
            return None

        protected_analysis = self._protected_feedback_analysis(
            analysis,
            protected_decision,
        )
        if protected_decision.next_action == "clarify":
            awaiting = self._feedback_event_from_decision(
                event,
                message=message,
                decision=protected_decision,
                status="awaiting_detail",
                clarification_count=1,
            )
            return _PreparedFeedback(
                context=context.model_copy(
                    update={"pending_feedback": awaiting},
                    deep=True,
                ),
                analysis=protected_analysis,
                decision=protected_decision,
                event=awaiting,
            )

        recovering = self._feedback_event_from_decision(
            event,
            message=message,
            decision=protected_decision,
            status="recovering",
            recovery_count=1,
        )
        await self._save_sessions([session], feedback_events=(recovering,))
        emit_stream_event(
            stage="补救准备",
            component="personal_feedback",
            status="success",
            title="补救状态已持久化",
            summary=protected_decision.next_action,
            details={
                "next_action": protected_decision.next_action,
                "recovery_count": recovering.recovery_count,
            },
        )
        return _PreparedFeedback(
            context=context.model_copy(
                update={"pending_feedback": recovering},
                deep=True,
            ),
            analysis=protected_analysis,
            decision=protected_decision,
            event=recovering,
        )

    async def _purge_feedback_raw_if_due(self) -> None:
        """同一自然日首次反馈请求尽力清理超过30天的原文。"""

        now = self._now()
        if self._last_feedback_raw_purge_date == now.date():
            return
        purge = getattr(
            self.conversation_store,
            "purge_feedback_raw_before",
            None,
        )
        if not callable(purge):
            self._last_feedback_raw_purge_date = now.date()
            return
        try:
            await purge(now - timedelta(days=30), purged_at=now)
            self._last_feedback_raw_purge_date = now.date()
        except Exception as exc:
            logger.warning(
                "个人反馈原文清理失败，保留数据等待下次请求",
                exception_type=type(exc).__name__,
            )

    @staticmethod
    def _protected_feedback_analysis(
        analysis: FeedbackAnalysis | None,
        decision: FeedbackDecision,
    ) -> FeedbackAnalysis:
        """把确定性降级追问表示成同一严格分析契约。"""

        if analysis is not None:
            return FeedbackAnalysis.model_validate(analysis).model_copy(deep=True)
        return FeedbackAnalysis(
            is_feedback=True,
            feedback_type=decision.feedback_type,
            completeness=decision.completeness,
            corrected_query=decision.protected_query,
            target_document_ids=decision.target_document_ids,
            missing_information=("reason",),
            suggested_action=decision.next_action,
            reason_code=decision.reason_code,
            confidence=1.0,
        )

    def _feedback_event_from_decision(
        self,
        event: PersonalFeedbackEvent,
        *,
        message: str,
        decision: FeedbackDecision,
        status: str,
        clarification_count: int | None = None,
        recovery_count: int | None = None,
    ) -> PersonalFeedbackEvent:
        """在保持事件身份和计数单调的前提下提交受保护决策。"""

        feedback_message = event.feedback_message
        if feedback_message != message:
            feedback_message = " ".join(
                part
                for part in (feedback_message, f"补充：{message}")
                if part
            )[:4000]
        memory_statuses = {
            route: "pending" for route in decision.memory_routes
        }
        return event.model_copy(
            update={
                "feedback_message": feedback_message,
                "feedback_type": decision.feedback_type,
                "completeness": decision.completeness,
                "corrected_query": decision.protected_query,
                "target_document_ids": decision.target_document_ids,
                "next_action": decision.next_action,
                "status": status,
                "clarification_count": (
                    event.clarification_count
                    if clarification_count is None
                    else clarification_count
                ),
                "recovery_count": (
                    event.recovery_count
                    if recovery_count is None
                    else recovery_count
                ),
                "recommendation_signals": decision.recommendation_signals,
                "memory_routes": decision.memory_routes,
                "memory_statuses": memory_statuses,
                "reason_code": decision.reason_code,
                "updated_at": self._now(),
            },
            deep=True,
        )

    async def _load_intent_memory_projection(self, user_id: str):
        """读取并投影长期记忆；异常时按无记忆继续当前请求。"""

        if self.intent_memory_repository is None:
            emit_stream_event(
                stage="记忆",
                component="intent_memory",
                status="skipped",
                title="长期意图记忆未配置",
                summary="本轮按当前消息与会话上下文继续",
            )
            return None
        try:
            async with self._intent_memory_guard(user_id):
                memory = await asyncio.to_thread(
                    self.intent_memory_repository.get,
                    user_id,
                )
            if memory is None:
                emit_stream_event(
                    stage="记忆",
                    component="intent_memory",
                    status="success",
                    title="长期意图记忆读取完成",
                    summary="当前用户尚无可用长期意图记忆",
                    details={"projection_available": False},
                )
                return None
            projection = self.intent_memory_service.project(memory)
            if (
                projection.default_recommendation_size is None
                and projection.dominant_intent is None
                and not projection.corrections
            ):
                emit_stream_event(
                    stage="记忆",
                    component="intent_memory",
                    status="success",
                    title="长期意图记忆读取完成",
                    summary="当前没有需要注入编排的偏好",
                    details={"projection_available": False},
                )
                return None
            emit_stream_event(
                stage="记忆",
                component="intent_memory",
                status="success",
                title="长期意图记忆读取完成",
                summary="已生成安全意图偏好投影",
                details={
                    "projection_available": True,
                    "default_recommendation_size": (
                        projection.default_recommendation_size
                    ),
                    "dominant_intent": projection.dominant_intent,
                    "correction_count": len(projection.corrections),
                },
            )
            return projection
        except Exception as exc:
            logger.warning(
                "用户意图记忆读取失败，按无长期记忆继续",
                exception_type=type(exc).__name__,
            )
            emit_stream_event(
                stage="记忆",
                component="intent_memory",
                status="degraded",
                title="长期意图记忆读取失败",
                summary="按无长期记忆继续",
                details={"error_type": type(exc).__name__},
            )
            return None

    async def _load_interaction_memory_projection(self, user_id: str):
        """加载回答习惯白名单投影；失败时按无习惯继续。"""

        if self.interaction_memory_repository is None:
            emit_stream_event(
                stage="记忆",
                component="interaction_memory",
                status="skipped",
                title="回答偏好记忆未配置",
                summary="本轮使用默认回答方式",
            )
            return None
        try:
            memory = await asyncio.to_thread(
                self.interaction_memory_repository.get_memory,
                user_id,
            )
            if memory is None:
                emit_stream_event(
                    stage="记忆",
                    component="interaction_memory",
                    status="success",
                    title="回答偏好记忆读取完成",
                    summary="当前用户尚无已学习偏好",
                    details={"projection_available": False},
                )
                return None
            projection = self.interaction_memory_service.project(memory)
            available = bool(projection.preferences)
            emit_stream_event(
                stage="记忆",
                component="interaction_memory",
                status="success",
                title="回答偏好记忆读取完成",
                summary=(
                    "已生成安全回答偏好投影"
                    if available
                    else "当前没有需要注入问答链的偏好"
                ),
                details={
                    "projection_available": available,
                    "preference_count": len(projection.preferences),
                },
            )
            return projection if available else None
        except Exception as exc:
            logger.warning(
                "用户交互记忆读取失败，按默认回答方式继续",
                exception_type=type(exc).__name__,
            )
            emit_stream_event(
                stage="记忆",
                component="interaction_memory",
                status="degraded",
                title="回答偏好记忆读取失败",
                summary="按默认回答方式继续",
                details={"error_type": type(exc).__name__},
            )
            return None

    async def _record_interaction_feedback(
        self,
        *,
        user_id: str,
        session_id: str,
        feedback_message: str,
        previous_exchange: tuple[str, str] | None,
        preanalyzed: ConversationFeedbackAnalysis | None = None,
        event_id: str | None = None,
    ) -> None:
        """在当前回复提交后尽力保存相邻反馈窗口。"""

        if (
            self.interaction_memory_repository is None
            or previous_exchange is None
            or (
                preanalyzed is None
                and not self.interaction_memory_service.is_feedback_candidate(
                    feedback_message,
                    has_previous_exchange=True,
                )
            )
        ):
            emit_stream_event(
                stage="记忆",
                component="interaction_memory",
                status="skipped",
                title="本轮无需记录交互反馈",
                summary="当前消息不是可学习的相邻反馈",
            )
            return
        previous_user_message, previous_assistant_message = previous_exchange
        try:
            event = ConversationFeedbackEvent(
                event_id=event_id or uuid4().hex,
                user_id=user_id,
                session_id=session_id,
                previous_user_message=previous_user_message,
                previous_assistant_message=previous_assistant_message,
                feedback_message=feedback_message,
                occurred_at=self._now(),
            )
            existing_getter = getattr(
                self.interaction_memory_repository,
                "get_event",
                None,
            )
            existing = (
                await asyncio.to_thread(existing_getter, event.event_id)
                if callable(existing_getter)
                else None
            )
            if existing is not None and existing.status == "analyzed":
                return
            await asyncio.to_thread(
                self.interaction_memory_repository.append_event,
                event,
            )
            if preanalyzed is not None:
                analysis = ConversationFeedbackAnalysis.model_validate(
                    preanalyzed
                )
                memory = await asyncio.to_thread(
                    self.interaction_memory_repository.get_memory,
                    user_id,
                )
                if memory is None:
                    memory = self.interaction_memory_service.empty(user_id)
                updated = self.interaction_memory_service.apply_analysis(
                    memory,
                    event=event,
                    analysis=analysis,
                )
                await asyncio.to_thread(
                    self.interaction_memory_repository.save_memory,
                    updated,
                )
                analyzed_event = ConversationFeedbackEvent.model_validate(
                    event.model_copy(
                        update={
                            "status": "analyzed",
                            "analysis": analysis,
                            "analysis_attempts": 1,
                            "analyzed_at": self._now(),
                        }
                    )
                )
                await asyncio.to_thread(
                    self.interaction_memory_repository.save_event,
                    analyzed_event,
                )
                emit_stream_event(
                    stage="记忆",
                    component="interaction_memory",
                    status="success",
                    title="回答方式偏好已即时更新",
                    summary="当前明确要求已保存为低优先级个人偏好",
                )
                return
            if self.interaction_memory_worker is not None:
                self.interaction_memory_worker.wake()
            emit_stream_event(
                stage="记忆",
                component="interaction_memory",
                status="success",
                title="交互反馈已提交",
                summary="后台偏好学习任务已唤醒",
            )
        except Exception as exc:
            logger.warning(
                "用户交互反馈记录失败，不影响当前业务结果",
                exception_type=type(exc).__name__,
            )
            emit_stream_event(
                stage="记忆",
                component="interaction_memory",
                status="degraded",
                title="交互反馈提交失败",
                summary="不影响当前聊天结果",
                details={"error_type": type(exc).__name__},
            )

    async def _analyze_interaction_feedback(
        self,
        *,
        user_id: str,
        session_id: str,
        feedback_id: str,
        feedback_message: str,
        previous_exchange: tuple[str, str] | None,
        deadline: float,
    ) -> ConversationFeedbackAnalysis | None:
        """用现有风格 Agent 同步分析回答方式，不复用质量分类结果。"""

        if self.interaction_feedback_agent is None or previous_exchange is None:
            return None
        previous_user_message, previous_assistant_message = previous_exchange
        event = ConversationFeedbackEvent(
            event_id=f"personal-{feedback_id}",
            user_id=user_id,
            session_id=session_id,
            previous_user_message=previous_user_message,
            previous_assistant_message=previous_assistant_message,
            feedback_message=feedback_message,
            occurred_at=self._now(),
        )
        try:
            return await asyncio.wait_for(
                self.interaction_feedback_agent.analyze(event),
                timeout=self._remaining_seconds(deadline),
            )
        except Exception as exc:
            logger.warning(
                "回答方式反馈分析失败，按无即时风格继续补救",
                error_type=type(exc).__name__,
            )
            emit_stream_event(
                stage="个人记忆更新",
                component="interaction_feedback_agent",
                status="degraded",
                title="回答方式偏好暂未解析",
                summary="本轮补救继续使用默认回答方式",
                details={"error_type": type(exc).__name__},
            )
            return None

    @staticmethod
    def _merge_interaction_memory(
        current: UserInteractionMemoryProjection,
        long_term: UserInteractionMemoryProjection | None,
    ) -> UserInteractionMemoryProjection:
        """本轮明确要求优先，并把长期白名单投影补足到最多三条。"""

        current_projection = UserInteractionMemoryProjection.model_validate(current)
        long_projection = (
            UserInteractionMemoryProjection.model_validate(long_term)
            if long_term is not None
            else UserInteractionMemoryProjection()
        )
        preferences = []
        seen_scopes: set[str] = set()
        for item in (*current_projection.preferences, *long_projection.preferences):
            if item.scope in seen_scopes:
                continue
            preferences.append(item.model_copy(deep=True))
            seen_scopes.add(item.scope)
            if len(preferences) == 3:
                break
        return UserInteractionMemoryProjection(preferences=preferences)

    async def _record_intent_memory(
        self,
        *,
        user_id: str,
        message: str,
        action: ArbitrationAction,
        previous_intent_state: IntentState,
        current_intent_state: IntentState,
    ) -> None:
        """在业务与 Session 均已提交后，尽力更新长期意图记忆。"""

        if self.intent_memory_repository is None or action not in {
            ArbitrationAction.NEW,
            ArbitrationAction.REFINE,
            ArbitrationAction.REPEAT,
            ArbitrationAction.KNOWLEDGE_ANSWER,
        }:
            emit_stream_event(
                stage="记忆",
                component="intent_memory",
                status="skipped",
                title="本轮无需更新长期意图记忆",
                summary=f"当前动作 {action.value} 不进入成功记忆",
            )
            return
        try:
            async with self._intent_memory_guard(user_id):
                current = await asyncio.to_thread(
                    self.intent_memory_repository.get,
                    user_id,
                )
                if current is None:
                    current = self.intent_memory_service.empty(user_id)
                updated = self.intent_memory_service.record_success(
                    current,
                    message=message,
                    action=action,
                    previous_intent_state=previous_intent_state,
                    current_intent_state=current_intent_state,
                )
                await asyncio.to_thread(
                    self.intent_memory_repository.save,
                    updated,
                )
            emit_stream_event(
                stage="记忆",
                component="intent_memory",
                status="success",
                title="长期意图记忆已更新",
                summary=f"已记录成功动作 {action.value}",
                details={
                    "action": action,
                    "previous_intent_state": previous_intent_state,
                    "current_intent_state": current_intent_state,
                },
            )
        except Exception as exc:
            logger.warning(
                "用户意图记忆更新失败，不影响当前业务结果",
                exception_type=type(exc).__name__,
            )
            emit_stream_event(
                stage="记忆",
                component="intent_memory",
                status="degraded",
                title="长期意图记忆更新失败",
                summary="不影响当前聊天结果",
                details={"error_type": type(exc).__name__},
            )

    @staticmethod
    def _should_create_article_child(
        session: ConversationSession,
        transition: object,
    ) -> bool:
        """只为主会话中成功定位到唯一文章的知识回答创建子会话。"""

        return (
            session.session_type == "main"
            and transition.reply.action is ArbitrationAction.KNOWLEDGE_ANSWER
            and not transition.reply.needs_clarification
            and len(getattr(transition, "knowledge_document_ids", ())) == 1
            and len(getattr(transition, "knowledge_document_titles", ())) == 1
        )

    @staticmethod
    def _is_explicit_child_close(message: str) -> bool:
        """识别不需要调用 LLM 或推荐链的明确子会话退出命令。"""

        normalized = " ".join(str(message).strip().split()).rstrip("。！？!?")
        return normalized in {
            "结束问答",
            "结束文章问答",
            "返回主会话",
            "退出问答",
        }

    async def _close_article_child(
        self,
        *,
        child: ConversationSession,
        parent: ConversationSession,
        message: str,
        deadline: float,
    ) -> ConversationReply:
        """显式关闭子会话并只向父会话追加结构化交接事件。"""

        response_message = (
            f"已结束《{child.focus_document_title}》文章问答并返回主会话。"
        )
        self._record_turn(child, message, response_message)
        await self._compress_session(child, deadline=deadline)
        child.session_status = "closed"
        child.handoff_summary = self._build_handoff_summary(child)
        parent.active_child_session_id = None
        parent.history.append(
            ConversationTurn(
                role="assistant",
                content=child.handoff_summary,
                message_type="child_handoff",
                related_session_id=child.session_id,
            )
        )
        compression = await self._compress_session(parent, deadline=deadline)
        await self._save_sessions([child, parent])
        reply = ConversationReply(
            session_id=parent.session_id,
            message=response_message,
            intent_source=RecognitionSource.RULE,
            action=ArbitrationAction.RETURN_TO_PARENT,
            intent_state=parent.intent_state,
            active_context=parent.active_context,
        )
        return self._reply_for_session(reply, parent, compression=compression)

    async def _commit_current_session(
        self,
        *,
        session: ConversationSession,
        message: str,
        transition: object,
        deadline: float,
    ) -> ConversationReply:
        """提交不发生父子切换的普通主会话或子会话轮次。"""

        self._apply_transition(session, transition)
        self._record_turn(session, message, transition.history_message)
        if session.session_type == "article_qa":
            self._update_child_knowledge_state(session, transition.reply)
        compression = await self._compress_session(session, deadline=deadline)
        snapshot = self._build_result_snapshot(
            session,
            getattr(transition, "result_snapshot_draft", None),
        )
        await self._save_sessions(
            [session],
            snapshots=(snapshot,) if snapshot is not None else (),
        )
        return self._reply_for_session(
            transition.reply,
            session,
            compression=compression,
        )

    async def _commit_feedback_session(
        self,
        *,
        session: ConversationSession,
        message: str,
        transition: object,
        prepared: _PreparedFeedback,
        deadline: float,
    ) -> ConversationReply:
        """追加追问或修正结果，并原子提交反馈生命周期状态。"""

        self._apply_transition(session, transition)
        self._record_turn(session, message, transition.history_message)
        if session.session_type == "article_qa":
            self._update_child_knowledge_state(session, transition.reply)
        compression = await self._compress_session(session, deadline=deadline)

        if prepared.decision.next_action == "clarify":
            await self._save_sessions(
                [session],
                feedback_events=(prepared.event,),
            )
            return self._reply_for_session(
                transition.reply,
                session,
                compression=compression,
            )

        snapshot = None
        if getattr(transition, "feedback_recovery_succeeded", None) is True:
            snapshot = self._build_result_snapshot(
                session,
                getattr(transition, "result_snapshot_draft", None),
            )
        succeeded = snapshot is not None
        terminal_event = prepared.event.model_copy(
            update={
                "status": "recovered" if succeeded else "recovery_failed",
                "recovery_result_id": snapshot.result_id if snapshot else None,
                "reason_code": (
                    prepared.event.reason_code
                    if succeeded
                    else "recovery_action_failed"
                ),
                "updated_at": self._now(),
            },
            deep=True,
        )
        session.pending_feedback_id = None
        await self._save_sessions(
            [session],
            snapshots=(snapshot,) if snapshot is not None else (),
            feedback_events=(terminal_event,),
        )
        emit_stream_event(
            stage="修正结果提交",
            component="personal_feedback",
            status="success" if succeeded else "degraded",
            title="修正结果已追加" if succeeded else "补救失败状态已提交",
            summary=terminal_event.status,
            details={
                "feedback_status": terminal_event.status,
                "result_snapshot_created": snapshot is not None,
                "recovery_count": terminal_event.recovery_count,
            },
        )
        return self._reply_for_session(
            transition.reply,
            session,
            compression=compression,
        )

    async def _commit_feedback_failure(
        self,
        *,
        session: ConversationSession,
        message: str,
        prepared: _PreparedFeedback,
        deadline: float,
    ) -> ConversationReply:
        """补救动作异常时追加安全失败消息并清理待处理状态。"""

        response_message = "本次补救未完成：相关服务暂时不可用，请稍后再试。"
        self._record_turn(session, message, response_message)
        compression = await self._compress_session(session, deadline=deadline)
        failed_event = prepared.event.model_copy(
            update={
                "status": "recovery_failed",
                "recovery_result_id": None,
                "reason_code": "recovery_action_failed",
                "updated_at": self._now(),
            },
            deep=True,
        )
        session.pending_feedback_id = None
        await self._save_sessions(
            [session],
            feedback_events=(failed_event,),
        )
        emit_stream_event(
            stage="修正结果提交",
            component="personal_feedback",
            status="degraded",
            title="补救失败状态已提交",
            summary=failed_event.status,
            details={
                "feedback_status": failed_event.status,
                "result_snapshot_created": False,
                "recovery_count": failed_event.recovery_count,
            },
        )
        reply = ConversationReply(
            session_id=session.session_id,
            message=response_message,
            intent_source=RecognitionSource.RULE,
            action=(
                ArbitrationAction.REFINE
                if prepared.decision.next_action == "retry_recommendation"
                else ArbitrationAction.KNOWLEDGE_ANSWER
            ),
            intent_state=session.intent_state,
            active_context=session.active_context,
            agent_statuses={"feedback_recovery": "failed"},
        )
        return self._reply_for_session(
            reply,
            session,
            compression=compression,
        )

    async def _enter_article_child(
        self,
        *,
        parent: ConversationSession,
        message: str,
        transition: object,
        deadline: float,
    ) -> ConversationReply:
        """把已定位到唯一文章的首轮问答只写入新子会话。"""

        document_id = tuple(transition.knowledge_document_ids)[0]
        document_title = tuple(transition.knowledge_document_titles)[0]
        child = ConversationSession(
            session_id=create_session_id(),
            user_id=parent.user_id,
            session_type="article_qa",
            parent_session_id=parent.session_id,
            focus_document_id=document_id,
            focus_document_title=document_title,
            intent_state=IntentState.KNOWLEDGE_QA,
        )
        self._record_turn(child, message, transition.history_message)
        self._update_child_knowledge_state(child, transition.reply)
        compression = await self._compress_session(child, deadline=deadline)
        snapshot = self._build_result_snapshot(
            child,
            getattr(transition, "result_snapshot_draft", None),
        )
        sessions_to_save: list[ConversationSession] = []
        if parent.active_child_session_id is not None:
            existing_child = await self.get_session(
                parent.user_id,
                parent.active_child_session_id,
            )
            if existing_child.session_type == "article_qa":
                existing_child.session_status = "suspended"
                sessions_to_save.append(existing_child)
        parent.active_child_session_id = child.session_id
        sessions_to_save.extend([parent, child])
        await self._save_sessions(
            sessions_to_save,
            snapshots=(snapshot,) if snapshot is not None else (),
        )
        return self._reply_for_session(
            transition.reply,
            child,
            compression=compression,
        )

    async def _return_to_parent(
        self,
        *,
        child: ConversationSession,
        parent: ConversationSession,
        message: str,
        transition: object,
        deadline: float,
    ) -> ConversationReply:
        """暂停文章子会话，写回交接事件并用父推荐上下文提交当前请求。"""

        child.session_status = "suspended"
        await self._compress_session(child, deadline=deadline)
        child.handoff_summary = self._build_handoff_summary(child)
        parent.active_child_session_id = None
        parent.history.append(
            ConversationTurn(
                role="assistant",
                content=child.handoff_summary,
                message_type="child_handoff",
                related_session_id=child.session_id,
            )
        )
        self._apply_transition(parent, transition)
        self._record_turn(parent, message, transition.history_message)
        compression = await self._compress_session(parent, deadline=deadline)
        snapshot = self._build_result_snapshot(
            parent,
            getattr(transition, "result_snapshot_draft", None),
        )
        await self._save_sessions(
            [child, parent],
            snapshots=(snapshot,) if snapshot is not None else (),
        )
        return self._reply_for_session(
            transition.reply,
            parent,
            compression=compression,
        )

    @staticmethod
    def _apply_transition(session: ConversationSession, transition: object) -> None:
        """只提交 Graph 明确允许更新的推荐上下文和意图状态。"""

        if transition.commit_context and transition.pending_context is not None:
            session.active_context = transition.pending_context.model_copy(deep=True)
        if (
            transition.commit_intent_state
            and transition.pending_intent_state is not None
        ):
            session.intent_state = transition.pending_intent_state

    @staticmethod
    def _update_child_knowledge_state(
        child: ConversationSession,
        reply: ConversationReply,
    ) -> None:
        """保存子会话引用身份和仍需澄清的问题，不把回答正文当事实。"""

        for citation in reply.citations:
            if citation.document_id not in child.cited_document_ids:
                child.cited_document_ids.append(citation.document_id)
        if reply.needs_clarification and child.history:
            question = child.history[-2].content
            if question not in child.unresolved_questions:
                child.unresolved_questions.append(question)

    @staticmethod
    def _build_handoff_summary(child: ConversationSession) -> str:
        """在专用摘要实现完成前使用确定性字段生成安全交接。"""

        questions = [
            turn.content
            for turn in child.history
            if turn.role == "user"
        ][-3:]
        parts = [
            "【文章问答交接 v1】",
            f"文章：{child.focus_document_title}",
        ]
        if child.summary and child.summary.startswith("【受保护滚动摘要"):
            parts.append("问答摘要：" + child.summary[:1000])
        if questions:
            parts.append("最近问题：" + "；".join(questions))
        if child.cited_document_ids:
            parts.append(
                "引用文档 ID：" + "；".join(child.cited_document_ids[-10:])
            )
        if child.unresolved_questions:
            parts.append(
                "未解决问题：" + "；".join(child.unresolved_questions[-3:])
            )
        return "\n".join(parts)[:2000]

    async def _save_sessions(
        self,
        sessions: list[ConversationSession],
        *,
        snapshots: Sequence[ConversationResultSnapshot] = (),
        feedback_events: Sequence[PersonalFeedbackEvent] = (),
    ) -> None:
        """统一映射会话 Store 提交失败，不暴露数据库异常。"""

        try:
            commit_recovery = getattr(
                self.conversation_store,
                "commit_recovery",
                None,
            )
            if (snapshots or feedback_events) and callable(commit_recovery):
                await commit_recovery(
                    sessions=sessions,
                    snapshots=snapshots,
                    feedback_events=feedback_events,
                )
            elif len(sessions) == 1:
                await self.conversation_store.save(sessions[0])
            else:
                await self.conversation_store.save_many(sessions)
            emit_stream_event(
                stage="会话提交",
                component="conversation_service",
                status="success",
                title="Session 已持久化",
                summary=f"已提交 {len(sessions)} 个会话状态",
                details={
                    "session_ids": [session.session_id for session in sessions],
                    "session_types": [session.session_type for session in sessions],
                    "snapshot_count": len(snapshots),
                    "feedback_event_count": len(feedback_events),
                },
            )
        except ConversationStoreError as exc:
            record_trace_event(
                "session.commit_failed",
                "conversation_service",
                status="error",
                error=exc,
            )
            raise ServiceUnavailableError("文章推荐服务暂时不可用") from None

    def _build_result_snapshot(
        self,
        session: ConversationSession,
        draft: ConversationResultSnapshotDraft | None,
    ) -> ConversationResultSnapshot | None:
        """用已追加的 assistant 序号补全 Graph 结果快照身份。"""

        if draft is None:
            return None
        protected = ConversationResultSnapshotDraft.model_validate(draft)
        return ConversationResultSnapshot(
            result_id=uuid4().hex,
            user_id=session.user_id,
            session_id=session.session_id,
            assistant_sequence_no=len(session.history) - 1,
            result_type=protected.result_type,
            query=protected.query,
            recommendation_document_ids=(
                protected.recommendation_document_ids
            ),
            citation_document_ids=protected.citation_document_ids,
            citation_chunk_ids=protected.citation_chunk_ids,
            knowledge_status=protected.knowledge_status,
            resolved_document_ids=protected.resolved_document_ids,
            created_at=self._now(),
        )

    @staticmethod
    def _reply_for_session(
        reply: ConversationReply,
        session: ConversationSession,
        *,
        compression: ConversationCompressionInfo,
    ) -> ConversationReply:
        """把实际活动会话的安全导航字段装入公开回复。"""

        return reply.model_copy(
            update={
                "session_id": session.session_id,
                "session_type": session.session_type,
                "parent_session_id": session.parent_session_id,
                "active_child_session_id": session.active_child_session_id,
                "focus_document_id": session.focus_document_id,
                "focus_document_title": session.focus_document_title,
                "session_status": session.session_status,
                "compression": compression,
                "intent_state": session.intent_state,
            },
            deep=True,
        )

    async def _compress_session(
        self,
        session: ConversationSession,
        *,
        deadline: float,
    ) -> ConversationCompressionInfo:
        """按摘要水位压缩模型上下文，始终保留完整原始消息。"""

        recent_start = max(0, len(session.history) - self.history_limit)
        summarize_start = session.summary_watermark + 1
        if summarize_start >= recent_start:
            compression = self._compression_info(session)
            emit_stream_event(
                stage="会话摘要",
                component="conversation_summary_agent",
                status="skipped",
                title="本轮无需压缩会话",
                summary=(
                    f"保留 {compression.retained_turn_count} 轮近期原文"
                ),
                details={
                    "status": compression.status,
                    "summarized_turn_count": compression.summarized_turn_count,
                    "retained_turn_count": compression.retained_turn_count,
                },
            )
            return compression

        summarize_end = min(recent_start, summarize_start + 24)
        turns_to_summarize = [
            turn.model_copy(deep=True)
            for turn in session.history[summarize_start:summarize_end]
        ]
        summary_result: ConversationSummaryResult | None = None
        emit_stream_event(
            stage="会话摘要",
            component="conversation_summary_agent",
            status="started",
            title="开始压缩较早会话",
            summary=f"待摘要 {len(turns_to_summarize)} 条消息",
            details={
                "existing_summary": bool(session.summary),
                "message_count": len(turns_to_summarize),
                "summary_mode": session.session_type,
            },
        )
        try:
            summary_result = ConversationSummaryResult.model_validate(
                await asyncio.wait_for(
                    self.summary_agent.run(
                        existing_summary=session.summary,
                        turns_to_summarize=turns_to_summarize,
                        active_context=(
                            session.active_context
                            if session.session_type == "main"
                            else None
                        ),
                        summary_mode=session.session_type,
                        focus_document_title=session.focus_document_title,
                        recent_recommendation_titles=(
                            self._recent_recommendation_titles(session.history)
                        ),
                        recent_citation_titles=(
                            self._recent_citation_titles(session.history)
                        ),
                        unresolved_questions=session.unresolved_questions,
                    ),
                    timeout=self._remaining_seconds(deadline),
                )
            )
        except Exception as exc:
            logger.warning(
                "会话摘要暂未完成，保留有界原文等待重试",
                error_type=type(exc).__name__,
            )

        if (
            summary_result is not None
            and summary_result.success
            and summary_result.summary is not None
        ):
            session.summary = summary_result.summary
            session.summary_watermark = summarize_end - 1
            session.summarized_turn_count = sum(
                turn.role == "user"
                for turn in session.history[:summarize_end]
            )
        compression = self._compression_info(session)
        emit_stream_event(
            stage="会话摘要",
            component="conversation_summary_agent",
            status=(
                "success"
                if summary_result is not None
                and summary_result.success
                and summary_result.summary is not None
                else "degraded"
            ),
            title=(
                "会话压缩完成"
                if compression.status == "compressed"
                else "会话压缩暂未完成"
            ),
            summary=(
                f"已摘要 {compression.summarized_turn_count} 轮，"
                f"保留 {compression.retained_turn_count} 轮近期原文"
            ),
            details={
                "status": compression.status,
                "summarized_turn_count": compression.summarized_turn_count,
                "retained_turn_count": compression.retained_turn_count,
            },
        )
        return compression

    def _compression_info(
        self,
        session: ConversationSession,
    ) -> ConversationCompressionInfo:
        """根据持久状态生成不暴露内部历史的公开压缩信息。"""

        recent_start = max(0, len(session.history) - self.history_limit)
        if session.summary_watermark + 1 < recent_start:
            status = "pending"
        elif session.summary is not None or session.summarized_turn_count > 0:
            status = "compressed"
        else:
            status = "not_needed"
        return ConversationCompressionInfo(
            status=status,
            summary=session.summary,
            summarized_turn_count=session.summarized_turn_count,
            retained_turn_count=sum(
                turn.role == "user"
                for turn in session.history[-self.history_limit :]
            ),
            dropped_turn_count=session.dropped_turn_count,
        )

    @staticmethod
    def _recent_recommendation_titles(
        history: list[ConversationTurn],
    ) -> list[str]:
        """从程序生成的推荐历史中确定性提取最近标题顺序。"""

        for turn in reversed(history):
            if turn.role != "assistant" or "推荐结果：" not in turn.content:
                continue
            text = turn.content.rsplit("推荐结果：", maxsplit=1)[1].splitlines()[0]
            return [
                title.strip()
                for title in text.split("；")
                if title.strip()
            ][:10]
        return []

    @staticmethod
    def _recent_citation_titles(
        history: list[ConversationTurn],
    ) -> list[str]:
        """从受保护引用文本中确定性提取最近问答标题。"""

        titles: list[str] = []
        for turn in reversed(history):
            if turn.role != "assistant" or "参考资料：" not in turn.content:
                continue
            for line in turn.content.splitlines():
                stripped = line.strip()
                if not stripped.startswith("[") or "] " not in stripped:
                    continue
                title = stripped.split("] ", maxsplit=1)[1].split("（", maxsplit=1)[0]
                if title and title not in titles:
                    titles.append(title)
            if titles:
                break
        return titles[:10]

    @staticmethod
    def _remaining_seconds(deadline: float) -> float:
        """返回正的剩余请求时间；截止后立即触发超时。"""

        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            raise TimeoutError("聊天请求超过总时限")
        return remaining

    async def _ensure_user_exists(self, user_id: str) -> None:
        """在创建或修改会话前确认用户属于当前模拟数据集。"""

        try:
            user = await self.user_store.get_user(user_id)
        except UserNotFoundError:
            raise
        except Exception as exc:
            logger.error(
                "用户查询依赖不可用",
                error_type=type(exc).__name__,
            )
            raise ServiceUnavailableError(
                "文章推荐服务暂时不可用"
            ) from None
        if user is None:
            raise UserNotFoundError("用户不存在")

    @asynccontextmanager
    async def _session_guard(
        self,
        key: tuple[str, str],
    ) -> AsyncIterator[None]:
        entry = self._session_locks.get(key)
        if entry is None:
            entry = _SessionLockEntry()
            self._session_locks[key] = entry
        entry.users += 1
        try:
            async with entry.lock:
                yield
        finally:
            entry.users -= 1
            if (
                entry.users == 0
                and not entry.lock.locked()
                and self._session_locks.get(key) is entry
            ):
                self._session_locks.pop(key, None)

    @asynccontextmanager
    async def _intent_memory_guard(self, user_id: str) -> AsyncIterator[None]:
        """串行化同一用户跨 Session 的进程内记忆读改写。"""

        entry = self._intent_memory_locks.get(user_id)
        if entry is None:
            entry = _SessionLockEntry()
            self._intent_memory_locks[user_id] = entry
        entry.users += 1
        try:
            async with entry.lock:
                yield
        finally:
            entry.users -= 1
            if (
                entry.users == 0
                and not entry.lock.locked()
                and self._intent_memory_locks.get(user_id) is entry
            ):
                self._intent_memory_locks.pop(user_id, None)

    @asynccontextmanager
    async def _session_guards(
        self,
        keys: list[tuple[str, str]],
    ) -> AsyncIterator[None]:
        """按调用方给出的父会话优先顺序获取多个进程内锁。"""

        unique_keys = list(dict.fromkeys(keys))
        async with AsyncExitStack() as stack:
            for key in unique_keys:
                await stack.enter_async_context(self._session_guard(key))
            yield

    def _record_turn(
        self,
        session: ConversationSession,
        user_message: str,
        assistant_message: str,
    ) -> None:
        session.history.extend(
            [
                ConversationTurn(role="user", content=user_message),
                ConversationTurn(role="assistant", content=assistant_message),
            ]
        )
        session.turn_count += 1

    @staticmethod
    def _latest_feedback_exchange(
        history: list[ConversationTurn],
    ) -> tuple[str, str] | None:
        """提取最近一组相邻普通用户消息与助手回答。"""

        for index in range(len(history) - 1, 0, -1):
            assistant_turn = history[index]
            user_turn = history[index - 1]
            if (
                assistant_turn.role == "assistant"
                and assistant_turn.message_type == "chat"
                and user_turn.role == "user"
                and user_turn.message_type == "chat"
            ):
                return user_turn.content, assistant_turn.content
        return None

    def _now(self) -> datetime:
        now = self._clock()
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("会话服务时钟必须包含时区")
        return now


def create_session_id() -> str:
    """生成不含用户信息的随机会话 ID。"""

    return uuid4().hex


__all__ = ["ConversationService", "ServiceUnavailableError", "create_session_id"]
