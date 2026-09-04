"""用户交互习惯记忆的模型、持久化、学习和运行闭环探针。"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from typing import Any
import unittest

from app.agents.conversation_feedback_agent import ConversationFeedbackAgent
from app.application.conversation_service import ConversationService
from app.application.user_interaction_memory import UserInteractionMemoryWorker
from app.domain.services.user_interaction_memory import (
    UserInteractionMemoryService,
)
from app.infrastructure.database.json.feature_store_models import UserBaseProfile
from app.infrastructure.database.sqlite.user_interaction_memory_repository import (
    SQLiteUserInteractionMemoryRepository,
)
from app.infrastructure.database.sqlite.user_profile_repository import (
    SQLiteUserProfileRepository,
)
from app.infrastructure.database.sqlite.conversation_store import (
    SQLiteConversationStore,
)
from app.models.conversation import (
    ConversationReply,
    ConversationSession,
    ConversationTurn,
)
from app.models.intent import (
    ArbitrationAction,
    IntentState,
    RecognitionSource,
)
from app.models.interaction_memory import (
    ConversationFeedbackAnalysis,
    ConversationFeedbackEvent,
    UserInteractionMemory,
)
from app.models.knowledge_qa import KnowledgeAnswerResult
from app.orchestration.conversation_nodes import _run_knowledge_qa


_NOW = datetime(2026, 8, 10, 8, 0, tzinfo=timezone.utc)


def _user(user_id: str) -> UserBaseProfile:
    return UserBaseProfile(
        user_id=user_id,
        topics=["Java"],
        blocked_topics=[],
        preferred_content_types=["technical_design"],
        preferred_difficulty="intermediate",
        preferred_reading_length="medium",
        followed_author_ids=[],
        blocked_author_ids=[],
        created_at=_NOW,
    )


def _event(
    event_id: str,
    *,
    user_id: str = "user-1",
    session_id: str = "session-1",
    occurred_at: datetime = _NOW,
    feedback_message: str = "我更关心项目背景和整体架构。",
) -> ConversationFeedbackEvent:
    return ConversationFeedbackEvent(
        event_id=event_id,
        user_id=user_id,
        session_id=session_id,
        previous_user_message="给我讲一下这个系统。",
        previous_assistant_message="这个系统用于知识问答和文章推荐。",
        feedback_message=feedback_message,
        occurred_at=occurred_at,
    )


def _focus_analysis() -> ConversationFeedbackAnalysis:
    return ConversationFeedbackAnalysis(
        is_preference_feedback=True,
        feedback_type="preference_refinement",
        scope="system_explanation",
        preferred_focus=["project_background", "architecture"],
        persistence="long_term_candidate",
        confidence=0.86,
        reason_code="answer_focus_refined",
        reason_summary="用户希望系统介绍优先说明项目背景和整体架构。",
    )


def _fact_correction_analysis() -> ConversationFeedbackAnalysis:
    return ConversationFeedbackAnalysis(
        is_preference_feedback=False,
        feedback_type="factual_correction",
        scope="knowledge_qa",
        persistence="current_turn_only",
        confidence=0.98,
        reason_code="topic_fact_correction",
        reason_summary="用户纠正当前主题为 Java，不代表长期偏好。",
    )


class _FakeFeedbackLlm:
    def __init__(self, response: object) -> None:
        self.response = response
        self.messages: list[object] = []
        self.calls = 0
        self.closed = False
        self.invoked = asyncio.Event()

    async def ainvoke(self, messages: list[object]) -> object:
        self.messages = list(messages)
        self.calls += 1
        self.invoked.set()
        return self.response

    async def aclose(self) -> None:
        self.closed = True


class _FailingFeedbackLlm:
    def __init__(self) -> None:
        self.calls = 0

    async def ainvoke(self, messages: list[object]) -> object:
        _ = messages
        self.calls += 1
        raise RuntimeError("不应持久化的模型失败详情")


class _KnownUserStore:
    async def get_user(self, user_id: str) -> object | None:
        return object() if user_id == "user-1" else None


class _InteractionWorkflow:
    def __init__(self) -> None:
        self.projections: list[object | None] = []

    async def run(self, **kwargs: Any) -> object:
        self.projections.append(kwargs.get("interaction_memory"))
        return SimpleNamespace(
            reply=ConversationReply(
                session_id=kwargs["session_id"],
                message="下面补充项目背景和整体架构。",
                intent_source=RecognitionSource.RULE,
                action=ArbitrationAction.KNOWLEDGE_ANSWER,
            ),
            history_message="下面补充项目背景和整体架构。",
            pending_context=None,
            commit_context=False,
            pending_intent_state=IntentState.KNOWLEDGE_QA,
            commit_intent_state=True,
            knowledge_document_ids=(),
            knowledge_document_titles=(),
            error_stage=None,
            trace=[],
        )


class _WakeRecorder:
    def __init__(self) -> None:
        self.calls = 0

    def wake(self) -> None:
        self.calls += 1


class UserInteractionMemoryRepositoryTests(unittest.TestCase):
    """原始反馈和长期交互记忆使用独立表与生命周期。"""

    def test_feedback_event_and_memory_survive_repository_reopen(self) -> None:
        with tempfile.TemporaryDirectory(prefix="interaction-memory-") as root:
            database_path = Path(root) / "user_profiles.sqlite3"
            user_repository = SQLiteUserProfileRepository(database_path)
            user_repository.replace_user(_user("user-1"))
            repository = SQLiteUserInteractionMemoryRepository(database_path)
            event = _event("event-1")
            memory = UserInteractionMemory.empty("user-1", now=_NOW)

            repository.append_event(event)
            repository.save_memory(memory)
            reopened = SQLiteUserInteractionMemoryRepository(database_path)

            self.assertEqual(reopened.get_event("event-1"), event)
            self.assertEqual(
                reopened.list_pending(now=_NOW, limit=10),
                (event,),
            )
            self.assertEqual(reopened.get_memory("user-1"), memory)
            self.assertIsNone(reopened.get_memory("user-2"))

    def test_analyzed_old_event_purges_raw_text_but_keeps_analysis(self) -> None:
        with tempfile.TemporaryDirectory(prefix="interaction-memory-") as root:
            database_path = Path(root) / "user_profiles.sqlite3"
            user_repository = SQLiteUserProfileRepository(database_path)
            user_repository.replace_user(_user("user-1"))
            repository = SQLiteUserInteractionMemoryRepository(database_path)
            event = _event(
                "event-old",
                occurred_at=_NOW - timedelta(days=31),
            ).model_copy(
                update={
                    "status": "analyzed",
                    "analysis": _focus_analysis(),
                    "analyzed_at": _NOW - timedelta(days=30),
                }
            )
            repository.append_event(event)

            purged = repository.purge_raw_before(
                _NOW - timedelta(days=30),
                purged_at=_NOW,
            )
            stored = repository.get_event("event-old")

            self.assertEqual(purged, 1)
            assert stored is not None
            self.assertIsNone(stored.previous_user_message)
            self.assertIsNone(stored.previous_assistant_message)
            self.assertIsNone(stored.feedback_message)
            self.assertEqual(stored.analysis, _focus_analysis())
            self.assertEqual(stored.raw_purged_at, _NOW)

    def test_unknown_user_and_duplicate_event_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="interaction-memory-") as root:
            database_path = Path(root) / "user_profiles.sqlite3"
            SQLiteUserProfileRepository(database_path)
            repository = SQLiteUserInteractionMemoryRepository(database_path)

            with self.assertRaisesRegex(ValueError, "用户"):
                repository.append_event(_event("event-missing"))

            user_repository = SQLiteUserProfileRepository(database_path)
            user_repository.replace_user(_user("user-1"))
            repository.append_event(_event("event-1"))
            with self.assertRaisesRegex(ValueError, "重复"):
                repository.append_event(_event("event-1"))


class UserInteractionMemoryServiceTests(unittest.TestCase):
    """只学习回答方式偏好，不把事实主题错误固化为用户习惯。"""

    def setUp(self) -> None:
        self.service = UserInteractionMemoryService(clock=lambda: _NOW)

    def test_system_explanation_focus_becomes_low_priority_projection(self) -> None:
        memory = self.service.apply_analysis(
            self.service.empty("user-1"),
            event=_event("event-1"),
            analysis=_focus_analysis(),
        )

        projection = self.service.project(memory)

        self.assertEqual(len(projection.preferences), 1)
        preference = projection.preferences[0]
        self.assertEqual(preference.scope, "system_explanation")
        self.assertEqual(
            preference.preferred_focus,
            ["project_background", "architecture"],
        )
        self.assertEqual(preference.evidence_count, 1)
        self.assertLessEqual(preference.confidence, 0.75)

    def test_same_event_is_idempotent_and_repetition_strengthens_preference(
        self,
    ) -> None:
        memory = self.service.empty("user-1")
        first_event = _event("event-1")
        memory = self.service.apply_analysis(
            memory,
            event=first_event,
            analysis=_focus_analysis(),
        )
        duplicate = self.service.apply_analysis(
            memory,
            event=first_event,
            analysis=_focus_analysis(),
        )
        strengthened = self.service.apply_analysis(
            duplicate,
            event=_event("event-2", session_id="session-2"),
            analysis=_focus_analysis(),
        )

        self.assertEqual(duplicate, memory)
        self.assertEqual(strengthened.preferences[0].evidence_count, 2)
        self.assertGreater(
            strengthened.preferences[0].confidence,
            memory.preferences[0].confidence,
        )

    def test_topic_or_fact_correction_does_not_create_long_term_preference(
        self,
    ) -> None:
        analysis = ConversationFeedbackAnalysis(
            is_preference_feedback=False,
            feedback_type="factual_correction",
            scope="knowledge_qa",
            persistence="current_turn_only",
            confidence=0.98,
            reason_code="topic_fact_correction",
            reason_summary="用户纠正当前主题为 Java，不代表长期偏好。",
        )

        memory = self.service.apply_analysis(
            self.service.empty("user-1"),
            event=_event(
                "event-topic",
                feedback_message="你答错了，我要的是 Java。",
            ),
            analysis=analysis,
        )

        self.assertEqual(memory.preferences, [])
        self.assertEqual(self.service.project(memory).preferences, [])
        self.assertEqual(
            self.service.project_analysis(analysis).preferences,
            [],
        )

    def test_explicit_style_analysis_projects_without_identity_or_reason(self) -> None:
        analysis = ConversationFeedbackAnalysis(
            is_preference_feedback=True,
            feedback_type="format_preference",
            scope="knowledge_qa",
            detail_level="brief",
            answer_structure="conclusion_first",
            persistence="explicit_long_term",
            confidence=0.96,
            reason_code="explicit_conclusion_first",
            reason_summary="用户明确要求以后先给结论。",
        )

        projection = self.service.project_analysis(analysis)

        self.assertEqual(len(projection.preferences), 1)
        self.assertEqual(
            projection.preferences[0].answer_structure,
            "conclusion_first",
        )
        self.assertEqual(projection.preferences[0].detail_level, "brief")
        self.assertNotIn("reason", projection.model_dump_json())
        self.assertNotIn("user", projection.model_dump_json())

    def test_candidate_gate_requires_previous_answer_and_feedback_signal(self) -> None:
        self.assertTrue(
            self.service.is_feedback_candidate(
                "我更关心项目背景和整体架构。",
                has_previous_exchange=True,
            )
        )
        self.assertFalse(
            self.service.is_feedback_candidate(
                "Java 怎么学习？",
                has_previous_exchange=True,
            )
        )
        self.assertFalse(
            self.service.is_feedback_candidate(
                "我更关心项目背景和整体架构。",
                has_previous_exchange=False,
            )
        )


class ConversationFeedbackAgentTests(unittest.IsolatedAsyncioTestCase):
    """LLM 负责语义归因，但只能输出受控回答偏好。"""

    async def test_agent_extracts_controlled_answer_focus(self) -> None:
        llm = _FakeFeedbackLlm(_focus_analysis().model_dump(mode="json"))
        agent = ConversationFeedbackAgent(llm=llm)

        analysis = await agent.analyze(_event("event-agent"))

        self.assertEqual(analysis, _focus_analysis())
        self.assertEqual(llm.calls, 1)
        system_prompt = getattr(llm.messages[0], "content")
        self.assertIn("当前事实或技术主题", system_prompt)
        self.assertIn("不得形成长期偏好", system_prompt)
        envelope = json.loads(getattr(llm.messages[1], "content"))
        self.assertEqual(set(envelope), {"contract", "input"})
        self.assertEqual(envelope["contract"]["name"], "conversation_feedback")
        self.assertEqual(envelope["contract"]["version"], 2)
        self.assertIsInstance(envelope["contract"]["output_schema"], dict)
        input_payload = envelope["input"]
        self.assertEqual(
            input_payload["feedback_message"],
            "我更关心项目背景和整体架构。",
        )

    async def test_agent_keeps_java_python_correction_current_turn_only(
        self,
    ) -> None:
        llm = _FakeFeedbackLlm(
            _fact_correction_analysis().model_dump(mode="json")
        )
        agent = ConversationFeedbackAgent(llm=llm)

        analysis = await agent.analyze(
            _event(
                "event-fact",
                feedback_message="你答错了，我要的是 Java，不是 Python。",
            )
        )

        assert analysis is not None
        self.assertFalse(analysis.is_preference_feedback)
        self.assertEqual(analysis.feedback_type, "factual_correction")
        self.assertEqual(analysis.persistence, "current_turn_only")
        self.assertEqual(analysis.preferred_focus, [])

    async def test_agent_rejects_unbounded_provider_reason_code(self) -> None:
        payload = _focus_analysis().model_dump(mode="json")
        payload["reason_code"] = "free_form_reason_from_prompt"
        agent = ConversationFeedbackAgent(
            llm=_FakeFeedbackLlm(payload)
        )

        with self.assertRaises(ValueError):
            await agent.analyze(_event("event-invalid-reason"))

    async def test_agent_without_llm_leaves_event_for_later(self) -> None:
        agent = ConversationFeedbackAgent(llm=None)

        self.assertIsNone(await agent.analyze(_event("event-disabled")))


class UserInteractionMemoryWorkerTests(unittest.IsolatedAsyncioTestCase):
    """后台处理失败不丢原文，成功后幂等更新长期偏好。"""

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory(
            prefix="interaction-worker-"
        )
        database_path = Path(self.temporary_directory.name) / "user_profiles.sqlite3"
        user_repository = SQLiteUserProfileRepository(database_path)
        user_repository.replace_user(_user("user-1"))
        self.repository = SQLiteUserInteractionMemoryRepository(database_path)
        self.service = UserInteractionMemoryService(clock=lambda: _NOW)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    async def test_successful_analysis_updates_event_and_memory_once(self) -> None:
        self.repository.append_event(_event("event-worker"))
        llm = _FakeFeedbackLlm(_focus_analysis().model_dump(mode="json"))
        worker = UserInteractionMemoryWorker(
            repository=self.repository,
            memory_service=self.service,
            feedback_agent=ConversationFeedbackAgent(llm=llm),
            clock=lambda: _NOW,
        )

        first_count = await worker.run_once()
        second_count = await worker.run_once()

        stored_event = self.repository.get_event("event-worker")
        stored_memory = self.repository.get_memory("user-1")
        self.assertEqual(first_count, 1)
        self.assertEqual(second_count, 0)
        self.assertEqual(llm.calls, 1)
        assert stored_event is not None
        assert stored_memory is not None
        self.assertEqual(stored_event.status, "analyzed")
        self.assertEqual(stored_event.analysis_attempts, 1)
        self.assertEqual(stored_event.analysis, _focus_analysis())
        self.assertEqual(stored_memory.preferences[0].evidence_count, 1)

    async def test_model_failure_keeps_pending_event_and_schedules_retry(
        self,
    ) -> None:
        self.repository.append_event(_event("event-retry"))
        worker = UserInteractionMemoryWorker(
            repository=self.repository,
            memory_service=self.service,
            feedback_agent=ConversationFeedbackAgent(
                llm=_FailingFeedbackLlm()
            ),
            clock=lambda: _NOW,
        )

        processed = await worker.run_once()

        stored = self.repository.get_event("event-retry")
        self.assertEqual(processed, 0)
        assert stored is not None
        self.assertEqual(stored.status, "pending")
        self.assertEqual(stored.analysis_attempts, 1)
        self.assertGreater(stored.next_attempt_at, _NOW)
        self.assertEqual(stored.last_error_type, "RuntimeError")
        self.assertEqual(
            stored.feedback_message,
            "我更关心项目背景和整体架构。",
        )
        self.assertNotIn(
            "不应持久化的模型失败详情",
            stored.model_dump_json(),
        )

    async def test_worker_purges_analyzed_raw_text_after_thirty_days(
        self,
    ) -> None:
        old_event = _event(
            "event-cleanup",
            occurred_at=_NOW - timedelta(days=31),
        ).model_copy(
            update={
                "status": "analyzed",
                "analysis": _focus_analysis(),
                "analysis_attempts": 1,
                "analyzed_at": _NOW - timedelta(days=30),
            }
        )
        self.repository.append_event(old_event)
        worker = UserInteractionMemoryWorker(
            repository=self.repository,
            memory_service=self.service,
            feedback_agent=ConversationFeedbackAgent(llm=None),
            clock=lambda: _NOW,
        )

        self.assertEqual(await worker.run_once(), 0)

        stored = self.repository.get_event("event-cleanup")
        assert stored is not None
        self.assertIsNone(stored.previous_user_message)
        self.assertIsNone(stored.previous_assistant_message)
        self.assertIsNone(stored.feedback_message)
        self.assertEqual(stored.raw_purged_at, _NOW)

    async def test_start_wakes_worker_and_stop_closes_feedback_llm(self) -> None:
        self.repository.append_event(_event("event-lifecycle"))
        llm = _FakeFeedbackLlm(_focus_analysis().model_dump(mode="json"))
        worker = UserInteractionMemoryWorker(
            repository=self.repository,
            memory_service=self.service,
            feedback_agent=ConversationFeedbackAgent(llm=llm),
            clock=lambda: _NOW,
            scan_interval_seconds=86400.0,
        )

        worker.start()
        await asyncio.wait_for(llm.invoked.wait(), timeout=1.0)
        await worker.stop()

        stored = self.repository.get_event("event-lifecycle")
        assert stored is not None
        self.assertEqual(stored.status, "analyzed")
        self.assertTrue(llm.closed)


class UserInteractionMemoryBootstrapTests(unittest.TestCase):
    """启动期必须托管 Worker，不能创建无人关闭的后台任务。"""

    def test_bootstrap_starts_and_stops_interaction_memory_worker(self) -> None:
        bootstrap_path = (
            Path(__file__).resolve().parents[1] / "app" / "bootstrap.py"
        )
        source = bootstrap_path.read_text(encoding="utf-8")

        self.assertIn("interaction_memory_worker.start()", source)
        self.assertIn("await interaction_memory_worker.stop()", source)
        self.assertLess(
            source.index("await interaction_memory_worker.stop()"),
            source.index("await knowledge_qa_service.aclose()"),
        )


class UserInteractionMemoryConversationTests(unittest.IsolatedAsyncioTestCase):
    """会话服务捕获相邻反馈并只向问答链加载精简投影。"""

    async def test_service_loads_projection_and_captures_previous_exchange(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(prefix="interaction-chat-") as root:
            root_path = Path(root)
            profile_path = root_path / "user_profiles.sqlite3"
            user_repository = SQLiteUserProfileRepository(profile_path)
            user_repository.replace_user(_user("user-1"))
            repository = SQLiteUserInteractionMemoryRepository(profile_path)
            memory_service = UserInteractionMemoryService(clock=lambda: _NOW)
            memory = memory_service.apply_analysis(
                memory_service.empty("user-1"),
                event=_event("seed-event"),
                analysis=_focus_analysis(),
            )
            repository.save_memory(memory)
            conversation_store = SQLiteConversationStore(
                root_path / "conversations.sqlite3"
            )
            await conversation_store.save(
                ConversationSession(
                    session_id="session-1",
                    user_id="user-1",
                    intent_state=IntentState.KNOWLEDGE_QA,
                    history=[
                        ConversationTurn(
                            role="user",
                            content="给我讲一下这个系统。",
                        ),
                        ConversationTurn(
                            role="assistant",
                            content="这个系统用于知识问答和文章推荐。",
                        ),
                    ],
                    turn_count=1,
                )
            )
            worker = _WakeRecorder()
            workflow = _InteractionWorkflow()
            service = ConversationService(
                user_store=_KnownUserStore(),
                recall_agent=object(),
                rerank_agent=object(),
                aggregator=object(),
                conversation_store=conversation_store,
                enable_llm=False,
                interaction_memory_repository=repository,
                interaction_memory_service=memory_service,
                interaction_memory_worker=worker,
                clock=lambda: _NOW,
            )
            service.workflow = workflow

            reply = await service.chat(
                "user-1",
                "我更关心项目背景和整体架构。",
                session_id="session-1",
            )

            self.assertEqual(reply.action, ArbitrationAction.KNOWLEDGE_ANSWER)
            self.assertEqual(len(workflow.projections), 1)
            projection = workflow.projections[0]
            self.assertIsNotNone(projection)
            self.assertEqual(
                projection.preferences[0].preferred_focus,
                ["project_background", "architecture"],
            )
            pending = repository.list_pending(now=_NOW, limit=10)
            self.assertEqual(len(pending), 1)
            self.assertEqual(
                pending[0].previous_user_message,
                "给我讲一下这个系统。",
            )
            self.assertEqual(
                pending[0].previous_assistant_message,
                "这个系统用于知识问答和文章推荐。",
            )
            self.assertEqual(
                pending[0].feedback_message,
                "我更关心项目背景和整体架构。",
            )
            self.assertEqual(worker.calls, 1)

    async def test_interaction_memory_failure_does_not_replace_reply(self) -> None:
        class _FailingRepository:
            def get_memory(self, user_id: str) -> None:
                _ = user_id
                raise RuntimeError("interaction memory unavailable")

            def append_event(self, event: object) -> None:
                _ = event
                raise RuntimeError("interaction memory unavailable")

        with tempfile.TemporaryDirectory(prefix="interaction-chat-") as root:
            conversation_store = SQLiteConversationStore(
                Path(root) / "conversations.sqlite3"
            )
            await conversation_store.save(
                ConversationSession(
                    session_id="session-1",
                    user_id="user-1",
                    history=[
                        ConversationTurn(role="user", content="介绍这个系统。"),
                        ConversationTurn(role="assistant", content="这是问答系统。"),
                    ],
                    turn_count=1,
                )
            )
            workflow = _InteractionWorkflow()
            service = ConversationService(
                user_store=_KnownUserStore(),
                recall_agent=object(),
                rerank_agent=object(),
                aggregator=object(),
                conversation_store=conversation_store,
                enable_llm=False,
                interaction_memory_repository=_FailingRepository(),
                interaction_memory_service=UserInteractionMemoryService(
                    clock=lambda: _NOW
                ),
                interaction_memory_worker=_WakeRecorder(),
                clock=lambda: _NOW,
            )
            service.workflow = workflow

            reply = await service.chat(
                "user-1",
                "我更关心整体架构。",
                session_id="session-1",
            )

            self.assertEqual(reply.action, ArbitrationAction.KNOWLEDGE_ANSWER)
            self.assertEqual(workflow.projections, [None])

    async def test_graph_forwards_projection_only_to_knowledge_service(
        self,
    ) -> None:
        projection = UserInteractionMemoryService(
            clock=lambda: _NOW
        ).project(
            UserInteractionMemoryService(clock=lambda: _NOW).apply_analysis(
                UserInteractionMemory.empty("user-1", now=_NOW),
                event=_event("event-node"),
                analysis=_focus_analysis(),
            )
        )

        class _KnowledgeService:
            def __init__(self) -> None:
                self.interaction_memory = None

            async def ask(self, question: str, **kwargs: Any) -> KnowledgeAnswerResult:
                _ = question
                self.interaction_memory = kwargs.get("interaction_memory")
                return KnowledgeAnswerResult(
                    status="success",
                    answer="受控回答。",
                    citations=(),
                )

        knowledge_service = _KnowledgeService()
        state = {
            "message": "给我讲一下这个系统。",
            "history": [],
            "conversation_summary": None,
            "knowledge_document_ids": (),
            "recognition": SimpleNamespace(rewritten_query=None),
            "interaction_memory": projection,
        }

        result = await _run_knowledge_qa(
            SimpleNamespace(knowledge_qa_service=knowledge_service),
            state,
        )

        self.assertNotIn("error_stage", result)
        self.assertEqual(knowledge_service.interaction_memory, projection)


if __name__ == "__main__":
    unittest.main()
