"""验证个人自然语言反馈的识别、补救、持久化和记忆隔离。"""

from __future__ import annotations

import asyncio
import json
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
from typing import Any

from app.agents.feedback_recovery_agent import FeedbackRecoveryAgent
from app.agents.conversation_feedback_agent import ConversationFeedbackAgent
from app.application.knowledge_qa import KnowledgeQaService
from app.application.conversation_service import ConversationService
from app.application.personal_feedback_learning import (
    PersonalFeedbackLearningService,
)
from app.domain.services.feedback_recovery_policy import FeedbackRecoveryPolicy
from app.infrastructure.database.conversation_store import ConversationStoreError
from app.infrastructure.database.json.feature_store import FeatureStore
from app.infrastructure.database.json.feature_store_models import UserBaseProfile
from app.infrastructure.database.sqlite.conversation_store import (
    SQLiteConversationStore,
)
from app.infrastructure.database.sqlite.user_profile_repository import (
    SQLiteUserProfileRepository,
)
from app.infrastructure.database.sqlite.user_interaction_memory_repository import (
    SQLiteUserInteractionMemoryRepository,
)
from app.models.conversation import (
    ConversationReply,
    ConversationSession,
    ConversationTurn,
)
from app.models.intent import IntentState
from app.models.intent import ArbitrationAction, RecognitionSource
from app.models.personal_feedback import (
    ConversationResultSnapshot,
    ConversationResultSnapshotDraft,
    FeedbackAnalysis,
    PersonalFeedbackEvent,
    RecommendationMemorySignal,
)
from app.agents.user_profile_agent import UserProfileAgent
from app.models.knowledge_qa import KnowledgeChunkRecord, KnowledgeGeneratedAnswer
from app.models.knowledge_qa import KnowledgeAnswerResult, KnowledgeCitation
from app.models.interaction_memory import ConversationFeedbackAnalysis
from app.domain.services.user_interaction_memory import UserInteractionMemoryService
from app.infrastructure.observability.conversation_trace import (
    ConversationStreamRecorder,
    conversation_stream_context,
)
from app.models.api import (
    ChatRequest,
    ChatResponse,
    ChatStreamErrorEvent,
    ChatStreamProcessEvent,
    ChatStreamResultEvent,
)
from app.orchestration.conversation_graph import ConversationGraph
from app.orchestration.conversation_state import ConversationGraphResult
from app.domain.services.conversation_arbitrator import ConversationArbitrator


_NOW = datetime(2026, 8, 13, 8, 0, tzinfo=timezone.utc)


def _recommendation_snapshot(
    document_ids: tuple[str, ...],
) -> ConversationResultSnapshot:
    """构造可验证推荐目标的结果快照。"""

    return ConversationResultSnapshot(
        result_id="result-1",
        user_id="user-1",
        session_id="session-1",
        assistant_sequence_no=1,
        result_type="recommendation",
        query="Java 并发",
        recommendation_document_ids=document_ids,
        created_at=_NOW,
    )


def _knowledge_snapshot(
    chunk_ids: tuple[str, ...],
) -> ConversationResultSnapshot:
    """构造带可信引用身份的知识回答快照。"""

    return ConversationResultSnapshot(
        result_id="result-2",
        user_id="user-1",
        session_id="session-1",
        assistant_sequence_no=3,
        result_type="knowledge_answer",
        query="Java 线程池如何配置",
        citation_document_ids=("doc-java",),
        citation_chunk_ids=chunk_ids,
        knowledge_status="success",
        resolved_document_ids=("doc-java",),
        created_at=_NOW,
    )


class PersonalFeedbackModelTests(unittest.TestCase):
    """严格模型应拒绝矛盾状态和不安全原始数据。"""

    def test_snapshot_requires_payload_matching_result_type(self) -> None:
        with self.assertRaisesRegex(ValueError, "推荐结果"):
            ConversationResultSnapshot(
                result_id="result-invalid",
                user_id="user-1",
                session_id="session-1",
                assistant_sequence_no=1,
                result_type="recommendation",
                query="Java",
                created_at=_NOW,
            )

    def test_snapshot_requires_timezone_aware_created_at(self) -> None:
        with self.assertRaisesRegex(ValueError, "时区"):
            _recommendation_snapshot(("doc-1",)).model_copy(
                update={"created_at": datetime(2026, 8, 13, 8, 0)}
            ).model_validate(
                {
                    **_recommendation_snapshot(("doc-1",)).model_dump(),
                    "created_at": datetime(2026, 8, 13, 8, 0),
                }
            )


class FeedbackRecoveryPolicyTests(unittest.TestCase):
    """确定性策略负责保护 LLM 候选并限制补救动作。"""

    def setUp(self) -> None:
        self.policy = FeedbackRecoveryPolicy()

    def test_policy_resolves_second_recommendation_from_snapshot(self) -> None:
        snapshot = _recommendation_snapshot(("doc-1", "doc-2", "doc-3"))
        decision = self.policy.protect(
            message="第二篇太基础了，换成更深入的线程池文章",
            snapshot=snapshot,
            pending_event=None,
            analysis=FeedbackAnalysis(
                is_feedback=True,
                feedback_type="recommendation_irrelevant",
                completeness="complete",
                corrected_query="深入的 Java 线程池调优",
                target_document_ids=(),
                suggested_action="retry_recommendation",
                recommendation_signals=(
                    RecommendationMemorySignal(
                        target_type="difficulty",
                        target_value="beginner",
                        source_document_ids=("doc-2",),
                        specific=True,
                        persistence="long_term_candidate",
                    ),
                ),
                reason_code="second_item_too_basic",
                confidence=0.94,
            ),
        )

        self.assertEqual(decision.target_document_ids, ("doc-2",))
        self.assertEqual(decision.excluded_document_ids, ("doc-2",))
        self.assertEqual(decision.next_action, "retry_recommendation")
        self.assertEqual(decision.memory_routes, ("recommendation_profile",))

    def test_policy_rejects_unknown_document_id_from_llm(self) -> None:
        snapshot = _recommendation_snapshot(("doc-1", "doc-2"))
        analysis = FeedbackAnalysis(
            is_feedback=True,
            feedback_type="recommendation_irrelevant",
            completeness="complete",
            corrected_query="深入的 Java 并发",
            target_document_ids=("forged-doc",),
            suggested_action="retry_recommendation",
            reason_code="forged_target",
            confidence=0.9,
        )

        with self.assertRaisesRegex(ValueError, "目标文档"):
            self.policy.protect(
                message="第二篇不相关",
                snapshot=snapshot,
                pending_event=None,
                analysis=analysis,
            )

    def test_fact_correction_cannot_create_long_term_topic_signal(self) -> None:
        decision = self.policy.protect(
            message="答错了，我问的是 Java，不是 Python",
            snapshot=_knowledge_snapshot(("chunk-1",)),
            pending_event=None,
            analysis=FeedbackAnalysis(
                is_feedback=True,
                feedback_type="answer_incorrect",
                completeness="complete",
                corrected_query="Java 线程池如何配置",
                suggested_action="retry_retrieval",
                recommendation_signals=(
                    RecommendationMemorySignal(
                        target_type="topic",
                        target_value="Java",
                        specific=True,
                        persistence="explicit_long_term",
                    ),
                ),
                reason_code="fact_correction",
                confidence=0.98,
            ),
        )

        self.assertEqual(decision.recommendation_signals, ())
        self.assertNotIn("recommendation_profile", decision.memory_routes)
        self.assertEqual(decision.next_action, "retry_retrieval")

    def test_generic_negative_feedback_clarifies_once(self) -> None:
        decision = self.policy.fallback_decision(
            "这些推荐不好",
            snapshot=_recommendation_snapshot(("doc-1", "doc-2")),
            pending_event=None,
        )

        self.assertTrue(decision.is_feedback)
        self.assertEqual(decision.completeness, "incomplete")
        self.assertEqual(decision.next_action, "clarify")
        self.assertIn("不相关", decision.clarification_question or "")

    def test_generic_long_term_candidate_can_reach_learning_threshold(self) -> None:
        signal = RecommendationMemorySignal(
            target_type="difficulty",
            target_value="beginner",
            specific=False,
            persistence="long_term_candidate",
        )

        self.assertFalse(signal.specific)


class PersonalFeedbackStoreTests(unittest.IsolatedAsyncioTestCase):
    """结果快照、反馈事件和 Session 指针必须原子持久化。"""

    @staticmethod
    def _session(*, pending_feedback_id: str | None) -> ConversationSession:
        return ConversationSession(
            session_id="session-1",
            user_id="user-1",
            history=[
                ConversationTurn(role="user", content="推荐 Java 并发文章"),
                ConversationTurn(role="assistant", content="已返回两篇文章"),
            ],
            turn_count=1,
            pending_feedback_id=pending_feedback_id,
        )

    @staticmethod
    def _awaiting_event() -> PersonalFeedbackEvent:
        return PersonalFeedbackEvent(
            feedback_id="feedback-1",
            user_id="user-1",
            session_id="session-1",
            source_result_id="result-1",
            feedback_message="这些推荐不好",
            feedback_type="recommendation_irrelevant",
            completeness="incomplete",
            next_action="clarify",
            status="awaiting_detail",
            clarification_count=1,
            reason_code="generic_negative_needs_detail",
            created_at=_NOW,
            updated_at=_NOW,
        )

    async def test_snapshot_feedback_and_pending_state_survive_reopen(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(prefix="personal-feedback-store-") as root:
            path = Path(root) / "conversations.sqlite3"
            store = SQLiteConversationStore(path)
            await store.commit_recovery(
                sessions=(self._session(pending_feedback_id="feedback-1"),),
                snapshots=(
                    _recommendation_snapshot(("doc-1", "doc-2")),
                ),
                feedback_events=(self._awaiting_event(),),
            )

            reopened = SQLiteConversationStore(path)
            loaded = await reopened.load("user-1", "session-1")
            context = await reopened.load_feedback_context(
                "user-1", "session-1"
            )
            events = await reopened.list_feedback_events("user-1")

            assert loaded is not None
            assert context.latest_result is not None
            assert context.pending_feedback is not None
            self.assertEqual(loaded.pending_feedback_id, "feedback-1")
            self.assertEqual(context.latest_result.result_id, "result-1")
            self.assertEqual(
                context.pending_feedback.feedback_id,
                "feedback-1",
            )
            self.assertEqual([event.feedback_id for event in events], ["feedback-1"])
            self.assertIsNone(
                await reopened.get_result_snapshot("user-2", "result-1")
            )

    async def test_failed_combined_commit_leaves_no_partial_state(self) -> None:
        with tempfile.TemporaryDirectory(prefix="personal-feedback-store-") as root:
            store = SQLiteConversationStore(Path(root) / "conversations.sqlite3")
            foreign_snapshot = _recommendation_snapshot(("doc-1",)).model_copy(
                update={"user_id": "user-2"}
            )

            with self.assertRaises(ConversationStoreError):
                await store.commit_recovery(
                    sessions=(self._session(pending_feedback_id=None),),
                    snapshots=(foreign_snapshot,),
                    feedback_events=(),
                )

            self.assertIsNone(await store.load("user-1", "session-1"))
            self.assertIsNone(
                await store.get_result_snapshot("user-2", "result-1")
            )

    async def test_completed_feedback_raw_text_is_purged_after_thirty_days(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(prefix="feedback-retention-") as root:
            store = SQLiteConversationStore(Path(root) / "conversations.sqlite3")
            old = _NOW - timedelta(days=31)
            snapshot = _recommendation_snapshot(("doc-1",)).model_copy(
                update={"created_at": old}
            )
            event = PersonalFeedbackEvent(
                feedback_id="feedback-old",
                user_id="user-1",
                session_id="session-1",
                source_result_id=snapshot.result_id,
                feedback_message="以后不要推荐这篇",
                feedback_type="recommendation_irrelevant",
                completeness="complete",
                corrected_query="Java 并发",
                target_document_ids=("doc-1",),
                next_action="retry_recommendation",
                status="recovered",
                recovery_count=1,
                recovery_result_id="result-recovery",
                reason_code="explicit_article_avoid",
                created_at=old,
                updated_at=old,
            )
            recovery = ConversationResultSnapshot(
                result_id="result-recovery",
                user_id="user-1",
                session_id="session-1",
                assistant_sequence_no=3,
                result_type="recommendation",
                query="Java 并发进阶",
                recommendation_document_ids=("doc-2",),
                created_at=old,
            )
            await store.commit_recovery(
                sessions=(self._session(pending_feedback_id=None),),
                snapshots=(snapshot, recovery),
                feedback_events=(event,),
            )

            changed = await store.purge_feedback_raw_before(
                _NOW - timedelta(days=30),
                purged_at=_NOW,
            )

            events = await store.list_feedback_events("user-1")
            purged_snapshot = await store.get_result_snapshot(
                "user-1",
                snapshot.result_id,
            )
            assert purged_snapshot is not None
            self.assertEqual(changed, 3)
            self.assertIsNone(events[0].feedback_message)
            self.assertIsNone(events[0].corrected_query)
            self.assertEqual(events[0].raw_purged_at, _NOW)
            self.assertIsNone(purged_snapshot.query)
            self.assertEqual(purged_snapshot.raw_purged_at, _NOW)

    async def test_session_delete_cascades_snapshot_and_feedback(self) -> None:
        with tempfile.TemporaryDirectory(prefix="feedback-cascade-") as root:
            store = SQLiteConversationStore(Path(root) / "conversations.sqlite3")
            await store.commit_recovery(
                sessions=(self._session(pending_feedback_id="feedback-1"),),
                snapshots=(_recommendation_snapshot(("doc-1", "doc-2")),),
                feedback_events=(self._awaiting_event(),),
            )

            await store.delete("user-1", "session-1")

            self.assertIsNone(await store.load("user-1", "session-1"))
            self.assertIsNone(
                await store.get_result_snapshot("user-1", "result-1")
            )
            self.assertEqual(await store.list_feedback_events("user-1"), ())

    async def test_cross_session_feedback_target_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="feedback-forged-session-") as root:
            store = SQLiteConversationStore(Path(root) / "conversations.sqlite3")
            foreign_session = ConversationSession(
                session_id="session-2",
                user_id="user-1",
                pending_feedback_id="feedback-1",
            )

            with self.assertRaises(ConversationStoreError):
                await store.commit_recovery(
                    sessions=(
                        self._session(pending_feedback_id=None),
                        foreign_session,
                    ),
                    snapshots=(_recommendation_snapshot(("doc-1", "doc-2")),),
                    feedback_events=(
                        self._awaiting_event().model_copy(
                            update={"session_id": "session-2"}
                        ),
                    ),
                )

            self.assertIsNone(await store.load("user-1", "session-1"))
            self.assertIsNone(await store.load("user-1", "session-2"))


class _LearningFeedbackStore:
    """只向学习协调器提供同用户结构化事件。"""

    def __init__(self) -> None:
        self.events: list[PersonalFeedbackEvent] = []

    async def list_feedback_events(
        self,
        user_id: str,
        *,
        limit: int = 100,
    ) -> tuple[PersonalFeedbackEvent, ...]:
        return tuple(
            event
            for event in reversed(self.events)
            if event.user_id == user_id
        )[:limit]


class _LearningDocumentRepository:
    """为画像学习提供两篇可验证 ready 文档事实。"""

    def get_document_facts(self, document_ids: tuple[str, ...]) -> dict[str, object]:
        from app.models.document import DocumentFact

        facts = {
            "java-post-002": DocumentFact(
                document_id="java-post-002",
                title="Java 虚拟线程",
                topics=["Java", "并发编程"],
                content_type="tutorial",
                difficulty="advanced",
                author_id="author-java",
                total_token_count=800,
            ),
            "spring-boot-post-001": DocumentFact(
                document_id="spring-boot-post-001",
                title="Spring Boot 入门",
                topics=["Spring Boot"],
                content_type="tutorial",
                difficulty="beginner",
                author_id="author-spring",
                total_token_count=600,
            ),
        }
        return {
            document_id: facts[document_id]
            for document_id in document_ids
            if document_id in facts
        }


def _learning_feature_store(root: str) -> FeatureStore:
    user_repository = SQLiteUserProfileRepository(
        Path(root) / "user_profiles.sqlite3"
    )
    user_repository.replace_user(
        UserBaseProfile(
            user_id="10001",
            topics=["Java"],
            blocked_topics=[],
            preferred_content_types=["tutorial"],
            preferred_difficulty="intermediate",
            preferred_reading_length="medium",
            followed_author_ids=[],
            blocked_author_ids=[],
            created_at=_NOW,
        )
    )
    return FeatureStore(
        auto_load_mock=True,
        clock=lambda: _NOW,
        user_repository=user_repository,
        document_repository=_LearningDocumentRepository(),
    )


def _learning_event(
    feedback_id: str,
    signal: RecommendationMemorySignal,
    *,
    user_id: str = "10001",
) -> PersonalFeedbackEvent:
    return PersonalFeedbackEvent(
        feedback_id=feedback_id,
        user_id=user_id,
        session_id="session-learning",
        source_result_id=f"source-{feedback_id}",
        feedback_message="明确的个人推荐负反馈",
        feedback_type="recommendation_irrelevant",
        completeness="complete",
        corrected_query="继续推荐相关内容",
        target_document_ids=signal.source_document_ids,
        next_action="retry_recommendation",
        status="recovered",
        recovery_count=1,
        recovery_result_id=f"recovery-{feedback_id}",
        recommendation_signals=(signal,),
        memory_routes=("recommendation_profile",),
        memory_statuses={"recommendation_profile": "pending"},
        reason_code="explicit_negative_preference",
        created_at=_NOW,
        updated_at=_NOW,
    )


class PersonalFeedbackLearningTests(unittest.IsolatedAsyncioTestCase):
    """推荐负反馈只能形成个人、低优先级且有界的排序事实。"""

    async def test_specific_article_feedback_does_not_block_article_topics(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(prefix="feedback-learning-") as root:
            feedback_store = _LearningFeedbackStore()
            feature_store = _learning_feature_store(root)
            learning = PersonalFeedbackLearningService(
                feedback_store=feedback_store,
                feature_store=feature_store,
            )
            event = _learning_event(
                "feedback-article",
                RecommendationMemorySignal(
                    target_type="article",
                    target_value="java-post-002",
                    source_document_ids=("java-post-002",),
                    specific=True,
                    persistence="long_term_candidate",
                ),
            )
            feedback_store.events.append(event)

            statuses = await learning.apply(event=event)
            profile = await UserProfileAgent(
                feature_store=feature_store,
                enable_llm=False,
                clock=lambda: _NOW,
            ).run(user_id="10001")
            assert profile.profile is not None

            self.assertEqual(statuses, {"recommendation_profile": "applied"})
            self.assertEqual(
                profile.profile.behavior_profile.negative_document_ids,
                ["java-post-002"],
            )
            self.assertNotIn(
                "Java",
                [
                    item.topic
                    for item in profile.profile.behavior_profile.negative_interests
                ],
            )

    async def test_specific_difficulty_feedback_applies_immediately(self) -> None:
        with tempfile.TemporaryDirectory(prefix="feedback-learning-") as root:
            feedback_store = _LearningFeedbackStore()
            feature_store = _learning_feature_store(root)
            learning = PersonalFeedbackLearningService(
                feedback_store=feedback_store,
                feature_store=feature_store,
            )
            event = _learning_event(
                "feedback-difficulty",
                RecommendationMemorySignal(
                    target_type="difficulty",
                    target_value="beginner",
                    source_document_ids=("spring-boot-post-001",),
                    specific=True,
                    persistence="long_term_candidate",
                ),
            )
            feedback_store.events.append(event)

            await learning.apply(event=event)
            profile = await UserProfileAgent(
                feature_store=feature_store,
                enable_llm=False,
                clock=lambda: _NOW,
            ).run(user_id="10001")
            assert profile.profile is not None

            negatives = (
                profile.profile.behavior_profile.negative_difficulty_preferences
            )
            self.assertEqual([item.value for item in negatives], ["beginner"])
            self.assertLess(negatives[0].weight, 0.0)

    async def test_generic_signal_needs_three_independent_events(self) -> None:
        with tempfile.TemporaryDirectory(prefix="feedback-learning-") as root:
            feedback_store = _LearningFeedbackStore()
            feature_store = _learning_feature_store(root)
            learning = PersonalFeedbackLearningService(
                feedback_store=feedback_store,
                feature_store=feature_store,
            )
            for index in range(3):
                event = _learning_event(
                    f"feedback-generic-{index}",
                    RecommendationMemorySignal(
                        target_type="difficulty",
                        target_value="beginner",
                        source_document_ids=("spring-boot-post-001",),
                        specific=False,
                        persistence="long_term_candidate",
                    ),
                )
                feedback_store.events.append(event)
                statuses = await learning.apply(event=event)
                if index < 2:
                    self.assertEqual(
                        statuses,
                        {"recommendation_profile": "skipped"},
                    )

            profile = await UserProfileAgent(
                feature_store=feature_store,
                enable_llm=False,
                clock=lambda: _NOW,
            ).run(user_id="10001")
            assert profile.profile is not None
            self.assertEqual(
                [
                    item.value
                    for item in profile.profile.behavior_profile.negative_difficulty_preferences
                ],
                ["beginner"],
            )

    async def test_duplicate_learning_uses_one_deterministic_behavior_id(self) -> None:
        with tempfile.TemporaryDirectory(prefix="feedback-learning-idempotent-") as root:
            feedback_store = _LearningFeedbackStore()
            feature_store = _learning_feature_store(root)
            learning = PersonalFeedbackLearningService(
                feedback_store=feedback_store,
                feature_store=feature_store,
            )
            event = _learning_event(
                "feedback-idempotent",
                RecommendationMemorySignal(
                    target_type="article",
                    target_value="java-post-002",
                    source_document_ids=("java-post-002",),
                    specific=True,
                    persistence="explicit_long_term",
                ),
            )
            feedback_store.events.append(event)

            first = await learning.apply(event=event)
            second = await learning.apply(event=event)
            stored = feature_store._user_repository.list_all_events()

            self.assertEqual(first, {"recommendation_profile": "applied"})
            self.assertEqual(second, {"recommendation_profile": "applied"})
            self.assertEqual(len(stored), 1)
            self.assertTrue(stored[0].event_id.startswith("feedback-"))


class _FeedbackLlm:
    """记录反馈 Agent 输入并返回固定结构化候选。"""

    def __init__(self, output: dict[str, Any]) -> None:
        self.output = output
        self.calls = 0
        self.messages: list[Any] = []
        self.closed = False

    async def ainvoke(self, messages: list[Any]) -> dict[str, Any]:
        self.calls += 1
        self.messages = list(messages)
        return dict(self.output)

    async def aclose(self) -> None:
        self.closed = True


class FeedbackRecoveryAgentTests(unittest.IsolatedAsyncioTestCase):
    """质量反馈 Agent 只能生成有界 Schema 候选。"""

    async def test_agent_receives_bounded_snapshot_without_user_profile(
        self,
    ) -> None:
        llm = _FeedbackLlm(
            {
                "feedback_type": "recommendation_irrelevant",
                "action": {
                    "kind": "retry_recommendation",
                    "corrected_query": "深入的 Java 线程池调优",
                    "target_document_ids": ["doc-2"],
                    "recommendation_signals": [],
                },
                "reason_code": "second_item_too_basic",
                "confidence": 0.95,
            }
        )
        agent = FeedbackRecoveryAgent(llm=llm)

        analysis = await agent.analyze(
            message="第二篇太基础了，想看更深入的线程池调优",
            snapshot=_recommendation_snapshot(("doc-1", "doc-2")),
            pending_event=None,
            previous_user_message="推荐 Java 并发文章",
            previous_assistant_message="已找到两篇文章",
        )

        assert analysis is not None
        self.assertEqual(analysis.feedback_type, "recommendation_irrelevant")
        self.assertEqual(llm.calls, 1)
        envelope = json.loads(llm.messages[-1].content)
        self.assertEqual(set(envelope), {"contract", "input"})
        self.assertEqual(envelope["contract"]["name"], "feedback_recovery")
        self.assertEqual(envelope["contract"]["version"], 2)
        schema = envelope["contract"]["output_schema"]
        self.assertIn("action", schema["properties"])
        payload = envelope["input"]
        self.assertNotIn("user_id", payload)
        self.assertNotIn("session_id", payload)
        self.assertNotIn("profile", payload)
        self.assertNotIn("prompt", payload)
        self.assertEqual(
            payload["snapshot"]["recommendation_document_ids"],
            ["doc-1", "doc-2"],
        )

    async def test_agent_rejects_free_tool_action_and_unknown_fields(self) -> None:
        agent = FeedbackRecoveryAgent(
            llm=_FeedbackLlm(
                {
                    "tool": "search_web",
                    "thought": "should_not_be_accepted",
                }
            )
        )

        with self.assertRaises(ValueError):
            await agent.analyze(
                message="第二篇不相关",
                snapshot=_recommendation_snapshot(("doc-1", "doc-2")),
                pending_event=None,
                previous_user_message="推荐 Java 并发文章",
                previous_assistant_message="已找到两篇文章",
            )

    async def test_agent_without_llm_returns_none_and_closes_owned_client(
        self,
    ) -> None:
        agent = FeedbackRecoveryAgent(llm=None)
        self.assertIsNone(
            await agent.analyze(
                message="这些推荐不好",
                snapshot=_recommendation_snapshot(("doc-1",)),
                pending_event=None,
                previous_user_message=None,
                previous_assistant_message=None,
            )
        )
        await agent.aclose()


def _chunk(chunk_id: str, *, position: int) -> KnowledgeChunkRecord:
    return KnowledgeChunkRecord(
        chunk_id=chunk_id,
        document_id="doc-1",
        title="Java 线程池",
        topics=["Java"],
        content_type="tutorial",
        difficulty="advanced",
        author_id="author-1",
        position=position,
        heading_path=("配置",),
        content=f"可信证据 {chunk_id}",
        content_hash=(str(position + 1) * 64)[:64],
        token_count=10,
    )


class _RecoveryRepository:
    def __init__(self, chunks: tuple[KnowledgeChunkRecord, ...]) -> None:
        self.records = {chunk.chunk_id: chunk for chunk in chunks}

    def get_chunks_by_ids(
        self,
        chunk_ids: tuple[str, ...],
    ) -> tuple[KnowledgeChunkRecord, ...]:
        return tuple(
            self.records[chunk_id]
            for chunk_id in chunk_ids
            if chunk_id in self.records
        )

    def list_ready_images_by_chunk_ids(self, chunk_ids: tuple[str, ...]) -> tuple:
        return ()


class _RecoveryAnswerAgent:
    def __init__(self) -> None:
        self.calls = 0
        self.evidence: tuple[KnowledgeChunkRecord, ...] = ()

    async def generate(self, **kwargs: Any) -> KnowledgeGeneratedAnswer:
        self.calls += 1
        self.evidence = tuple(kwargs["evidence"])
        return KnowledgeGeneratedAnswer(
            answer="修正后的可信回答。",
            cited_chunk_ids=tuple(chunk.chunk_id for chunk in self.evidence),
        )

    async def aclose(self) -> None:
        return None


class KnowledgeRecoveryTests(unittest.IsolatedAsyncioTestCase):
    """已验证证据再回答不得重新搜索或接受失效 Chunk。"""

    async def test_regenerate_answer_reuses_ready_chunks_without_search(
        self,
    ) -> None:
        repository = _RecoveryRepository(
            (_chunk("chunk-1", position=0), _chunk("chunk-2", position=1))
        )
        answer_agent = _RecoveryAnswerAgent()
        service = KnowledgeQaService(
            repository=repository,
            answer_agent=answer_agent,
        )

        result = await service.regenerate_from_evidence(
            "请补充关键步骤并先给结论",
            chunk_ids=("chunk-1", "chunk-2"),
            request_route="/api/v1/chat",
        )

        self.assertEqual(result.status, "success")
        self.assertEqual(answer_agent.calls, 1)
        self.assertEqual(
            tuple(citation.chunk_id for citation in result.citations),
            ("chunk-1", "chunk-2"),
        )
        self.assertEqual(
            tuple(chunk.chunk_id for chunk in answer_agent.evidence),
            ("chunk-1", "chunk-2"),
        )

    async def test_regenerate_answer_rejects_deleted_or_changed_chunk(
        self,
    ) -> None:
        repository = _RecoveryRepository((_chunk("chunk-1", position=0),))
        answer_agent = _RecoveryAnswerAgent()
        service = KnowledgeQaService(
            repository=repository,
            answer_agent=answer_agent,
        )

        result = await service.regenerate_from_evidence(
            "补充说明",
            chunk_ids=("chunk-1", "chunk-2"),
            request_route="/api/v1/chat",
        )

        self.assertEqual(result.status, "insufficient_evidence")
        self.assertEqual(answer_agent.calls, 0)


class _GraphFeedbackAgent:
    """为 Graph 返回固定质量反馈候选。"""

    def __init__(self, analysis: FeedbackAnalysis | None) -> None:
        self.analysis = analysis
        self.calls = 0

    async def analyze(self, **_: Any) -> FeedbackAnalysis | None:
        self.calls += 1
        return self.analysis


class _FailingGraphFeedbackAgent:
    def __init__(self) -> None:
        self.calls = 0

    async def analyze(self, **_: Any) -> FeedbackAnalysis | None:
        self.calls += 1
        raise ValueError("模拟反馈 Schema 校验失败")


class _ForgedTargetFeedbackAgent:
    async def analyze(self, **_: Any) -> FeedbackAnalysis:
        return FeedbackAnalysis(
            is_feedback=True,
            feedback_type="recommendation_irrelevant",
            completeness="complete",
            corrected_query="Java 线程池调优",
            target_document_ids=("forged-doc",),
            suggested_action="retry_recommendation",
            reason_code="forged_target",
            confidence=0.99,
        )


class _GraphKnowledgeService:
    """分别记录普通问答、重新检索和证据再回答。"""

    def __init__(self) -> None:
        self.ask_calls: list[dict[str, Any]] = []
        self.regenerate_calls: list[dict[str, Any]] = []

    async def ask(self, question: str, **kwargs: Any) -> KnowledgeAnswerResult:
        self.ask_calls.append({"question": question, **kwargs})
        return KnowledgeAnswerResult(
            status="success",
            answer="普通知识回答。",
            citations=(
                KnowledgeCitation(
                    citation_id="1",
                    document_id="doc-1",
                    title="Java 线程池",
                    chunk_id="chunk-1",
                    heading_path=("配置",),
                    excerpt="可信证据",
                ),
            ),
            resolved_document_ids=("doc-1",),
            resolved_document_titles=("Java 线程池",),
        )

    async def regenerate_from_evidence(
        self,
        question: str,
        **kwargs: Any,
    ) -> KnowledgeAnswerResult:
        self.regenerate_calls.append({"question": question, **kwargs})
        return KnowledgeAnswerResult(
            status="success",
            answer="更简洁的可信回答。",
            citations=(
                KnowledgeCitation(
                    citation_id="1",
                    document_id="doc-1",
                    title="Java 线程池",
                    chunk_id="chunk-1",
                    heading_path=("配置",),
                    excerpt="可信证据",
                ),
            ),
            resolved_document_ids=("doc-1",),
            resolved_document_titles=("Java 线程池",),
        )


def _classifying_event(snapshot: ConversationResultSnapshot) -> PersonalFeedbackEvent:
    return PersonalFeedbackEvent(
        feedback_id="feedback-graph",
        user_id=snapshot.user_id,
        session_id=snapshot.session_id,
        source_result_id=snapshot.result_id,
        feedback_message="这些推荐不好",
        status="classifying",
        reason_code="classifying_pending",
        created_at=_NOW,
        updated_at=_NOW,
    )


class PersonalFeedbackGraphTests(unittest.IsolatedAsyncioTestCase):
    """Graph 应在原意图链前执行一次有限反馈路由。"""

    @staticmethod
    def _graph(
        feedback_agent: _GraphFeedbackAgent,
        knowledge_service: _GraphKnowledgeService,
    ) -> ConversationGraph:
        from app.agents.intent_recognition_agent import IntentRecognitionAgent

        return ConversationGraph(
            intent_agent=IntentRecognitionAgent(enable_llm=False),
            arbitrator=ConversationArbitrator(),
            recall_agent=object(),
            rerank_agent=object(),
            aggregator=object(),
            knowledge_qa_service=knowledge_service,
            feedback_agent=feedback_agent,
            feedback_policy=FeedbackRecoveryPolicy(),
        )

    async def test_normal_follow_up_skips_feedback_agent(self) -> None:
        feedback_agent = _GraphFeedbackAgent(None)
        knowledge_service = _GraphKnowledgeService()
        graph = self._graph(feedback_agent, knowledge_service)

        result = await graph.run(
            user_id="user-1",
            session_id="session-1",
            message="第二篇主要讲了什么？",
            history=[],
            previous_context=None,
            intent_state=IntentState.RECOMMENDATION,
            feedback_context=None,
        )

        self.assertEqual(feedback_agent.calls, 0)
        self.assertIn("recognize_intent", result.trace)

    async def test_incomplete_feedback_returns_one_clarification(self) -> None:
        snapshot = _recommendation_snapshot(("doc-1", "doc-2"))
        feedback_agent = _GraphFeedbackAgent(
            FeedbackAnalysis(
                is_feedback=True,
                feedback_type="recommendation_irrelevant",
                completeness="incomplete",
                missing_information=("reason",),
                suggested_action="clarify",
                reason_code="generic_negative",
                confidence=0.92,
            )
        )
        graph = self._graph(feedback_agent, _GraphKnowledgeService())

        analysis, decision = await graph.classify_feedback(
            message="这些推荐不好",
            history=[],
            feedback_context={
                "latest_result": snapshot,
                "pending_feedback": _classifying_event(snapshot),
            },
        )

        self.assertIsNotNone(analysis)
        self.assertEqual(decision.next_action, "clarify")
        self.assertIn("不相关", decision.clarification_question or "")

    async def test_classification_failure_uses_safe_fallback(self) -> None:
        snapshot = _recommendation_snapshot(("doc-1", "doc-2"))
        feedback_agent = _FailingGraphFeedbackAgent()
        graph = self._graph(feedback_agent, _GraphKnowledgeService())

        analysis, decision = await graph.classify_feedback(
            message="这些推荐不好",
            history=[],
            feedback_context={
                "latest_result": snapshot,
                "pending_feedback": _classifying_event(snapshot),
            },
        )

        self.assertIsNone(analysis)
        self.assertEqual(feedback_agent.calls, 1)
        self.assertEqual(decision.next_action, "clarify")
        self.assertEqual(decision.reason_code, "generic_negative_needs_detail")

    async def test_policy_rejection_closes_without_executing_action(self) -> None:
        snapshot = _recommendation_snapshot(("doc-1", "doc-2"))
        graph = self._graph(
            _ForgedTargetFeedbackAgent(),
            _GraphKnowledgeService(),
        )

        analysis, decision = await graph.classify_feedback(
            message="第二篇不相关",
            history=[],
            feedback_context={
                "latest_result": snapshot,
                "pending_feedback": _classifying_event(snapshot),
            },
        )

        self.assertIsNone(analysis)
        self.assertFalse(decision.is_feedback)
        self.assertEqual(decision.next_action, "normal")
        self.assertEqual(decision.reason_code, "fallback_no_high_precision_feedback")

    async def test_answer_style_feedback_reuses_snapshot_evidence(self) -> None:
        snapshot = _knowledge_snapshot(("chunk-1",))
        pending = _classifying_event(snapshot)
        feedback_agent = _GraphFeedbackAgent(
            FeedbackAnalysis(
                is_feedback=True,
                feedback_type="answer_style",
                completeness="complete",
                corrected_query="请先给结论并简短说明",
                suggested_action="retry_answer_from_evidence",
                reason_code="style_too_verbose",
                confidence=0.96,
            )
        )
        knowledge_service = _GraphKnowledgeService()
        graph = self._graph(feedback_agent, knowledge_service)
        context = {
            "latest_result": snapshot,
            "pending_feedback": pending,
        }
        analysis, decision = await graph.classify_feedback(
            message="太啰嗦了，以后先给结论",
            history=[],
            feedback_context=context,
        )

        result = await graph.run(
            user_id="user-1",
            session_id="session-1",
            message="太啰嗦了，以后先给结论",
            history=[],
            previous_context=None,
            feedback_context=context,
            protected_feedback_analysis=analysis,
            protected_feedback_decision=decision,
        )

        self.assertEqual(len(knowledge_service.ask_calls), 0)
        self.assertEqual(len(knowledge_service.regenerate_calls), 1)
        self.assertEqual(
            knowledge_service.regenerate_calls[0]["chunk_ids"],
            ("chunk-1",),
        )
        self.assertTrue(result.reply.message.startswith("已根据你的反馈修正"))
        self.assertEqual(result.feedback_decision, decision)
        assert result.result_snapshot_draft is not None
        self.assertEqual(result.result_snapshot_draft.result_type, "knowledge_answer")


class _KnownUserStore:
    async def get_user(self, user_id: str) -> object:
        return object() if user_id == "user-1" else None


class _ConversationFeedbackWorkflow:
    """检查会话服务是否严格执行先记录、再分类、再补救。"""

    def __init__(
        self,
        store: SQLiteConversationStore,
        *,
        fail_recovery: bool = False,
    ) -> None:
        self.store = store
        self.fail_recovery = fail_recovery
        self.classify_calls = 0
        self.run_calls = 0
        self.recovery_runs = 0

    async def classify_feedback(
        self,
        *,
        message: str,
        history: list[ConversationTurn],
        feedback_context: object,
    ) -> tuple[FeedbackAnalysis | None, object]:
        del history
        self.classify_calls += 1
        context = await self.store.load_feedback_context("user-1", "session-1")
        assert context.pending_feedback is not None
        if self.classify_calls == 1:
            assert context.pending_feedback.status == "classifying"
            analysis = FeedbackAnalysis(
                is_feedback=True,
                feedback_type="recommendation_irrelevant",
                completeness="incomplete",
                missing_information=("reason",),
                suggested_action="clarify",
                reason_code="generic_negative",
                confidence=0.9,
            )
        else:
            assert context.pending_feedback.status == "awaiting_detail"
            analysis = FeedbackAnalysis(
                is_feedback=True,
                feedback_type="recommendation_irrelevant",
                completeness="complete",
                corrected_query="Java 线程池调优",
                suggested_action="retry_recommendation",
                reason_code="corrected_topic",
                confidence=0.96,
            )
        decision = FeedbackRecoveryPolicy().protect(
            message,
            snapshot=context.latest_result,
            pending_event=context.pending_feedback,
            analysis=analysis,
        )
        return analysis, decision

    async def run(self, **kwargs: Any) -> ConversationGraphResult:
        self.run_calls += 1
        decision = kwargs.get("protected_feedback_decision")
        analysis = kwargs.get("protected_feedback_analysis")
        if decision is None:
            reply = ConversationReply(
                session_id="session-1",
                message="已返回两篇文章。",
                intent_source=RecognitionSource.RULE,
                action=ArbitrationAction.NEW,
                recommendations=[],
            )
            return ConversationGraphResult(
                reply=reply,
                history_message="已返回两篇文章。 推荐结果：文章一；文章二",
                pending_context=None,
                result_snapshot_draft=ConversationResultSnapshotDraft(
                    result_type="recommendation",
                    query="Java 并发",
                    recommendation_document_ids=("doc-1", "doc-2"),
                ),
                trace=["normal_result"],
            )
        if decision.next_action == "clarify":
            reply = ConversationReply(
                session_id="session-1",
                message=decision.clarification_question,
                intent_source=RecognitionSource.RULE,
                action=ArbitrationAction.CLARIFY,
                needs_clarification=True,
            )
            return ConversationGraphResult(
                reply=reply,
                history_message=decision.clarification_question,
                feedback_analysis=analysis,
                feedback_decision=decision,
                trace=["feedback_clarify"],
            )
        if self.fail_recovery:
            raise RuntimeError("模拟补救执行失败")
        self.recovery_runs += 1
        context = await self.store.load_feedback_context("user-1", "session-1")
        assert context.pending_feedback is not None
        assert context.pending_feedback.status == "recovering"
        reply = ConversationReply(
            session_id="session-1",
            message="已根据你的反馈修正：返回线程池调优文章。",
            intent_source=RecognitionSource.RULE,
            action=ArbitrationAction.REFINE,
        )
        return ConversationGraphResult(
            reply=reply,
            history_message=(
                "已根据你的反馈修正：返回线程池调优文章。 推荐结果：文章三"
            ),
            feedback_analysis=analysis,
            feedback_decision=decision,
            feedback_recovery_succeeded=True,
            result_snapshot_draft=ConversationResultSnapshotDraft(
                result_type="recommendation",
                query="Java 线程池调优",
                recommendation_document_ids=("doc-3",),
            ),
            trace=["feedback_recovery"],
        )


class PersonalFeedbackConversationTests(unittest.IsolatedAsyncioTestCase):
    """会话服务应原子持久化追问和追加修正结果。"""

    async def test_full_feedback_chain_persists_and_appends_corrected_result(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(prefix="feedback-conversation-") as root:
            store = SQLiteConversationStore(Path(root) / "conversations.sqlite3")
            service = ConversationService(
                user_store=_KnownUserStore(),
                recall_agent=object(),
                rerank_agent=object(),
                conversation_store=store,
                enable_llm=False,
                feedback_policy=FeedbackRecoveryPolicy(),
                clock=lambda: _NOW,
            )
            workflow = _ConversationFeedbackWorkflow(store)
            service.workflow = workflow

            await service.chat(
                "user-1",
                "推荐 Java 并发文章",
                session_id="session-1",
            )
            first_context = await store.load_feedback_context(
                "user-1", "session-1"
            )
            assert first_context.latest_result is not None
            self.assertEqual(
                first_context.latest_result.recommendation_document_ids,
                ("doc-1", "doc-2"),
            )

            clarification = await service.chat(
                "user-1",
                "这些推荐不好",
                session_id="session-1",
            )
            awaiting = await store.load_feedback_context("user-1", "session-1")
            assert awaiting.pending_feedback is not None
            self.assertTrue(clarification.needs_clarification)
            self.assertEqual(awaiting.pending_feedback.status, "awaiting_detail")
            self.assertEqual(awaiting.pending_feedback.clarification_count, 1)

            corrected = await service.chat(
                "user-1",
                "都和线程池调优无关",
                session_id="session-1",
            )
            final_context = await store.load_feedback_context("user-1", "session-1")
            events = await store.list_feedback_events("user-1")
            session = await store.load("user-1", "session-1")

            assert session is not None
            assert final_context.latest_result is not None
            self.assertIsNone(final_context.pending_feedback)
            self.assertIsNone(session.pending_feedback_id)
            self.assertEqual(events[0].status, "recovered")
            self.assertEqual(events[0].recovery_count, 1)
            self.assertEqual(
                events[0].recovery_result_id,
                final_context.latest_result.result_id,
            )
            self.assertEqual(
                final_context.latest_result.recommendation_document_ids,
                ("doc-3",),
            )
            self.assertTrue(corrected.message.startswith("已根据你的反馈修正"))
            self.assertEqual(len(session.history), 6)
            self.assertEqual(workflow.classify_calls, 2)

    async def test_stale_recovering_is_failed_without_replaying_action(self) -> None:
        with tempfile.TemporaryDirectory(prefix="feedback-stale-") as root:
            store = SQLiteConversationStore(Path(root) / "conversations.sqlite3")
            snapshot = _recommendation_snapshot(("doc-1", "doc-2"))
            event = PersonalFeedbackEvent(
                feedback_id="feedback-stale",
                user_id="user-1",
                session_id="session-1",
                source_result_id=snapshot.result_id,
                feedback_message="都和线程池调优无关",
                feedback_type="recommendation_irrelevant",
                completeness="complete",
                corrected_query="Java 线程池调优",
                next_action="retry_recommendation",
                status="recovering",
                recovery_count=1,
                reason_code="corrected_topic",
                created_at=_NOW,
                updated_at=_NOW,
            )
            session = ConversationSession(
                session_id="session-1",
                user_id="user-1",
                history=[
                    ConversationTurn(role="user", content="推荐 Java 并发文章"),
                    ConversationTurn(role="assistant", content="已返回两篇文章。"),
                ],
                turn_count=1,
                pending_feedback_id=event.feedback_id,
            )
            await store.commit_recovery(
                sessions=(session,),
                snapshots=(snapshot,),
                feedback_events=(event,),
            )
            service = ConversationService(
                user_store=_KnownUserStore(),
                recall_agent=object(),
                rerank_agent=object(),
                conversation_store=store,
                enable_llm=False,
                feedback_policy=FeedbackRecoveryPolicy(),
                clock=lambda: _NOW,
            )
            workflow = _ConversationFeedbackWorkflow(store)
            service.workflow = workflow

            await service.chat(
                "user-1",
                "重新推荐一组 Java 并发文章",
                session_id="session-1",
            )

            events = await store.list_feedback_events("user-1")
            stored = await store.load("user-1", "session-1")
            assert stored is not None
            self.assertEqual(events[0].status, "recovery_failed")
            self.assertEqual(events[0].recovery_count, 1)
            self.assertIsNone(stored.pending_feedback_id)
            self.assertEqual(workflow.classify_calls, 0)
            self.assertEqual(workflow.run_calls, 1)

    async def test_recovery_exception_appends_failure_and_closes_event(self) -> None:
        with tempfile.TemporaryDirectory(prefix="feedback-action-failed-") as root:
            store = SQLiteConversationStore(Path(root) / "conversations.sqlite3")
            service = ConversationService(
                user_store=_KnownUserStore(),
                recall_agent=object(),
                rerank_agent=object(),
                conversation_store=store,
                enable_llm=False,
                feedback_policy=FeedbackRecoveryPolicy(),
                clock=lambda: _NOW,
            )
            workflow = _ConversationFeedbackWorkflow(store, fail_recovery=True)
            service.workflow = workflow

            await service.chat(
                "user-1",
                "推荐 Java 并发文章",
                session_id="session-1",
            )
            await service.chat(
                "user-1",
                "这些推荐不好",
                session_id="session-1",
            )
            failed = await service.chat(
                "user-1",
                "都和线程池调优无关",
                session_id="session-1",
            )

            context = await store.load_feedback_context("user-1", "session-1")
            events = await store.list_feedback_events("user-1")
            session = await store.load("user-1", "session-1")

            assert session is not None
            self.assertTrue(failed.message.startswith("本次补救未完成"))
            self.assertIsNone(context.pending_feedback)
            self.assertIsNone(session.pending_feedback_id)
            self.assertEqual(events[0].status, "recovery_failed")
            self.assertEqual(events[0].reason_code, "recovery_action_failed")
            self.assertEqual(events[0].recovery_count, 1)
            self.assertIsNone(events[0].recovery_result_id)
            self.assertEqual(len(session.history), 6)
            self.assertEqual(
                context.latest_result.recommendation_document_ids,
                ("doc-1", "doc-2"),
            )

    async def test_concurrent_duplicate_detail_executes_recovery_once(self) -> None:
        with tempfile.TemporaryDirectory(prefix="feedback-concurrent-") as root:
            store = SQLiteConversationStore(Path(root) / "conversations.sqlite3")
            service = ConversationService(
                user_store=_KnownUserStore(),
                recall_agent=object(),
                rerank_agent=object(),
                conversation_store=store,
                enable_llm=False,
                feedback_policy=FeedbackRecoveryPolicy(),
                clock=lambda: _NOW,
            )
            workflow = _ConversationFeedbackWorkflow(store)
            service.workflow = workflow

            await service.chat(
                "user-1",
                "推荐 Java 并发文章",
                session_id="session-1",
            )
            await service.chat(
                "user-1",
                "这些推荐不好",
                session_id="session-1",
            )

            replies = await asyncio.gather(
                service.chat(
                    "user-1",
                    "都和线程池调优无关",
                    session_id="session-1",
                ),
                service.chat(
                    "user-1",
                    "都和线程池调优无关",
                    session_id="session-1",
                ),
            )

            events = await store.list_feedback_events("user-1")
            session = await store.load("user-1", "session-1")
            assert session is not None
            self.assertEqual(workflow.recovery_runs, 1)
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0].status, "recovered")
            self.assertEqual(len(session.history), 6)
            self.assertEqual(replies[0].message, replies[1].message)


class _ImmediateLearningWorkflow:
    def __init__(self, store: SQLiteConversationStore) -> None:
        self.store = store

    async def classify_feedback(self, **kwargs: Any) -> tuple[FeedbackAnalysis, object]:
        context = kwargs["feedback_context"]
        analysis = FeedbackAnalysis(
            is_feedback=True,
            feedback_type="recommendation_irrelevant",
            completeness="complete",
            corrected_query="Java 并发进阶",
            target_document_ids=("doc-1",),
            suggested_action="retry_recommendation",
            recommendation_signals=(
                RecommendationMemorySignal(
                    target_type="article",
                    target_value="doc-1",
                    source_document_ids=("doc-1",),
                    specific=True,
                    persistence="explicit_long_term",
                ),
            ),
            reason_code="explicit_article_avoid",
            confidence=0.98,
        )
        return analysis, FeedbackRecoveryPolicy().protect(
            kwargs["message"],
            snapshot=context.latest_result,
            pending_event=context.pending_feedback,
            analysis=analysis,
        )

    async def run(self, **kwargs: Any) -> ConversationGraphResult:
        if kwargs.get("protected_feedback_decision") is None:
            reply = ConversationReply(
                session_id="session-1",
                message="已返回两篇文章。",
                intent_source=RecognitionSource.RULE,
                action=ArbitrationAction.NEW,
            )
            return ConversationGraphResult(
                reply=reply,
                history_message="已返回两篇文章。",
                result_snapshot_draft=ConversationResultSnapshotDraft(
                    result_type="recommendation",
                    query="Java 并发",
                    recommendation_document_ids=("doc-1", "doc-2"),
                ),
            )
        context = await self.store.load_feedback_context("user-1", "session-1")
        assert context.pending_feedback is not None
        assert context.pending_feedback.status == "recovering"
        reply = ConversationReply(
            session_id="session-1",
            message="已根据你的反馈修正：返回新的进阶文章。",
            intent_source=RecognitionSource.RULE,
            action=ArbitrationAction.REFINE,
        )
        return ConversationGraphResult(
            reply=reply,
            history_message=reply.message,
            feedback_analysis=kwargs["protected_feedback_analysis"],
            feedback_decision=kwargs["protected_feedback_decision"],
            feedback_recovery_succeeded=True,
            result_snapshot_draft=ConversationResultSnapshotDraft(
                result_type="recommendation",
                query="Java 并发进阶",
                recommendation_document_ids=("doc-3",),
            ),
        )


class _RecoveryLearningRecorder:
    def __init__(self, store: SQLiteConversationStore) -> None:
        self.store = store
        self.calls = 0

    async def apply(self, *, event: PersonalFeedbackEvent) -> dict[str, str]:
        self.calls += 1
        stored = next(
            item
            for item in await self.store.list_feedback_events(event.user_id)
            if item.feedback_id == event.feedback_id
        )
        assert stored.status == "recovered"
        return {"recommendation_profile": "applied"}


class PersonalFeedbackPostCommitLearningTests(unittest.IsolatedAsyncioTestCase):
    """个人画像学习只能发生在修正结果成功提交之后。"""

    async def test_recovered_event_advances_memory_status_after_commit(self) -> None:
        with tempfile.TemporaryDirectory(prefix="feedback-post-learning-") as root:
            store = SQLiteConversationStore(Path(root) / "conversations.sqlite3")
            learning = _RecoveryLearningRecorder(store)
            service = ConversationService(
                user_store=_KnownUserStore(),
                recall_agent=object(),
                rerank_agent=object(),
                conversation_store=store,
                enable_llm=False,
                feedback_policy=FeedbackRecoveryPolicy(),
                personal_feedback_learning=learning,
                clock=lambda: _NOW,
            )
            service.workflow = _ImmediateLearningWorkflow(store)

            await service.chat(
                "user-1",
                "推荐 Java 并发文章",
                session_id="session-1",
            )
            await service.chat(
                "user-1",
                "第一篇我不感兴趣，以后不要再推荐",
                session_id="session-1",
            )

            event = (await store.list_feedback_events("user-1"))[0]
            self.assertEqual(learning.calls, 1)
            self.assertEqual(event.status, "recovered")
            self.assertIsNotNone(event.recovery_result_id)
            self.assertEqual(
                event.memory_statuses,
                {"recommendation_profile": "applied"},
            )

    async def test_feedback_chain_emits_safe_stream_business_stages(self) -> None:
        with tempfile.TemporaryDirectory(prefix="feedback-stream-") as root:
            store = SQLiteConversationStore(Path(root) / "conversations.sqlite3")
            learning = _RecoveryLearningRecorder(store)
            service = ConversationService(
                user_store=_KnownUserStore(),
                recall_agent=object(),
                rerank_agent=object(),
                conversation_store=store,
                enable_llm=False,
                feedback_policy=FeedbackRecoveryPolicy(),
                personal_feedback_learning=learning,
                clock=lambda: _NOW,
            )
            service.workflow = _ImmediateLearningWorkflow(store)
            await service.chat(
                "user-1",
                "推荐 Java 并发文章",
                session_id="session-1",
            )
            recorder = ConversationStreamRecorder()

            with conversation_stream_context(recorder):
                await service.chat(
                    "user-1",
                    "第一篇我不感兴趣，以后不要再推荐",
                    session_id="session-1",
                )

            events = recorder.snapshot()
            stages = {event["stage"] for event in events}
            self.assertTrue(
                {
                    "反馈检测",
                    "反馈分类",
                    "信息完整度",
                    "补救准备",
                    "修正结果提交",
                    "个人记忆更新",
                }.issubset(stages)
            )
            serialized = json.dumps(events, ensure_ascii=False).casefold()
            self.assertNotIn("prompt", serialized)
            self.assertNotIn("raw_response", serialized)
            self.assertNotIn("chain_of_thought", serialized)


class _StyleFeedbackLlm:
    def __init__(self, *, fail: bool = False) -> None:
        self.calls = 0
        self.fail = fail

    async def ainvoke(self, messages: list[object]) -> dict[str, object]:
        del messages
        self.calls += 1
        if self.fail:
            raise ValueError("模拟回答方式分析失败")
        return ConversationFeedbackAnalysis(
            is_preference_feedback=True,
            feedback_type="format_preference",
            scope="knowledge_qa",
            detail_level="brief",
            answer_structure="conclusion_first",
            persistence="explicit_long_term",
            confidence=0.96,
            reason_code="answer_structure_refined",
            reason_summary="用户明确要求以后先给结论并简短说明。",
        ).model_dump(mode="json")


class _NoWakeWorker:
    def __init__(self) -> None:
        self.calls = 0

    def wake(self) -> None:
        self.calls += 1


class _StyleFeedbackWorkflow:
    def __init__(self, store: SQLiteConversationStore) -> None:
        self.store = store
        self.projection = None

    async def classify_feedback(self, **kwargs: Any) -> tuple[FeedbackAnalysis, object]:
        context = kwargs["feedback_context"]
        analysis = FeedbackAnalysis(
            is_feedback=True,
            feedback_type="answer_style",
            completeness="complete",
            corrected_query="解释系统架构",
            suggested_action="retry_answer_from_evidence",
            reason_code="style_too_verbose",
            confidence=0.96,
        )
        decision = FeedbackRecoveryPolicy().protect(
            kwargs["message"],
            snapshot=context.latest_result,
            pending_event=context.pending_feedback,
            analysis=analysis,
        )
        return analysis, decision

    async def run(self, **kwargs: Any) -> ConversationGraphResult:
        self.projection = kwargs.get("interaction_memory")
        reply = ConversationReply(
            session_id="session-1",
            message="已根据你的反馈修正：结论：系统由编排、检索和回答层组成。",
            intent_source=RecognitionSource.RULE,
            action=ArbitrationAction.KNOWLEDGE_ANSWER,
            intent_state=IntentState.KNOWLEDGE_QA,
        )
        return ConversationGraphResult(
            reply=reply,
            history_message=reply.message,
            feedback_analysis=kwargs["protected_feedback_analysis"],
            feedback_decision=kwargs["protected_feedback_decision"],
            feedback_recovery_succeeded=True,
            result_snapshot_draft=ConversationResultSnapshotDraft(
                result_type="knowledge_answer",
                query="解释系统架构",
                citation_document_ids=("doc-1",),
                citation_chunk_ids=("chunk-1",),
                knowledge_status="success",
                resolved_document_ids=("doc-1",),
            ),
        )


class PersonalFeedbackStyleMemoryTests(unittest.IsolatedAsyncioTestCase):
    """回答方式反馈应即时采用并直接落入现有交互记忆。"""

    async def test_preanalyzed_style_feedback_updates_without_worker_reanalysis(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(prefix="feedback-style-") as root:
            root_path = Path(root)
            user_repository = SQLiteUserProfileRepository(
                root_path / "user_profiles.sqlite3"
            )
            user_repository.replace_user(
                UserBaseProfile(
                    user_id="user-1",
                    topics=[],
                    blocked_topics=[],
                    preferred_content_types=[],
                    preferred_difficulty="",
                    preferred_reading_length="",
                    followed_author_ids=[],
                    blocked_author_ids=[],
                    created_at=_NOW,
                )
            )
            interaction_repository = SQLiteUserInteractionMemoryRepository(
                user_repository.path
            )
            interaction_service = UserInteractionMemoryService(clock=lambda: _NOW)
            style_llm = _StyleFeedbackLlm()
            worker = _NoWakeWorker()
            store = SQLiteConversationStore(root_path / "conversations.sqlite3")
            snapshot = _knowledge_snapshot(("chunk-1",)).model_copy(
                update={"assistant_sequence_no": 1}
            )
            session = ConversationSession(
                session_id="session-1",
                user_id="user-1",
                intent_state=IntentState.KNOWLEDGE_QA,
                history=[
                    ConversationTurn(role="user", content="解释系统架构"),
                    ConversationTurn(role="assistant", content="一段很长的回答"),
                ],
                turn_count=1,
            )
            await store.commit_recovery(
                sessions=(session,),
                snapshots=(snapshot,),
            )
            service = ConversationService(
                user_store=_KnownUserStore(),
                recall_agent=object(),
                rerank_agent=object(),
                conversation_store=store,
                enable_llm=False,
                feedback_policy=FeedbackRecoveryPolicy(),
                interaction_memory_repository=interaction_repository,
                interaction_memory_service=interaction_service,
                interaction_memory_worker=worker,
                interaction_feedback_agent=ConversationFeedbackAgent(llm=style_llm),
                clock=lambda: _NOW,
            )
            workflow = _StyleFeedbackWorkflow(store)
            service.workflow = workflow

            await service.chat(
                "user-1",
                "太啰嗦了，以后先给结论并简短说明",
                session_id="session-1",
            )

            assert workflow.projection is not None
            self.assertEqual(
                workflow.projection.preferences[0].answer_structure,
                "conclusion_first",
            )
            self.assertEqual(style_llm.calls, 1)
            self.assertEqual(worker.calls, 0)
            memory = interaction_repository.get_memory("user-1")
            assert memory is not None
            self.assertEqual(
                memory.preferences[0].answer_structure,
                "conclusion_first",
            )
            stored_event = interaction_repository.get_event(
                next(iter(memory.preferences[0].source_event_ids))
            )
            assert stored_event is not None
            self.assertEqual(stored_event.status, "analyzed")

    async def test_style_analysis_failure_does_not_block_recovery(self) -> None:
        with tempfile.TemporaryDirectory(prefix="feedback-style-failed-") as root:
            root_path = Path(root)
            store = SQLiteConversationStore(root_path / "conversations.sqlite3")
            snapshot = _knowledge_snapshot(("chunk-1",)).model_copy(
                update={"assistant_sequence_no": 1}
            )
            session = ConversationSession(
                session_id="session-1",
                user_id="user-1",
                intent_state=IntentState.KNOWLEDGE_QA,
                history=[
                    ConversationTurn(role="user", content="解释系统架构"),
                    ConversationTurn(role="assistant", content="一段很长的回答"),
                ],
                turn_count=1,
            )
            await store.commit_recovery(
                sessions=(session,),
                snapshots=(snapshot,),
            )
            style_llm = _StyleFeedbackLlm(fail=True)
            service = ConversationService(
                user_store=_KnownUserStore(),
                recall_agent=object(),
                rerank_agent=object(),
                conversation_store=store,
                enable_llm=False,
                feedback_policy=FeedbackRecoveryPolicy(),
                interaction_feedback_agent=ConversationFeedbackAgent(llm=style_llm),
                clock=lambda: _NOW,
            )
            workflow = _StyleFeedbackWorkflow(store)
            service.workflow = workflow

            reply = await service.chat(
                "user-1",
                "太啰嗦了，以后先给结论并简短说明",
                session_id="session-1",
            )

            event = (await store.list_feedback_events("user-1"))[0]
            self.assertTrue(reply.message.startswith("已根据你的反馈修正"))
            self.assertEqual(style_llm.calls, 1)
            self.assertIsNone(workflow.projection)
            self.assertEqual(event.status, "recovered")


class _RetentionConversationStore:
    def __init__(self) -> None:
        self.calls: list[tuple[datetime, datetime]] = []

    async def purge_feedback_raw_before(
        self,
        cutoff: datetime,
        *,
        purged_at: datetime,
    ) -> int:
        self.calls.append((cutoff, purged_at))
        return 0


class PersonalFeedbackRetentionTests(unittest.IsolatedAsyncioTestCase):
    """运行期原文清理同一自然日最多执行一次。"""

    async def test_runtime_purge_runs_once_per_natural_day(self) -> None:
        current = [_NOW]
        store = _RetentionConversationStore()
        service = ConversationService(
            user_store=_KnownUserStore(),
            recall_agent=object(),
            rerank_agent=object(),
            conversation_store=store,
            enable_llm=False,
            clock=lambda: current[0],
        )

        await service._purge_feedback_raw_if_due()
        await service._purge_feedback_raw_if_due()
        current[0] = _NOW + timedelta(days=1)
        await service._purge_feedback_raw_if_due()

        self.assertEqual(len(store.calls), 2)
        self.assertEqual(store.calls[0][0], _NOW - timedelta(days=30))


class PersonalFeedbackBootstrapTests(unittest.TestCase):
    """启动期应装配独立 Agent、学习协调器、清理和关闭生命周期。"""

    def test_bootstrap_wires_independent_feedback_components(self) -> None:
        source = (
            Path(__file__).resolve().parents[1] / "app" / "bootstrap.py"
        ).read_text(encoding="utf-8")

        self.assertIn("FeedbackRecoveryAgent.from_settings(settings)", source)
        self.assertGreaterEqual(
            source.count("ConversationFeedbackAgent.from_settings(settings)"),
            2,
        )
        self.assertIn("PersonalFeedbackLearningService(", source)
        self.assertIn("await conversation_store.purge_feedback_raw_before(", source)
        self.assertIn("await feedback_recovery_agent.aclose()", source)
        self.assertIn("await interaction_feedback_agent.aclose()", source)

    def test_http_and_stream_schema_fields_remain_unchanged(self) -> None:
        self.assertEqual(
            set(ChatRequest.model_fields),
            {"user_id", "message", "session_id"},
        )
        self.assertNotIn("feedback", ChatResponse.model_fields)
        self.assertEqual(
            set(ChatStreamProcessEvent.model_fields),
            {
                "event",
                "trace_id",
                "sequence",
                "elapsed_ms",
                "stage",
                "component",
                "status",
                "title",
                "summary",
                "details",
            },
        )
        self.assertEqual(
            set(ChatStreamResultEvent.model_fields),
            {"event", "trace_id", "sequence", "elapsed_ms", "response"},
        )
        self.assertEqual(
            set(ChatStreamErrorEvent.model_fields),
            {"event", "trace_id", "sequence", "elapsed_ms", "error"},
        )


if __name__ == "__main__":
    unittest.main()
