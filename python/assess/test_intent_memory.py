"""验证用户级意图记忆的存储、学习、路由和会话集成。"""

from __future__ import annotations

import sqlite3
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from app.agents.intent_recognition_agent import IntentRecognitionAgent
from app.application.conversation_service import ConversationService
from app.infrastructure.database.json.feature_store_models import UserBaseProfile
from app.infrastructure.database.sqlite.user_intent_memory_repository import (
    SQLiteUserIntentMemoryRepository,
)
from app.infrastructure.database.sqlite.user_profile_repository import (
    SQLiteUserProfileRepository,
)
from app.infrastructure.database.sqlite.conversation_store import (
    SQLiteConversationStore,
)
from app.domain.services.user_intent_memory import UserIntentMemoryService
from app.models.intent import (
    ArbitrationAction,
    IntentName,
    IntentRecognition,
    IntentState,
    RecognitionSource,
    RecommendationContext,
    RelationHint,
)
from app.models.conversation import ConversationReply
from app.models.intent_memory import UserIntentMemory, UserIntentMemoryProjection
from app.orchestration.conversation_nodes import _recognize_intent


_NOW = datetime(2026, 8, 10, 8, 0, tzinfo=timezone.utc)


def _user(user_id: str) -> UserBaseProfile:
    """构造只用于临时 SQLite 的最小已知用户。"""

    return UserBaseProfile(
        user_id=user_id,
        topics=[],
        blocked_topics=[],
        preferred_content_types=[],
        preferred_difficulty="",
        preferred_reading_length="",
        followed_author_ids=[],
        blocked_author_ids=[],
        created_at=_NOW,
    )


class _CountingIntentLlm:
    """记录结构化意图调用及其 Prompt 输入。"""

    def __init__(self, output: dict[str, Any]) -> None:
        self.output = output
        self.calls = 0
        self.messages: list[Any] = []

    async def ainvoke(self, messages: list[Any]) -> dict[str, Any]:
        self.calls += 1
        self.messages = list(messages)
        return dict(self.output)


class UserIntentMemoryRepositoryTests(unittest.TestCase):
    """长期意图记忆与推荐画像共库但使用独立表。"""

    def test_empty_memory_round_trips_and_users_are_isolated(self) -> None:
        with tempfile.TemporaryDirectory(prefix="intent-memory-") as root:
            database_path = Path(root) / "user_profiles.sqlite3"
            user_repository = SQLiteUserProfileRepository(database_path)
            user_repository.replace_user(_user("user-1"))
            user_repository.replace_user(_user("user-2"))
            repository = SQLiteUserIntentMemoryRepository(database_path)

            memory = UserIntentMemory.empty("user-1", now=_NOW)
            repository.save(memory)

            self.assertEqual(repository.get("user-1"), memory)
            self.assertIsNone(repository.get("user-2"))
            self.assertIsNone(repository.get("missing-user"))
            reopened = SQLiteUserIntentMemoryRepository(database_path)
            self.assertEqual(reopened.get("user-1"), memory)

    def test_unknown_user_cannot_receive_memory(self) -> None:
        with tempfile.TemporaryDirectory(prefix="intent-memory-") as root:
            database_path = Path(root) / "user_profiles.sqlite3"
            SQLiteUserProfileRepository(database_path)
            repository = SQLiteUserIntentMemoryRepository(database_path)

            with self.assertRaisesRegex(ValueError, "用户"):
                repository.save(
                    UserIntentMemory.empty("missing-user", now=_NOW)
                )

    def test_invalid_persisted_memory_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="intent-memory-") as root:
            database_path = Path(root) / "user_profiles.sqlite3"
            user_repository = SQLiteUserProfileRepository(database_path)
            user_repository.replace_user(_user("user-1"))
            repository = SQLiteUserIntentMemoryRepository(database_path)
            repository.save(UserIntentMemory.empty("user-1", now=_NOW))
            with sqlite3.connect(database_path) as connection:
                connection.execute("PRAGMA ignore_check_constraints = ON")
                connection.execute(
                    "UPDATE user_intent_memories SET memory_json = ? "
                    "WHERE user_id = ?",
                    ('{"recommendation_count": "invalid"}', "user-1"),
                )

            with self.assertRaisesRegex(ValueError, "意图记忆"):
                repository.get("user-1")


class UserIntentMemoryServiceTests(unittest.TestCase):
    """确定性服务只从成功行为积累受保护习惯。"""

    def setUp(self) -> None:
        self.service = UserIntentMemoryService(clock=lambda: _NOW)

    def test_repeated_sizes_become_default_and_explicit_preference_overrides(self) -> None:
        memory = UserIntentMemory.empty("user-1", now=_NOW)

        for _ in range(2):
            memory = self.service.record_success(
                memory,
                message="推荐 3 篇 Java 文章",
                action=ArbitrationAction.NEW,
                previous_intent_state=IntentState.RECOMMENDATION,
                current_intent_state=IntentState.RECOMMENDATION,
            )
        self.assertIsNone(
            self.service.project(memory).default_recommendation_size
        )

        memory = self.service.record_success(
            memory,
            message="推荐三篇 Java 文章",
            action=ArbitrationAction.NEW,
            previous_intent_state=IntentState.RECOMMENDATION,
            current_intent_state=IntentState.RECOMMENDATION,
        )
        self.assertEqual(
            self.service.project(memory).default_recommendation_size,
            3,
        )

        memory = self.service.record_success(
            memory,
            message="以后默认推荐 4 篇",
            action=ArbitrationAction.REFINE,
            previous_intent_state=IntentState.RECOMMENDATION,
            current_intent_state=IntentState.RECOMMENDATION,
        )
        projection = self.service.project(memory)
        self.assertEqual(projection.default_recommendation_size, 4)
        evidence = next(
            item for item in memory.recommendation_sizes if item.size == 4
        )
        self.assertTrue(evidence.explicit)

    def test_success_counts_create_dominant_intent_and_correction_evidence(self) -> None:
        original = UserIntentMemory.empty("user-1", now=_NOW)
        memory = original
        for _ in range(3):
            memory = self.service.record_success(
                memory,
                message="虚拟线程是什么？",
                action=ArbitrationAction.KNOWLEDGE_ANSWER,
                previous_intent_state=IntentState.KNOWLEDGE_QA,
                current_intent_state=IntentState.KNOWLEDGE_QA,
            )
        memory = self.service.record_success(
            memory,
            message="不是问答，我想继续推荐",
            action=ArbitrationAction.REPEAT,
            previous_intent_state=IntentState.KNOWLEDGE_QA,
            current_intent_state=IntentState.RECOMMENDATION,
        )

        projection = self.service.project(memory)

        self.assertEqual(original.knowledge_qa_count, 0)
        self.assertEqual(memory.knowledge_qa_count, 3)
        self.assertEqual(memory.recommendation_count, 1)
        self.assertEqual(projection.dominant_intent, "knowledge_qa")
        self.assertGreaterEqual(projection.dominant_intent_confidence, 0.6)
        self.assertEqual(len(projection.corrections), 1)
        self.assertEqual(
            projection.corrections[0].from_intent,
            IntentState.KNOWLEDGE_QA,
        )
        self.assertEqual(
            projection.corrections[0].to_intent,
            IntentState.RECOMMENDATION,
        )

    def test_non_business_actions_do_not_create_positive_evidence(self) -> None:
        memory = UserIntentMemory.empty("user-1", now=_NOW)

        for action in (
            ArbitrationAction.CLARIFY,
            ArbitrationAction.UNSUPPORTED,
            ArbitrationAction.RETURN_TO_PARENT,
        ):
            memory = self.service.record_success(
                memory,
                message="无法判断",
                action=action,
                previous_intent_state=IntentState.RECOMMENDATION,
                current_intent_state=IntentState.RECOMMENDATION,
            )

        self.assertEqual(memory.recommendation_count, 0)
        self.assertEqual(memory.knowledge_qa_count, 0)
        self.assertEqual(memory.recommendation_sizes, [])
        self.assertEqual(memory.corrections, [])

    def test_ambiguous_multiple_sizes_do_not_create_size_evidence(self) -> None:
        memory = self.service.record_success(
            UserIntentMemory.empty("user-1", now=_NOW),
            message="推荐 3 篇或 5 篇 Java 文章",
            action=ArbitrationAction.NEW,
            previous_intent_state=IntentState.RECOMMENDATION,
            current_intent_state=IntentState.RECOMMENDATION,
        )

        self.assertEqual(memory.recommendation_count, 1)
        self.assertEqual(memory.recommendation_sizes, [])


class IntentMemoryRoutingTests(unittest.IsolatedAsyncioTestCase):
    """长期记忆只补充默认值和歧义上下文，不覆盖当前明确表达。"""

    async def test_rule_recommendation_uses_memory_default_without_llm(self) -> None:
        llm = _CountingIntentLlm({})
        memory = UserIntentMemoryProjection(default_recommendation_size=3)

        result = await IntentRecognitionAgent(llm=llm).run(
            "推荐 Java 虚拟线程文章",
            intent_memory=memory,
        )

        self.assertEqual(result.intent, IntentName.RECOMMEND_ARTICLES)
        assert result.resolved_intent is not None
        self.assertEqual(result.resolved_intent.size, 3)
        self.assertEqual(llm.calls, 0)

    async def test_explicit_size_and_current_context_override_memory(self) -> None:
        llm = _CountingIntentLlm({})
        memory = UserIntentMemoryProjection(default_recommendation_size=3)

        explicit = await IntentRecognitionAgent(llm=llm).run(
            "推荐 5 篇 Java 文章",
            intent_memory=memory,
        )
        repeated = await IntentRecognitionAgent(llm=llm).run(
            "换一批",
            active_context=RecommendationContext(query="Java", size=4),
            intent_memory=memory,
        )

        assert explicit.resolved_intent is not None
        assert repeated.resolved_intent is not None
        self.assertEqual(explicit.resolved_intent.size, 5)
        self.assertEqual(repeated.resolved_intent.size, 4)
        self.assertEqual(llm.calls, 0)

    async def test_ambiguous_llm_input_contains_only_memory_projection(self) -> None:
        llm = _CountingIntentLlm(
            {
                "intent": "recommend_articles",
                "relation": "refine",
                "rewritten_query": "推荐更深入的 Java 文章",
                "updated_intent": {"resource_type": "article", "size": 3},
                "confidence": 0.85,
            }
        )
        memory = UserIntentMemoryProjection(
            default_recommendation_size=3,
            dominant_intent="recommend_articles",
            dominant_intent_confidence=0.8,
        )

        await IntentRecognitionAgent(llm=llm).run(
            "更深入一点",
            active_context=RecommendationContext(query="Java", size=5),
            intent_memory=memory,
        )

        system_prompt = str(llm.messages[0].content)
        input_prompt = str(llm.messages[1].content)
        self.assertIn("当前消息", system_prompt)
        self.assertIn("长期记忆", system_prompt)
        self.assertIn("user_intent_memory", input_prompt)
        self.assertIn("default_recommendation_size", input_prompt)
        self.assertNotIn("user_id", input_prompt)

    async def test_graph_node_forwards_memory_projection_to_agent(self) -> None:
        memory = UserIntentMemoryProjection(default_recommendation_size=3)

        class _RecordingAgent:
            def __init__(self) -> None:
                self.intent_memory: UserIntentMemoryProjection | None = None

            async def run(self, message: str, **kwargs: Any) -> IntentRecognition:
                del message
                self.intent_memory = kwargs.get("intent_memory")
                return IntentRecognition(
                    intent=IntentName.NO_ACTION,
                    source=RecognitionSource.RULE,
                    relation=RelationHint.UNCLEAR,
                    confidence=1.0,
                )

        agent = _RecordingAgent()
        state = {
            "message": "你好",
            "history": [],
            "conversation_summary": None,
            "previous_context": None,
            "current_intent_state": IntentState.RECOMMENDATION,
            "intent_memory": memory,
        }

        await _recognize_intent(SimpleNamespace(intent_agent=agent), state)

        self.assertEqual(agent.intent_memory, memory)


class _KnownUserStore:
    """会话集成探针使用的最小已知用户目录。"""

    async def get_user(self, user_id: str) -> object | None:
        return object() if user_id.startswith("user-") else None


class _MemoryWorkflow:
    """返回固定成功动作并记录会话服务传入的记忆投影。"""

    def __init__(self, action: ArbitrationAction) -> None:
        self.action = action
        self.intent_memories: list[UserIntentMemoryProjection | None] = []

    async def run(self, **kwargs: Any) -> object:
        self.intent_memories.append(kwargs.get("intent_memory"))
        is_recommendation = self.action in {
            ArbitrationAction.NEW,
            ArbitrationAction.REFINE,
            ArbitrationAction.REPEAT,
        }
        is_knowledge = self.action is ArbitrationAction.KNOWLEDGE_ANSWER
        context = (
            RecommendationContext(query="Java", size=3)
            if is_recommendation
            else None
        )
        return SimpleNamespace(
            reply=ConversationReply(
                session_id=kwargs["session_id"],
                message="已完成。",
                intent_source=RecognitionSource.RULE,
                action=self.action,
                active_context=context,
                needs_clarification=self.action is ArbitrationAction.CLARIFY,
            ),
            history_message="已完成。",
            pending_context=context,
            commit_context=is_recommendation,
            pending_intent_state=(
                IntentState.KNOWLEDGE_QA
                if is_knowledge
                else IntentState.RECOMMENDATION
                if is_recommendation
                else None
            ),
            commit_intent_state=is_recommendation or is_knowledge,
            knowledge_document_ids=(),
            knowledge_document_titles=(),
            error_stage=None,
            trace=[],
        )


def _conversation_service(
    *,
    root: Path,
    memory_repository: Any,
    action: ArbitrationAction,
) -> tuple[ConversationService, _MemoryWorkflow]:
    workflow = _MemoryWorkflow(action)
    service = ConversationService(
        user_store=_KnownUserStore(),
        recall_agent=object(),
        rerank_agent=object(),
        aggregator=object(),
        conversation_store=SQLiteConversationStore(
            root / "conversations.sqlite3"
        ),
        enable_llm=False,
        intent_memory_repository=memory_repository,
        intent_memory_service=UserIntentMemoryService(clock=lambda: _NOW),
    )
    service.workflow = workflow
    return service, workflow


class ConversationIntentMemoryTests(unittest.IsolatedAsyncioTestCase):
    """会话服务在业务提交成功后才读写长期意图记忆。"""

    async def test_service_loads_projection_and_persists_successful_sizes(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(prefix="intent-memory-service-") as root:
            root_path = Path(root)
            profile_path = root_path / "user_profiles.sqlite3"
            user_repository = SQLiteUserProfileRepository(profile_path)
            user_repository.replace_user(_user("user-1"))
            memory_repository = SQLiteUserIntentMemoryRepository(profile_path)
            memory_service = UserIntentMemoryService(clock=lambda: _NOW)
            seeded = UserIntentMemory.empty("user-1", now=_NOW)
            for _ in range(3):
                seeded = memory_service.record_success(
                    seeded,
                    message="推荐 3 篇 Java 文章",
                    action=ArbitrationAction.NEW,
                    previous_intent_state=IntentState.RECOMMENDATION,
                    current_intent_state=IntentState.RECOMMENDATION,
                )
            memory_repository.save(seeded)
            service, workflow = _conversation_service(
                root=root_path,
                memory_repository=memory_repository,
                action=ArbitrationAction.NEW,
            )

            await service.chat(
                "user-1",
                "推荐 3 篇 Java 文章",
                session_id="session-1",
            )

            self.assertEqual(
                workflow.intent_memories[0].default_recommendation_size,
                3,
            )
            stored = memory_repository.get("user-1")
            assert stored is not None
            self.assertEqual(stored.recommendation_count, 4)
            size_evidence = next(
                item for item in stored.recommendation_sizes if item.size == 3
            )
            self.assertEqual(size_evidence.evidence_count, 4)

    async def test_knowledge_switch_records_correction_but_clarify_does_not_learn(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(prefix="intent-memory-service-") as root:
            root_path = Path(root)
            profile_path = root_path / "user_profiles.sqlite3"
            user_repository = SQLiteUserProfileRepository(profile_path)
            user_repository.replace_user(_user("user-1"))
            memory_repository = SQLiteUserIntentMemoryRepository(profile_path)
            knowledge_service, _ = _conversation_service(
                root=root_path,
                memory_repository=memory_repository,
                action=ArbitrationAction.KNOWLEDGE_ANSWER,
            )

            await knowledge_service.chat(
                "user-1",
                "不是推荐，我想问虚拟线程是什么",
                session_id="knowledge-session",
            )

            stored = memory_repository.get("user-1")
            assert stored is not None
            self.assertEqual(stored.knowledge_qa_count, 1)
            self.assertEqual(len(stored.corrections), 1)

            clarify_service, _ = _conversation_service(
                root=root_path,
                memory_repository=memory_repository,
                action=ArbitrationAction.CLARIFY,
            )
            await clarify_service.chat(
                "user-1",
                "无法判断",
                session_id="clarify-session",
            )
            unchanged = memory_repository.get("user-1")
            self.assertEqual(unchanged, stored)

    async def test_memory_failures_do_not_replace_successful_reply(self) -> None:
        class _FailingRepository:
            def get(self, user_id: str) -> UserIntentMemory | None:
                del user_id
                raise RuntimeError("memory unavailable")

            def save(self, memory: UserIntentMemory) -> None:
                del memory
                raise RuntimeError("memory unavailable")

        with tempfile.TemporaryDirectory(prefix="intent-memory-service-") as root:
            service, workflow = _conversation_service(
                root=Path(root),
                memory_repository=_FailingRepository(),
                action=ArbitrationAction.NEW,
            )

            reply = await service.chat(
                "user-1",
                "推荐 3 篇 Java 文章",
                session_id="session-1",
            )

            self.assertEqual(reply.action, ArbitrationAction.NEW)
            self.assertEqual(workflow.intent_memories, [None])


if __name__ == "__main__":
    unittest.main()
