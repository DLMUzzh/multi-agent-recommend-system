"""推荐链和会话压缩关键边界的持久回归验证。"""

from __future__ import annotations

import asyncio
from copy import deepcopy
import importlib
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import app.models.schemas as schemas
from app.agents.intent_recognition_agent import IntentRecognitionAgent
from app.agents.user_profile_agent import UserProfileAgent
from app.config import Settings
from app.api.routers.chat import _to_chat_response, _to_session_history_response
from app.models.schemas import (
    ArbitrationAction,
    ConversationReply,
    ConversationCompressionInfo,
    ConversationSession,
    ConversationSummaryResult,
    ConversationTurn,
    IntentState,
    RecognitionSource,
    RecommendationContext,
)
from app.orchestration.conversation_graph import ConversationGraph
from app.application.conversation_service import ConversationService, ServiceUnavailableError
from app.infrastructure.database.json.feature_store import FeatureStore
from app.infrastructure.database.sqlite.conversation_store import (
    SQLiteConversationStore,
)


class ChatTimeoutSettingsTests(unittest.TestCase):
    """验证聊天总截止时间的环境配置边界。"""

    def test_chat_request_timeout_uses_default_and_environment_override(self) -> None:
        default_settings = Settings(_env_file=None)
        with patch.dict(
            "os.environ",
            {"ARTICLE_REC_CHAT_REQUEST_TIMEOUT_SECONDS": "12.5"},
        ):
            overridden = Settings(_env_file=None)

        self.assertEqual(
            getattr(default_settings, "chat_request_timeout_seconds", None),
            45.0,
        )
        self.assertEqual(
            getattr(overridden, "chat_request_timeout_seconds", None),
            12.5,
        )

    def test_timeout_example_and_summary_client_shutdown_are_wired(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        env_example = (project_root / "python" / ".env.example").read_text(
            encoding="utf-8"
        )
        bootstrap_source = (
            project_root / "python" / "app" / "bootstrap.py"
        ).read_text(
            encoding="utf-8"
        )

        self.assertIn("ARTICLE_REC_CHAT_REQUEST_TIMEOUT_SECONDS=45", env_example)
        self.assertIn(
            "conversation_service.summary_agent",
            bootstrap_source,
        )


class CompressionResponseContractTests(unittest.TestCase):
    """验证聊天和会话查询公开相同压缩状态。"""

    def test_chat_response_exposes_compression_result(self) -> None:
        reply = ConversationReply(
            session_id="public-compression",
            message="已完成本轮推荐。",
            intent_source=RecognitionSource.FALLBACK,
            action=ArbitrationAction.CLARIFY,
            compression=ConversationCompressionInfo(
                status="compressed",
                summary="用户长期关注 Java 工程实践。",
                summarized_turn_count=2,
                retained_turn_count=6,
                dropped_turn_count=0,
            ),
        )

        response = _to_chat_response(reply)

        compression = getattr(response, "compression", None)
        self.assertIsNotNone(compression)
        self.assertEqual(compression.status, "compressed")
        self.assertEqual(compression.summary, "用户长期关注 Java 工程实践。")

    def test_session_response_marks_more_than_six_turns_as_pending(self) -> None:
        session = ConversationSession(
            session_id="pending-compression",
            user_id="10001",
            history=_history(7),
            turn_count=9,
            summary="用户长期关注 Java 工程实践。",
            summarized_turn_count=2,
            dropped_turn_count=1,
        )

        response = _to_session_history_response(session)

        compression = getattr(response, "compression", None)
        self.assertIsNotNone(compression)
        self.assertEqual(compression.status, "pending")
        self.assertEqual(compression.retained_turn_count, 6)
        self.assertEqual(compression.dropped_turn_count, 1)

    def test_session_response_uses_summary_watermark_without_hiding_history(
        self,
    ) -> None:
        session = ConversationSession(
            session_id="stable-history",
            user_id="10001",
            history=_history(7),
            summary="【受保护滚动摘要 v2】\n模式：主会话",
            summary_watermark=1,
            summarized_turn_count=1,
        )

        response = _to_session_history_response(session)

        self.assertEqual(len(response.history), 14)
        self.assertEqual(response.compression.status, "compressed")
        self.assertEqual(response.compression.retained_turn_count, 6)


class IntentStateContractTests(unittest.TestCase):
    """验证会话只增加一个独立、受限的业务意图状态。"""

    def test_session_defaults_to_recommendation_and_accepts_knowledge_qa(self) -> None:
        intent_state = getattr(schemas, "IntentState", None)

        self.assertIsNotNone(intent_state)
        default_session = ConversationSession(
            session_id="default-intent-state",
            user_id="10001",
        )
        knowledge_session = ConversationSession(
            session_id="knowledge-intent-state",
            user_id="10001",
            intent_state="knowledge_qa",
        )

        self.assertEqual(default_session.intent_state.value, "recommendation")
        self.assertEqual(knowledge_session.intent_state.value, "knowledge_qa")


class CompressionWindowTests(unittest.TestCase):
    """验证调试窗口公开压缩进度和摘要结果。"""

    def test_chat_window_contains_compression_progress_and_result_rendering(
        self,
    ) -> None:
        html = (
            Path(__file__).resolve().parents[2] / "Test" / "chat_window.html"
        ).read_text(encoding="utf-8")

        self.assertIn('id="compression-status"', html)
        self.assertIn("正在压缩会话", html)
        self.assertIn("本次压缩暂未完成，将在后续对话重试", html)
        self.assertIn("function renderCompression", html)
        self.assertIn("compressionSummary.textContent", html)

    def test_chat_window_only_renders_unified_document_recommendations(self) -> None:
        html = (
            Path(__file__).resolve().parents[2] / "Test" / "chat_window.html"
        ).read_text(encoding="utf-8")

        self.assertIn("article.document_id", html)
        self.assertIn("article.excerpt", html)
        self.assertIn("article.reason", html)
        self.assertNotIn("article.description", html)
        self.assertNotIn("article.topics", html)
        self.assertNotIn("article.article_id", html)
        self.assertNotIn('id="similar-mode-button"', html)
        self.assertNotIn("/similar", html)


class _FakeStructuredLlm:
    def __init__(self, response: object) -> None:
        self.response = response
        self.messages: list[Any] = []

    async def ainvoke(self, messages: list[Any]) -> object:
        self.messages = list(messages)
        return self.response


class _FailingStructuredLlm:
    def __init__(self) -> None:
        self.calls = 0

    async def ainvoke(self, messages: list[Any]) -> object:
        _ = messages
        self.calls += 1
        raise RuntimeError("受控画像 LLM 失败")


class _FixedProfileFeatureStore:
    def __init__(self, features: dict[str, Any]) -> None:
        self.features = deepcopy(features)
        self.cache: dict[str, Any] | None = None
        self.history: list[dict[str, Any]] = []
        self.feature_calls = 0

    async def get_user_features(
        self,
        user_id: str,
        *,
        as_of: object,
    ) -> dict[str, Any]:
        _ = user_id, as_of
        self.feature_calls += 1
        return deepcopy(self.features)

    async def get_cached_profile(self, user_id: str) -> dict[str, Any] | None:
        _ = user_id
        return deepcopy(self.cache)

    async def set_cached_profile(self, user_id: str, profile: object) -> None:
        _ = user_id
        if hasattr(profile, "model_dump"):
            self.cache = profile.model_dump(mode="json")
        else:
            self.cache = deepcopy(profile)

    async def invalidate_cached_profile(self, user_id: str) -> None:
        _ = user_id
        self.cache = None

    async def archive_profile(self, user_id: str, profile: object) -> None:
        _ = user_id
        if hasattr(profile, "model_dump"):
            self.history.append(profile.model_dump(mode="json"))
        else:
            self.history.append(deepcopy(profile))


def _semantic_profile_output(*, rogue_interest: bool = False) -> dict[str, Any]:
    emerging_interests = []
    if rogue_interest:
        emerging_interests.append(
            {
                "topic": "量子计算",
                "strength": 0.8,
                "source_topics": ["不存在的主题"],
                "reason": "没有可信来源的测试兴趣",
            }
        )
    return {
        "semantic_profile": {
            "reader_profile": {
                "reader_type": "深度技术读者",
            },
            "interest_analysis": {
                "core_interests": [
                    {
                        "topic": "Java 后端工程",
                        "strength": 0.92,
                        "source_topics": ["Java", "Spring Boot"],
                        "reason": "长期主题共同指向 Java 后端工程",
                    }
                ],
                "emerging_interests": emerging_interests,
                "fading_interests": [],
                "negative_interests": [
                    {
                        "topic": "娱乐八卦",
                        "strength": 1.0,
                        "source_topics": ["娱乐八卦"],
                        "reason": "用户显式屏蔽该主题",
                    }
                ],
                "expansion_interests": [
                    {
                        "topic": "JVM 性能工程",
                        "based_on": ["Java"],
                        "exploration_confidence": 0.7,
                    }
                ],
            },
            "reading_preferences": {
                "recommended_difficulty": "intermediate",
                "content_depth": "deep",
                "preferred_reading_length": "medium",
                "preferred_content_types": ["tutorial", "case_study"],
                "technical_density": "high",
                "reason": "行为证据显示偏好中高级实战内容",
            },
            "exploration_strategy": {
                "mode": "balanced",
                "diversity_level": "medium",
                "reason": "保持核心主题并进行有限探索",
            },
            "recommendation_strategy": {
                "primary_topics": ["Java 后端工程"],
                "secondary_topics": [],
                "exploration_topics": ["JVM 性能工程"],
                "author_strategy": "保持已有作者偏好",
                "ranking_notes": ["优先高技术密度实战内容"],
            },
            "preference_conflicts": [],
        }
    }


class UserProfileSemanticEnrichmentTests(unittest.IsolatedAsyncioTestCase):
    """验证画像 Agent 只在可信事实之上执行受保护的语义增强。"""

    @staticmethod
    def _clock() -> datetime:
        return datetime(2026, 8, 8, tzinfo=timezone.utc)

    async def _features(self) -> dict[str, Any]:
        store = FeatureStore(clock=self._clock)
        return await store.get_user_features(
            "10001",
            as_of="2026-08-08T00:00:00+00:00",
        )

    async def test_cache_miss_applies_llm_once_without_changing_facts(
        self,
    ) -> None:
        features = await self._features()
        settings = Settings(_env_file=None)
        baseline = await UserProfileAgent(
            feature_store=_FixedProfileFeatureStore(features),
            enable_llm=False,
            clock=self._clock,
            settings=settings,
        ).run(user_id="10001")
        fake_llm = _FakeStructuredLlm(_semantic_profile_output())

        result = await UserProfileAgent(
            feature_store=_FixedProfileFeatureStore(features),
            llm=fake_llm,
            clock=self._clock,
            settings=settings,
        ).run(user_id="10001")

        self.assertTrue(result.success)
        self.assertIsNotNone(result.profile)
        self.assertIsNotNone(baseline.profile)
        assert result.profile is not None and baseline.profile is not None
        self.assertEqual(len(fake_llm.messages), 2)
        self.assertEqual(result.data["semantic_enrichment_status"], "applied")
        self.assertTrue(result.data["llm_applied"])
        self.assertEqual(result.profile.base_profile, baseline.profile.base_profile)
        self.assertEqual(
            result.profile.behavior_profile,
            baseline.profile.behavior_profile,
        )
        self.assertEqual(result.profile.evidence, baseline.profile.evidence)
        self.assertEqual(
            result.profile.profile_confidence,
            baseline.profile.profile_confidence,
        )
        prompt = str(fake_llm.messages[-1].content)
        self.assertNotIn("10001", prompt)
        self.assertNotIn('"age"', prompt)
        self.assertNotIn('"region"', prompt)
        self.assertNotIn("event_id", prompt)

    async def test_prompt_includes_actual_schema_and_fact_constraints(self) -> None:
        features = await self._features()
        fake_llm = _FakeStructuredLlm(_semantic_profile_output())

        result = await UserProfileAgent(
            feature_store=_FixedProfileFeatureStore(features),
            llm=fake_llm,
            clock=self._clock,
            settings=Settings(_env_file=None),
        ).run(user_id="10001")

        self.assertTrue(result.success)
        system_prompt = str(fake_llm.messages[0].content)
        self.assertIn("待处理数据", system_prompt)
        self.assertIn("程序恢复", system_prompt)
        envelope = json.loads(str(fake_llm.messages[1].content))
        self.assertEqual(set(envelope), {"contract", "input"})
        self.assertEqual(
            envelope["contract"]["name"],
            "user_profile_enrichment",
        )
        self.assertEqual(envelope["contract"]["version"], 2)
        schema = envelope["contract"]["output_schema"]
        payload = envelope["input"]
        input_text = json.dumps(payload, ensure_ascii=False)
        self.assertIn("semantic_profile", schema["properties"])
        schema_text = json.dumps(schema, ensure_ascii=False)
        self.assertNotIn("activity_level", schema_text)
        self.assertNotIn("analysis_confidence", schema_text)
        self.assertNotIn("evidence_sources", schema_text)
        self.assertNotIn("focus_ratio", schema_text)
        self.assertNotIn("exploration_ratio", schema_text)
        self.assertNotIn("excluded_topics", schema_text)
        self.assertIn("activity", payload)
        self.assertNotIn("10001", input_text)
        self.assertNotIn('"age"', input_text)
        self.assertNotIn('"region"', input_text)
        self.assertNotIn("event_id", input_text)

    async def test_cache_hit_skips_feature_read_and_second_llm_call(self) -> None:
        features = await self._features()
        store = _FixedProfileFeatureStore(features)
        fake_llm = _FakeStructuredLlm(_semantic_profile_output())
        agent = UserProfileAgent(
            feature_store=store,
            llm=fake_llm,
            clock=self._clock,
            settings=Settings(_env_file=None),
        )

        first = await agent.run(user_id="10001")
        second = await agent.run(user_id="10001")

        self.assertEqual(first.data["semantic_enrichment_status"], "applied")
        self.assertEqual(len(fake_llm.messages), 2)
        self.assertEqual(store.feature_calls, 1)
        self.assertTrue(second.data["cache_hit"])

    async def test_untrusted_interest_is_removed_from_applied_profile(self) -> None:
        fake_llm = _FakeStructuredLlm(
            _semantic_profile_output(rogue_interest=True)
        )
        result = await UserProfileAgent(
            feature_store=_FixedProfileFeatureStore(await self._features()),
            llm=fake_llm,
            clock=self._clock,
            settings=Settings(_env_file=None),
        ).run(user_id="10001")

        self.assertEqual(result.data["semantic_enrichment_status"], "applied")
        self.assertIsNotNone(result.profile)
        assert result.profile is not None
        emerging = result.profile.semantic_profile.interest_analysis.emerging_interests
        self.assertNotIn("量子计算", [item.topic for item in emerging])

    async def test_provider_cannot_override_activity_or_confidence(
        self,
    ) -> None:
        output = _semantic_profile_output()
        output["semantic_profile"]["reader_profile"].update(
            {
                "activity_level": "new_user",
                "analysis_confidence": 1.0,
            }
        )
        result = await UserProfileAgent(
            feature_store=_FixedProfileFeatureStore(await self._features()),
            llm=_FakeStructuredLlm(output),
            clock=self._clock,
            settings=Settings(_env_file=None),
        ).run(user_id="10001")

        self.assertTrue(result.success)
        self.assertEqual(result.data["semantic_enrichment_status"], "failed")
        self.assertFalse(result.data["llm_applied"])
        self.assertIsNotNone(result.profile)

    async def test_failed_llm_returns_usable_minimal_profile(self) -> None:
        failing_llm = _FailingStructuredLlm()
        result = await UserProfileAgent(
            feature_store=_FixedProfileFeatureStore(await self._features()),
            llm=failing_llm,
            clock=self._clock,
            settings=Settings(_env_file=None),
        ).run(user_id="10001")

        self.assertTrue(result.success)
        self.assertEqual(failing_llm.calls, 1)
        self.assertEqual(result.data["semantic_enrichment_status"], "failed")
        self.assertIsNotNone(result.profile)

    async def test_failed_enrichment_is_not_reused_from_cache(self) -> None:
        store = _FixedProfileFeatureStore(await self._features())
        failing_llm = _FailingStructuredLlm()
        agent = UserProfileAgent(
            feature_store=store,
            llm=failing_llm,
            clock=self._clock,
            settings=Settings(_env_file=None),
        )

        first = await agent.run(user_id="10001")
        second = await agent.run(user_id="10001")

        self.assertEqual(first.data["semantic_enrichment_status"], "failed")
        self.assertEqual(second.data["semantic_enrichment_status"], "failed")
        self.assertEqual(failing_llm.calls, 2)
        self.assertEqual(store.feature_calls, 2)
        self.assertIsNone(store.cache)

    async def test_disabled_llm_is_reported_without_failing_profile(self) -> None:
        result = await UserProfileAgent(
            feature_store=_FixedProfileFeatureStore(await self._features()),
            enable_llm=False,
            clock=self._clock,
            settings=Settings(_env_file=None),
        ).run(user_id="10001")

        self.assertTrue(result.success)
        self.assertEqual(result.data["semantic_enrichment_status"], "disabled")
        self.assertIsNotNone(result.profile)

    async def test_available_llm_refreshes_disabled_cache(self) -> None:
        store = _FixedProfileFeatureStore(await self._features())
        disabled = await UserProfileAgent(
            feature_store=store,
            enable_llm=False,
            clock=self._clock,
            settings=Settings(_env_file=None),
        ).run(user_id="10001")
        fake_llm = _FakeStructuredLlm(_semantic_profile_output())

        refreshed = await UserProfileAgent(
            feature_store=store,
            llm=fake_llm,
            clock=self._clock,
            settings=Settings(_env_file=None),
        ).run(user_id="10001")

        self.assertEqual(disabled.data["semantic_enrichment_status"], "disabled")
        self.assertEqual(refreshed.data["semantic_enrichment_status"], "applied")
        self.assertFalse(refreshed.data["cache_hit"])
        self.assertEqual(len(fake_llm.messages), 2)
        self.assertEqual(store.feature_calls, 2)

    async def test_no_semantic_evidence_skips_llm(self) -> None:
        features = await self._features()
        explicit = deepcopy(features["explicit_preferences"])
        for key in (
            "topics",
            "blocked_topics",
            "preferred_content_types",
            "followed_author_ids",
            "blocked_author_ids",
        ):
            explicit[key] = []
        explicit["preferred_difficulty"] = ""
        explicit["preferred_reading_length"] = ""
        features["user_id"] = "cold-user"
        features["explicit_preferences"] = explicit
        features["user_description"] = deepcopy(explicit)
        for key in (
            "short_term_topic_evidence",
            "long_term_topic_evidence",
            "negative_topic_evidence",
            "content_type_evidence",
            "difficulty_evidence",
            "reading_length_evidence",
            "author_evidence",
            "search_queries",
        ):
            features[key] = []
        features["data_quality"]["valid_event_count"] = 0
        features["data_quality"]["strong_signal_count"] = 0
        features["confidence_inputs"] = {
            "valid_event_count": 0,
            "metadata_completeness": 0.0,
            "strong_signal_count": 0,
            "recency_score": 0.0,
            "has_consumption_signal": False,
            "topic_metadata_ratio": 0.0,
        }
        features["latest_event_at"] = None
        features["offline_profile_at"] = None
        features["realtime_event_count"] = 0
        features["offline_features"]["previous_profile"] = {"available": False}
        features["activity"] = {
            "recency_score": 0.0,
            "frequency_score": 0.0,
            "engagement_score": 0.0,
            "level": "new_user",
            "active_days_30d": 0,
            "effective_read_count_30d": 0,
            "strong_interaction_count_30d": 0,
            "average_read_quality": 0.0,
            "distinct_topic_count_30d": 0,
        }
        fake_llm = _FakeStructuredLlm(_semantic_profile_output())

        result = await UserProfileAgent(
            feature_store=_FixedProfileFeatureStore(features),
            llm=fake_llm,
            clock=self._clock,
            settings=Settings(_env_file=None),
        ).run(user_id="cold-user")

        self.assertTrue(result.success)
        self.assertEqual(result.data["semantic_enrichment_status"], "not_needed")
        self.assertEqual(fake_llm.messages, [])
        self.assertEqual(result.profile.profile_status, "cold_start")

    async def test_v2_cache_is_invalidated_before_llm_enrichment(self) -> None:
        features = await self._features()
        store = _FixedProfileFeatureStore(features)
        baseline = await UserProfileAgent(
            feature_store=store,
            enable_llm=False,
            clock=self._clock,
            settings=Settings(_env_file=None),
        ).run(user_id="10001")
        self.assertIsNotNone(baseline.profile)
        assert baseline.profile is not None
        stale = baseline.profile.model_copy(
            update={
                "profile_version": "v2",
                "semantic_enrichment_status": "not_needed",
            },
            deep=True,
        )
        await store.set_cached_profile("10001", stale)
        fake_llm = _FakeStructuredLlm(_semantic_profile_output())

        result = await UserProfileAgent(
            feature_store=store,
            llm=fake_llm,
            clock=self._clock,
            settings=Settings(_env_file=None),
        ).run(user_id="10001")

        self.assertEqual(len(fake_llm.messages), 2)
        self.assertEqual(result.data["semantic_enrichment_status"], "applied")
        self.assertEqual(len(store.history), 1)
        self.assertNotEqual(result.profile.profile_version, "v2")



class ConversationSummaryAgentTests(unittest.IsolatedAsyncioTestCase):
    """验证滚动摘要 Agent 的结构化输出和安全失败。"""

    def _summary_module(self) -> Any:
        try:
            return importlib.import_module(
                "app.agents.conversation_summary_agent"
            )
        except ModuleNotFoundError:
            self.fail("缺少 app.agents.conversation_summary_agent")

    def test_compression_models_expose_public_state(self) -> None:
        compression_type = getattr(schemas, "ConversationCompressionInfo", None)
        summary_result_type = getattr(schemas, "ConversationSummaryResult", None)

        self.assertIsNotNone(compression_type)
        self.assertIsNotNone(summary_result_type)
        assert compression_type is not None
        assert summary_result_type is not None
        compression = compression_type(
            status="pending",
            summary="用户持续关注多Agent工程实践。",
            summarized_turn_count=3,
            retained_turn_count=7,
            dropped_turn_count=1,
        )
        result = summary_result_type(
            success=True,
            summary="用户持续关注多Agent工程实践。",
        )

        self.assertEqual(compression.status, "pending")
        self.assertEqual(compression.retained_turn_count, 7)
        self.assertEqual(result.agent_name, "conversation_summary")

    def test_summary_output_rejects_indexes_shared_across_groups(self) -> None:
        with self.assertRaises(ValueError):
            schemas.LlmConversationSummaryOutput(
                selected_turn_indexes=[0, 1],
                user_constraint_indexes=[1],
                unresolved_question_indexes=[2],
            )

    async def test_summary_agent_renders_context_and_selected_source_turns(
        self,
    ) -> None:
        module = self._summary_module()
        fake_llm = _FakeStructuredLlm({"selected_turn_indexes": [0]})
        agent = module.ConversationSummaryAgent(llm=fake_llm)
        context = RecommendationContext(
            query="请查找多Agent工程实践文章，不包含娱乐八卦",
        )

        result = await agent.run(
            existing_summary="用户偏好中文技术文章。",
            turns_to_summarize=[
                ConversationTurn(role="user", content="不要娱乐八卦"),
                ConversationTurn(role="assistant", content="已排除娱乐八卦。"),
            ],
            active_context=context,
        )

        self.assertTrue(result.success)
        assert result.summary is not None
        self.assertIn("多Agent", result.summary)
        self.assertIn("娱乐八卦", result.summary)
        self.assertIn("不要娱乐八卦", result.summary)
        self.assertNotIn("偏好中文", result.summary)
        self.assertFalse(result.data["existing_summary_accepted"])
        self.assertEqual(len(fake_llm.messages), 2)
        payload_text = fake_llm.messages[1].content
        envelope = json.loads(payload_text)
        self.assertEqual(set(envelope), {"contract", "input"})
        self.assertEqual(envelope["contract"]["name"], "conversation_summary")
        self.assertEqual(envelope["contract"]["version"], 2)
        self.assertIsInstance(envelope["contract"]["output_schema"], dict)
        payload = envelope["input"]
        self.assertNotIn("existing_summary", payload)
        self.assertEqual(len(payload["turns_to_summarize"]), 2)
        self.assertEqual(
            payload["active_context"]["query"],
            "请查找多Agent工程实践文章，不包含娱乐八卦",
        )

    async def test_summary_agent_rejects_out_of_range_source_index(self) -> None:
        module = self._summary_module()
        agent = module.ConversationSummaryAgent(
            llm=_FakeStructuredLlm({"selected_turn_indexes": [2]}),
        )

        result = await agent.run(
            existing_summary=None,
            turns_to_summarize=[
                ConversationTurn(role="user", content="推荐 Python 文章"),
                ConversationTurn(role="assistant", content="已找到相关文章。"),
            ],
            active_context=None,
        )

        self.assertFalse(result.success)
        self.assertIsNone(result.summary)

    async def test_summary_agent_accepts_seven_pending_turns_after_recovery(
        self,
    ) -> None:
        module = self._summary_module()
        agent = module.ConversationSummaryAgent(
            llm=_FakeStructuredLlm({"selected_turn_indexes": [0, 12]}),
        )

        result = await agent.run(
            existing_summary="已有摘要。",
            turns_to_summarize=_history(7),
            active_context=None,
        )

        self.assertTrue(result.success)
        assert result.summary is not None
        self.assertIn("用户消息 0", result.summary)
        self.assertIn("用户消息 6", result.summary)
        self.assertNotIn("已有摘要", result.summary)

    async def test_summary_agent_without_llm_returns_retryable_failure(self) -> None:
        module = self._summary_module()
        agent = module.ConversationSummaryAgent(enable_llm=False)

        result = await agent.run(
            existing_summary=None,
            turns_to_summarize=[
                ConversationTurn(role="user", content="推荐 Java 文章"),
                ConversationTurn(role="assistant", content="已找到相关文章。"),
            ],
            active_context=None,
        )

        self.assertFalse(result.success)
        self.assertIsNone(result.summary)

    async def test_summary_agent_separates_constraints_and_unresolved_questions(
        self,
    ) -> None:
        module = self._summary_module()
        agent = module.ConversationSummaryAgent(
            llm=_FakeStructuredLlm(
                {
                    "selected_turn_indexes": [1],
                    "user_constraint_indexes": [0],
                    "unresolved_question_indexes": [2],
                }
            )
        )

        result = await agent.run(
            existing_summary=None,
            turns_to_summarize=[
                ConversationTurn(role="user", content="请只依据文章原文回答"),
                ConversationTurn(role="assistant", content="已经进入文章问答"),
                ConversationTurn(role="user", content="生产环境如何回滚？"),
            ],
            active_context=None,
            summary_mode="article_qa",
            focus_document_title="Spring Boot 部署",
            recent_recommendation_titles=["Spring Boot 部署", "Spring Boot 测试"],
            recent_citation_titles=["Spring Boot 部署"],
            unresolved_questions=["文档是否说明灰度发布？"],
        )

        self.assertTrue(result.success)
        assert result.summary is not None
        self.assertIn("模式：文章聚焦问答", result.summary)
        self.assertIn("聚焦文章：Spring Boot 部署", result.summary)
        self.assertIn("最近推荐标题：Spring Boot 部署；Spring Boot 测试", result.summary)
        self.assertIn("最近问答引用：Spring Boot 部署", result.summary)
        self.assertIn("用户明确约束：[0:user]请只依据文章原文回答", result.summary)
        self.assertIn("生产环境如何回滚", result.summary)
        self.assertIn("文档是否说明灰度发布", result.summary)


class IntentSummaryContextTests(unittest.IsolatedAsyncioTestCase):
    """验证意图 LLM 只接收摘要和最近六轮原文。"""

    async def test_intent_payload_contains_summary_and_caps_raw_history(self) -> None:
        fake_llm = _FakeStructuredLlm(
            {
                "intent": "no_action",
                "relation": "unclear",
                "updated_intent": None,
                "confidence": 0.95,
            }
        )
        agent = IntentRecognitionAgent(llm=fake_llm)

        try:
            await agent.run(
                "推荐 Java，再解释第二篇",
                history=_history(8),
                conversation_summary="用户长期关注 Java 工程实践。",
            )
        except TypeError:
            self.fail("IntentRecognitionAgent 尚未接收 conversation_summary")

        envelope = json.loads(fake_llm.messages[1].content)
        self.assertEqual(set(envelope), {"contract", "input"})
        payload = envelope["input"]
        self.assertEqual(
            payload["conversation_summary"],
            "用户长期关注 Java 工程实践。",
        )
        self.assertEqual(len(payload["conversation_history"]), 12)
        self.assertEqual(
            payload["conversation_history"][0]["content"],
            "用户消息 2",
        )

    async def test_knowledge_question_is_recognized_with_current_state(self) -> None:
        fake_llm = _FakeStructuredLlm(
            {
                "intent": "knowledge_qa",
                "relation": "new",
                "rewritten_query": "Spring 事务为什么会失效？",
                "updated_intent": None,
                "confidence": 0.95,
            }
        )
        agent = IntentRecognitionAgent(llm=fake_llm)

        result = await agent.run(
            "为什么会失效？",
            intent_state="recommendation",
        )

        self.assertEqual(result.intent.value, "knowledge_qa")
        self.assertIsNone(result.resolved_intent)
        envelope = json.loads(fake_llm.messages[1].content)
        self.assertEqual(set(envelope), {"contract", "input"})
        payload = envelope["input"]
        self.assertEqual(payload["current_intent_state"], "recommendation")

    async def test_invalid_knowledge_output_with_recommendation_payload_falls_back(
        self,
    ) -> None:
        fake_llm = _FakeStructuredLlm(
            {
                "intent": "knowledge_qa",
                "relation": "new",
                "rewritten_query": "Spring 事务为什么会失效？",
                "updated_intent": {
                    "resource_type": "article",
                    "size": 5,
                },
                "confidence": 0.95,
            }
        )

        result = await IntentRecognitionAgent(llm=fake_llm).run(
            "为什么会失效？",
            intent_state="recommendation",
        )

        self.assertEqual(result.intent.value, "unknown")
        self.assertEqual(result.source.value, "fallback")


class ProfileSoftEnhancementStatusTests(unittest.TestCase):
    """验证语义增强失败不会被误判为画像 Agent 整体失败。"""

    def test_successful_profile_ignores_legacy_degraded_status_branch(self) -> None:
        result = SimpleNamespace(
            success=True,
            degraded_reason=None,
            data={"semantic_enrichment_status": "failed"},
        )

        self.assertEqual(ConversationGraph._result_status(result), "success")


def _history(turn_count: int) -> list[ConversationTurn]:
    history: list[ConversationTurn] = []
    for index in range(turn_count):
        history.extend(
            [
                ConversationTurn(role="user", content=f"用户消息 {index}"),
                ConversationTurn(role="assistant", content=f"助手回复 {index}"),
            ]
        )
    return history


class _FakeWorkflow:
    def __init__(self, *, delay: float = 0.0) -> None:
        self.delay = delay
        self.histories: list[list[ConversationTurn]] = []
        self.summaries: list[str | None] = []

    async def run(
        self,
        *,
        user_id: str,
        session_id: str,
        message: str,
        history: list[ConversationTurn],
        previous_context: RecommendationContext | None,
        conversation_summary: str | None = None,
        intent_state: IntentState = IntentState.RECOMMENDATION,
    ) -> Any:
        _ = user_id, previous_context, intent_state
        self.histories.append([turn.model_copy(deep=True) for turn in history])
        self.summaries.append(conversation_summary)
        if self.delay:
            await asyncio.sleep(self.delay)
        return SimpleNamespace(
            reply=ConversationReply(
                session_id=session_id,
                message="已完成本轮推荐。",
                intent_source=RecognitionSource.FALLBACK,
                action=ArbitrationAction.CLARIFY,
            ),
            history_message=f"助手处理：{message}",
            pending_context=None,
            commit_context=False,
            pending_intent_state=None,
            commit_intent_state=False,
            error_stage=None,
        )


class _FakeSummaryAgent:
    def __init__(self, result: ConversationSummaryResult) -> None:
        self.result = result
        self.calls: list[dict[str, Any]] = []

    async def run(self, **kwargs: Any) -> ConversationSummaryResult:
        self.calls.append(kwargs)
        return self.result.model_copy(deep=True)


class _CompressionUnusedAgent:
    """会话压缩探针中不会进入的业务 Agent。"""

    async def run(self, **_: Any) -> object:
        raise AssertionError("会话压缩探针不应执行推荐 Agent")


class _CompressionUnusedAggregator:
    """会话压缩探针中不会进入的结果聚合器。"""

    def aggregate(self, **_: Any) -> list[object]:
        raise AssertionError("会话压缩探针不应执行推荐聚合")


def _conversation_service(
    path: Path,
) -> tuple[ConversationService, SQLiteConversationStore, _FakeWorkflow]:
    store = SQLiteConversationStore(path)
    service = ConversationService(
        user_store=FeatureStore(),
        recall_agent=_CompressionUnusedAgent(),
        rerank_agent=_CompressionUnusedAgent(),
        aggregator=_CompressionUnusedAggregator(),
        conversation_store=store,
        enable_llm=False,
    )
    workflow = _FakeWorkflow()
    service.workflow = workflow
    return service, store, workflow


class ConversationCompressionServiceTests(unittest.IsolatedAsyncioTestCase):
    """验证六轮窗口、滚动摘要、失败重试和硬上限。"""

    async def test_seventh_turn_is_summarized_and_recent_six_are_persisted(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(prefix="article-rec-compression-") as root:
            service, store, _ = _conversation_service(
                Path(root) / "conversations.sqlite3"
            )
            await store.save(
                ConversationSession(
                    session_id="summary-success",
                    user_id="10001",
                    history=_history(6),
                    turn_count=6,
                )
            )
            summary_agent = _FakeSummaryAgent(
                ConversationSummaryResult(
                    success=True,
                    summary="用户早期主要关注 Java 教程。",
                )
            )
            service.summary_agent = summary_agent

            reply = await service.chat(
                "10001",
                "继续推荐",
                session_id="summary-success",
            )
            stored = await store.load("10001", "summary-success")

            self.assertIsNotNone(stored)
            assert stored is not None
            self.assertEqual(len(stored.history), 14)
            self.assertEqual(stored.history[0].content, "用户消息 0")
            self.assertEqual(stored.summary, "用户早期主要关注 Java 教程。")
            self.assertEqual(stored.summarized_turn_count, 1)
            self.assertEqual(stored.summary_watermark, 1)
            self.assertTrue(
                all(turn.message_id is not None for turn in stored.history)
            )
            self.assertEqual(len(summary_agent.calls), 1)
            compression = getattr(reply, "compression", None)
            self.assertIsNotNone(compression)
            self.assertEqual(compression.status, "compressed")
            self.assertEqual(compression.retained_turn_count, 6)

    async def test_failed_summary_keeps_at_most_twelve_turns_and_tracks_drop(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(prefix="article-rec-compression-") as root:
            service, store, _ = _conversation_service(
                Path(root) / "conversations.sqlite3"
            )
            await store.save(
                ConversationSession(
                    session_id="summary-failure",
                    user_id="10001",
                    history=_history(12),
                    turn_count=12,
                )
            )
            service.summary_agent = _FakeSummaryAgent(
                ConversationSummaryResult(
                    success=False,
                    error="TimeoutError",
                )
            )

            reply = await service.chat(
                "10001",
                "继续推荐",
                session_id="summary-failure",
            )
            stored = await store.load("10001", "summary-failure")

            self.assertIsNotNone(stored)
            assert stored is not None
            self.assertEqual(len(stored.history), 26)
            self.assertEqual(stored.history[0].content, "用户消息 0")
            self.assertEqual(stored.dropped_turn_count, 0)
            self.assertEqual(stored.summary_watermark, -1)
            compression = getattr(reply, "compression", None)
            self.assertIsNotNone(compression)
            self.assertEqual(compression.status, "pending")
            self.assertEqual(compression.retained_turn_count, 6)

    async def test_workflow_receives_only_recent_six_raw_turns(self) -> None:
        with tempfile.TemporaryDirectory(prefix="article-rec-compression-") as root:
            service, store, workflow = _conversation_service(
                Path(root) / "conversations.sqlite3"
            )
            await store.save(
                ConversationSession(
                    session_id="bounded-context",
                    user_id="10001",
                    history=_history(12),
                    turn_count=12,
                )
            )
            service.summary_agent = _FakeSummaryAgent(
                ConversationSummaryResult(success=False, error="TimeoutError")
            )

            await service.chat(
                "10001",
                "继续推荐",
                session_id="bounded-context",
            )

            self.assertEqual(len(workflow.histories), 1)
            self.assertEqual(len(workflow.histories[0]), 12)
            self.assertEqual(workflow.histories[0][0].content, "用户消息 6")

    async def test_workflow_receives_persisted_summary(self) -> None:
        with tempfile.TemporaryDirectory(prefix="article-rec-compression-") as root:
            service, store, workflow = _conversation_service(
                Path(root) / "conversations.sqlite3"
            )
            await store.save(
                ConversationSession(
                    session_id="summary-context",
                    user_id="10001",
                    history=_history(6),
                    turn_count=8,
                    summary="用户长期关注 Java 工程实践。",
                    summarized_turn_count=2,
                )
            )
            service.summary_agent = _FakeSummaryAgent(
                ConversationSummaryResult(
                    success=True,
                    summary="用户长期关注 Java 工程实践。",
                )
            )

            await service.chat(
                "10001",
                "继续推荐",
                session_id="summary-context",
            )

            self.assertEqual(
                workflow.summaries,
                ["用户长期关注 Java 工程实践。"],
            )

    async def test_chat_deadline_stops_slow_core_workflow(self) -> None:
        with tempfile.TemporaryDirectory(prefix="article-rec-compression-") as root:
            service, _, _ = _conversation_service(
                Path(root) / "conversations.sqlite3"
            )
            service.workflow = _FakeWorkflow(delay=0.05)
            service.chat_request_timeout_seconds = 0.01

            with self.assertRaises(ServiceUnavailableError):
                await service.chat(
                    "10001",
                    "继续推荐",
                    session_id="slow-workflow",
                )



if __name__ == "__main__":
    unittest.main()
