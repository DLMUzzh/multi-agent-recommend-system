"""验证死代码、日志保留、Controller 和依赖工程边界。"""

from __future__ import annotations

import asyncio
import importlib
import importlib.util
import inspect
import json
import os
import re
import sqlite3
import subprocess
import tempfile
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.agents import user_profile_agent
from app.api import errors as error_responses
from app.api.routers import chat as chat_controller
from app.infrastructure.database.json.feature_store import FeatureStore
from app.infrastructure.database.json import feature_store_models
from app.infrastructure.observability.conversation_trace import (
    ConversationTraceWriter,
    record_trace_event,
)
from assess import pipeline_evaluation


class LargeModuleSplitBoundaryTests(unittest.TestCase):
    """批准的大文件拆分必须保留旧入口并形成真实职责模块。"""

    @staticmethod
    def _import_split_modules(names: tuple[str, ...]) -> dict[str, object]:
        modules: dict[str, object] = {}
        for name in names:
            if importlib.util.find_spec(name) is None:
                raise AssertionError(f"缺少批准的职责模块：{name}")
            modules[name] = importlib.import_module(name)
        return modules

    def test_model_contracts_are_split_behind_compatibility_module(self) -> None:
        import app.models.schemas as schemas

        modules = self._import_split_modules(
            (
                "app.models.common",
                "app.models.article",
                "app.models.intent",
                "app.models.profile",
                "app.models.conversation",
                "app.models.api",
            )
        )
        representative_exports = {
            "app.models.common": ("AgentResult",),
            "app.models.article": (
                "DocumentCandidate",
                "DocumentRecallResult",
                "DocumentRerankResult",
                "RankedDocument",
            ),
            "app.models.intent": (
                "IntentRecognition",
                "IntentState",
                "RecommendationContext",
                "ArbitrationDecision",
            ),
            "app.models.profile": ("UserProfile", "UserProfileResult"),
            "app.models.conversation": (
                "ConversationTurn",
                "ConversationSession",
                "ConversationReply",
            ),
            "app.models.api": (
                "ChatRequest",
                "ChatResponse",
                "ChatStreamErrorEvent",
                "ChatStreamProcessEvent",
                "ChatStreamResultEvent",
                "ErrorResponse",
                "SimilarDocumentResponse",
            ),
        }
        for module_name, exports in representative_exports.items():
            for name in exports:
                self.assertIs(
                    getattr(schemas, name),
                    getattr(modules[module_name], name),
                )

        self.assertEqual(
            schemas.__all__,
            [
                "ActivityLevel",
                "ActivityProfile",
                "AgentResult",
                "ArbitrationAction",
                "ArbitrationDecision",
                "AuthorAffinity",
                "BaseProfileSnapshot",
                "BehaviorProfile",
                "ChatRequest",
                "ChatResponse",
                "ChatStreamErrorEvent",
                "ChatStreamProcessEvent",
                "ChatStreamResultEvent",
                "ConversationReply",
                "ConversationCompressionInfo",
                "ConversationSession",
                "ConversationSummaryResult",
                "ConversationTurn",
                "DocumentCandidate",
                "DocumentRecallResult",
                "DocumentRecommendation",
                "DocumentRerankResult",
                "DifficultyRecommendation",
                "DegradedComponent",
                "ErrorDetail",
                "ErrorResponse",
                "EvidenceSource",
                "ExpansionInterest",
                "ExplorationStrategy",
                "IntentName",
                "IntentRecognition",
                "IntentState",
                "InterestAnalysis",
                "KnowledgePlanCoverage",
                "KnowledgePlanDecision",
                "KnowledgePlanFacet",
                "KnowledgePlanReasonCode",
                "KnowledgePlanStep",
                "KnowledgePlanStepResult",
                "KnowledgePlanStepStatus",
                "KnowledgePlanSupportLevel",
                "KnowledgePlanTraceStep",
                "KnowledgeReasoningPlan",
                "KnowledgeReasoningStrategy",
                "LlmConversationSummaryOutput",
                "PROFILE_VERSION",
                "PreferenceConflict",
                "ProfileEvidence",
                "PublicArticleRecommendation",
                "PublicRecommendationContext",
                "RankedDocument",
                "ReaderProfileAnalysis",
                "ReadingPreferencesAnalysis",
                "RecognitionSource",
                "RecommendationContext",
                "RecommendationStrategy",
                "RelationHint",
                "SemanticInterest",
                "SemanticEnrichmentStatus",
                "SemanticProfile",
                "SessionHistoryResponse",
                "SimilarDocumentResponse",
                "TopicInterest",
                "UserProfile",
                "UserProfileResult",
                "ValuePreference",
                "HealthResponse",
            ],
        )
        self.assertLessEqual(len(inspect.getsourcelines(schemas)[0]), 180)

    def test_conversation_graph_delegates_to_split_modules(self) -> None:
        graph_module = importlib.import_module(
            "app.orchestration.conversation_graph"
        )
        modules = self._import_split_modules(
            (
                "app.orchestration.conversation_state",
                "app.orchestration.conversation_nodes",
                "app.orchestration.conversation_responses",
            )
        )

        self.assertIs(
            graph_module.ConversationGraphResult,
            modules["app.orchestration.conversation_state"].ConversationGraphResult,
        )
        self.assertTrue(hasattr(graph_module.ConversationGraph, "run"))
        self.assertFalse(
            hasattr(graph_module.ConversationGraph, "_run_article_rerank_agent")
        )
        self.assertTrue(hasattr(graph_module.ConversationGraph, "_result_status"))
        self.assertLessEqual(len(inspect.getsourcelines(graph_module)[0]), 450)

    def test_feature_store_delegates_to_json_service_modules(self) -> None:
        feature_store_module = importlib.import_module(
            "app.infrastructure.database.json.feature_store"
        )
        modules = self._import_split_modules(
            (
                "app.infrastructure.database.json.feature_store_models",
                "app.infrastructure.database.json.feature_store_repository",
                "app.infrastructure.database.json.feature_store_tags",
                "app.infrastructure.database.json.feature_store_features",
            )
        )

        self.assertIs(
            feature_store_module.UserNotFoundError,
            modules[
                "app.infrastructure.database.json.feature_store_models"
            ].UserNotFoundError,
        )
        for method_name in (
            "reload_mock_data",
            "record_behavior",
            "refresh_daily_tags",
            "get_user_features",
            "compact_user_features",
            "normalize_score",
        ):
            self.assertTrue(
                hasattr(feature_store_module.FeatureStore, method_name),
                method_name,
            )
        self.assertEqual(feature_store_module.FeatureStore.normalize_score(5.0), 0.5)
        self.assertLessEqual(
            len(inspect.getsourcelines(feature_store_module)[0]),
            750,
        )


class UserProfileDeadCodeTests(unittest.TestCase):
    """画像模块只保留当前受保护的 LLM 增强链。"""

    def test_legacy_llm_profile_surface_is_removed(self) -> None:
        removed_methods = {
            "_analyze_with_llm",
            "_apply_llm_output",
            "_topic_evidence_sources",
            "_negative_allowed_topics",
            "_validated_semantic_interests",
            "_validated_expansion_interests",
            "_normalized_exploration_strategy",
            "_validated_topic_list",
            "_clean_text",
            "_parse_llm_output",
            "_allowed_topics",
            "_build_rule_semantic_profile",
            "_rule_evidence_sources",
            "_rule_difficulty",
            "_rule_reading_length",
            "_rule_summary",
            "_clean_string_list",
        }

        self.assertFalse(hasattr(user_profile_agent, "LlmProfileOutput"))
        self.assertNotIn("LlmProfileOutput", user_profile_agent.__all__)
        self.assertTrue(
            hasattr(user_profile_agent, "LlmProfileEnrichmentOutput")
        )
        self.assertNotIn(
            "LlmProfileEnrichmentOutput",
            user_profile_agent.__all__,
        )
        for method_name in removed_methods:
            self.assertFalse(
                hasattr(user_profile_agent.UserProfileAgent, method_name),
                method_name,
            )

        source = inspect.getsource(user_profile_agent)
        self.assertNotIn("class LlmProfileOutput", source)
        self.assertNotIn("_llm_compatibility_enabled", source)


class FeatureStoreDeadCodeTests(unittest.IsolatedAsyncioTestCase):
    """Feature Store 保留当前公开特征契约，不再携带不可达旧计算链。"""

    async def test_current_public_feature_contract_is_stable(self) -> None:
        store = FeatureStore(
            clock=lambda: datetime(2026, 8, 3, tzinfo=timezone.utc)
        )

        features = await store.get_user_features(
            "10001",
            as_of="2026-08-03T00:00:00+00:00",
        )

        self.assertEqual(features["user_id"], "10001")
        self.assertEqual(
            features["explicit_preferences"],
            features["user_description"],
        )
        self.assertEqual(features["rfe_activity"], features["activity"])
        self.assertEqual(features["rfe_activity"]["level"], "new_user")
        self.assertEqual(features["search_queries"], [])
        self.assertEqual(features["long_term_topic_evidence"], [])
        self.assertNotIn("profile_confidence", features)
        self.assertEqual(
            features["confidence_inputs"],
            {
                "valid_event_count": 0,
                "metadata_completeness": 0.0,
                "strong_signal_count": 0,
                "recency_score": 0.0,
                "has_consumption_signal": False,
                "topic_metadata_ratio": 0.0,
            },
        )

    def test_unreachable_legacy_feature_chain_is_removed(self) -> None:
        removed_methods = {
            "get_articles",
            "get_offline_feature_state",
            "set_offline_feature_state",
            "get_offline_profile",
            "set_offline_profile",
            "get_recent_behaviors",
            "merge_offline_tags",
            "_calculate_features",
            "_merge_offline_and_realtime",
            "_vectors_from_features",
            "_vector_values_from_rows",
            "_offline_vector_rows",
            "_merge_score_rows",
            "_merge_author_rows",
            "_build_daily_buckets",
            "_merge_daily_buckets",
            "_calculate_rfe_from_buckets",
            "_apply_negative_feedback",
            "_add_score",
            "_score_rows",
            "_author_rows",
            "_calculate_rfe",
            "_metadata_completeness",
            "_relationship_warnings",
            "_offline_profile_time",
        }

        for method_name in removed_methods:
            self.assertFalse(hasattr(FeatureStore, method_name), method_name)
        self.assertFalse(hasattr(FeatureStore, "_calculate_confidence"))


class RetiredArticleChainRemovalTests(unittest.TestCase):
    """新文档推荐链成为唯一实现，退役模块不再可导入。"""

    def test_document_recommendation_modules_replace_article_modules(self) -> None:
        for module_name in (
            "app.agents.document_recall_agent",
            "app.agents.document_rerank_agent",
            "app.api.routers.similar_documents",
            "app.application.similar_document_recommendation",
            "app.domain.services.document_result_aggregator",
        ):
            self.assertIsNotNone(importlib.util.find_spec(module_name), module_name)

        for module_name in (
            "app.agents.article_recall_agent",
            "app.agents.article_rerank_agent",
            "app.agents.recommendation_reason_agent",
            "app.api.routers.similar_articles",
            "app.application.similar_article_recommendation",
            "app.application.document_ingestion",
            "app.domain.services.article_result_aggregator",
            "app.domain.services.document_chunker",
            "app.infrastructure.database.json.article_catalog",
            "app.infrastructure.database.json.conversation_store",
            "app.infrastructure.retrieval.article_search",
            "assess.filtered_control",
            "assess.similar_article_evaluation",
        ):
            self.assertIsNone(importlib.util.find_spec(module_name), module_name)

    def test_legacy_settings_and_json_migration_are_removed(self) -> None:
        settings_module = importlib.import_module("app.config.settings")
        removed_settings = {
            "app_name",
            "debug",
            "llm_reason_temperature",
            "llm_reason_max_tokens",
            "recall_candidate_multiplier",
            "recall_bm25_title_weight",
            "recall_bm25_topic_weight",
            "recall_bm25_description_weight",
            "recall_primary_query_weight",
            "recall_expanded_query_weight",
            "recall_empty_query_score",
            "recall_topic_match_ratio",
            "recall_mode",
            "embedding_min_similarity",
            "recall_vector_query_timeout_seconds",
            "recall_primary_database_url",
            "recall_secondary_database_url",
        }
        for field_name in removed_settings:
            self.assertNotIn(
                field_name,
                settings_module.Settings.model_fields,
                field_name,
            )

        conversation_module = importlib.import_module(
            "app.infrastructure.database.sqlite.conversation_store"
        )
        self.assertFalse(
            hasattr(conversation_module.SQLiteConversationStore, "bootstrap_from_json")
        )


class FeatureStoreDocumentBehaviorTests(unittest.IsolatedAsyncioTestCase):
    """行为事实使用文档身份，缺失属性由独立文档画像目录解释。"""

    def test_behavior_event_uses_document_id_and_rejects_article_id(self) -> None:
        event_type = feature_store_models.BehaviorEvent
        common = {
            "event_id": "evt-document-1",
            "user_id": "user-1",
            "event_type": "read",
            "occurred_at": "2026-08-07T12:00:00+08:00",
            "metadata": {"dwell_seconds": 120, "completion_rate": 0.8},
        }

        event = event_type.model_validate({**common, "document_id": "document-1"})

        self.assertEqual(event.document_id, "document-1")
        with self.assertRaises(ValueError):
            event_type.model_validate({**common, "article_id": "legacy-article-1"})
        with self.assertRaises(ValueError):
            event_type.model_validate(
                {
                    **common,
                    "document_id": "document-1",
                    "author_id": "legacy-author",
                }
            )

    def test_behavior_event_does_not_duplicate_document_author(self) -> None:
        event = feature_store_models.BehaviorEvent.model_validate(
            {
                "event_id": "evt-sqlite-1",
                "user_id": "user-1",
                "event_type": "read",
                "occurred_at": "2026-08-07T12:00:00+08:00",
                "document_id": "document-1",
                "metadata": {"dwell_seconds": 120, "completion_rate": 0.8},
            }
        )

        self.assertNotIn("author_id", event.model_fields)

    async def test_feature_store_reads_sqlite_and_ignores_legacy_files(
        self,
    ) -> None:
        from app.application.knowledge_qa import KnowledgeQaService
        from app.infrastructure.retrieval.knowledge_search import (
            InMemoryKnowledgeSearch,
        )
        from app.infrastructure.database.sqlite.knowledge_repository import (
            SQLiteKnowledgeRepository,
        )
        from app.infrastructure.database.sqlite.user_profile_repository import (
            SQLiteUserProfileRepository,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            data_dir = Path(temporary_directory)
            document_repository = SQLiteKnowledgeRepository(
                data_dir / "documents.sqlite3"
            )
            user_repository = SQLiteUserProfileRepository(
                data_dir / "user_profiles.sqlite3"
            )
            knowledge_service = KnowledgeQaService(
                repository=document_repository,
                search=InMemoryKnowledgeSearch(),
                clock=lambda: datetime(2026, 8, 6, tzinfo=timezone.utc),
            )
            await knowledge_service.ingest_document(
                document_id="document-1",
                title="Spring Boot 部署",
                content_markdown="# Spring Boot\n\n部署需要构建可执行 Jar。",
                topics=["Spring Boot"],
                content_type="tutorial",
                difficulty="intermediate",
                author_id="author-1",
            )
            user_repository.replace_user(
                feature_store_models.UserBaseProfile.model_validate(
                    {
                        "user_id": "user-1",
                        "created_at": "2026-08-01T00:00:00+08:00",
                    }
                )
            )
            user_repository.append_event(
                feature_store_models.BehaviorEvent.model_validate(
                    {
                        "event_id": "evt-document-1",
                        "user_id": "user-1",
                        "event_type": "read",
                        "occurred_at": "2026-08-07T12:00:00+08:00",
                        "document_id": "document-1",
                        "metadata": {
                            "dwell_seconds": 120,
                            "completion_rate": 0.8,
                        },
                    }
                )
            )
            for filename, payload in {
                "mock_users.json": [{"user_id": "legacy-user"}],
                "mock_document_profiles.json": [
                    {"document_id": "legacy-document"}
                ],
                "mock_behavior_events.json": [
                    {"event_id": "legacy-event"}
                ],
            }.items():
                (data_dir / filename).write_text(
                    json.dumps(payload, ensure_ascii=False),
                    encoding="utf-8",
                )

            store = FeatureStore(
                data_dir=data_dir,
                user_repository=user_repository,
                document_repository=document_repository,
            )
            documents = document_repository.get_document_facts(
                ("document-1", "legacy-article-1")
            )
            features = await store.get_user_features(
                "user-1",
                as_of="2026-08-07T13:00:00+08:00",
            )

        self.assertEqual(set(documents), {"document-1"})
        self.assertEqual(documents["document-1"].document_id, "document-1")
        self.assertEqual(
            [item["topic"] for item in features["short_term_topic_evidence"]],
            ["Spring Boot"],
        )

    async def test_repository_behavior_data_matches_ready_sqlite_documents(
        self,
    ) -> None:
        project_root = Path(__file__).resolve().parents[2]
        data_dir = project_root / "data"
        with sqlite3.connect(data_dir / "documents.sqlite3") as connection:
            connection.row_factory = sqlite3.Row
            document_rows = connection.execute(
                """
                SELECT document_id, topics, content_type, difficulty, author_id
                FROM documents
                WHERE status = 'ready'
                """
            ).fetchall()
            document_columns = {
                row[1] for row in connection.execute("PRAGMA table_info(documents)")
            }
        document_by_id = {row["document_id"]: row for row in document_rows}
        with sqlite3.connect(data_dir / "user_profiles.sqlite3") as connection:
            connection.row_factory = sqlite3.Row
            events = connection.execute(
                """
                SELECT event_id, user_id, event_type, occurred_at,
                       document_id, metadata_json
                FROM user_behavior_events
                ORDER BY occurred_at, event_id
                """
            ).fetchall()
            users = connection.execute("SELECT * FROM users ORDER BY user_id").fetchall()
            event_columns = {
                row[1]
                for row in connection.execute(
                    "PRAGMA table_info(user_behavior_events)"
                )
            }
            user_columns = {
                row[1] for row in connection.execute("PRAGMA table_info(users)")
            }

        self.assertTrue(
            {"topics", "content_type", "difficulty", "author_id"}
            <= document_columns
        )
        self.assertNotIn("author_id", event_columns)
        self.assertTrue({"age", "region"}.isdisjoint(user_columns))
        self.assertTrue(document_by_id)
        forbidden_metadata = {
            "topics",
            "content_type",
            "difficulty",
            "language",
            "author_id",
        }
        user_ids: set[str] = set()
        for event in events:
            self.assertFalse(str(event["document_id"]).startswith("300"))
            self.assertIn(event["document_id"], document_by_id)
            metadata = json.loads(event["metadata_json"])
            self.assertTrue(forbidden_metadata.isdisjoint(metadata))
            self.assertNotIn("selected_result_article_id", metadata)
            selected_id = metadata.get("selected_result_document_id")
            if selected_id is not None:
                self.assertIn(selected_id, document_by_id)
            user_ids.add(event["user_id"])

        self.assertEqual(user_ids, {"10001", "10002"})
        self.assertEqual({row["user_id"] for row in users}, user_ids)
        self.assertTrue(
            all(
                "long_read" not in json.loads(row["preferred_content_types"])
                for row in users
            )
        )
        for retired_path in (
            data_dir / "mock_articles.json",
            data_dir / "mock_offline_features.json",
            data_dir / "redis" / "user_behavior.json",
            data_dir / "similar_article_evaluation_cases.json",
            data_dir / "mock_document_profiles.json",
            data_dir / "mock_users.json",
            data_dir / "mock_behavior_events.json",
        ):
            self.assertFalse(retired_path.exists(), str(retired_path))

        store = FeatureStore(clock=lambda: datetime(2026, 8, 8, tzinfo=timezone.utc))
        first = await store.get_user_features(
            "10001",
            as_of="2026-08-08T00:00:00+00:00",
        )
        second = await store.get_user_features(
            "10002",
            as_of="2026-08-08T00:00:00+00:00",
        )
        first_topics = {
            item["topic"] for item in first["short_term_topic_evidence"]
        }
        second_topics = {
            item["topic"] for item in second["short_term_topic_evidence"]
        }
        self.assertTrue(first_topics)
        self.assertTrue(second_topics)
        self.assertNotEqual(first_topics, second_topics)
        first_profile = await user_profile_agent.UserProfileAgent(
            feature_store=store,
            enable_llm=False,
            clock=lambda: datetime(2026, 8, 8, tzinfo=timezone.utc),
        ).run(user_id="10001")
        second_profile = await user_profile_agent.UserProfileAgent(
            feature_store=store,
            enable_llm=False,
            clock=lambda: datetime(2026, 8, 8, tzinfo=timezone.utc),
        ).run(user_id="10002")
        assert first_profile.profile is not None
        assert second_profile.profile is not None
        self.assertGreater(first_profile.profile.profile_confidence, 0.5)
        self.assertGreater(second_profile.profile.profile_confidence, 0.5)


class SQLiteUserProfileRepositoryContractTests(unittest.TestCase):
    """用户和行为事实由独立 SQLite 仓储持久化。"""

    def test_user_and_behavior_round_trip(self) -> None:
        from app.infrastructure.database.sqlite.user_profile_repository import (
            SQLiteUserProfileRepository,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            repository = SQLiteUserProfileRepository(
                Path(temporary_directory) / "user_profiles.sqlite3"
            )
            user = feature_store_models.UserBaseProfile.model_validate(
                {
                    "user_id": "user-1",
                    "topics": ["Spring Boot"],
                    "blocked_topics": [],
                    "preferred_content_types": ["tutorial"],
                    "preferred_difficulty": "intermediate",
                    "preferred_reading_length": "medium",
                    "followed_author_ids": ["author-1"],
                    "blocked_author_ids": [],
                    "created_at": "2026-08-01T00:00:00+08:00",
                }
            )
            event = feature_store_models.BehaviorEvent.model_validate(
                {
                    "event_id": "evt-1",
                    "user_id": "user-1",
                    "event_type": "read",
                    "occurred_at": "2026-08-07T12:00:00+08:00",
                    "document_id": "document-1",
                    "metadata": {"dwell_seconds": 120, "completion_rate": 0.8},
                }
            )

            repository.replace_user(user)
            repository.append_event(event)

            self.assertEqual(repository.get_user("user-1"), user)
            self.assertEqual(
                repository.list_events(
                    "user-1",
                    since=datetime(2026, 8, 7, 0, 0, tzinfo=timezone.utc),
                    until=datetime(2026, 8, 8, 0, 0, tzinfo=timezone.utc),
                ),
                (event,),
            )
            with self.assertRaisesRegex(ValueError, "event_id"):
                repository.append_event(event)


class SharedControllerErrorTests(unittest.IsolatedAsyncioTestCase):
    """聊天 HTTP 入口复用共享公开校验与错误体实现。"""

    def test_controllers_use_shared_error_primitives(self) -> None:
        try:
            from app.api import errors as shared_errors
        except (ImportError, ModuleNotFoundError):
            self.fail("缺少共享 Controller 错误响应模块")

        self.assertIs(chat_controller.required_text, shared_errors.required_text)
        self.assertIs(
            chat_controller.validation_error_handler,
            shared_errors.validation_error_handler,
        )
        self.assertFalse(hasattr(chat_controller, "_PublicValidationError"))

    async def test_chat_route_keeps_shared_validation_contract(self) -> None:
        app = FastAPI()
        app.include_router(chat_controller.router)
        chat_controller.register_error_handlers(app)
        transport = ASGITransport(app=app, raise_app_exceptions=False)

        async with AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            chat_response = await client.post(
                "/api/v1/chat",
                json={"user_id": "user-1", "message": "   "},
            )

        expected = {
            "error": {
                "code": "VALIDATION_ERROR",
                "message": "请求参数无效",
            }
        }
        self.assertEqual(chat_response.status_code, 422)
        self.assertEqual(chat_response.json(), expected)


class KnowledgeImageBootstrapBoundaryTests(unittest.TestCase):
    """启动装配必须把固定本地图片根目录注入知识问答服务。"""

    def test_bootstrap_uses_fixed_knowledge_image_root(self) -> None:
        bootstrap = importlib.import_module("app.bootstrap")
        paths = importlib.import_module("app.config.paths")
        source = inspect.getsource(bootstrap)

        self.assertEqual(
            paths.KNOWLEDGE_IMAGE_ROOT,
            paths.DATA_ROOT / "knowledge_images",
        )
        self.assertIn("LocalKnowledgeImageStore", source)
        self.assertIn("KNOWLEDGE_IMAGE_ROOT", source)
        self.assertIn("image_store=image_store", source)


class KnowledgePlanAndExecuteBoundaryTests(unittest.TestCase):
    """复杂知识推理必须保持角色依赖、共享资源与公共配置边界。"""

    def test_bootstrap_shares_search_and_reranker_for_plan_executor(self) -> None:
        bootstrap = importlib.import_module("app.bootstrap")
        source = inspect.getsource(bootstrap)

        self.assertIn("KnowledgeReasoningPlannerAgent", source)
        self.assertIn("KnowledgePlanExecutor", source)
        self.assertIn("KnowledgePlanCoverageChecker", source)
        self.assertIn("chunk_rerank_agent = KnowledgeChunkRerankAgent", source)
        self.assertIn("reranker=chunk_rerank_agent", source)
        self.assertIn("chunk_rerank_agent=chunk_rerank_agent", source)
        self.assertIn("search=document_search", source)
        self.assertIn(
            "request_timeout_seconds=settings.chat_request_timeout_seconds",
            source,
        )

    def test_planner_and_coverage_keep_role_dependencies(self) -> None:
        planner = importlib.import_module(
            "app.agents.knowledge_reasoning_planner_agent"
        )
        coverage = importlib.import_module(
            "app.domain.services.knowledge_plan_coverage"
        )
        planner_source = inspect.getsource(planner)
        coverage_source = inspect.getsource(coverage)

        for forbidden in (
            "app.application",
            "knowledge_search",
            "knowledge_repository",
            "knowledge_answer_agent",
        ):
            self.assertNotIn(forbidden, planner_source)
        self.assertNotIn("app.application", coverage_source)
        self.assertNotIn("app.agents", coverage_source)
        self.assertIn("app.models.knowledge_qa", coverage_source)

    def test_settings_and_public_routes_remain_unchanged(self) -> None:
        settings_module = importlib.import_module("app.config.settings")
        main_module = importlib.import_module("app.main")

        self.assertEqual(
            set(settings_module.Settings.model_fields),
            {
                "llm_provider",
                "llm_api_key",
                "llm_base_url",
                "llm_model",
                "llm_small_model",
                "llm_large_model",
                "llm_temperature",
                "llm_max_tokens",
                "llm_request_timeout_seconds",
                "llm_max_retries",
                "llm_intent_temperature",
                "llm_intent_max_tokens",
                "llm_profile_temperature",
                "llm_profile_max_tokens",
                "llm_rerank_temperature",
                "llm_rerank_max_tokens",
                "recall_bm25_k1",
                "recall_bm25_b",
                "embedding_api_key",
                "embedding_base_url",
                "embedding_model",
                "embedding_dimensions",
                "embedding_batch_size",
                "embedding_request_timeout_seconds",
                "embedding_max_retries",
                "recall_rrf_k",
                "agent_timeout_user_profile",
                "chat_request_timeout_seconds",
            },
        )
        self.assertEqual(
            {route.path for route in main_module.app.routes},
            {
                "/api/v1/chat",
                "/api/v1/chat/stream",
                "/api/v1/documents/{document_id}/similar",
                "/api/v1/knowledge/ask",
                "/api/v1/knowledge/documents",
                "/api/v1/knowledge/images/{image_id}",
                "/api/v1/sessions/{session_id}",
                "/docs",
                "/docs/oauth2-redirect",
                "/health",
                "/openapi.json",
                "/redoc",
            },
        )


class ConversationTraceRetentionTests(unittest.IsolatedAsyncioTestCase):
    """Trace 写入后清理现行 JSON 和旧版 Markdown，失败不影响业务。"""

    async def test_write_removes_expired_and_excess_trace_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "log"
            date_directory = root / datetime.now(timezone.utc).strftime("%Y-%m-%d")
            date_directory.mkdir(parents=True)
            now = time.time()

            expired = date_directory / "expired.json"
            expired.write_text("{}", encoding="utf-8")
            os.utime(expired, (now - 8 * 86400, now - 8 * 86400))
            expired_legacy = date_directory / "expired.md"
            expired_legacy.write_text("# 旧版追踪日志", encoding="utf-8")
            os.utime(
                expired_legacy,
                (now - 8 * 86400, now - 8 * 86400),
            )

            recent_files: list[Path] = []
            for index in range(4):
                suffix = ".md" if index % 2 else ".json"
                path = date_directory / f"recent-{index}{suffix}"
                path.write_text(
                    "# 旧版追踪日志" if suffix == ".md" else "{}",
                    encoding="utf-8",
                )
                age_seconds = 10 - index
                os.utime(path, (now - age_seconds, now - age_seconds))
                recent_files.append(path)

            try:
                writer = ConversationTraceWriter(
                    root,
                    retention_days=7,
                    max_files=3,
                )
            except TypeError:
                self.fail("ConversationTraceWriter 缺少日志保留参数")
            await self._write_one_trace(writer)

            remaining = sorted(
                path
                for path in root.glob("*/*")
                if path.suffix in {".json", ".md"}
            )
            self.assertEqual(len(remaining), 3)
            self.assertNotIn(expired, remaining)
            self.assertNotIn(expired_legacy, remaining)
            self.assertNotIn(recent_files[0], remaining)
            self.assertNotIn(recent_files[1], remaining)
            self.assertIn(recent_files[2], remaining)
            self.assertIn(recent_files[3], remaining)

    async def test_cleanup_failure_does_not_change_business_result(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "log"
            writer = ConversationTraceWriter(root)
            cleanup_called = False

            def fail_cleanup() -> None:
                nonlocal cleanup_called
                cleanup_called = True
                raise OSError("清理失败")

            writer._cleanup_sync = fail_cleanup  # type: ignore[attr-defined]

            result = await self._write_one_trace(writer)

            self.assertEqual(result, "业务成功")
            self.assertTrue(cleanup_called)
            self.assertEqual(len(list(root.glob("*/*.json"))), 1)

    @staticmethod
    async def _write_one_trace(writer: ConversationTraceWriter) -> str:
        async with writer.trace_request(
            user_id="user-1",
            message="推荐异步文章",
            supplied_session_id=None,
        ):
            record_trace_event(
                "agent.completed",
                "document_recall_agent",
                output_data={"document_ids": ["document-1"]},
                status="success",
            )
            await asyncio.sleep(0)
            return "业务成功"


class DependencyAndVerificationEntryTests(unittest.TestCase):
    """依赖使用兼容区间，统一验证入口可供本地和 Gitee 调用。"""

    def test_runtime_dependencies_use_bounded_direct_ranges(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        requirement_path = project_root / "python" / "requirements.txt"
        requirements = [
            line.strip()
            for line in requirement_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        package_names = {
            re.split(r"[<>=!~]", requirement, maxsplit=1)[0].casefold()
            for requirement in requirements
        }

        self.assertNotIn("langchain", package_names)
        self.assertNotIn("langchain-openai", package_names)
        self.assertIn("langchain-core", package_names)
        for requirement in requirements:
            self.assertIn(">=", requirement, requirement)
            self.assertIn("<", requirement, requirement)

    def test_dev_dependencies_and_verify_script_are_available(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        dev_path = project_root / "python" / "requirements-dev.txt"
        verify_path = project_root / "scripts" / "verify.sh"

        self.assertTrue(dev_path.is_file(), "缺少开发依赖文件")
        dev_content = dev_path.read_text(encoding="utf-8")
        self.assertIn("-r requirements.txt", dev_content)
        self.assertRegex(dev_content, r"(?m)^ruff>=.+<.+$")

        self.assertTrue(verify_path.is_file(), "缺少统一验证脚本")
        self.assertNotEqual(verify_path.stat().st_mode & 0o111, 0)
        verify_content = verify_path.read_text(encoding="utf-8")
        for required_command in (
            "python -m ruff check",
            "python -m unittest discover",
            "python -m compileall",
            "git diff --check",
            "git diff --cached --check",
        ):
            self.assertIn(required_command, verify_content)
        self.assertIn(
            "python -m unittest discover -s assess -t .",
            verify_content,
            "unittest 必须固定 python 包顶层，确保 assess 与 app 使用稳定绝对导入",
        )
        self.assertIn("json.loads", verify_content)
        self.assertIn("Markdown 本地链接检查通过", verify_content)


class ProjectSkillBoundaryTests(unittest.TestCase):
    """交互记忆维护规则必须作为可发现的仓库级 Skill 固化。"""

    def test_interaction_memory_skill_has_safety_and_ownership_contract(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        skill_root = (
            project_root
            / ".agents"
            / "skills"
            / "managing-user-interaction-memory"
        )
        skill_path = skill_root / "SKILL.md"
        metadata_path = skill_root / "agents" / "openai.yaml"

        self.assertTrue(skill_path.is_file(), "缺少交互记忆仓库级 Skill")
        self.assertTrue(metadata_path.is_file(), "缺少 Skill UI 元数据")
        ignored = subprocess.run(
            ["git", "check-ignore", "-q", str(skill_path.relative_to(project_root))],
            cwd=project_root,
            check=False,
        )
        self.assertEqual(ignored.returncode, 1, "交互记忆 Skill 不得被 Git 忽略")
        skill = skill_path.read_text(encoding="utf-8")
        metadata = metadata_path.read_text(encoding="utf-8")

        self.assertIn("name: managing-user-interaction-memory", skill)
        self.assertRegex(skill, r"description:\s+Use when")
        for required in (
            "当前明确要求 > 当前 Session > 用户交互记忆 > 默认回答策略",
            "conversation_feedback_events",
            "user_interaction_memories",
            "KnowledgeAnswerAgent",
            "30 天",
            "事实纠错",
            "推荐画像",
            "意图识别",
        ):
            self.assertIn(required, skill)
        self.assertIn('display_name: "用户交互记忆维护"', metadata)
        self.assertIn("$managing-user-interaction-memory", metadata)


class P2ResponsibilityBoundaryTests(unittest.TestCase):
    """验证长编排和公开降级映射已经收敛。"""

    def test_knowledge_query_analysis_retires_split_agents_and_names(self) -> None:
        knowledge_models = importlib.import_module("app.models.knowledge_qa")
        api_models = importlib.import_module("app.models.api")

        self.assertIsNone(
            importlib.util.find_spec("app.agents.knowledge_query_rewrite_agent")
        )
        self.assertIsNone(
            importlib.util.find_spec("app.agents.knowledge_query_plan_agent")
        )
        self.assertFalse(hasattr(knowledge_models, "KnowledgeQueryRewriteResult"))
        self.assertFalse(hasattr(knowledge_models, "KnowledgeQueryPlan"))
        self.assertIn("knowledge_query_analysis", api_models.DegradedComponent.__args__)
        self.assertNotIn("knowledge_query_rewrite", api_models.DegradedComponent.__args__)
        self.assertEqual(
            error_responses.degraded_components(
                {"knowledge_query_analysis": "degraded"}
            ),
            ["knowledge_query_analysis"],
        )

    def test_image_store_and_execution_writer_protocols_do_not_overlap(self) -> None:
        knowledge_service = importlib.import_module("app.application.knowledge_qa")

        self.assertTrue(hasattr(knowledge_service.KnowledgeImageStore, "resolve"))
        self.assertTrue(
            hasattr(knowledge_service.KnowledgeImageStore, "delete_unreferenced")
        )
        self.assertFalse(
            hasattr(knowledge_service.KnowledgeExecutionRecordWriter, "resolve")
        )
        self.assertFalse(
            hasattr(
                knowledge_service.KnowledgeExecutionRecordWriter,
                "delete_unreferenced",
            )
        )

    def test_chat_controller_uses_shared_degraded_component_mapping(self) -> None:
        self.assertIs(
            chat_controller.degraded_components,
            error_responses.degraded_components,
        )
        statuses = {
            "document_recall": "degraded",
            "document_rerank": "failed",
        }
        self.assertEqual(
            error_responses.degraded_components(statuses),
            ["document_recall", "document_rerank"],
        )

    def test_long_business_orchestrators_delegate_by_stage(self) -> None:
        self.assertLessEqual(
            len(inspect.getsourcelines(pipeline_evaluation.evaluate_pipeline)[0]),
            100,
        )
        self.assertLessEqual(
            len(inspect.getsourcelines(FeatureStore.get_user_features)[0]),
            90,
        )
        self.assertLessEqual(
            len(
                inspect.getsourcelines(
                    user_profile_agent.UserProfileAgent._protected_semantic_profile
                )[0]
            ),
            100,
        )

    def test_pipeline_regressions_no_longer_owns_intent_and_retrieval_suites(
        self,
    ) -> None:
        root = Path(__file__).resolve().parent
        regressions = (root / "test_pipeline_regressions.py").read_text(
            encoding="utf-8"
        )
        retrieval = (root / "test_retrieval_components.py").read_text(
            encoding="utf-8"
        )
        intent_routing = (root / "test_intent_routing.py").read_text(
            encoding="utf-8"
        )
        profile_rerank = (root / "test_profile_rerank.py").read_text(
            encoding="utf-8"
        )
        pipeline = (root / "test_pipeline_evaluation.py").read_text(
            encoding="utf-8"
        )

        for class_name in (
            "IntentDecisionTreeTests",
            "IntentDecisionTreeGraphTests",
            "RecallDegradationContractTests",
        ):
            self.assertNotIn(f"class {class_name}", regressions)
            self.assertIn(f"class {class_name}", retrieval)
        for class_name in (
            "IntentRoutingContractTests",
            "RuleFirstIntentRoutingTests",
            "IntentRoutingArbitrationTests",
            "KnowledgePreparedQueryTests",
        ):
            self.assertNotIn(f"class {class_name}", regressions)
            self.assertIn(f"class {class_name}", intent_routing)
        for retired_name in (
            "RuntimeRerankRoutingTests",
            "ArticleAggregationRegressionTests",
            "ArticleRerankProgramWeightTests",
            "FilteredControlResultTests",
        ):
            self.assertNotIn(retired_name, regressions)
            self.assertNotIn(retired_name, retrieval)
            self.assertNotIn(retired_name, profile_rerank)
            self.assertNotIn(retired_name, pipeline)
        self.assertIn("class PipelineEvaluationTests", pipeline)
        self.assertIn("SQLite Chunk 统一推荐评估", pipeline)
        self.assertLess(len(regressions.splitlines()), 1200)


if __name__ == "__main__":
    unittest.main()
