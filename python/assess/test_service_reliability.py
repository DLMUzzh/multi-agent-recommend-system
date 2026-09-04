"""验证应用服务总截止时间和 Agent 重试边界。"""

from __future__ import annotations

import asyncio
import importlib
import json
import sqlite3
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from app.agents.base_agent import BaseAgent
from app.api.routers.chat import _to_chat_response, _to_session_history_response
from app.config import Settings
from app.models.schemas import (
    AgentResult,
    ArbitrationAction,
    ChatRequest,
    ConversationReply,
    ConversationSession,
    ConversationTurn,
    IntentState,
    RecognitionSource,
    RecommendationContext,
)
from app.application.conversation_service import (
    ConversationService,
    ServiceUnavailableError,
)
from app.infrastructure.database.conversation_store import ConversationStoreError
from app.infrastructure.database.json.feature_store import FeatureStore
from app.infrastructure.database.sqlite.conversation_store import (
    SQLiteConversationStore,
)
from app.main import create_app
from app.models.knowledge_qa import (
    KnowledgeAnswerResult,
    KnowledgeCitation,
    KnowledgeDocumentIngestResult,
    KnowledgeImageUploadResult,
)
from app.models.personal_feedback import ConversationResultSnapshotDraft


class LlmModelRoleSettingsTests(unittest.TestCase):
    """验证大小模型配置保持旧环境兼容并按角色选择模型。"""

    def test_model_roles_fall_back_to_legacy_model(self) -> None:
        from app.infrastructure.llm import client as llm_client

        settings = Settings(
            _env_file=None,
            llm_model="legacy-chat",
            llm_small_model="",
            llm_large_model="",
        )

        self.assertEqual(
            llm_client.llm_model_name(settings, "small"),
            "legacy-chat",
        )
        self.assertEqual(
            llm_client.llm_model_name(settings, "large"),
            "legacy-chat",
        )

    def test_chat_client_uses_requested_model_role(self) -> None:
        from app.infrastructure.llm import client as llm_client

        captured: list[dict[str, Any]] = []

        def factory(**kwargs: Any) -> object:
            captured.append(kwargs)
            return object()

        settings = Settings(
            _env_file=None,
            llm_api_key="test-key",
            llm_model="legacy-chat",
            llm_small_model="fast-chat",
            llm_large_model="quality-chat",
        )

        llm_client.create_chat_llm(
            settings=settings,
            model_role="small",
            model_factory=factory,
        )
        llm_client.create_chat_llm(
            settings=settings,
            model_role="large",
            model_factory=factory,
        )

        self.assertEqual(
            [call["model"] for call in captured],
            ["fast-chat", "quality-chat"],
        )
        self.assertEqual(
            llm_client.public_llm_config(settings)["models"],
            {"small": "fast-chat", "large": "quality-chat"},
        )

    def test_same_model_name_disables_upgrade_client(self) -> None:
        from app.infrastructure.llm import client as llm_client

        class SameModelOutput(BaseModel):
            value: str

        settings = Settings(
            _env_file=None,
            llm_api_key="test-key",
            llm_small_model="shared-chat",
            llm_large_model="shared-chat",
        )
        small_client = object()

        with patch.object(
            llm_client,
            "create_structured_llm",
            return_value=small_client,
        ) as factory:
            primary, upgrade = llm_client.create_controlled_structured_llms(
                SameModelOutput,
                settings=settings,
            )

        self.assertIs(primary, small_client)
        self.assertIsNone(upgrade)
        self.assertEqual(factory.call_count, 1)
        self.assertEqual(factory.call_args.args[0], SameModelOutput)
        self.assertEqual(factory.call_args.kwargs["model_role"], "small")


class ControlledLlmUpgradeTests(unittest.IsolatedAsyncioTestCase):
    """验证请求级受控升级只处理能力型失败且共享一次额度。"""

    async def test_invalid_output_uses_large_model_once(self) -> None:
        from app.infrastructure.llm import client as llm_client

        calls: list[str] = []

        async def operation(llm: str, model_role: str) -> str:
            calls.append(f"{model_role}:{llm}")
            if model_role == "small":
                raise ValueError("结构不完整")
            return "large-ok"

        deadline = asyncio.get_running_loop().time() + 60.0
        with llm_client.llm_upgrade_scope(deadline=deadline):
            result = await llm_client.invoke_with_controlled_upgrade(
                stage="query_analysis",
                small_llm="small-client",
                large_llm="large-client",
                operation=operation,
            )

        self.assertEqual(result, "large-ok")
        self.assertEqual(
            calls,
            ["small:small-client", "large:large-client"],
        )

    async def test_timeout_does_not_upgrade_same_provider(self) -> None:
        from app.infrastructure.llm import client as llm_client

        calls: list[str] = []

        async def operation(llm: str, model_role: str) -> str:
            calls.append(f"{model_role}:{llm}")
            raise TimeoutError("provider timeout")

        deadline = asyncio.get_running_loop().time() + 60.0
        with llm_client.llm_upgrade_scope(deadline=deadline):
            with self.assertRaises(TimeoutError):
                await llm_client.invoke_with_controlled_upgrade(
                    stage="query_analysis",
                    small_llm="small-client",
                    large_llm="large-client",
                    operation=operation,
                )

        self.assertEqual(calls, ["small:small-client"])
        self.assertEqual(
            llm_client.llm_failure_kind(TimeoutError()),
            "timeout",
        )

    def test_low_confidence_is_an_upgradeable_capability_failure(self) -> None:
        from app.infrastructure.llm import client as llm_client

        error = llm_client.LlmLowConfidenceError("置信度不足")

        self.assertEqual(llm_client.llm_failure_kind(error), "low_confidence")

    async def test_two_stages_share_only_one_upgrade(self) -> None:
        from app.infrastructure.llm import client as llm_client

        calls: list[str] = []

        async def operation(llm: str, model_role: str) -> str:
            calls.append(f"{model_role}:{llm}")
            if model_role == "small":
                raise ValueError("能力型失败")
            return "upgraded"

        deadline = asyncio.get_running_loop().time() + 60.0
        with llm_client.llm_upgrade_scope(deadline=deadline):
            first = await llm_client.invoke_with_controlled_upgrade(
                stage="query_analysis",
                small_llm="small-client",
                large_llm="large-client",
                operation=operation,
            )
            with self.assertRaises(ValueError):
                await llm_client.invoke_with_controlled_upgrade(
                    stage="chunk_rerank",
                    small_llm="small-client",
                    large_llm="large-client",
                    operation=operation,
                )

        self.assertEqual(first, "upgraded")
        self.assertEqual(
            calls,
            [
                "small:small-client",
                "large:large-client",
                "small:small-client",
            ],
        )

    async def test_insufficient_remaining_time_skips_upgrade(self) -> None:
        from app.infrastructure.llm import client as llm_client

        calls: list[str] = []

        async def operation(llm: str, model_role: str) -> str:
            calls.append(f"{model_role}:{llm}")
            raise ValueError("结构不完整")

        deadline = asyncio.get_running_loop().time() + 1.0
        with llm_client.llm_upgrade_scope(deadline=deadline):
            with self.assertRaises(ValueError):
                await llm_client.invoke_with_controlled_upgrade(
                    stage="query_analysis",
                    small_llm="small-client",
                    large_llm="large-client",
                    operation=operation,
                )

        self.assertEqual(calls, ["small:small-client"])

    async def test_nested_scope_reuses_consumed_outer_upgrade_budget(self) -> None:
        from app.infrastructure.llm import client as llm_client

        calls: list[str] = []

        async def operation(llm: str, model_role: str) -> str:
            calls.append(f"{model_role}:{llm}")
            if model_role == "small":
                raise ValueError("能力型失败")
            return "upgraded"

        deadline = asyncio.get_running_loop().time() + 60.0
        with llm_client.llm_upgrade_scope(deadline=deadline):
            first = await llm_client.invoke_with_controlled_upgrade(
                stage="conversation_query_analysis",
                small_llm="small-client",
                large_llm="large-client",
                operation=operation,
            )
            with llm_client.llm_upgrade_scope(deadline=deadline - 1.0):
                with self.assertRaises(ValueError):
                    await llm_client.invoke_with_controlled_upgrade(
                        stage="knowledge_chunk_rerank",
                        small_llm="small-client",
                        large_llm="large-client",
                        operation=operation,
                    )

        self.assertEqual(first, "upgraded")
        self.assertEqual(
            calls,
            [
                "small:small-client",
                "large:large-client",
                "small:small-client",
            ],
        )

    async def test_structured_client_trace_excludes_prompt_and_model_output(
        self,
    ) -> None:
        from app.infrastructure.llm import client as llm_client

        class TraceOutput(BaseModel):
            value: str

        class FakeChatModel:
            model_name = "fast-chat"
            model_role = "small"

            def _request_payload(
                self,
                messages: list[Any],
                *,
                json_mode: bool,
            ) -> dict[str, Any]:
                _ = messages, json_mode
                return {
                    "model": self.model_name,
                    "messages": [
                        {"role": "user", "content": "private-prompt-value"}
                    ],
                }

            async def _post_with_retries(
                self,
                payload: dict[str, Any],
            ) -> dict[str, Any]:
                _ = payload
                return {
                    "choices": [
                        {
                            "message": {
                                "content": '{"value":"private-output-value"}'
                            }
                        }
                    ]
                }

            @staticmethod
            def _response_content(response_payload: dict[str, Any]) -> str:
                return str(response_payload["choices"][0]["message"]["content"])

            async def aclose(self) -> None:
                return None

        events: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

        def capture(*args: Any, **kwargs: Any) -> None:
            events.append((args, kwargs))

        structured = llm_client._DeepSeekStructuredClient(
            FakeChatModel(),
            TraceOutput,
        )
        with patch.object(llm_client, "record_trace_event", side_effect=capture):
            output = await structured.ainvoke(
                [HumanMessage(content="private-prompt-value")]
            )

        self.assertEqual(output.value, "private-output-value")
        serialized = json.dumps(events, ensure_ascii=False, default=str)
        self.assertNotIn("private-prompt-value", serialized)
        self.assertNotIn("private-output-value", serialized)
        self.assertNotIn("raw_response", serialized)
        self.assertNotIn("validated_output", serialized)
        self.assertIn("fast-chat", serialized)
        self.assertIn("small", serialized)
        completed = next(
            kwargs
            for args, kwargs in events
            if args[0] == "llm.completed"
        )
        self.assertGreaterEqual(completed["output_data"]["latency_ms"], 0.0)

    async def test_upgrade_trace_excludes_exception_text_and_records_safe_models(
        self,
    ) -> None:
        from app.infrastructure.llm import client as llm_client

        class FakeLlm:
            def __init__(self, model_name: str, model_role: str) -> None:
                self.model_name = model_name
                self.model_role = model_role

        small = FakeLlm("fast-chat", "small")
        large = FakeLlm("quality-chat", "large")
        events: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

        async def operation(llm: FakeLlm, model_role: str) -> str:
            if model_role == "small":
                raise ValueError("private-small-output")
            raise ValueError("private-large-output")

        def capture(*args: Any, **kwargs: Any) -> None:
            events.append((args, kwargs))

        deadline = asyncio.get_running_loop().time() + 60.0
        with (
            patch.object(llm_client, "record_trace_event", side_effect=capture),
            llm_client.llm_upgrade_scope(deadline=deadline),
            self.assertRaises(ValueError),
        ):
            await llm_client.invoke_with_controlled_upgrade(
                stage="query_analysis",
                small_llm=small,
                large_llm=large,
                operation=operation,
            )

        serialized = json.dumps(events, ensure_ascii=False, default=str)
        self.assertNotIn("private-small-output", serialized)
        self.assertNotIn("private-large-output", serialized)
        self.assertIn("fast-chat", serialized)
        self.assertIn("quality-chat", serialized)
        failed = next(
            kwargs
            for args, kwargs in events
            if args[0] == "llm.upgrade_failed"
        )
        self.assertNotIn("error", failed)
        self.assertGreaterEqual(failed["output_data"]["latency_ms"], 0.0)


class AgentModelRoleWiringTests(unittest.TestCase):
    """验证后台任务、在线辅助任务和最终回答使用正确模型角色。"""

    @staticmethod
    def _settings() -> Settings:
        return Settings(
            _env_file=None,
            llm_api_key="test-key",
            llm_small_model="fast-chat",
            llm_large_model="quality-chat",
        )

    def test_fixed_small_agents_request_only_small_model(self) -> None:
        from app.agents import conversation_feedback_agent
        from app.agents import conversation_summary_agent
        from app.agents import feedback_recovery_agent
        from app.agents import intent_recognition_agent
        from app.agents import user_profile_agent

        cases = (
            (
                intent_recognition_agent,
                lambda settings: intent_recognition_agent.IntentRecognitionAgent(
                    settings=settings
                ),
            ),
            (
                user_profile_agent,
                lambda settings: user_profile_agent.UserProfileAgent(
                    settings=settings
                ),
            ),
            (
                conversation_summary_agent,
                lambda settings: conversation_summary_agent.ConversationSummaryAgent(
                    settings=settings
                ),
            ),
            (
                conversation_feedback_agent,
                lambda settings: conversation_feedback_agent.ConversationFeedbackAgent.from_settings(
                    settings
                ),
            ),
            (
                feedback_recovery_agent,
                lambda settings: feedback_recovery_agent.FeedbackRecoveryAgent.from_settings(
                    settings
                ),
            ),
        )
        for module, build in cases:
            with self.subTest(module=module.__name__), patch.object(
                module,
                "create_structured_llm",
                return_value=object(),
            ) as factory:
                build(self._settings())
                self.assertEqual(factory.call_args.kwargs["model_role"], "small")

    def test_answer_agent_requests_only_large_model(self) -> None:
        from app.agents import knowledge_answer_agent

        with patch.object(
            knowledge_answer_agent,
            "create_structured_llm",
            return_value=object(),
        ) as factory:
            knowledge_answer_agent.KnowledgeAnswerAgent.from_settings(
                self._settings()
            )

        self.assertEqual(factory.call_args.kwargs["model_role"], "large")

    def test_upgradeable_agents_request_small_and_distinct_large_models(
        self,
    ) -> None:
        from app.agents import knowledge_chunk_rerank_agent
        from app.agents import knowledge_query_analysis_agent
        from app.agents import knowledge_reasoning_planner_agent

        cases = (
            (
                knowledge_query_analysis_agent,
                knowledge_query_analysis_agent.KnowledgeQueryAnalysisAgent.from_settings,
            ),
            (
                knowledge_reasoning_planner_agent,
                knowledge_reasoning_planner_agent.KnowledgeReasoningPlannerAgent.from_settings,
            ),
            (
                knowledge_chunk_rerank_agent,
                knowledge_chunk_rerank_agent.KnowledgeChunkRerankAgent.from_settings,
            ),
        )
        for module, build in cases:
            with self.subTest(module=module.__name__), patch.object(
                module,
                "create_controlled_structured_llms",
                return_value=(object(), object()),
            ) as factory:
                build(self._settings())
                expected_calls = (
                    2
                    if module.__name__.endswith("knowledge_chunk_rerank_agent")
                    else 1
                )
                self.assertEqual(factory.call_count, expected_calls)


class SharedLlmHttpClientTests(unittest.IsolatedAsyncioTestCase):
    """验证同一 Provider 连接配置只建立一套底层异步连接池。"""

    async def test_small_and_large_clients_share_pool_until_last_close(self) -> None:
        from app.infrastructure.llm import client as llm_client

        class FakeHttpClient:
            def __init__(self, **kwargs: Any) -> None:
                self.kwargs = kwargs
                self.close_count = 0

            async def aclose(self) -> None:
                self.close_count += 1

        created: list[FakeHttpClient] = []

        def factory(**kwargs: Any) -> FakeHttpClient:
            client = FakeHttpClient(**kwargs)
            created.append(client)
            return client

        unique_suffix = str(time.time_ns())
        settings = Settings(
            _env_file=None,
            llm_api_key=f"shared-key-{unique_suffix}",
            llm_base_url=f"https://shared-{unique_suffix}.invalid/v1",
            llm_small_model="fast-chat",
            llm_large_model="quality-chat",
        )
        with patch.object(llm_client.httpx, "AsyncClient", side_effect=factory):
            small = llm_client.create_chat_llm(
                settings=settings,
                model_role="small",
            )
            large = llm_client.create_chat_llm(
                settings=settings,
                model_role="large",
            )

        self.assertEqual(len(created), 1)
        self.assertIs(small._http_client, large._http_client)
        await small.aclose()
        self.assertEqual(created[0].close_count, 0)
        await large.aclose()
        self.assertEqual(created[0].close_count, 1)
        await large.aclose()
        self.assertEqual(created[0].close_count, 1)

    async def test_structured_wrapper_exposes_safe_model_identity(self) -> None:
        from app.infrastructure.llm import client as llm_client

        class FakeHttpClient:
            async def aclose(self) -> None:
                return None

        class TraceOutput(BaseModel):
            value: str

        unique_suffix = str(time.time_ns())
        settings = Settings(
            _env_file=None,
            llm_api_key=f"identity-key-{unique_suffix}",
            llm_base_url=f"https://identity-{unique_suffix}.invalid/v1",
            llm_small_model="fast-chat",
            llm_large_model="quality-chat",
        )
        with patch.object(
            llm_client.httpx,
            "AsyncClient",
            return_value=FakeHttpClient(),
        ):
            structured = llm_client.create_structured_llm(
                TraceOutput,
                settings=settings,
                model_role="large",
            )

        self.assertEqual(structured.model_name, "quality-chat")
        self.assertEqual(structured.model_role, "large")
        self.assertEqual(llm_client._safe_model_name(structured), "quality-chat")
        await structured.aclose()


class LlmStreamProjectionTests(unittest.TestCase):
    """验证页面只展示安全模型元数据和受控升级结果。"""

    def test_llm_projection_keeps_role_model_reason_and_latency(self) -> None:
        from app.infrastructure.observability import conversation_trace

        projection = conversation_trace._stream_projection(
            "llm.upgrade_completed",
            "knowledge_query_analysis_agent",
            input_data=None,
            output_data={
                "model_role": "large",
                "model_name": "quality-chat",
                "reason": "invalid_output",
                "latency_ms": 12.5,
            },
            status="success",
            error=None,
        )

        self.assertIsNotNone(projection)
        assert projection is not None
        self.assertEqual(projection["details"]["model_role"], "large")
        self.assertEqual(projection["details"]["model_name"], "quality-chat")
        self.assertEqual(projection["details"]["reason"], "invalid_output")
        self.assertEqual(projection["details"]["latency_ms"], 12.5)
        serialized = json.dumps(projection, ensure_ascii=False)
        self.assertNotIn("request", serialized)
        self.assertNotIn("raw_response", serialized)
        self.assertNotIn("validated_output", serialized)

    def test_upgrade_started_projection_is_not_reported_as_completed(self) -> None:
        from app.infrastructure.observability import conversation_trace

        projection = conversation_trace._stream_projection(
            "llm.upgrade_started",
            "knowledge_query_analysis_agent",
            input_data={
                "from_model_role": "small",
                "from_model_name": "fast-chat",
                "to_model_role": "large",
                "to_model_name": "quality-chat",
                "reason": "low_confidence",
            },
            output_data=None,
            status=None,
            error=None,
        )

        self.assertIsNotNone(projection)
        assert projection is not None
        self.assertEqual(projection["status"], "started")
        self.assertEqual(projection["title"], "辅助大模型升级开始")


class UnifiedApplicationTests(unittest.IsolatedAsyncioTestCase):
    """验证文档导入、单轮问答和健康状态统一由主应用提供。"""

    def test_main_app_registers_knowledge_routes(self) -> None:
        app = create_app()

        route_paths = {route.path for route in app.routes}

        self.assertIn("/api/v1/knowledge/documents", route_paths)
        self.assertIn("/api/v1/knowledge/ask", route_paths)
        self.assertIn("/api/v1/knowledge/images/{image_id}", route_paths)
        self.assertIn("/api/v1/chat/stream", route_paths)

    async def test_main_health_reports_embedding_configuration(self) -> None:
        app = create_app()
        transport = ASGITransport(app=app, raise_app_exceptions=False)

        async with AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            response = await client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertIsInstance(response.json()["embedding_configured"], bool)


class ChatWindowLauncherTests(unittest.TestCase):
    """验证唯一测试窗口的本地静态服务不会被浏览器代理阻塞。"""

    @staticmethod
    def _launcher_module() -> Any:
        project_root = Path(__file__).resolve().parents[2]
        module_path = project_root / "Test" / "chat_window.py"
        spec = importlib.util.spec_from_file_location(
            "chat_window_launcher",
            module_path,
        )
        if spec is None or spec.loader is None:
            raise AssertionError("无法加载唯一测试窗口启动器")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_browser_open_does_not_block_static_server_start(self) -> None:
        module = self._launcher_module()
        browser_started = threading.Event()
        release_browser = threading.Event()

        def blocking_open(*_: Any, **__: Any) -> bool:
            browser_started.set()
            release_browser.wait(timeout=1.0)
            return True

        try:
            with patch.object(module.webbrowser, "open", side_effect=blocking_open):
                started_at = time.monotonic()
                module._open_browser("http://127.0.0.1:12345/chat_window.html")
                elapsed = time.monotonic() - started_at

                self.assertLess(elapsed, 0.2)
                self.assertTrue(browser_started.wait(timeout=0.2))
        finally:
            release_browser.set()


class UnifiedTestWindowTests(unittest.TestCase):
    """验证唯一测试窗口覆盖完整会话、文档导入和安全知识诊断。"""

    def test_only_chat_window_remains_as_current_test_entry(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        self.assertTrue((project_root / "Test" / "chat_window.html").is_file())
        self.assertTrue((project_root / "Test" / "chat_window.py").is_file())
        self.assertFalse(
            (project_root / "Test" / "knowledge_chat_window.html").exists()
        )
        self.assertFalse(
            (project_root / "Test" / "knowledge_chat_window.py").exists()
        )

    def test_chat_window_contains_all_approved_test_capabilities(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        html = (project_root / "Test" / "chat_window.html").read_text(
            encoding="utf-8"
        )
        launcher = (project_root / "Test" / "chat_window.py").read_text(
            encoding="utf-8"
        )

        self.assertIn("http://127.0.0.1:8000", html)
        self.assertIn("/api/v1/knowledge/documents", html)
        self.assertIn("/api/v1/chat/stream", html)
        self.assertIn('id="session-id"', html)
        self.assertIn("function renderHistory", html)
        self.assertIn("function renderCompression", html)
        self.assertIn("function renderSessionNavigation", html)
        self.assertIn("response.recommendations", html)
        self.assertIn("response.images", html)
        self.assertIn('document.createElement("img")', html)
        self.assertIn("knowledgeImageUrl", html)
        self.assertIn("图片加载失败", html)
        self.assertIn("citation.document_id", html)
        self.assertIn("citation.chunk_id", html)
        self.assertIn('id="stream-process"', html)
        self.assertIn('id="stream-thought-list"', html)
        self.assertIn('id="stream-event-log"', html)
        self.assertIn('id="stream-process-list"', html)
        self.assertIn("function startStreamTimeline", html)
        self.assertIn("function appendStreamEvent", html)
        self.assertIn("function updateReasoningSummary", html)
        self.assertIn("function reasoningPhaseForEvent", html)
        self.assertIn("function renderStreamDetails", html)
        self.assertIn("async function consumeChatStream", html)
        self.assertIn("response.body.getReader()", html)
        self.assertIn("new TextDecoder", html)
        self.assertIn("response.execution_trace", html)
        self.assertIn("response.execution_trace?.route", html)
        self.assertIn("function renderKnowledgePlanTrace(trace)", html)
        self.assertIn("reasoning_strategy", html)
        self.assertIn("plan_revision_count", html)
        self.assertIn("plan_steps", html)
        self.assertIn("coverage_ratio", html)
        self.assertIn("replanned", html)
        self.assertIn('document.createElement("details")', html)
        self.assertIn("streamProcessView.open = true", html)
        self.assertIn("思考过程（实时）", html)
        self.assertIn("已完成思考", html)
        self.assertIn("详细执行链路", html)
        self.assertIn("由安全执行轨迹确定性生成，不是模型隐藏思维链", html)
        self.assertIn("安全业务轨迹", html)
        self.assertIn("用户问题", html)
        self.assertIn("历史消息", html)
        self.assertIn("会话摘要", html)
        self.assertIn("限定文档", html)
        self.assertIn("推荐", html)
        self.assertIn("请求接口", html)
        self.assertIn("问题类型", html)
        self.assertIn("检索策略", html)
        self.assertIn("独立查询", html)
        self.assertIn("发送给检索", html)
        self.assertIn("召回 Chunk", html)
        self.assertIn("文档证据", html)
        self.assertIn("降级组件", html)
        self.assertNotIn('apiRequest("/api/v1/chat",', html)
        self.assertNotIn("function renderRouteTrace", html)
        self.assertIn("app.main:app", launcher)
        self.assertIn('"8000"', launcher)
        self.assertFalse(
            (project_root / "python" / "app" / "knowledge_main.py").exists()
        )



class _CountingAgent(BaseAgent):
    def __init__(self, failures: list[Exception], *, max_retries: int) -> None:
        super().__init__(
            name="counting",
            timeout=5.0,
            max_retries=max_retries,
        )
        self.failures = list(failures)
        self.attempts = 0

    async def _execute(self, **_: Any) -> AgentResult:
        self.attempts += 1
        if self.failures:
            raise self.failures.pop(0)
        return AgentResult(agent_name=self.name)


class BaseAgentRetryBoundaryTests(unittest.IsolatedAsyncioTestCase):
    """重试次数和异常分类必须符合公开参数语义。"""

    async def test_zero_retries_executes_once(self) -> None:
        try:
            agent = _CountingAgent([TimeoutError("暂时超时")], max_retries=0)
        except ValueError:
            self.fail("max_retries=0 应表示禁用重试")

        result = await agent.run()

        self.assertFalse(result.success)
        self.assertEqual(agent.attempts, 1)

    async def test_retryable_error_uses_initial_attempt_plus_retries(self) -> None:
        agent = _CountingAgent(
            [TimeoutError("第一次"), TimeoutError("第二次")],
            max_retries=2,
        )

        result = await agent.run()

        self.assertTrue(result.success)
        self.assertEqual(agent.attempts, 3)

    async def test_deterministic_error_is_not_retried(self) -> None:
        agent = _CountingAgent([ValueError("输入无效")], max_retries=2)

        result = await agent.run()

        self.assertFalse(result.success)
        self.assertEqual(agent.attempts, 1)


class _MemoryConversationStore:
    """为会话提交可靠性探针提供深拷贝内存边界。"""

    def __init__(self, session: ConversationSession) -> None:
        self.session = session.model_copy(deep=True)

    async def load(
        self,
        user_id: str,
        session_id: str,
    ) -> ConversationSession | None:
        if (
            self.session.user_id != user_id
            or self.session.session_id != session_id
        ):
            return None
        return self.session.model_copy(deep=True)

    async def save(self, session: ConversationSession) -> None:
        self.session = session.model_copy(deep=True)

    async def delete(self, user_id: str, session_id: str) -> None:
        if self.session.user_id == user_id and self.session.session_id == session_id:
            self.session = ConversationSession(
                session_id=session_id,
                user_id=user_id,
            )


class _UnusedAgent:
    async def run(self, **_: Any) -> object:
        raise AssertionError("受控会话验证不会执行推荐 Agent")


class _UnusedAggregator:
    def aggregate(self, **_: Any) -> list[object]:
        raise AssertionError("受控会话验证不会执行推荐聚合")


class _IntentStateWorkflow:
    def __init__(
        self,
        *,
        pending_intent_state: IntentState | None,
        commit_intent_state: bool,
    ) -> None:
        self.pending_intent_state = pending_intent_state
        self.commit_intent_state = commit_intent_state
        self.received_intent_states: list[IntentState] = []

    async def run(
        self,
        *,
        user_id: str,
        session_id: str,
        message: str,
        history: list[ConversationTurn],
        previous_context: RecommendationContext | None,
        conversation_summary: str | None,
        intent_state: IntentState,
    ) -> object:
        del user_id, history, previous_context, conversation_summary
        self.received_intent_states.append(intent_state)
        return SimpleNamespace(
            reply=ConversationReply(
                session_id=session_id,
                message="已完成受控路由。",
                intent_source=RecognitionSource.RULE,
                action=ArbitrationAction.REPEAT,
            ),
            history_message=f"助手处理：{message}",
            pending_context=None,
            commit_context=False,
            pending_intent_state=self.pending_intent_state,
            commit_intent_state=self.commit_intent_state,
            error_stage=None,
        )


def _intent_state_service(
    workflow: _IntentStateWorkflow,
) -> tuple[ConversationService, _MemoryConversationStore]:
    store = _MemoryConversationStore(
        ConversationSession(
            session_id="intent-route",
            user_id="10001",
            intent_state=IntentState.KNOWLEDGE_QA,
        )
    )
    service = ConversationService(
        user_store=FeatureStore(),
        recall_agent=_UnusedAgent(),
        rerank_agent=_UnusedAgent(),
        aggregator=_UnusedAggregator(),
        conversation_store=store,
        enable_llm=False,
    )
    service.workflow = workflow
    return service, store


class ConversationIntentStateCommitTests(unittest.IsolatedAsyncioTestCase):
    """会话服务只提交 Graph 明确确认的业务路由。"""

    async def test_workflow_receives_and_commits_intent_state(self) -> None:
        workflow = _IntentStateWorkflow(
            pending_intent_state=IntentState.RECOMMENDATION,
            commit_intent_state=True,
        )
        service, store = _intent_state_service(workflow)

        await service.chat("10001", "继续推荐", session_id="intent-route")

        self.assertEqual(
            workflow.received_intent_states,
            [IntentState.KNOWLEDGE_QA],
        )
        self.assertEqual(store.session.intent_state, IntentState.RECOMMENDATION)

    async def test_non_committing_transition_preserves_intent_state(self) -> None:
        workflow = _IntentStateWorkflow(
            pending_intent_state=None,
            commit_intent_state=False,
        )
        service, store = _intent_state_service(workflow)

        await service.chat("10001", "无法判断的请求", session_id="intent-route")

        self.assertEqual(store.session.intent_state, IntentState.KNOWLEDGE_QA)


class KnowledgeChatPublicContractTests(unittest.TestCase):
    """验证主聊天复用原请求，并公开问答状态与受保护引用。"""

    def test_chat_request_contract_is_unchanged(self) -> None:
        properties = ChatRequest.model_json_schema()["properties"]

        self.assertEqual(
            set(properties),
            {"user_id", "session_id", "message"},
        )

    def test_plan_degraded_components_are_publicly_mapped(self) -> None:
        from app.api.errors import degraded_components

        self.assertEqual(
            degraded_components(
                {
                    "knowledge_planner": "degraded",
                    "knowledge_plan_execution": "failed",
                    "knowledge_plan_coverage": "degraded",
                }
            ),
            [
                "knowledge_planner",
                "knowledge_plan_execution",
                "knowledge_plan_coverage",
            ],
        )

    def test_public_responses_expose_child_session_navigation(self) -> None:
        reply = ConversationReply(
            session_id="child-qa",
            message="已进入文章问答。",
            intent_source=RecognitionSource.RULE,
            action=ArbitrationAction.KNOWLEDGE_ANSWER,
            intent_state=IntentState.KNOWLEDGE_QA,
            session_type="article_qa",
            parent_session_id="main-session",
            focus_document_id="doc-spring",
            focus_document_title="Spring Boot 部署",
            session_status="active",
        )
        session = ConversationSession(
            session_id="main-session",
            user_id="10001",
            active_child_session_id="child-qa",
        )

        chat_response = _to_chat_response(reply)
        session_response = _to_session_history_response(session)

        self.assertEqual(chat_response.session_type, "article_qa")
        self.assertEqual(chat_response.parent_session_id, "main-session")
        self.assertEqual(chat_response.focus_document_id, "doc-spring")
        self.assertEqual(chat_response.focus_document_title, "Spring Boot 部署")
        self.assertEqual(chat_response.session_status, "active")
        self.assertEqual(session_response.session_type, "main")
        self.assertEqual(session_response.active_child_session_id, "child-qa")

    def test_conversation_model_protects_parent_child_invariants(self) -> None:
        child = ConversationSession(
            session_id="child-qa",
            user_id="10001",
            session_type="article_qa",
            parent_session_id="main-session",
            focus_document_id="doc-spring",
            focus_document_title="Spring Boot 部署",
        )

        self.assertEqual(child.session_status, "active")
        with self.assertRaises(ValueError):
            ConversationSession(
                session_id="invalid-child",
                user_id="10001",
                session_type="article_qa",
            )
        with self.assertRaises(ValueError):
            ConversationSession(
                session_id="invalid-main",
                user_id="10001",
                focus_document_id="doc-spring",
            )

    def test_chat_and_session_responses_expose_knowledge_state(self) -> None:
        citation = KnowledgeCitation(
            citation_id="1",
            document_id="doc-spring",
            title="Spring 事务实践",
            chunk_id="chunk-1",
            heading_path=("事务边界",),
            excerpt="Spring 事务通过代理边界生效。",
        )
        reply = ConversationReply(
            session_id="knowledge-public",
            message="Spring 事务通过代理边界生效。",
            intent_source=RecognitionSource.RULE,
            action="knowledge_answer",
            intent_state=IntentState.KNOWLEDGE_QA,
            citations=[citation],
        )
        session = ConversationSession(
            session_id="knowledge-public",
            user_id="10001",
            intent_state=IntentState.KNOWLEDGE_QA,
        )

        chat_response = _to_chat_response(reply)
        session_response = _to_session_history_response(session)

        self.assertEqual(chat_response.intent_state, IntentState.KNOWLEDGE_QA)
        self.assertEqual(chat_response.citations, [citation])
        self.assertEqual(chat_response.citations[0].citation_id, "1")
        self.assertEqual(session_response.intent_state, IntentState.KNOWLEDGE_QA)
        self.assertNotIn(
            "citations",
            session_response.model_dump(),
        )

    def test_chat_response_exposes_current_knowledge_images_and_trace(self) -> None:
        from app.models.knowledge_qa import (
            KnowledgeExecutionResult,
            KnowledgeExecutionTrace,
            KnowledgeImageCitation,
        )

        image = KnowledgeImageCitation(
            citation_id="图1",
            image_id="img-" + "a" * 32,
            document_id="doc-hmi",
            title="HMI 刷机",
            heading_path=("传输到跳板机",),
            caption="HMI 刷机传输进度界面",
            url=(
                "/api/v1/knowledge/images/img-"
                + "a" * 32
                + "?v="
                + "b" * 12
            ),
        )
        trace = KnowledgeExecutionTrace(
            trace_id="c" * 32,
            request_route="/api/v1/chat",
            question="HMI 刷机传输进度是什么界面？",
            standalone_query="HMI 刷机传输到跳板机进度界面",
            result=KnowledgeExecutionResult(
                status="success",
                citation_count=1,
                image_count=1,
                elapsed_ms=12.5,
            ),
        )
        reply = ConversationReply(
            session_id="knowledge-trace",
            message="传输进度为 40%。[1][图1]",
            intent_source=RecognitionSource.RULE,
            action=ArbitrationAction.KNOWLEDGE_ANSWER,
            intent_state=IntentState.KNOWLEDGE_QA,
            images=[image],
            execution_trace=trace,
        )

        response = _to_chat_response(reply)

        self.assertEqual(response.images, [image])
        self.assertEqual(response.execution_trace, trace)

    def test_chat_window_has_safe_knowledge_result_and_citation_rendering(
        self,
    ) -> None:
        project_root = Path(__file__).resolve().parents[2]
        html = (project_root / "Test" / "chat_window.html").read_text(
            encoding="utf-8"
        )

        self.assertIn('id="knowledge-citations"', html)
        self.assertIn("function renderKnowledgeResult", html)
        self.assertIn("function renderKnowledgePlanTrace(trace)", html)
        self.assertIn("renderKnowledgePlanTrace(response.execution_trace)", html)
        self.assertIn("trace.reasoning_strategy", html)
        self.assertIn("trace.plan_revision_count", html)
        self.assertIn("trace.plan_steps", html)
        self.assertIn("trace.coverage.coverage_ratio", html)
        self.assertIn("trace.coverage.replanned", html)
        self.assertIn("first.citation_id", html)
        self.assertIn("`[${first.citation_id}] ${first.title}", html)
        self.assertNotIn("`[${citation.citation_id}] ${path}", html)
        self.assertIn("citation.excerpt", html)
        self.assertIn("response.degraded_components", html)
        self.assertIn('id="session-type"', html)
        self.assertIn('id="parent-session-id"', html)
        self.assertIn('id="end-article-qa"', html)
        self.assertIn('id="return-parent-session"', html)
        self.assertIn("function renderSessionNavigation", html)
        self.assertIn("turn.related_session_id", html)
        self.assertNotIn("innerHTML", html)

    def test_production_bootstrap_shares_sqlite_search_without_json_catalog(
        self,
    ) -> None:
        project_root = Path(__file__).resolve().parents[2]
        source = (project_root / "python" / "app" / "bootstrap.py").read_text(
            encoding="utf-8"
        )

        self.assertNotIn("catalog = JsonArticleCatalog()", source)
        self.assertIn("SQLiteKnowledgeRepository", source)
        self.assertIn("DocumentRecallAgent", source)
        self.assertIn("search=document_search", source)
        self.assertIn("knowledge_qa_service=knowledge_qa_service", source)


class ChatStreamApiTests(unittest.IsolatedAsyncioTestCase):
    """验证聊天流按真实阶段输出 NDJSON，并以完整响应结束。"""

    async def test_stream_emits_process_events_before_final_response(self) -> None:
        from app.infrastructure.observability.conversation_trace import (
            record_trace_event,
        )

        class _Service:
            async def chat(
                self,
                user_id: str,
                message: str,
                *,
                session_id: str | None = None,
            ) -> ConversationReply:
                record_trace_event(
                    "agent.started",
                    "intent_recognition_agent",
                    input_data={"message": message, "history": []},
                )
                await asyncio.sleep(0)
                record_trace_event(
                    "agent.completed",
                    "intent_recognition_agent",
                    output_data={
                        "intent": "recommend_articles",
                        "source": "rule",
                        "confidence": 1.0,
                        "rewritten_query": "Java 入门",
                    },
                )
                return ConversationReply(
                    session_id=session_id or "stream-session",
                    message="找到 1 篇。",
                    intent_source=RecognitionSource.RULE,
                    action=ArbitrationAction.NEW,
                )

        app = create_app()
        app.state.conversation_service = _Service()
        app.state.conversation_trace_writer = None
        app.state.knowledge_test_record_writer = None
        transport = ASGITransport(app=app, raise_app_exceptions=False)

        async with AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            response = await client.post(
                "/api/v1/chat/stream",
                json={"user_id": "10001", "message": "推荐 Java 入门"},
            )

        events = [json.loads(line) for line in response.text.splitlines()]
        self.assertEqual(response.status_code, 200)
        self.assertTrue(
            response.headers["content-type"].startswith("application/x-ndjson")
        )
        self.assertEqual(events[0]["event"], "process")
        self.assertEqual(events[0]["title"], "收到请求")
        self.assertEqual(events[-1]["event"], "result")
        self.assertEqual(events[-1]["response"]["session_id"], "stream-session")
        self.assertEqual(
            [event["sequence"] for event in events],
            list(range(1, len(events) + 1)),
        )
        self.assertLess(
            next(
                index
                for index, event in enumerate(events)
                if event["component"] == "intent_recognition_agent"
                and event["status"] == "started"
            ),
            len(events) - 1,
        )
        self.assertNotIn("Prompt", response.text)
        self.assertNotIn("traceback", response.text.casefold())

    async def test_stream_failure_ends_with_safe_error_event(self) -> None:
        class _Service:
            async def chat(self, *_: Any, **__: Any) -> ConversationReply:
                raise ServiceUnavailableError("不得输出的内部路径 /mnt/private")

        app = create_app()
        app.state.conversation_service = _Service()
        app.state.conversation_trace_writer = None
        app.state.knowledge_test_record_writer = None
        transport = ASGITransport(app=app, raise_app_exceptions=False)

        async with AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            response = await client.post(
                "/api/v1/chat/stream",
                json={"user_id": "10001", "message": "测试失败"},
            )

        events = [json.loads(line) for line in response.text.splitlines()]
        self.assertEqual(events[-1]["event"], "error")
        self.assertEqual(events[-1]["error"]["code"], "SERVICE_UNAVAILABLE")
        self.assertNotIn("/mnt/private", response.text)

    async def test_stream_records_full_chain_before_final_result(self) -> None:
        class _Service:
            async def chat(self, *_: Any, **__: Any) -> ConversationReply:
                return ConversationReply(
                    session_id="stream-record-session",
                    message="找到 1 篇。",
                    intent_source=RecognitionSource.RULE,
                    action=ArbitrationAction.NEW,
                )

        class _Writer:
            def __init__(self) -> None:
                self.calls: list[dict[str, Any]] = []

            async def append_stream(self, **kwargs: Any) -> bool:
                self.calls.append(kwargs)
                return True

        writer = _Writer()
        app = create_app()
        app.state.conversation_service = _Service()
        app.state.conversation_trace_writer = None
        app.state.knowledge_test_record_writer = writer
        transport = ASGITransport(app=app, raise_app_exceptions=False)

        async with AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            response = await client.post(
                "/api/v1/chat/stream",
                json={"user_id": "10001", "message": "推荐 Java 入门"},
            )

        events = [json.loads(line) for line in response.text.splitlines()]
        self.assertEqual(len(writer.calls), 1)
        self.assertEqual(
            writer.calls[0]["response"].session_id,
            "stream-record-session",
        )
        record_index = next(
            index
            for index, event in enumerate(events)
            if event.get("title") == "全链路测试记录已写入"
        )
        self.assertLess(record_index, len(events) - 1)
        self.assertEqual(events[-1]["event"], "result")

    async def test_stream_record_failure_does_not_replace_chat_result(self) -> None:
        class _Service:
            async def chat(self, *_: Any, **__: Any) -> ConversationReply:
                return ConversationReply(
                    session_id="stream-record-failure",
                    message="聊天结果仍然可用。",
                    intent_source=RecognitionSource.RULE,
                    action=ArbitrationAction.NEW,
                )

        class _Writer:
            async def append_stream(self, **_: Any) -> bool:
                raise OSError("不得影响聊天结果 /mnt/private")

        app = create_app()
        app.state.conversation_service = _Service()
        app.state.conversation_trace_writer = None
        app.state.knowledge_test_record_writer = _Writer()
        transport = ASGITransport(app=app, raise_app_exceptions=False)

        async with AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            response = await client.post(
                "/api/v1/chat/stream",
                json={"user_id": "10001", "message": "推荐 Java 入门"},
            )

        events = [json.loads(line) for line in response.text.splitlines()]
        self.assertEqual(events[-1]["event"], "result")
        self.assertEqual(
            events[-1]["response"]["session_id"],
            "stream-record-failure",
        )
        self.assertIn(
            "全链路测试记录写入失败",
            [event.get("title") for event in events],
        )
        self.assertNotIn("/mnt/private", response.text)


class ConversationStreamProjectionTests(unittest.IsolatedAsyncioTestCase):
    """验证推荐链投影保留 Chunk、画像和各阶段评分。"""

    async def test_recommendation_projection_contains_auditable_details(
        self,
    ) -> None:
        from app.infrastructure.observability.conversation_trace import (
            ConversationStreamRecorder,
            conversation_stream_context,
            record_trace_event,
        )

        recorder = ConversationStreamRecorder()
        with conversation_stream_context(recorder):
            record_trace_event(
                "agent.completed",
                "user_profile_agent",
                output_data={
                    "profile_available": True,
                    "profile_status": "ready",
                    "profile_confidence": 0.91,
                    "core_topics": ["Java", "Spring"],
                    "negative_topics": ["营销"],
                    "preferred_difficulty": "intermediate",
                    "preferred_reading_length": "medium",
                    "activity_level": "active_reader",
                },
            )
            record_trace_event(
                "agent.completed",
                "document_recall_agent",
                output_data={
                    "retrieval_mode": "hybrid",
                    "retrieval_diagnostics": {
                        "bm25_status": "success",
                        "vector_status": "success",
                    },
                    "candidates": [
                        {
                            "document_id": "doc-java",
                            "title": "Java 入门",
                            "matched_chunk_ids": ["chunk-1", "chunk-2"],
                            "recall_score": 0.88,
                            "excerpt": "Java 基础语法与工程实践。",
                        }
                    ],
                },
            )
            record_trace_event(
                "agent.completed",
                "document_rerank_agent",
                output_data={
                    "llm_status": "success",
                    "profile_status": "ready",
                    "profile_confidence": 0.91,
                    "document_scores": [
                        {
                            "document_id": "doc-java",
                            "relevance_score": 0.85,
                            "profile_score": 0.72,
                            "llm_score": 0.9,
                            "final_score": 0.82,
                            "rerank_reason": "主题和难度匹配",
                        }
                    ],
                },
            )

        events = recorder.snapshot()
        profile = events[0]["details"]
        recall = events[1]["details"]
        rerank = events[2]["details"]["document_scores"][0]
        self.assertEqual(profile["core_topics"], ["Java", "Spring"])
        self.assertEqual(profile["negative_topics"], ["营销"])
        self.assertEqual(
            recall["candidates"][0]["matched_chunk_ids"],
            ["chunk-1", "chunk-2"],
        )
        self.assertEqual(
            recall["retrieval_diagnostics"]["vector_status"],
            "success",
        )
        self.assertEqual(rerank["llm_score"], 0.9)
        self.assertEqual(rerank["rerank_reason"], "主题和难度匹配")

def _sqlite_conversation_store_module() -> Any:
    try:
        return importlib.import_module(
            "app.infrastructure.database.sqlite.conversation_store"
        )
    except ModuleNotFoundError as exc:
        raise AssertionError("SQLite 会话 Store 尚未实现") from exc


class SQLiteConversationStoreTests(unittest.IsolatedAsyncioTestCase):
    """验证会话 SQLite 的完整往返和原子提交。"""

    @staticmethod
    def _session() -> ConversationSession:
        return ConversationSession(
            session_id="sqlite-session",
            user_id="10001",
            intent_state=IntentState.KNOWLEDGE_QA,
            active_context=RecommendationContext(
                query="查找 Java 相关文章",
                size=3,
                seen_article_ids=["30001"],
            ),
            history=[
                ConversationTurn(role="user", content="推荐 Java 文章"),
                ConversationTurn(role="assistant", content="已完成推荐。"),
            ],
            turn_count=1,
            summary="用户关注 Java。",
            summarized_turn_count=2,
            dropped_turn_count=1,
        )

    async def test_round_trip_uses_two_tables_and_delete_is_idempotent(
        self,
    ) -> None:
        module = _sqlite_conversation_store_module()
        with tempfile.TemporaryDirectory(prefix="article-rec-session-sqlite-") as root:
            path = Path(root) / "conversations.sqlite3"
            store = module.SQLiteConversationStore(path)

            await store.save(self._session())
            loaded = await store.load("10001", "sqlite-session")
            await store.delete("10001", "sqlite-session")
            await store.delete("10001", "sqlite-session")
            deleted = await store.load("10001", "sqlite-session")
            with sqlite3.connect(path) as connection:
                tables = {
                    row[0]
                    for row in connection.execute(
                        "SELECT name FROM sqlite_master "
                        "WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
                    )
                }

            assert loaded is not None
            expected = self._session()
            self.assertEqual(
                loaded.model_dump(exclude={"history"}),
                expected.model_dump(exclude={"history"}),
            )
            self.assertEqual(
                [(turn.role, turn.content) for turn in loaded.history],
                [(turn.role, turn.content) for turn in expected.history],
            )
            self.assertTrue(
                all(turn.message_id is not None for turn in loaded.history)
            )
            self.assertIsNone(deleted)
            self.assertEqual(
                tables,
                {
                    "article_qa_sessions",
                    "conversations",
                    "conversation_messages",
                    "conversation_result_snapshots",
                    "personal_feedback_events",
                },
            )
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)

    async def test_existing_messages_keep_stable_identity_when_appending(
        self,
    ) -> None:
        module = _sqlite_conversation_store_module()
        with tempfile.TemporaryDirectory(prefix="article-rec-session-sqlite-") as root:
            store = module.SQLiteConversationStore(
                Path(root) / "conversations.sqlite3"
            )
            await store.save(self._session())
            first = await store.load("10001", "sqlite-session")
            assert first is not None
            original_identity = [
                (turn.message_id, turn.sequence_no, turn.created_at)
                for turn in first.history
            ]
            first.history.extend(
                [
                    ConversationTurn(role="user", content="继续提问"),
                    ConversationTurn(role="assistant", content="继续回答"),
                ]
            )
            first.turn_count += 1

            await store.save(first)
            second = await store.load("10001", "sqlite-session")

            assert second is not None
            self.assertEqual(len(second.history), 4)
            self.assertEqual(
                [
                    (turn.message_id, turn.sequence_no, turn.created_at)
                    for turn in second.history[:2]
                ],
                original_identity,
            )
            self.assertTrue(
                all(turn.message_id is not None for turn in second.history)
            )
            self.assertEqual(
                [turn.sequence_no for turn in second.history],
                [0, 1, 2, 3],
            )

    async def test_initialization_migrates_legacy_two_table_database(self) -> None:
        module = _sqlite_conversation_store_module()
        with tempfile.TemporaryDirectory(prefix="article-rec-session-sqlite-") as root:
            path = Path(root) / "conversations.sqlite3"
            with sqlite3.connect(path) as connection:
                connection.executescript(
                    """
                    CREATE TABLE conversations (
                        user_id TEXT NOT NULL,
                        session_id TEXT NOT NULL,
                        intent_state TEXT NOT NULL,
                        active_context_json TEXT,
                        turn_count INTEGER NOT NULL,
                        summary TEXT,
                        summarized_turn_count INTEGER NOT NULL,
                        dropped_turn_count INTEGER NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        PRIMARY KEY (user_id, session_id)
                    );
                    CREATE TABLE conversation_messages (
                        message_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT NOT NULL,
                        session_id TEXT NOT NULL,
                        sequence_no INTEGER NOT NULL,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        UNIQUE (user_id, session_id, sequence_no)
                    );
                    """
                )

            module.SQLiteConversationStore(path)

            with sqlite3.connect(path) as connection:
                conversation_columns = {
                    row[1]
                    for row in connection.execute(
                        "PRAGMA table_info(conversations)"
                    )
                }
                message_columns = {
                    row[1]
                    for row in connection.execute(
                        "PRAGMA table_info(conversation_messages)"
                    )
                }
                child_table = connection.execute(
                    "SELECT name FROM sqlite_master "
                    "WHERE type = 'table' AND name = 'article_qa_sessions'"
                ).fetchone()

            self.assertIn("summary_watermark", conversation_columns)
            self.assertIn("message_type", message_columns)
            self.assertIn("related_session_id", message_columns)
            self.assertIsNotNone(child_table)

    async def test_parent_child_round_trip_and_delete_boundaries(self) -> None:
        module = _sqlite_conversation_store_module()
        with tempfile.TemporaryDirectory(prefix="article-rec-session-sqlite-") as root:
            store = module.SQLiteConversationStore(
                Path(root) / "conversations.sqlite3"
            )
            parent = ConversationSession(
                session_id="main-session",
                user_id="10001",
                active_child_session_id="child-session",
            )
            child = ConversationSession(
                session_id="child-session",
                user_id="10001",
                session_type="article_qa",
                parent_session_id="main-session",
                focus_document_id="doc-spring",
                focus_document_title="Spring Boot 部署",
                intent_state=IntentState.KNOWLEDGE_QA,
                cited_document_ids=["doc-spring"],
                unresolved_questions=["生产环境如何回滚？"],
            )

            await store.save_many([parent, child])
            loaded_parent = await store.load("10001", "main-session")
            loaded_child = await store.load("10001", "child-session")

            assert loaded_parent is not None
            assert loaded_child is not None
            self.assertEqual(
                loaded_parent.active_child_session_id,
                "child-session",
            )
            self.assertEqual(loaded_child.parent_session_id, "main-session")
            self.assertEqual(loaded_child.focus_document_id, "doc-spring")
            self.assertEqual(
                loaded_child.unresolved_questions,
                ["生产环境如何回滚？"],
            )

            await store.delete("10001", "child-session")
            self.assertIsNotNone(await store.load("10001", "main-session"))
            self.assertIsNone(await store.load("10001", "child-session"))

            await store.save_many([parent, child])
            await store.delete("10001", "main-session")
            self.assertIsNone(await store.load("10001", "main-session"))
            self.assertIsNone(await store.load("10001", "child-session"))

    async def test_failed_message_replace_keeps_previous_session(self) -> None:
        module = _sqlite_conversation_store_module()
        with tempfile.TemporaryDirectory(prefix="article-rec-session-sqlite-") as root:
            path = Path(root) / "conversations.sqlite3"
            store = module.SQLiteConversationStore(path)
            original = self._session()
            await store.save(original)
            with sqlite3.connect(path) as connection:
                connection.execute(
                    """
                    CREATE TRIGGER reject_assistant_message
                    BEFORE INSERT ON conversation_messages
                    WHEN NEW.role = 'assistant'
                    BEGIN
                        SELECT RAISE(ABORT, 'reject assistant');
                    END
                    """
                )

            changed = original.model_copy(
                update={"summary": "不应被部分提交。"},
                deep=True,
            )
            changed.history.extend(
                [
                    ConversationTurn(role="user", content="新增问题"),
                    ConversationTurn(role="assistant", content="新增回答"),
                ]
            )
            changed.turn_count += 1
            with self.assertRaises(ConversationStoreError):
                await store.save(changed)
            loaded = await store.load("10001", "sqlite-session")

            assert loaded is not None
            self.assertEqual(loaded.summary, original.summary)
            self.assertEqual(
                [(turn.role, turn.content) for turn in loaded.history],
                [(turn.role, turn.content) for turn in original.history],
            )

    async def test_save_many_failure_rolls_back_parent_and_child(self) -> None:
        module = _sqlite_conversation_store_module()
        with tempfile.TemporaryDirectory(prefix="article-rec-session-sqlite-") as root:
            path = Path(root) / "conversations.sqlite3"
            store = module.SQLiteConversationStore(path)
            parent = ConversationSession(
                session_id="main-session",
                user_id="10001",
                active_child_session_id="child-session",
            )
            child = ConversationSession(
                session_id="child-session",
                user_id="10001",
                session_type="article_qa",
                parent_session_id="main-session",
                focus_document_id="doc-spring",
                focus_document_title="Spring Boot 部署",
                intent_state=IntentState.KNOWLEDGE_QA,
            )
            await store.save_many([parent, child])
            with sqlite3.connect(path) as connection:
                connection.execute(
                    """
                    CREATE TRIGGER reject_child_assistant_message
                    BEFORE INSERT ON conversation_messages
                    WHEN NEW.session_id = 'child-session'
                         AND NEW.role = 'assistant'
                    BEGIN
                        SELECT RAISE(ABORT, 'reject child assistant');
                    END
                    """
                )

            changed_parent = parent.model_copy(
                update={"summary": "父会话不应被部分提交。"},
                deep=True,
            )
            changed_parent.history.append(
                ConversationTurn(role="user", content="父会话新增消息")
            )
            changed_child = child.model_copy(
                update={"summary": "子会话不应被部分提交。"},
                deep=True,
            )
            changed_child.history.append(
                ConversationTurn(role="assistant", content="子会话新增消息")
            )

            with self.assertRaises(ConversationStoreError):
                await store.save_many([changed_parent, changed_child])
            loaded_parent = await store.load("10001", "main-session")
            loaded_child = await store.load("10001", "child-session")

            assert loaded_parent is not None
            assert loaded_child is not None
            self.assertIsNone(loaded_parent.summary)
            self.assertIsNone(loaded_child.summary)
            self.assertEqual(loaded_parent.history, [])
            self.assertEqual(loaded_child.history, [])


class _KnowledgeApiService:
    def __init__(
        self,
        error: Exception | None = None,
        *,
        image_file: Any = None,
    ) -> None:
        self.error = error
        self.image_file = image_file
        self.ingest_calls: list[dict[str, str]] = []
        self.ask_calls: list[str] = []
        self.upload_calls: list[dict[str, Any]] = []

    async def ingest_document(
        self,
        *,
        document_id: str,
        title: str,
        content_markdown: str,
        topics: list[str],
        content_type: str,
        difficulty: str,
        author_id: str,
    ) -> KnowledgeDocumentIngestResult:
        if self.error is not None:
            raise self.error
        self.ingest_calls.append(
            {
                "document_id": document_id,
                "title": title,
                "content_markdown": content_markdown,
                "topics": topics,
                "content_type": content_type,
                "difficulty": difficulty,
                "author_id": author_id,
            }
        )
        return KnowledgeDocumentIngestResult(
            document_id=document_id,
            title=title,
            content_hash="a" * 64,
            chunk_count=1,
        )

    async def ask(self, question: str, *, limit: int = 5) -> KnowledgeAnswerResult:
        del limit
        if self.error is not None:
            raise self.error
        self.ask_calls.append(question)
        return KnowledgeAnswerResult(
            status="degraded",
            answer="事件循环负责调度协程。",
            citations=(
                KnowledgeCitation(
                    citation_id="1",
                    document_id="doc-python",
                    title="Python 异步编程",
                    chunk_id="chunk-1",
                    heading_path=("Python", "事件循环"),
                    excerpt="事件循环负责调度协程。",
                ),
            ),
            degraded_components=("answer",),
        )

    async def upload_image(
        self,
        *,
        image_id: str,
        content: bytes,
        mime_type: str,
    ) -> KnowledgeImageUploadResult:
        if self.error is not None:
            raise self.error
        self.upload_calls.append(
            {
                "image_id": image_id,
                "content": content,
                "mime_type": mime_type,
            }
        )
        return KnowledgeImageUploadResult(
            image_id=image_id,
            content_hash="b" * 64,
            mime_type="image/png",
            byte_size=len(content),
        )

    def get_image_file(self, image_id: str) -> Any:
        del image_id
        if self.error is not None:
            raise self.error
        return self.image_file

    async def aclose(self) -> None:
        return None


class KnowledgeApiTests(unittest.IsolatedAsyncioTestCase):
    """验证主应用中的无会话知识 API 与错误响应保持稳定。"""

    async def test_document_ingest_and_ask_use_unified_service(self) -> None:
        service = _KnowledgeApiService()
        app = create_app()
        app.state.knowledge_qa_service = service
        transport = ASGITransport(app=app, raise_app_exceptions=False)

        async with AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            ingest_response = await client.post(
                "/api/v1/knowledge/documents",
                json={
                    "document_id": "doc-python",
                    "title": "Python 异步编程",
                    "content_markdown": "# Python\n\n事件循环负责调度协程。",
                    "topics": ["Python", "异步编程"],
                    "content_type": "tutorial",
                    "difficulty": "intermediate",
                    "author_id": "author-python",
                },
            )
            answer_response = await client.post(
                "/api/v1/knowledge/ask",
                json={"question": "事件循环负责什么？"},
            )

        self.assertEqual(ingest_response.status_code, 201)
        self.assertEqual(answer_response.status_code, 200)
        self.assertEqual(
            answer_response.json()["citations"][0]["chunk_id"],
            "chunk-1",
        )
        self.assertEqual(service.ask_calls, ["事件循环负责什么？"])

    async def test_value_error_is_mapped_to_safe_400(self) -> None:
        app = create_app()
        app.state.knowledge_qa_service = _KnowledgeApiService(
            ValueError("文档正文没有可切分内容")
        )
        transport = ASGITransport(app=app, raise_app_exceptions=False)

        async with AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            response = await client.post(
                "/api/v1/knowledge/ask",
                json={"question": "有效问题"},
            )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(
            response.json(),
            {
                "error": {
                    "code": "INVALID_KNOWLEDGE_REQUEST",
                    "message": "文档正文没有可切分内容",
                }
            },
        )

    async def test_internal_error_is_mapped_to_safe_503(self) -> None:
        app = create_app()
        app.state.knowledge_qa_service = _KnowledgeApiService(
            RuntimeError("内部数据库路径")
        )
        transport = ASGITransport(app=app, raise_app_exceptions=False)

        async with AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            response = await client.post(
                "/api/v1/knowledge/ask",
                json={"question": "有效问题"},
            )

        self.assertEqual(response.status_code, 503)
        self.assertEqual(
            response.json(),
            {
                "error": {
                    "code": "KNOWLEDGE_SERVICE_UNAVAILABLE",
                    "message": "知识问答服务暂时不可用",
                }
            },
        )

    async def test_image_upload_and_read_use_safe_binary_contract(self) -> None:
        image_id = "img-" + "a" * 32
        png_bytes = b"\x89PNG\r\n\x1a\nfixture"
        with tempfile.TemporaryDirectory() as temporary_directory:
            image_path = Path(temporary_directory) / "image.png"
            image_path.write_bytes(png_bytes)
            service = _KnowledgeApiService(
                image_file=SimpleNamespace(
                    path=image_path,
                    mime_type="image/png",
                    content_hash="b" * 64,
                )
            )
            app = create_app()
            app.state.knowledge_qa_service = service
            transport = ASGITransport(app=app, raise_app_exceptions=False)

            async with AsyncClient(
                transport=transport,
                base_url="http://testserver",
            ) as client:
                upload_response = await client.put(
                    f"/api/v1/knowledge/images/{image_id}",
                    content=png_bytes,
                    headers={"Content-Type": "image/png"},
                )
                image_response = await client.get(
                    f"/api/v1/knowledge/images/{image_id}"
                )

        self.assertEqual(upload_response.status_code, 200)
        self.assertEqual(upload_response.json()["image_id"], image_id)
        self.assertEqual(service.upload_calls[0]["content"], png_bytes)
        self.assertEqual(service.upload_calls[0]["mime_type"], "image/png")
        self.assertEqual(image_response.status_code, 200)
        self.assertEqual(image_response.content, png_bytes)
        self.assertEqual(image_response.headers["content-type"], "image/png")
        self.assertEqual(image_response.headers["x-content-type-options"], "nosniff")
        self.assertEqual(image_response.headers["etag"], '"' + "b" * 64 + '"')
        self.assertEqual(
            image_response.headers["cache-control"],
            "private, max-age=3600",
        )

    async def test_image_upload_limit_and_missing_file_are_safe(self) -> None:
        image_id = "img-" + "c" * 32
        service = _KnowledgeApiService()
        app = create_app()
        app.state.knowledge_qa_service = service
        transport = ASGITransport(app=app, raise_app_exceptions=False)

        async with AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            oversized_response = await client.put(
                f"/api/v1/knowledge/images/{image_id}",
                content=b"x" * (8 * 1024 * 1024 + 1),
                headers={"Content-Type": "image/png"},
            )
            missing_response = await client.get(
                f"/api/v1/knowledge/images/{image_id}"
            )

        self.assertEqual(oversized_response.status_code, 400)
        self.assertEqual(service.upload_calls, [])
        self.assertNotIn("path", oversized_response.text.lower())
        self.assertEqual(missing_response.status_code, 404)
        self.assertEqual(
            missing_response.json(),
            {
                "error": {
                    "code": "KNOWLEDGE_IMAGE_NOT_FOUND",
                    "message": "知识图片不存在或尚未就绪",
                }
            },
        )


class SimilarDocumentApiTests(unittest.IsolatedAsyncioTestCase):
    """验证独立相似文章 API 的最小契约与安全错误。"""

    @staticmethod
    def _router_module() -> Any:
        module_name = "app.api.routers.similar_documents"
        if importlib.util.find_spec(module_name) is None:
            raise AssertionError(f"缺少批准的 Router：{module_name}")
        return importlib.import_module(module_name)

    @staticmethod
    def _app(module: Any, service: Any) -> FastAPI:
        from app.api.routers.chat import register_error_handlers

        app = FastAPI()
        app.state.similar_document_recommendation_service = service
        app.include_router(module.router)
        register_error_handlers(app)
        module.register_similar_document_error_handlers(app)
        return app

    async def test_success_returns_fixed_minimal_recommendations_and_degradation(
        self,
    ) -> None:
        module = self._router_module()
        article_models = importlib.import_module("app.models.article")

        class _Service:
            def __init__(self) -> None:
                self.calls: list[tuple[str, str]] = []

            async def recommend(
                self,
                *,
                user_id: str,
                document_id: str,
            ) -> Any:
                self.calls.append((user_id, document_id))
                return SimpleNamespace(
                    source_document_id=document_id,
                    recommendations=(
                        article_models.DocumentRecommendation(
                            document_id="doc-related",
                            title="相关文档",
                            excerpt="与源文档相关的可信 Chunk 摘录。",
                            score=0.82,
                            reason="查询与候选摘录直接相关。",
                        ),
                    ),
                    agent_statuses={
                        "user_profile": "failed",
                        "document_recall": "success",
                        "document_rerank": "degraded",
                    },
                )

        service = _Service()
        transport = ASGITransport(
            app=self._app(module, service),
            raise_app_exceptions=False,
        )

        async with AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            response = await client.get(
                "/api/v1/documents/doc-source/similar",
                params={"user_id": "10001"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(service.calls, [("10001", "doc-source")])
        self.assertEqual(
            response.json(),
            {
                "source_document_id": "doc-source",
                "recommendations": [
                    {
                        "document_id": "doc-related",
                        "title": "相关文档",
                        "excerpt": "与源文档相关的可信 Chunk 摘录。",
                        "score": 0.82,
                        "reason": "查询与候选摘录直接相关。",
                    }
                ],
                "degraded": True,
                "degraded_components": ["user_profile", "document_rerank"],
            },
        )

    async def test_missing_user_and_document_use_stable_404_codes(self) -> None:
        module = self._router_module()
        application_module = importlib.import_module(
            "app.application.similar_document_recommendation"
        )
        feature_store_module = importlib.import_module(
            "app.infrastructure.database.json.feature_store"
        )

        class _Service:
            async def recommend(
                self,
                *,
                user_id: str,
                document_id: str,
            ) -> Any:
                if user_id == "missing-user":
                    raise feature_store_module.UserNotFoundError(user_id)
                raise application_module.DocumentNotFoundError(document_id)

        transport = ASGITransport(
            app=self._app(module, _Service()),
            raise_app_exceptions=False,
        )

        async with AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            missing_user = await client.get(
                "/api/v1/documents/doc-source/similar",
                params={"user_id": "missing-user"},
            )
            missing_document = await client.get(
                "/api/v1/documents/missing-doc/similar",
                params={"user_id": "10001"},
            )

        self.assertEqual(missing_user.status_code, 404)
        self.assertEqual(
            missing_user.json()["error"]["code"],
            "USER_NOT_FOUND",
        )
        self.assertEqual(missing_document.status_code, 404)
        self.assertEqual(
            missing_document.json()["error"]["code"],
            "DOCUMENT_NOT_FOUND",
        )

    async def test_core_failure_and_blank_parameter_are_safe(self) -> None:
        module = self._router_module()

        class _Service:
            async def recommend(self, **_: Any) -> Any:
                raise ServiceUnavailableError("受控内部路径")

        transport = ASGITransport(
            app=self._app(module, _Service()),
            raise_app_exceptions=False,
        )

        async with AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            unavailable = await client.get(
                "/api/v1/documents/doc-source/similar",
                params={"user_id": "10001"},
            )
            invalid = await client.get(
                "/api/v1/documents/%20/similar",
                params={"user_id": "   "},
            )

        self.assertEqual(unavailable.status_code, 503)
        self.assertEqual(
            unavailable.json(),
            {
                "error": {
                    "code": "SERVICE_UNAVAILABLE",
                    "message": "文章推荐服务暂时不可用",
                }
            },
        )
        self.assertEqual(invalid.status_code, 422)
        self.assertEqual(invalid.json()["error"]["code"], "VALIDATION_ERROR")

class _ChildSessionWorkflow:
    """按顺序返回固定转换并记录实际会话上下文。"""

    def __init__(self, transitions: list[SimpleNamespace]) -> None:
        self.transitions = list(transitions)
        self.calls: list[dict[str, Any]] = []

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
        knowledge_document_ids: tuple[str, ...] = (),
    ) -> SimpleNamespace:
        self.calls.append(
            {
                "user_id": user_id,
                "session_id": session_id,
                "message": message,
                "history": [turn.model_copy(deep=True) for turn in history],
                "previous_context": (
                    previous_context.model_copy(deep=True)
                    if previous_context is not None
                    else None
                ),
                "conversation_summary": conversation_summary,
                "intent_state": intent_state,
                "knowledge_document_ids": knowledge_document_ids,
            }
        )
        return self.transitions.pop(0)


def _knowledge_transition(
    session_id: str,
    *,
    answer: str,
    document_ids: tuple[str, ...],
    document_titles: tuple[str, ...],
) -> SimpleNamespace:
    return SimpleNamespace(
        reply=ConversationReply(
            session_id=session_id,
            message=answer,
            intent_source=RecognitionSource.RULE,
            action=ArbitrationAction.KNOWLEDGE_ANSWER,
            intent_state=IntentState.KNOWLEDGE_QA,
        ),
        history_message=answer,
        pending_context=None,
        commit_context=False,
        pending_intent_state=IntentState.KNOWLEDGE_QA,
        commit_intent_state=True,
        knowledge_document_ids=document_ids,
        knowledge_document_titles=document_titles,
        citations=[],
        error_stage=None,
    )


def _recommendation_transition(
    session_id: str,
    context: RecommendationContext,
) -> SimpleNamespace:
    return SimpleNamespace(
        reply=ConversationReply(
            session_id=session_id,
            message="已继续原推荐。",
            intent_source=RecognitionSource.RULE,
            action=ArbitrationAction.REPEAT,
            intent_state=IntentState.RECOMMENDATION,
            active_context=context,
        ),
        history_message="已继续原推荐。",
        pending_context=context,
        commit_context=True,
        pending_intent_state=IntentState.RECOMMENDATION,
        commit_intent_state=True,
        knowledge_document_ids=(),
        knowledge_document_titles=(),
        citations=[],
        error_stage=None,
    )


class ArticleQaChildSessionTests(unittest.IsolatedAsyncioTestCase):
    """验证文章问答使用独立会话并在切换时恢复父推荐上下文。"""

    @staticmethod
    def _service(
        path: Path,
        workflow: _ChildSessionWorkflow,
    ) -> tuple[ConversationService, SQLiteConversationStore]:
        store = SQLiteConversationStore(path)
        service = ConversationService(
            user_store=FeatureStore(),
            recall_agent=_UnusedAgent(),
            rerank_agent=_UnusedAgent(),
            aggregator=_UnusedAggregator(),
            conversation_store=store,
            enable_llm=False,
        )
        service.workflow = workflow
        return service, store

    def test_handoff_uses_protected_summary_and_structured_facts(self) -> None:
        child = ConversationSession(
            session_id="child-session",
            user_id="10001",
            session_type="article_qa",
            parent_session_id="main-session",
            focus_document_id="doc-spring",
            focus_document_title="Spring Boot 部署",
            intent_state=IntentState.KNOWLEDGE_QA,
            summary="【受保护滚动摘要 v2】\n模式：文章聚焦问答",
            cited_document_ids=["doc-spring"],
            unresolved_questions=["生产环境如何回滚？"],
            history=[
                ConversationTurn(role="user", content="它如何部署？"),
                ConversationTurn(role="assistant", content="不应复制的完整助手回答"),
            ],
        )

        handoff = ConversationService._build_handoff_summary(child)

        self.assertIn("问答摘要：", handoff)
        self.assertIn("引用文档 ID：doc-spring", handoff)
        self.assertIn("未解决问题：生产环境如何回滚？", handoff)
        self.assertNotIn("不应复制的完整助手回答", handoff)

    async def test_article_question_creates_isolated_child_and_follow_up_uses_it(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(prefix="article-qa-child-") as root:
            parent_context = RecommendationContext(
                query="查找 Spring Boot 部署相关文章",
                size=3,
            )
            first_transition = _knowledge_transition(
                "main-session",
                answer="可以使用 Jar 部署。",
                document_ids=("doc-spring",),
                document_titles=("Spring Boot 部署",),
            )
            first_transition.result_snapshot_draft = ConversationResultSnapshotDraft(
                result_type="knowledge_answer",
                query="第一篇如何部署？",
                citation_document_ids=("doc-spring",),
                citation_chunk_ids=("chunk-spring-deploy",),
                knowledge_status="success",
                resolved_document_ids=("doc-spring",),
            )
            workflow = _ChildSessionWorkflow(
                [
                    first_transition,
                    _knowledge_transition(
                        "placeholder-child",
                        answer="也可以使用容器部署。",
                        document_ids=("doc-spring",),
                        document_titles=("Spring Boot 部署",),
                    ),
                ]
            )
            service, store = self._service(
                Path(root) / "conversations.sqlite3",
                workflow,
            )
            parent = ConversationSession(
                session_id="main-session",
                user_id="10001",
                active_context=parent_context,
                history=[
                    ConversationTurn(role="user", content="推荐 Spring Boot 文章"),
                    ConversationTurn(
                        role="assistant",
                        content="推荐结果：Spring Boot 部署；Spring Boot 测试",
                    ),
                ],
                turn_count=1,
                summary="父会话推荐摘要，不得进入子会话。",
            )
            await store.save(parent)

            entered = await service.chat(
                "10001",
                "第一篇如何部署？",
                session_id="main-session",
            )
            child_session_id = entered.session_id
            followed = await service.chat(
                "10001",
                "它支持容器吗？",
                session_id=child_session_id,
            )
            stored_parent = await store.load("10001", "main-session")
            stored_child = await store.load("10001", child_session_id)
            feedback_context = await store.load_feedback_context(
                "10001",
                child_session_id,
            )

            self.assertEqual(entered.session_type, "article_qa")
            self.assertEqual(entered.parent_session_id, "main-session")
            self.assertEqual(entered.focus_document_id, "doc-spring")
            self.assertNotEqual(child_session_id, "main-session")
            self.assertEqual(followed.session_id, child_session_id)
            assert stored_parent is not None
            assert stored_child is not None
            assert feedback_context.latest_result is not None
            self.assertEqual(
                feedback_context.latest_result.session_id,
                child_session_id,
            )
            self.assertEqual(
                feedback_context.latest_result.citation_chunk_ids,
                ("chunk-spring-deploy",),
            )
            self.assertEqual(len(stored_parent.history), 2)
            self.assertEqual(len(stored_child.history), 4)
            self.assertEqual(stored_child.parent_session_id, "main-session")
            self.assertEqual(
                [turn.content for turn in workflow.calls[1]["history"]],
                ["第一篇如何部署？", "可以使用 Jar 部署。"],
            )
            self.assertIsNone(workflow.calls[1]["conversation_summary"])
            self.assertEqual(
                workflow.calls[1]["knowledge_document_ids"],
                ("doc-spring",),
            )
            self.assertEqual(
                workflow.calls[1]["previous_context"],
                parent_context,
            )

    async def test_recommendation_intent_suspends_child_and_returns_parent(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(prefix="article-qa-child-") as root:
            parent_context = RecommendationContext(
                query="查找 Spring Boot 部署相关文章",
                size=3,
            )
            workflow = _ChildSessionWorkflow(
                [
                    _knowledge_transition(
                        "main-session",
                        answer="可以使用 Jar 部署。",
                        document_ids=("doc-spring",),
                        document_titles=("Spring Boot 部署",),
                    ),
                    _recommendation_transition("placeholder-child", parent_context),
                ]
            )
            service, store = self._service(
                Path(root) / "conversations.sqlite3",
                workflow,
            )
            await store.save(
                ConversationSession(
                    session_id="main-session",
                    user_id="10001",
                    active_context=parent_context,
                )
            )
            entered = await service.chat(
                "10001",
                "《Spring Boot 部署》讲了什么？",
                session_id="main-session",
            )

            returned = await service.chat(
                "10001",
                "继续推荐",
                session_id=entered.session_id,
            )
            stored_parent = await store.load("10001", "main-session")
            stored_child = await store.load("10001", entered.session_id)

            self.assertEqual(returned.session_id, "main-session")
            self.assertEqual(returned.session_type, "main")
            assert stored_parent is not None
            assert stored_child is not None
            self.assertIsNone(stored_parent.active_child_session_id)
            self.assertEqual(stored_child.session_status, "suspended")
            self.assertEqual(workflow.calls[1]["previous_context"], parent_context)
            self.assertEqual(
                [turn.content for turn in stored_parent.history[-2:]],
                ["继续推荐", "已继续原推荐。"],
            )

    async def test_failed_recommendation_keeps_child_active_without_handoff(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(prefix="article-qa-child-") as root:
            parent_context = RecommendationContext(
                query="查找 Spring Boot 部署相关文章",
                size=3,
            )
            failed_transition = _recommendation_transition(
                "placeholder-child",
                parent_context,
            )
            failed_transition.error_stage = "document_recall"
            workflow = _ChildSessionWorkflow(
                [
                    _knowledge_transition(
                        "main-session",
                        answer="可以使用 Jar 部署。",
                        document_ids=("doc-spring",),
                        document_titles=("Spring Boot 部署",),
                    ),
                    failed_transition,
                ]
            )
            service, store = self._service(
                Path(root) / "conversations.sqlite3",
                workflow,
            )
            await store.save(
                ConversationSession(
                    session_id="main-session",
                    user_id="10001",
                    active_context=parent_context,
                )
            )
            entered = await service.chat(
                "10001",
                "《Spring Boot 部署》讲了什么？",
                session_id="main-session",
            )

            with self.assertRaises(ServiceUnavailableError):
                await service.chat(
                    "10001",
                    "继续推荐",
                    session_id=entered.session_id,
                )
            parent = await store.load("10001", "main-session")
            child = await store.load("10001", entered.session_id)

            assert parent is not None
            assert child is not None
            self.assertEqual(parent.active_child_session_id, entered.session_id)
            self.assertEqual(child.session_status, "active")
            self.assertFalse(
                any(turn.message_type == "child_handoff" for turn in parent.history)
            )

    async def test_same_article_new_process_creates_new_child_session(self) -> None:
        with tempfile.TemporaryDirectory(prefix="article-qa-child-") as root:
            workflow = _ChildSessionWorkflow(
                [
                    _knowledge_transition(
                        "main-session",
                        answer="第一次问答。",
                        document_ids=("doc-spring",),
                        document_titles=("Spring Boot 部署",),
                    ),
                    _knowledge_transition(
                        "main-session",
                        answer="第二次问答。",
                        document_ids=("doc-spring",),
                        document_titles=("Spring Boot 部署",),
                    ),
                ]
            )
            service, store = self._service(
                Path(root) / "conversations.sqlite3",
                workflow,
            )
            await store.save(
                ConversationSession(
                    session_id="main-session",
                    user_id="10001",
                )
            )

            first = await service.chat(
                "10001",
                "《Spring Boot 部署》第一次问答",
                session_id="main-session",
            )
            second = await service.chat(
                "10001",
                "《Spring Boot 部署》开始新的问答",
                session_id="main-session",
            )
            parent = await store.load("10001", "main-session")
            first_child = await store.load("10001", first.session_id)
            second_child = await store.load("10001", second.session_id)

            assert parent is not None
            assert first_child is not None
            assert second_child is not None
            self.assertNotEqual(first.session_id, second.session_id)
            self.assertEqual(first_child.session_status, "suspended")
            self.assertEqual(second_child.session_status, "active")
            self.assertEqual(parent.active_child_session_id, second.session_id)
            self.assertEqual(
                first_child.focus_document_id,
                second_child.focus_document_id,
            )

    async def test_open_knowledge_question_remains_in_parent_session(self) -> None:
        with tempfile.TemporaryDirectory(prefix="article-qa-child-") as root:
            workflow = _ChildSessionWorkflow(
                [
                    _knowledge_transition(
                        "main-session",
                        answer="可以从多个文档解释。",
                        document_ids=(),
                        document_titles=(),
                    )
                ]
            )
            service, store = self._service(
                Path(root) / "conversations.sqlite3",
                workflow,
            )

            reply = await service.chat(
                "10001",
                "Spring Boot 通常如何部署？",
                session_id="main-session",
            )
            parent = await store.load("10001", "main-session")

            self.assertEqual(reply.session_id, "main-session")
            self.assertEqual(reply.session_type, "main")
            assert parent is not None
            self.assertIsNone(parent.active_child_session_id)
            self.assertEqual(len(parent.history), 2)

    async def test_explicit_end_closes_child_without_running_recommendation(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(prefix="article-qa-child-") as root:
            workflow = _ChildSessionWorkflow(
                [
                    _knowledge_transition(
                        "main-session",
                        answer="可以使用 Jar 部署。",
                        document_ids=("doc-spring",),
                        document_titles=("Spring Boot 部署",),
                    )
                ]
            )
            service, store = self._service(
                Path(root) / "conversations.sqlite3",
                workflow,
            )
            await store.save(
                ConversationSession(
                    session_id="main-session",
                    user_id="10001",
                )
            )
            entered = await service.chat(
                "10001",
                "《Spring Boot 部署》讲了什么？",
                session_id="main-session",
            )

            returned = await service.chat(
                "10001",
                "结束问答",
                session_id=entered.session_id,
            )
            parent = await store.load("10001", "main-session")
            child = await store.load("10001", entered.session_id)

            self.assertEqual(len(workflow.calls), 1)
            self.assertEqual(returned.session_id, "main-session")
            self.assertEqual(returned.action, ArbitrationAction.RETURN_TO_PARENT)
            assert parent is not None
            assert child is not None
            self.assertEqual(child.session_status, "closed")
            self.assertIsNone(parent.active_child_session_id)
            self.assertEqual(parent.history[-1].message_type, "child_handoff")
            self.assertEqual(
                parent.history[-1].related_session_id,
                entered.session_id,
            )


if __name__ == "__main__":
    unittest.main()
