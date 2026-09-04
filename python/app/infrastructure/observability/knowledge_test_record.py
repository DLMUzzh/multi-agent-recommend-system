"""把知识问答安全执行摘要追加为运行期 Markdown 测试记录。"""

from __future__ import annotations

import asyncio
import json
import os
import re
import threading
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog

from app.config.paths import KNOWLEDGE_TEST_RECORD_PATH
from app.models.knowledge_qa import KnowledgeExecutionTrace
from app.models.schemas import ChatResponse, ChatStreamProcessEvent


logger = structlog.get_logger()
_SENSITIVE_KEY_PARTS = (
    "answer_draft",
    "authorization",
    "cookie",
    "embedding",
    "password",
    "prompt",
    "raw_output",
    "secret",
    "stack",
    "token",
    "traceback",
)
_INTERNAL_PATH = re.compile(
    r"(?<![\w:/])/(?:mnt|home|root|private|tmp|var|usr|opt|srv|workspace)"
    r"(?:/[^\s,;:()\[\]{}<>\"']+)+"
)


class KnowledgeTestRecordWriter:
    """并发安全地追加已通过严格 DTO 校验的测试摘要。"""

    def __init__(
        self,
        path: str | Path = KNOWLEDGE_TEST_RECORD_PATH,
    ) -> None:
        self.path = Path(path)
        self._write_lock = threading.Lock()

    async def append(self, trace: KnowledgeExecutionTrace) -> bool:
        """追加一轮 Markdown；失败只返回 False，不改变问答结果。"""

        try:
            protected = KnowledgeExecutionTrace.model_validate(trace).model_copy(
                deep=True
            )
            document = self._render(protected)
            await asyncio.to_thread(self._append_sync, document)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "知识问答测试记录写入失败",
                exception_type=type(exc).__name__,
            )
            return False
        return True

    async def append_stream(
        self,
        *,
        trace_id: str,
        events: Sequence[Mapping[str, Any]],
        response: ChatResponse,
    ) -> bool:
        """按事件顺序追加一轮完整聊天链路与最终公开结果。"""

        try:
            protected_events = tuple(
                ChatStreamProcessEvent.model_validate(event).model_copy(deep=True)
                for event in events
            )
            protected_response = ChatResponse.model_validate(response).model_copy(
                deep=True
            )
            document = self._render_stream(
                trace_id=str(trace_id),
                events=protected_events,
                response=protected_response,
            )
            await asyncio.to_thread(self._append_sync, document)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "聊天全链路测试记录写入失败",
                exception_type=type(exc).__name__,
            )
            return False
        return True

    def _append_sync(self, document: str) -> None:
        payload = document.encode("utf-8")
        with self._write_lock:
            self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            os.chmod(self.path.parent, 0o700)
            descriptor = os.open(
                self.path,
                os.O_CREAT | os.O_APPEND | os.O_WRONLY,
                0o600,
            )
            try:
                os.write(descriptor, payload)
            finally:
                os.close(descriptor)

    @classmethod
    def _render(cls, trace: KnowledgeExecutionTrace) -> str:
        timestamp = datetime.now(timezone.utc).astimezone().isoformat(
            timespec="seconds"
        )
        requested_documents = ", ".join(trace.input.requested_document_ids) or "无"
        sub_queries = "；".join(cls._line(query) for query in trace.sub_queries) or "无"
        search_queries = (
            "；".join(cls._line(query) for query in trace.search_queries) or "无"
        )
        chunks = "\n".join(
            "- "
            f"#{item.rank} · {cls._line(item.title)} · {item.document_id} · "
            f"{item.chunk_id}\n"
            f"  - 分数 {item.score:.4f} · "
            f"{'进入最终证据' if item.selected else '未选'}\n"
            f"  - 章节："
            f"{' > '.join(cls._line(part) for part in item.heading_path) or '正文'}\n"
            f"  - 摘录：{cls._line(item.excerpt)}"
            for item in trace.retrieved_chunks
        ) or "无"
        documents = "\n".join(
            "- "
            + cls._line(item.title)
            + f" ({item.document_id}) · 文档分 {item.score:.4f}\n"
            + "  - 召回 Chunk："
            + (", ".join(item.retrieved_chunk_ids) or "无")
            + "\n  - 最终证据 Chunk："
            + (", ".join(item.selected_chunk_ids) or "无")
            for item in trace.documents
        ) or "- 无"
        plan_and_coverage = cls._render_plan_and_coverage(trace)
        degraded = ", ".join(trace.result.degraded_components) or "无"
        return (
            f"## {timestamp} · {trace.trace_id}\n"
            "### 1. 发送信息\n"
            f"- 用户问题：{cls._line(trace.question)}\n"
            f"- 历史消息：{trace.input.history_message_count} 条\n"
            f"- 会话摘要：{'已携带' if trace.input.has_conversation_summary else '未携带'}\n"
            f"- 上游准备查询：{'已使用' if trace.input.prepared_query else '未使用'}\n"
            f"- 限定文档：{requested_documents}\n\n"
            "### 2. 路由与问题分析\n"
            f"- 路由：{trace.route} · {trace.request_route}\n"
            f"- 问题类型：{trace.question_type}\n"
            f"- 检索策略：{trace.strategy}\n"
            f"- 独立查询：{cls._line(trace.standalone_query)}\n"
            f"- 历史参与：{'是' if trace.uses_history else '否'}\n"
            f"- 置信度：{trace.confidence:.2f}\n"
            f"- 子查询：{sub_queries}\n\n"
            "### 3. 计划与覆盖\n"
            f"{plan_and_coverage}\n\n"
            "### 4. 实际检索查询\n"
            f"- {search_queries}\n\n"
            "### 5. 召回 Chunk\n"
            f"{chunks}\n\n"
            "### 6. 文档证据\n"
            f"{documents}\n\n"
            "### 7. 最终结果\n"
            f"- 结果：{trace.result.status} · {trace.result.citation_count} 个引用 · "
            f"{trace.result.image_count} 张图片 · {trace.result.elapsed_ms:.1f} ms\n"
            f"- 检索通道：BM25 {trace.diagnostics.bm25_status} · "
            f"Vector {trace.diagnostics.vector_status}\n"
            f"- 降级：{degraded}\n\n---\n\n"
        )

    @classmethod
    def _render_plan_and_coverage(
        cls,
        trace: KnowledgeExecutionTrace,
    ) -> str:
        """渲染严格计划 DTO 中的白名单字段；简单问题显示未启用。"""

        if (
            trace.reasoning_strategy is None
            and not trace.plan_steps
            and trace.coverage is None
        ):
            return "- 未启用"
        lines = [
            "- 推理策略："
            + cls._line(str(trace.reasoning_strategy or "未启用")),
            f"- 计划修订：{trace.plan_revision_count} 版",
        ]
        for step in trace.plan_steps:
            chunk_ids = (
                ", ".join(
                    cls._line(chunk_id)
                    for chunk_id in step.selected_chunk_ids
                )
                or "无"
            )
            lines.extend(
                (
                    f"- 第 {step.revision} 版 · {cls._line(step.step_id)} · "
                    f"维度 {cls._line(step.facet)} · "
                    f"查询：{cls._line(step.query)}",
                    "  - "
                    f"必需：{'是' if step.required else '否'} · "
                    f"状态：{cls._line(step.status)} · "
                    f"原因：{cls._line(step.reason_code)}",
                    f"  - Chunk：{chunk_ids}",
                )
            )
        coverage = trace.coverage
        if coverage is not None:
            lines.extend(
                (
                    "- 最终覆盖："
                    f"必需步骤 {coverage.covered_required_steps}/"
                    f"{coverage.required_steps} · "
                    f"全部覆盖 {coverage.covered_steps}/"
                    f"{len(coverage.step_results)}",
                    f"  - 覆盖率：{coverage.coverage_ratio:.4f}",
                    f"  - 已重规划：{'是' if coverage.replanned else '否'}",
                    f"  - 最终动作：{cls._line(coverage.decision)}",
                )
            )
        return "\n".join(lines)

    @classmethod
    def _render_stream(
        cls,
        *,
        trace_id: str,
        events: Sequence[ChatStreamProcessEvent],
        response: ChatResponse,
    ) -> str:
        """把白名单流事件渲染为紧凑、可按时间复盘的 Markdown。"""

        timestamp = datetime.now(timezone.utc).astimezone().isoformat(
            timespec="seconds"
        )
        event_lines: list[str] = []
        for event in events:
            summary = f" · {cls._line(event.summary)}" if event.summary else ""
            event_lines.append(
                f"- #{event.sequence} · {event.elapsed_ms:.1f} ms · "
                f"{cls._line(event.stage)} · {cls._line(event.component)} · "
                f"{event.status} · {cls._line(event.title)}{summary}"
            )
            details = cls._safe_stream_value(event.details)
            if details:
                event_lines.append(
                    "  - 详情："
                    + json.dumps(details, ensure_ascii=False, sort_keys=True)
                )
        reasoning_lines = cls._stream_reasoning_summary(events, response)
        degraded = ", ".join(response.degraded_components) or "无"
        return (
            f"## {timestamp} · {cls._line(trace_id)} · 全链路\n"
            "### 思考过程摘要\n"
            "> 由安全执行轨迹确定性生成，不是模型隐藏思维链。\n\n"
            + "\n".join(f"- {line}" for line in reasoning_lines)
            + "\n\n### 详细执行链路\n"
            + ("\n".join(event_lines) or "- 无过程事件")
            + "\n\n### 最终公开结果\n"
            f"- 最终结果：{response.action.value}\n"
            f"- Session：{cls._line(response.session_id)}\n"
            f"- 意图状态：{response.intent_state.value}\n"
            f"- 推荐：{len(response.recommendations)} 篇\n"
            f"- 引用：{len(response.citations)} 个\n"
            f"- 图片：{len(response.images)} 张\n"
            f"- 降级组件：{degraded}\n\n---\n\n"
        )

    @classmethod
    def _stream_reasoning_summary(
        cls,
        events: Sequence[ChatStreamProcessEvent],
        response: ChatResponse,
    ) -> tuple[str, ...]:
        """从白名单事件生成可复核摘要，不推断模型内部思考。"""

        details = [event.details for event in events]
        request = next(
            (
                item
                for item in details
                if isinstance(item.get("message"), str)
            ),
            {},
        )
        message = cls._line(str(request.get("message") or "当前请求"))
        history_count = next(
            (
                item.get("history_message_count")
                for item in details
                if isinstance(item.get("history_message_count"), int)
            ),
            0,
        )
        lines = [
            f"理解问题：已接收“{message}”，并加载 {history_count} 条历史消息。"
        ]

        analysis = next(
            (
                item
                for item in reversed(details)
                if item.get("question_type") or item.get("strategy")
            ),
            {},
        )
        intent = next(
            (
                item
                for item in reversed(details)
                if item.get("intent")
            ),
            {},
        )
        if analysis:
            question_type = cls._line(
                str(analysis.get("question_type") or "unknown")
            )
            strategy = cls._line(str(analysis.get("strategy") or "direct"))
            query = cls._line(
                str(analysis.get("standalone_query") or message)
            )
            lines.append(
                "判断路由：进入知识问答，"
                f"问题类型为 {question_type}，采用 {strategy} 策略，"
                f"检索查询为“{query}”。"
            )
        elif intent:
            lines.append(
                "判断路由：识别为 "
                f"{cls._line(str(intent.get('intent')))}，继续对应业务链。"
            )

        recalled_count = sum(
            event.title.startswith("召回 Chunk #") for event in events
        )
        if recalled_count == 0:
            recalled_count = sum(
                1
                for item in details
                if isinstance(item.get("chunk_id"), str)
            )
        if recalled_count:
            lines.append(
                f"召回候选：已召回 {recalled_count} 个 Chunk，"
                "保留精确 Chunk ID、来源文档和排序信息。"
            )

        selected = next(
            (
                item.get("selected_chunk_ids")
                for item in reversed(details)
                if isinstance(item.get("selected_chunk_ids"), Sequence)
                and not isinstance(item.get("selected_chunk_ids"), str)
            ),
            None,
        )
        documents = next(
            (
                item.get("documents")
                for item in reversed(details)
                if isinstance(item.get("documents"), Sequence)
                and not isinstance(item.get("documents"), str)
            ),
            None,
        )
        if selected is not None or documents is not None:
            selected_count = len(selected or ())
            document_count = len(documents or ())
            lines.append(
                f"核验证据：选中 {selected_count} 个证据 Chunk，"
                f"按 {document_count} 篇文档组织，回答仍可回溯到原 Chunk。"
            )

        image_items = next(
            (
                item.get("images")
                for item in reversed(details)
                if isinstance(item.get("images"), Sequence)
                and not isinstance(item.get("images"), str)
            ),
            None,
        )
        if image_items is not None or response.images:
            lines.append(
                "检查图片：核验图片与最终证据 Chunk 的关联关系，"
                f"保留 {len(image_items or response.images)} 张可展示图片。"
            )

        if response.action.value == "knowledge_answer":
            lines.append(
                "组织结果：形成知识回答，保留 "
                f"{len(response.citations)} 条精确 Chunk 引用，"
                "相同文档复用同一索引。"
            )
        else:
            lines.append(
                "组织结果：形成推荐结果，返回 "
                f"{len(response.recommendations)} 篇文档。"
            )
        return tuple(lines[:7])

    @classmethod
    def _safe_stream_value(cls, value: Any, *, depth: int = 0) -> Any:
        """再次限制记录详情，避免未来调用方绕过流式投影边界。"""

        if depth >= 6:
            return "[内容层级已截断]"
        if isinstance(value, Mapping):
            protected: dict[str, Any] = {}
            for key, item in list(value.items())[:30]:
                normalized_key = str(key)[:100]
                compact_key = normalized_key.casefold().replace("_", "")
                if any(
                    part.replace("_", "") in compact_key
                    for part in _SENSITIVE_KEY_PARTS
                ):
                    protected[normalized_key] = "[REDACTED]"
                else:
                    protected[normalized_key] = cls._safe_stream_value(
                        item,
                        depth=depth + 1,
                    )
            return protected
        if isinstance(value, Sequence) and not isinstance(
            value,
            (str, bytes, bytearray),
        ):
            return [
                cls._safe_stream_value(item, depth=depth + 1)
                for item in list(value)[:20]
            ]
        if isinstance(value, str):
            return _INTERNAL_PATH.sub("[REDACTED_PATH]", value)[:1200]
        if value is None or isinstance(value, int | float | bool):
            return value
        return str(type(value).__name__)

    @staticmethod
    def _line(value: str) -> str:
        """压平换行和控制空白，避免业务文本改变记录结构。"""

        return " ".join(str(value).split())


__all__ = ["KnowledgeTestRecordWriter"]
