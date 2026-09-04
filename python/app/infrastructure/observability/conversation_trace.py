"""按请求聚合并安全写入只含最终 Agent 输出的 JSON 日志。"""

from __future__ import annotations

import asyncio
import json
import os
import re
import threading
import time
import uuid
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from dataclasses import asdict, is_dataclass
from datetime import date, datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import structlog
from pydantic import BaseModel, SecretStr

from app.config.paths import LOG_ROOT

DEFAULT_CONVERSATION_TRACE_PATH = LOG_ROOT
_PERSISTED_EVENT_TYPES = {"agent.completed", "agent.failed"}
_TRACE_FILE_SUFFIXES = {".json", ".md"}
_AGENT_DISPLAY_NAMES = {
    "intent_recognition_agent": "IntentRecognitionAgent",
    "user_profile_agent": "UserProfileAgent",
    "document_recall_agent": "DocumentRecallAgent",
    "document_rerank_agent": "DocumentRerankAgent",
}
_REDACTED = "[REDACTED]"
_REDACTED_PATH = "[REDACTED_PATH]"
_SENSITIVE_KEYS = {
    "access_token",
    "api_key",
    "authorization",
    "cookie",
    "cwd",
    "file_path",
    "filename",
    "id_token",
    "password",
    "path",
    "proxy_authorization",
    "refresh_token",
    "secret",
    "set_cookie",
    "stack",
    "stack_trace",
    "token",
    "traceback",
}
_SENSITIVE_COMPACT_KEYS = {
    value.replace("_", "") for value in _SENSITIVE_KEYS
}
_INTERNAL_POSIX_PATH = re.compile(
    r"(?<![\w:/])/(?:mnt|home|root|private|tmp|var|usr|opt|srv|workspace)"
    r"(?:/[^\s,;:()\[\]{}<>\"']+)+"
)
_INTERNAL_WINDOWS_PATH = re.compile(
    r"(?<![\w])(?:[A-Za-z]:\\(?:[^\\\s,;:()\[\]{}<>\"']+\\)*"
    r"[^\\\s,;:()\[\]{}<>\"']+)"
)
_CURRENT_TRACE: ContextVar[ConversationTrace | None] = ContextVar(
    "conversation_trace",
    default=None,
)
_CURRENT_STREAM: ContextVar[ConversationStreamRecorder | None] = ContextVar(
    "conversation_stream",
    default=None,
)
logger = structlog.get_logger()


class ConversationStreamRecorder:
    """把现有业务埋点投影为可实时消费的安全阶段事件。"""

    _MAX_PROCESS_EVENTS = 200

    def __init__(self) -> None:
        self.trace_id = uuid.uuid4().hex
        self._started = time.perf_counter()
        self._loop = asyncio.get_running_loop()
        self._queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._events: list[dict[str, Any]] = []
        self._sequence = 0
        self._event_lock = threading.Lock()

    def emit(
        self,
        *,
        stage: str,
        component: str,
        status: str,
        title: str,
        summary: str = "",
        details: Mapping[str, Any] | None = None,
    ) -> None:
        """发布一条有界、已脱敏的过程事件。"""

        with self._event_lock:
            if len(self._events) >= self._MAX_PROCESS_EVENTS:
                return
            self._sequence += 1
            event = {
                "event": "process",
                "trace_id": self.trace_id,
                "sequence": self._sequence,
                "elapsed_ms": round(self.elapsed_ms(), 1),
                "stage": str(stage)[:80],
                "component": str(component)[:100],
                "status": _stream_status(status),
                "title": str(title)[:120],
                "summary": str(summary)[:500],
                "details": _bound_stream_value(_sanitize(dict(details or {}))),
            }
            self._events.append(event)
        self._loop.call_soon_threadsafe(self._queue.put_nowait, event)

    def record(
        self,
        event_type: str,
        component: str,
        *,
        input_data: Any | None,
        output_data: Any | None,
        status: str | None,
        error: BaseException | Mapping[str, Any] | None,
    ) -> None:
        """把组件埋点转换为白名单阶段摘要，忽略原始 Prompt 与响应。"""

        projection = _stream_projection(
            event_type,
            component,
            input_data=input_data,
            output_data=output_data,
            status=status,
            error=error,
        )
        if projection is not None:
            self.emit(component=component, **projection)

    def terminal(self, event: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        """为 result 或 error 生成连续序号并发布终止事件。"""

        with self._event_lock:
            self._sequence += 1
            terminal = {
                "event": event,
                "trace_id": self.trace_id,
                "sequence": self._sequence,
                "elapsed_ms": round(self.elapsed_ms(), 1),
                **dict(payload),
            }
        self._loop.call_soon_threadsafe(self._queue.put_nowait, terminal)
        return terminal

    async def next_event(self) -> dict[str, Any]:
        """等待下一条过程或终止事件。"""

        return await self._queue.get()

    def snapshot(self) -> tuple[dict[str, Any], ...]:
        """返回当前已发布过程事件的安全副本。"""

        with self._event_lock:
            return tuple(_sanitize(event) for event in self._events)

    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self._started) * 1000


class ConversationTrace:
    """保存一次聊天请求的有序事件和最终结果。"""

    def __init__(
        self,
        *,
        user_id: str,
        message: str,
        supplied_session_id: str | None,
    ) -> None:
        self.trace_id = uuid.uuid4().hex
        self.started_at = datetime.now(timezone.utc)
        _ = user_id, message, supplied_session_id
        self.resolved_session_id: str | None = None
        self.status = "in_progress"
        self.events: list[dict[str, Any]] = []
        self._sequence = 0
        self._event_lock = threading.Lock()

    def record(
        self,
        event_type: str,
        component: str,
        *,
        input_data: Any | None = None,
        output_data: Any | None = None,
        status: str | None = None,
        error: BaseException | Mapping[str, Any] | None = None,
    ) -> None:
        """只添加 Agent 终止事件；输入和 LLM 事件始终丢弃。"""

        normalized_event_type = str(event_type)
        if normalized_event_type not in _PERSISTED_EVENT_TYPES:
            return

        try:
            event: dict[str, Any] = {
                "sequence": 0,
                "component": str(component),
                "status": str(status or _event_default_status(normalized_event_type)),
                "output": _sanitize(output_data),
            }
            if error is not None:
                event["error"] = _safe_error(error)
            with self._event_lock:
                self._sequence += 1
                event["sequence"] = self._sequence
                self.events.append(event)
        except Exception as exc:
            logger.warning(
                "对话追踪事件记录失败",
                exception_type=type(exc).__name__,
            )

    def set_resolved_session_id(self, session_id: str) -> None:
        """保存服务端最终采用的会话 ID。"""

        self.resolved_session_id = str(session_id)

    def set_response(self, response: Any) -> None:
        """标记业务成功；公开响应不写入 Agent 日志。"""

        _ = response
        self.status = "success"

    def mark_error(self, exc: BaseException) -> None:
        """只记录异常类型，不保存异常正文、堆栈或内部路径。"""

        self.status = "error"
        _ = exc

    def document(self) -> dict[str, Any]:
        """生成只包含 Agent 最终输出的安全内存快照。"""

        completed_at = datetime.now(timezone.utc)
        if self.status == "in_progress":
            self.status = "success"
        return {
            "schema_version": 2,
            "trace_id": self.trace_id,
            "started_at": _datetime_text(self.started_at),
            "completed_at": _datetime_text(completed_at),
            "status": _final_trace_status(self),
            "session_id": self.resolved_session_id,
            "agents": [
                {
                    "sequence": event["sequence"],
                    "agent": _agent_display_name(str(event["component"])),
                    "status": event["status"],
                    "output": event["output"],
                    **(
                        {"error": event["error"]}
                        if "error" in event
                        else {}
                    ),
                }
                for event in self.events
            ],
        }


class ConversationTraceWriter:
    """为每个 HTTP 聊天请求生成一份只含 Agent 最终输出的 JSON 日志。"""

    def __init__(
        self,
        path: str | Path = DEFAULT_CONVERSATION_TRACE_PATH,
        *,
        retention_days: int = 7,
        max_files: int = 500,
    ) -> None:
        if retention_days < 1:
            raise ValueError("retention_days 不能小于 1")
        if max_files < 1:
            raise ValueError("max_files 不能小于 1")
        self.path = Path(path)
        self.retention_days = retention_days
        self.max_files = max_files
        self._write_lock = threading.Lock()

    @asynccontextmanager
    async def trace_request(
        self,
        *,
        user_id: str,
        message: str,
        supplied_session_id: str | None,
    ) -> AsyncIterator[ConversationTrace]:
        """建立请求级上下文，并在退出边界时安全写入 Agent 日志。"""

        trace = ConversationTrace(
            user_id=user_id,
            message=message,
            supplied_session_id=supplied_session_id,
        )
        token = _CURRENT_TRACE.set(trace)
        try:
            yield trace
        except BaseException as exc:
            trace.mark_error(exc)
            raise
        finally:
            _CURRENT_TRACE.reset(token)
            await self._try_append(trace)

    async def _try_append(self, trace: ConversationTrace) -> None:
        if not trace.events:
            return
        try:
            payload = json.dumps(
                trace.document(),
                ensure_ascii=False,
                indent=2,
            ) + "\n"
            await asyncio.to_thread(self._write_sync, trace, payload)
        except Exception as exc:
            logger.warning(
                "对话追踪日志写入失败",
                exception_type=type(exc).__name__,
            )
            return
        try:
            await asyncio.to_thread(self._cleanup_sync)
        except Exception as exc:
            logger.warning(
                "对话追踪日志清理失败",
                exception_type=type(exc).__name__,
            )

    def _write_sync(self, trace: ConversationTrace, document: str) -> None:
        date_directory = self.path / trace.started_at.strftime("%Y-%m-%d")
        filename = (
            trace.started_at.strftime("%H-%M-%S-%f")
            + f"_{trace.trace_id}.json"
        )
        target = date_directory / filename
        payload = document.encode("utf-8")
        with self._write_lock:
            self.path.mkdir(mode=0o700, parents=True, exist_ok=True)
            os.chmod(self.path, 0o700)
            date_directory.mkdir(mode=0o700, parents=True, exist_ok=True)
            os.chmod(date_directory, 0o700)
            descriptor = os.open(
                target,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                0o600,
            )
            try:
                os.fchmod(descriptor, 0o600)
                offset = 0
                while offset < len(payload):
                    offset += os.write(descriptor, payload[offset:])
                os.fsync(descriptor)
            finally:
                os.close(descriptor)

    def _cleanup_sync(self) -> None:
        """清理受控日期目录中过期或超量的现行及旧版追踪日志。"""

        cutoff = time.time() - self.retention_days * 86400
        date_pattern = re.compile(r"\d{4}-\d{2}-\d{2}")
        with self._write_lock:
            if not self.path.is_dir():
                return
            files: list[tuple[int, Path]] = []
            date_directories: list[Path] = []
            for directory in self.path.iterdir():
                if (
                    not directory.is_dir()
                    or date_pattern.fullmatch(directory.name) is None
                ):
                    continue
                date_directories.append(directory)
                for target in directory.iterdir():
                    if (
                        not target.is_file()
                        or target.suffix not in _TRACE_FILE_SUFFIXES
                    ):
                        continue
                    stat_result = target.stat()
                    if stat_result.st_mtime < cutoff:
                        target.unlink(missing_ok=True)
                        continue
                    files.append((stat_result.st_mtime_ns, target))

            files.sort(key=lambda item: (item[0], str(item[1])), reverse=True)
            for _, target in files[self.max_files :]:
                target.unlink(missing_ok=True)

            for directory in date_directories:
                try:
                    directory.rmdir()
                except OSError:
                    pass


def _event_default_status(event_type: str) -> str:
    return "failed" if event_type.endswith(".failed") else "success"


def _final_trace_status(trace: ConversationTrace) -> str:
    if trace.status == "error":
        return "error"
    degraded_statuses = {"degraded", "failed", "error", "invalid_response"}
    if any(str(event.get("status")) in degraded_statuses for event in trace.events):
        return "degraded"
    return "success"


def _agent_display_name(component: str) -> str:
    if component in _AGENT_DISPLAY_NAMES:
        return _AGENT_DISPLAY_NAMES[component]
    return "".join(part.capitalize() for part in component.split("_") if part)


def current_conversation_trace() -> ConversationTrace | None:
    """返回当前异步请求的追踪对象；请求外返回 ``None``。"""

    return _CURRENT_TRACE.get()


def current_conversation_stream() -> ConversationStreamRecorder | None:
    """返回当前流式聊天的事件记录器；普通请求返回 ``None``。"""

    return _CURRENT_STREAM.get()


@contextmanager
def conversation_stream_context(
    recorder: ConversationStreamRecorder,
):
    """在当前异步任务树中绑定流式事件记录器。"""

    token = _CURRENT_STREAM.set(recorder)
    try:
        yield recorder
    finally:
        _CURRENT_STREAM.reset(token)


def emit_stream_event(
    *,
    stage: str,
    component: str,
    status: str,
    title: str,
    summary: str = "",
    details: Mapping[str, Any] | None = None,
) -> None:
    """从业务代码发布一条安全流式事件；非流式请求为空操作。"""

    recorder = current_conversation_stream()
    if recorder is None:
        return
    recorder.emit(
        stage=stage,
        component=component,
        status=status,
        title=title,
        summary=summary,
        details=details,
    )


def record_trace_event(
    event_type: str,
    component: str,
    *,
    input_data: Any | None = None,
    output_data: Any | None = None,
    status: str | None = None,
    error: BaseException | Mapping[str, Any] | None = None,
) -> None:
    """同时投递流式安全事件，并按旧契约保存 Agent 终止输出。"""

    trace = current_conversation_trace()
    if trace is not None:
        trace.record(
            event_type,
            component,
            input_data=input_data,
            output_data=output_data,
            status=status,
            error=error,
        )
    stream = current_conversation_stream()
    if stream is not None:
        stream.record(
            event_type,
            component,
            input_data=input_data,
            output_data=output_data,
            status=status,
            error=error,
        )


def _stream_status(value: str | None) -> str:
    normalized = str(value or "success").casefold()
    if normalized in {"started", "success", "degraded", "failed", "skipped"}:
        return normalized
    if normalized in {"error", "invalid_response"}:
        return "failed"
    return "success"


def _stream_projection(
    event_type: str,
    component: str,
    *,
    input_data: Any | None,
    output_data: Any | None,
    status: str | None,
    error: BaseException | Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """把内部埋点转换为不含 Prompt、原始响应和完整历史的页面摘要。"""

    normalized_type = str(event_type)
    phase = (
        "started"
        if normalized_type.endswith((".started", "_started"))
        else "failed"
        if normalized_type.endswith((".failed", "_failed"))
        else _stream_status(status)
    )
    if normalized_type.startswith("llm."):
        inputs = input_data if isinstance(input_data, Mapping) else {}
        outputs = output_data if isinstance(output_data, Mapping) else {}
        safe_fields = (
            "schema",
            "provider",
            "model_role",
            "model_name",
            "from_model_role",
            "from_model_name",
            "to_model_role",
            "to_model_name",
            "reason",
            "failure_kind",
            "latency_ms",
        )
        details = {
            key: outputs.get(key, inputs.get(key))
            for key in safe_fields
            if outputs.get(key, inputs.get(key)) not in (None, "")
        }
        schema = str(details.get("schema") or "")[:100]
        is_upgrade = normalized_type.startswith("llm.upgrade_")
        if is_upgrade:
            title = (
                "辅助大模型升级开始"
                if phase == "started"
                else "辅助大模型升级失败"
                if phase == "failed"
                else "辅助大模型升级完成"
            )
            summary = " → ".join(
                value
                for value in (
                    str(details.get("from_model_name") or ""),
                    str(
                        details.get("to_model_name")
                        or details.get("model_name")
                        or ""
                    ),
                )
                if value
            ) or "受控模型升级"
        else:
            title = (
                "结构化模型调用开始"
                if phase == "started"
                else "结构化模型调用失败"
                if phase == "failed"
                else "结构化模型调用完成"
            )
            summary = schema or str(details.get("model_name") or "结构化模型调用")
        return {
            "stage": "模型调用",
            "status": phase,
            "title": title,
            "summary": summary,
            "details": _bound_stream_value(_sanitize(details)),
        }

    protected_input = _sanitize(input_data)
    protected_output = _sanitize(output_data)
    inputs = protected_input if isinstance(protected_input, Mapping) else {}
    outputs = protected_output if isinstance(protected_output, Mapping) else {}
    definitions: dict[str, tuple[str, str]] = {
        "conversation_service": ("会话", "会话服务"),
        "conversation_graph": ("编排", "会话编排"),
        "intent_recognition_agent": ("问题分析", "意图识别"),
        "conversation_arbitrator": ("路由", "业务路由仲裁"),
        "user_profile_agent": ("用户信息", "用户画像"),
        "document_recall_agent": ("召回", "推荐 Chunk 召回"),
        "document_rerank_agent": ("重排", "推荐文档重排"),
        "document_result_aggregator": ("聚合", "推荐结果聚合"),
        "knowledge_qa": ("知识问答", "知识问答链"),
    }
    stage, base_title = definitions.get(
        component,
        ("业务执行", "业务组件"),
    )
    title = (
        f"{base_title}开始"
        if phase == "started"
        else f"{base_title}失败"
        if phase == "failed"
        else f"{base_title}完成"
    )
    details: dict[str, Any] = {}
    summary = ""
    if component == "intent_recognition_agent":
        if phase == "started":
            history = inputs.get("history")
            details = {
                "history_message_count": (
                    len(history) if isinstance(history, Sequence) else 0
                ),
                "has_conversation_summary": bool(
                    inputs.get("conversation_summary")
                ),
                "current_intent_state": inputs.get("intent_state"),
            }
            summary = "正在判断推荐、知识问答或澄清路由"
        else:
            recognized = (
                outputs.get("validated_output")
                if isinstance(outputs.get("validated_output"), Mapping)
                else outputs
            )
            details = {
                key: recognized.get(key)
                for key in (
                    "intent",
                    "source",
                    "relation",
                    "confidence",
                    "rewritten_query",
                )
                if recognized.get(key) is not None
            }
            summary = str(details.get("intent") or phase)
    elif component == "conversation_arbitrator":
        decision = outputs
        details = {
            key: decision.get(key)
            for key in ("action", "reason", "context", "clarification_question")
            if decision.get(key) is not None
        }
        summary = str(details.get("action") or phase)
    elif component == "user_profile_agent":
        details = {
            key: outputs.get(key)
            for key in (
                "profile_available",
                "profile_status",
                "profile_confidence",
                "agent_status",
                "core_topics",
                "negative_topics",
                "preferred_difficulty",
                "preferred_reading_length",
                "preferred_content_types",
                "activity_level",
            )
            if outputs.get(key) is not None
        }
        summary = (
            "画像可用" if details.get("profile_available") else "未使用画像"
        )
    elif component == "document_recall_agent":
        if phase == "started":
            details = {
                key: inputs.get(key)
                for key in ("query", "size", "seen_document_count")
                if inputs.get(key) is not None
            }
            summary = str(details.get("query") or "开始共享 Chunk 检索")
        else:
            candidates = outputs.get("candidates", [])
            if not isinstance(candidates, Sequence) or isinstance(candidates, str):
                candidates = []
            details = {
                "retrieval_mode": outputs.get("retrieval_mode"),
                "retrieval_diagnostics": outputs.get("retrieval_diagnostics"),
                "candidate_count": len(candidates),
                "candidates": list(candidates)[:20],
            }
            summary = f"召回 {len(candidates)} 个文档候选"
    elif component == "document_rerank_agent" and phase != "started":
        scores = outputs.get("document_scores", [])
        details = {
            key: outputs.get(key)
            for key in (
                "llm_status",
                "profile_status",
                "profile_confidence",
                "blend_weights",
            )
            if outputs.get(key) is not None
        }
        details["document_scores"] = scores if isinstance(scores, list) else []
        summary = f"重排 {len(details['document_scores'])} 个文档"
    elif component == "document_result_aggregator" and phase != "started":
        documents = outputs.get("documents", [])
        details = {
            "documents": documents if isinstance(documents, list) else [],
            "kept_document_ids": outputs.get("kept_document_ids", []),
            "discarded_document_ids": outputs.get("discarded_document_ids", []),
        }
        summary = f"保留 {len(details['documents'])} 篇推荐"
    elif component == "knowledge_qa" and phase != "started":
        result = outputs
        details = {
            "status": result.get("status"),
            "degraded_components": result.get("degraded_components", []),
        }
        summary = str(result.get("status") or phase)
    elif component == "conversation_service":
        if normalized_type == "session.resolved":
            title = "Session 已确定"
            details = {"session_id": outputs.get("session_id")}
        elif normalized_type == "session.loaded":
            title = "Session 与历史已加载"
            session = outputs.get("session")
            if isinstance(session, Mapping):
                details = {
                    "session_id": session.get("session_id"),
                    "session_type": session.get("session_type"),
                    "intent_state": session.get("intent_state"),
                    "history_message_count": len(session.get("history") or []),
                    "has_summary": bool(session.get("summary")),
                }
        elif normalized_type == "service.completed":
            title = "会话结果已提交"
            reply = outputs.get("reply")
            if isinstance(reply, Mapping):
                details = {
                    "session_id": reply.get("session_id"),
                    "action": reply.get("action"),
                    "intent_state": reply.get("intent_state"),
                    "recommendation_count": len(reply.get("recommendations") or []),
                    "citation_count": len(reply.get("citations") or []),
                    "image_count": len(reply.get("images") or []),
                }
                summary = str(details.get("action") or "完成")
    elif component == "conversation_graph" and phase != "started":
        details = {
            key: outputs.get(key)
            for key in (
                "commit_context",
                "pending_intent_state",
                "commit_intent_state",
                "knowledge_document_ids",
                "error_stage",
                "trace",
            )
            if outputs.get(key) is not None
        }
        summary = "会话路由和业务分支已汇合"
    if error is not None:
        details["error_type"] = (
            type(error).__name__
            if isinstance(error, BaseException)
            else str(error.get("type") or "UnknownError")
        )
    return {
        "stage": stage,
        "status": phase,
        "title": title,
        "summary": summary,
        "details": details,
    }


def _bound_stream_value(value: Any, *, depth: int = 0) -> Any:
    """限制单条流式事件体积，同时保留调试所需结构。"""

    if depth >= 6:
        return "[内容层级已截断]"
    if isinstance(value, str):
        return value[:1200]
    if isinstance(value, Mapping):
        return {
            str(key)[:100]: _bound_stream_value(item, depth=depth + 1)
            for key, item in list(value.items())[:30]
        }
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return [
            _bound_stream_value(item, depth=depth + 1)
            for item in list(value)[:20]
        ]
    return value


def _sanitize(value: Any, *, field_name: str | None = None) -> Any:
    if field_name is not None and _is_sensitive_key(field_name):
        return _REDACTED
    if isinstance(value, SecretStr):
        return _REDACTED
    if isinstance(value, Path):
        return _REDACTED_PATH
    if isinstance(value, BaseException):
        return {"type": type(value).__name__}
    if isinstance(value, BaseModel):
        return _sanitize(value.model_dump(mode="json"))
    if is_dataclass(value) and not isinstance(value, type):
        return _sanitize(asdict(value))
    if isinstance(value, Enum):
        return _sanitize(value.value)
    if isinstance(value, datetime):
        return _datetime_text(value)
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {
            str(key): _sanitize(item, field_name=str(key))
            for key, item in value.items()
        }
    if isinstance(value, set | frozenset):
        return [_sanitize(item) for item in sorted(value, key=str)]
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return [_sanitize(item) for item in value]
    if isinstance(value, bytes | bytearray):
        return {"type": type(value).__name__, "length": len(value)}
    if isinstance(value, str):
        return _redact_internal_paths(value)
    if value is None or isinstance(value, int | float | bool):
        return value
    return {"type": type(value).__name__}


def _safe_error(error: BaseException | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(error, BaseException):
        return {"type": type(error).__name__}
    sanitized = _sanitize(error)
    return sanitized if isinstance(sanitized, dict) else {"type": "UnknownError"}


def _is_sensitive_key(value: str) -> bool:
    normalized = value.strip().casefold().replace("-", "_")
    compact = "".join(character for character in normalized if character.isalnum())
    return (
        normalized in _SENSITIVE_KEYS
        or compact in _SENSITIVE_COMPACT_KEYS
        or normalized.endswith("_api_key")
        or normalized.endswith("_access_token")
        or normalized.endswith("_refresh_token")
        or normalized.endswith("_token")
        or normalized.endswith("_password")
        or normalized.endswith("_secret")
        or compact.endswith("apikey")
        or compact.endswith("accesstoken")
        or compact.endswith("refreshtoken")
        or compact.endswith("token")
        or compact.endswith("password")
        or compact.endswith("secret")
    )


def _redact_internal_paths(value: str) -> str:
    sanitized = _INTERNAL_POSIX_PATH.sub(_REDACTED_PATH, value)
    return _INTERNAL_WINDOWS_PATH.sub(_REDACTED_PATH, sanitized)


def _datetime_text(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        return value.isoformat()
    normalized = value.astimezone(timezone.utc)
    return normalized.isoformat().replace("+00:00", "Z")


__all__ = [
    "ConversationTrace",
    "ConversationTraceWriter",
    "ConversationStreamRecorder",
    "DEFAULT_CONVERSATION_TRACE_PATH",
    "conversation_stream_context",
    "current_conversation_stream",
    "current_conversation_trace",
    "emit_stream_event",
    "record_trace_event",
]
