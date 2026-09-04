"""共享 DeepSeek 客户端、请求级追踪与不含密钥的公开诊断。"""

from __future__ import annotations

import asyncio
import hashlib
import time
import threading
from collections.abc import Awaitable, Callable, Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Generic, Literal, TypeVar

import httpx
from langchain_core.messages import AIMessage, BaseMessage
from pydantic import BaseModel, SecretStr

from app.config import Settings, get_settings
from app.infrastructure.observability.conversation_trace import record_trace_event


DEEPSEEK_PROVIDER = "deepseek"
SchemaT = TypeVar("SchemaT", bound=BaseModel)
ResultT = TypeVar("ResultT")
LlmModelRole = Literal["small", "large"]
LlmFailureKind = Literal[
    "invalid_output",
    "low_confidence",
    "timeout",
    "rate_limited",
    "provider_error",
    "transport_error",
]
_UPGRADE_RESERVE_SECONDS = 12.0


@dataclass(slots=True)
class _LlmUpgradeState:
    """保存单个请求的辅助大模型升级额度和截止时间。"""

    deadline: float | None
    remaining_upgrades: int = 1


@dataclass(slots=True)
class _SharedHttpClient:
    """保存相同连接配置共享的异步 HTTP 客户端及引用数。"""

    client: httpx.AsyncClient
    reference_count: int = 1


_llm_upgrade_state: ContextVar[_LlmUpgradeState | None] = ContextVar(
    "llm_upgrade_state",
    default=None,
)
_shared_http_clients: dict[tuple[str, str, float], _SharedHttpClient] = {}
_shared_http_clients_lock = threading.Lock()


class LlmLowConfidenceError(ValueError):
    """表示结构合法但能力置信度不足，允许受控升级。"""


class _DeepSeekAsyncClient:
    """通过纯异步 HTTP 调用 DeepSeek Chat Completions。"""

    _MESSAGE_ROLES = {
        "system": "system",
        "human": "user",
        "ai": "assistant",
    }
    _RETRYABLE_STATUS_CODES = {408, 409, 429}

    def __init__(
        self,
        *,
        api_key: SecretStr,
        base_url: str,
        model: str,
        model_role: LlmModelRole,
        temperature: float,
        max_tokens: int,
        timeout: float,
        max_retries: int,
    ) -> None:
        self._model = model
        self._model_role = model_role
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._max_retries = max_retries
        self._http_client_key, self._http_client = _acquire_http_client(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
        )
        self._closed = False

    async def ainvoke(self, messages: Sequence[BaseMessage]) -> AIMessage:
        """返回与 LangChain 消息契约兼容的模型文本。"""

        content = await self._invoke_text(messages)
        return AIMessage(content=content)

    async def aclose(self) -> None:
        """关闭进程级共享异步 HTTP 客户端。"""

        if self._closed:
            return
        self._closed = True
        await _release_http_client(self._http_client_key)

    @property
    def model_name(self) -> str:
        """返回不含认证信息的模型名称，供安全诊断使用。"""

        return self._model

    @property
    def model_role(self) -> LlmModelRole:
        """返回当前客户端承担的小模型或大模型角色。"""

        return self._model_role

    async def _invoke_text(
        self,
        messages: Sequence[BaseMessage],
        *,
        json_mode: bool = False,
    ) -> str:
        payload = self._request_payload(messages, json_mode=json_mode)
        response_payload = await self._post_with_retries(payload)
        return self._response_content(response_payload)

    def _request_payload(
        self,
        messages: Sequence[BaseMessage],
        *,
        json_mode: bool,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [self._message_payload(message) for message in messages],
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        return payload

    @staticmethod
    def _response_content(response_payload: dict[str, Any]) -> str:
        choices = response_payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError("LLM 响应缺少候选结果")
        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            raise ValueError("LLM 响应候选结构无效")
        message = first_choice.get("message")
        if not isinstance(message, dict):
            raise ValueError("LLM 响应消息结构无效")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("LLM 响应缺少文本内容")
        return content.strip()

    async def _post_with_retries(self, payload: dict[str, Any]) -> dict[str, Any]:
        for attempt in range(self._max_retries + 1):
            try:
                response = await self._http_client.post(
                    "chat/completions",
                    json=payload,
                )
                retryable_status = (
                    response.status_code in self._RETRYABLE_STATUS_CODES
                    or response.status_code >= 500
                )
                if retryable_status and attempt < self._max_retries:
                    await asyncio.sleep(self._retry_delay(attempt))
                    continue
                response.raise_for_status()
                response_payload = response.json()
                if not isinstance(response_payload, dict):
                    raise ValueError("LLM 响应顶层必须是对象")
                return response_payload
            except (httpx.TimeoutException, httpx.TransportError):
                if attempt >= self._max_retries:
                    raise
                await asyncio.sleep(self._retry_delay(attempt))
        raise RuntimeError("LLM 请求重试状态无效")

    @classmethod
    def _message_payload(cls, message: BaseMessage) -> dict[str, str]:
        role = cls._MESSAGE_ROLES.get(message.type)
        if role is None:
            raise ValueError(f"不支持的 LLM 消息类型：{message.type}")
        if not isinstance(message.content, str):
            raise ValueError("LLM 消息内容必须是文本")
        return {"role": role, "content": message.content}

    @staticmethod
    def _retry_delay(attempt: int) -> float:
        return min(0.5 * (2**attempt), 4.0)


class _DeepSeekStructuredClient(Generic[SchemaT]):
    """在当前事件循环内校验 DeepSeek JSON mode 输出。"""

    def __init__(
        self,
        chat_model: _DeepSeekAsyncClient,
        schema: type[SchemaT],
    ) -> None:
        self._chat_model = chat_model
        self._schema = schema

    async def ainvoke(self, messages: Sequence[BaseMessage]) -> SchemaT:
        """调用 JSON mode，并返回经过 Pydantic 校验的结构化结果。"""

        started_at = time.perf_counter()
        request_payload = self._chat_model._request_payload(
            messages,
            json_mode=True,
        )
        component = self._schema.__module__.rsplit(".", 1)[-1]
        trace_metadata = {
            "schema": self._schema.__name__,
            "provider": DEEPSEEK_PROVIDER,
            "model_role": self._chat_model.model_role,
            "model_name": self._chat_model.model_name,
        }
        record_trace_event(
            "llm.started",
            component,
            input_data=trace_metadata,
        )
        response_payload: dict[str, Any] | None = None
        try:
            response_payload = await self._chat_model._post_with_retries(
                request_payload
            )
            content = self._chat_model._response_content(response_payload)
            if content.startswith("```"):
                content = content.split("\n", 1)[1].rsplit("```", 1)[0]
            output = self._schema.model_validate_json(content)
        except asyncio.CancelledError:
            record_trace_event(
                "llm.failed",
                component,
                output_data={
                    **trace_metadata,
                    "failure_kind": "cancelled",
                    "latency_ms": _elapsed_ms(started_at),
                },
                status="error",
            )
            raise
        except Exception as exc:
            record_trace_event(
                "llm.failed",
                component,
                output_data={
                    **trace_metadata,
                    "failure_kind": llm_failure_kind(exc),
                    "latency_ms": _elapsed_ms(started_at),
                },
                status=(
                    "invalid_response" if response_payload is not None else "error"
                ),
            )
            raise
        record_trace_event(
            "llm.completed",
            component,
            output_data={
                **trace_metadata,
                "latency_ms": _elapsed_ms(started_at),
            },
            status="success",
        )
        return output

    @property
    def model_name(self) -> str:
        """透传底层非敏感模型名，供受控升级与安全 Trace 使用。"""

        return self._chat_model.model_name

    @property
    def model_role(self) -> LlmModelRole:
        """透传底层模型角色，不暴露认证或请求内容。"""

        return self._chat_model.model_role

    async def aclose(self) -> None:
        """关闭底层共享异步 HTTP 客户端。"""

        await self._chat_model.aclose()


def is_llm_configured(settings: Settings | None = None) -> bool:
    """判断是否配置了非空 API Key，且不暴露其内容。"""

    current = settings or get_settings()
    return bool(_secret_value(current.llm_api_key).strip())


def public_llm_config(settings: Settings | None = None) -> dict[str, Any]:
    """返回适合终端诊断的非敏感配置。"""

    current = settings or get_settings()
    return {
        "provider": current.llm_provider,
        "model": current.llm_model,
        "models": {
            "small": llm_model_name(current, "small"),
            "large": llm_model_name(current, "large"),
        },
        "base_url": current.llm_base_url,
        "configured": is_llm_configured(current),
        "request_timeout_seconds": current.llm_request_timeout_seconds,
        "max_retries": current.llm_max_retries,
    }


def create_chat_llm(
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    enable_llm: bool | None = None,
    settings: Settings | None = None,
    model_factory: Callable[..., Any] | None = None,
    model_role: LlmModelRole = "large",
) -> Any | None:
    """创建纯异步 DeepSeek 客户端，未配置时返回 ``None``。

    ``enable_llm=False`` 始终禁用模型。默认或显式启用时，缺少 Key 会返回
    ``None``，使终端和自动测试继续使用确定性降级。
    """

    current = settings or get_settings()
    if enable_llm is False or not is_llm_configured(current):
        return None

    provider = current.llm_provider.strip().casefold()
    if provider != DEEPSEEK_PROVIDER:
        raise ValueError(f"不支持的 LLM 提供方：{current.llm_provider}")

    output_limit = max_tokens if max_tokens is not None else current.llm_max_tokens
    effective_temperature = (
        current.llm_temperature if temperature is None else temperature
    )
    model_name = llm_model_name(current, model_role)
    if model_factory is not None:
        return model_factory(
            api_key=current.llm_api_key,
            base_url=current.llm_base_url.rstrip("/"),
            model=model_name,
            temperature=effective_temperature,
            timeout=current.llm_request_timeout_seconds,
            max_retries=current.llm_max_retries,
            use_responses_api=False,
            extra_body={"max_tokens": output_limit},
        )
    return _DeepSeekAsyncClient(
        api_key=current.llm_api_key,
        base_url=current.llm_base_url,
        model=model_name,
        model_role=model_role,
        temperature=effective_temperature,
        max_tokens=output_limit,
        timeout=current.llm_request_timeout_seconds,
        max_retries=current.llm_max_retries,
    )


def create_structured_llm(
    schema: type[BaseModel],
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    enable_llm: bool | None = None,
    settings: Settings | None = None,
    model_factory: Callable[..., Any] | None = None,
    model_role: LlmModelRole = "large",
) -> Any | None:
    """创建使用 ``schema`` 校验 JSON mode 输出的 DeepSeek 客户端。"""

    chat_model = create_chat_llm(
        temperature=temperature,
        max_tokens=max_tokens,
        enable_llm=enable_llm,
        settings=settings,
        model_factory=model_factory,
        model_role=model_role,
    )
    if chat_model is None:
        return None
    if isinstance(chat_model, _DeepSeekAsyncClient):
        return _DeepSeekStructuredClient(chat_model, schema)
    return chat_model.with_structured_output(schema, method="json_mode")


def create_controlled_structured_llms(
    schema: type[BaseModel],
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    enable_llm: bool | None = None,
    settings: Settings | None = None,
    model_factory: Callable[..., Any] | None = None,
) -> tuple[Any | None, Any | None]:
    """创建小模型主客户端和仅在模型名不同才启用的大模型升级客户端。"""

    current = settings or get_settings()
    small_llm = create_structured_llm(
        schema,
        temperature=temperature,
        max_tokens=max_tokens,
        enable_llm=enable_llm,
        settings=current,
        model_factory=model_factory,
        model_role="small",
    )
    if llm_model_name(current, "large") == llm_model_name(current, "small"):
        return small_llm, None
    large_llm = create_structured_llm(
        schema,
        temperature=temperature,
        max_tokens=max_tokens,
        enable_llm=enable_llm,
        settings=current,
        model_factory=model_factory,
        model_role="large",
    )
    return small_llm, large_llm


def safe_llm_error(exc: Exception) -> str:
    """只返回异常类型，不回显请求详情或密钥。"""

    return type(exc).__name__


def llm_model_name(
    settings: Settings,
    model_role: LlmModelRole,
) -> str:
    """按角色返回模型名；空角色配置兼容回退到原统一模型。"""

    if model_role == "small":
        configured = settings.llm_small_model
    elif model_role == "large":
        configured = settings.llm_large_model
    else:
        raise ValueError("LLM 模型角色无效")
    resolved = configured.strip() or settings.llm_model.strip()
    if not resolved:
        raise ValueError("LLM 模型名称不能为空")
    return resolved


@contextmanager
def llm_upgrade_scope(
    *,
    deadline: float | None,
    max_upgrades: int = 1,
) -> Iterator[None]:
    """为当前请求建立最多一次辅助大模型升级额度。"""

    if max_upgrades < 0:
        raise ValueError("LLM 升级额度不能小于零")
    current = _llm_upgrade_state.get()
    if current is not None:
        previous_deadline = current.deadline
        if deadline is not None and (
            previous_deadline is None or deadline < previous_deadline
        ):
            current.deadline = deadline
        try:
            yield
        finally:
            current.deadline = previous_deadline
        return
    token = _llm_upgrade_state.set(
        _LlmUpgradeState(
            deadline=deadline,
            remaining_upgrades=max_upgrades,
        )
    )
    try:
        yield
    finally:
        _llm_upgrade_state.reset(token)


async def invoke_with_controlled_upgrade(
    *,
    stage: str,
    small_llm: Any,
    large_llm: Any,
    operation: Callable[[Any, LlmModelRole], Awaitable[ResultT]],
) -> ResultT:
    """先调用小模型，只在能力型失败且额度、预算允许时升级一次。"""

    try:
        return await operation(small_llm, "small")
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        failure_kind = llm_failure_kind(exc)
        state = _llm_upgrade_state.get()
        if (
            failure_kind not in {"invalid_output", "low_confidence"}
            or large_llm is None
            or state is None
            or state.remaining_upgrades < 1
            or not _has_upgrade_budget(state.deadline)
        ):
            raise
        state.remaining_upgrades -= 1
        upgrade_started_at = time.perf_counter()
        record_trace_event(
            "llm.upgrade_started",
            stage,
            input_data={
                "from_model_role": "small",
                "to_model_role": "large",
                "from_model_name": _safe_model_name(small_llm),
                "to_model_name": _safe_model_name(large_llm),
                "reason": failure_kind,
            },
        )
        try:
            timeout = _upgrade_timeout(state.deadline)
            if timeout is None:
                result = await operation(large_llm, "large")
            else:
                async with asyncio.timeout(timeout):
                    result = await operation(large_llm, "large")
        except asyncio.CancelledError:
            raise
        except Exception as upgrade_exc:
            record_trace_event(
                "llm.upgrade_failed",
                stage,
                status="error",
                output_data={
                    "model_role": "large",
                    "model_name": _safe_model_name(large_llm),
                    "reason": failure_kind,
                    "failure_kind": llm_failure_kind(upgrade_exc),
                    "latency_ms": _elapsed_ms(upgrade_started_at),
                },
            )
            raise
        record_trace_event(
            "llm.upgrade_completed",
            stage,
            output_data={
                "model_role": "large",
                "model_name": _safe_model_name(large_llm),
                "reason": failure_kind,
                "latency_ms": _elapsed_ms(upgrade_started_at),
            },
            status="success",
        )
        return result


def llm_failure_kind(exc: BaseException) -> LlmFailureKind:
    """把模型失败归一为不含响应正文的安全类别。"""

    if isinstance(exc, LlmLowConfidenceError):
        return "low_confidence"
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError, httpx.TimeoutException)):
        return "timeout"
    if isinstance(exc, httpx.TransportError):
        return "transport_error"
    if isinstance(exc, httpx.HTTPStatusError):
        status_code = exc.response.status_code
        if status_code == 429:
            return "rate_limited"
        return "provider_error"
    if isinstance(exc, (ValueError, TypeError)):
        return "invalid_output"
    return "provider_error"


def _has_upgrade_budget(deadline: float | None) -> bool:
    if deadline is None:
        return True
    remaining = deadline - asyncio.get_running_loop().time()
    return remaining > _UPGRADE_RESERVE_SECONDS


def _upgrade_timeout(deadline: float | None) -> float | None:
    if deadline is None:
        return None
    remaining = deadline - asyncio.get_running_loop().time()
    return max(0.001, remaining - _UPGRADE_RESERVE_SECONDS)


def _secret_value(value: SecretStr | str) -> str:
    if isinstance(value, SecretStr):
        return value.get_secret_value()
    return value


def _acquire_http_client(
    *,
    api_key: SecretStr,
    base_url: str,
    timeout: float,
) -> tuple[tuple[str, str, float], httpx.AsyncClient]:
    """按连接身份共享池；模型和采样参数继续留在每次请求 payload。"""

    normalized_base_url = base_url.rstrip("/") + "/"
    secret = api_key.get_secret_value()
    key = (
        normalized_base_url,
        hashlib.sha256(secret.encode("utf-8")).hexdigest(),
        float(timeout),
    )
    with _shared_http_clients_lock:
        shared = _shared_http_clients.get(key)
        if shared is None:
            shared = _SharedHttpClient(
                client=httpx.AsyncClient(
                    base_url=normalized_base_url,
                    headers={
                        "Authorization": f"Bearer {secret}",
                        "Content-Type": "application/json",
                    },
                    timeout=timeout,
                )
            )
            _shared_http_clients[key] = shared
        else:
            shared.reference_count += 1
        return key, shared.client


async def _release_http_client(key: tuple[str, str, float]) -> None:
    """仅在最后一个模型包装器关闭时释放共享连接池。"""

    client: httpx.AsyncClient | None = None
    with _shared_http_clients_lock:
        shared = _shared_http_clients.get(key)
        if shared is None:
            return
        shared.reference_count -= 1
        if shared.reference_count <= 0:
            client = shared.client
            del _shared_http_clients[key]
    if client is not None:
        await client.aclose()


def _elapsed_ms(started_at: float) -> float:
    """返回有界非负耗时，避免 Trace 接触请求或响应正文。"""

    return max(0.0, (time.perf_counter() - started_at) * 1000)


def _safe_model_name(llm: Any) -> str:
    """从客户端读取非敏感模型名，Fake 或外部客户端缺失时返回空串。"""

    value = getattr(llm, "model_name", "")
    return str(value)[:200] if value is not None else ""


__all__ = [
    "LlmLowConfidenceError",
    "create_chat_llm",
    "create_controlled_structured_llms",
    "create_structured_llm",
    "invoke_with_controlled_upgrade",
    "is_llm_configured",
    "llm_failure_kind",
    "llm_model_name",
    "llm_upgrade_scope",
    "public_llm_config",
    "safe_llm_error",
]
