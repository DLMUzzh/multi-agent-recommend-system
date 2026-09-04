"""共享 Chunk 检索使用的 OpenAI-compatible Embedding 客户端。"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Sequence
from typing import Any, Protocol

import httpx
from pydantic import SecretStr

from app.config import Settings, get_settings


class EmbeddingClient(Protocol):
    """外部 Embedding 服务的最小异步契约。"""

    async def embed(
        self,
        texts: list[str],
    ) -> Sequence[Sequence[float]]:
        """批量返回与输入顺序一致的向量。"""

        ...


class OpenAICompatibleEmbeddingClient:
    """通过 OpenAI-compatible API 批量生成并校验文本向量。"""

    _RETRYABLE_STATUS_CODES = {408, 409, 429}

    def __init__(
        self,
        *,
        api_key: SecretStr,
        base_url: str,
        model: str,
        dimensions: int,
        timeout: float,
        max_retries: int,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if not base_url.strip():
            raise ValueError("Embedding API 地址不能为空")
        self._model = model
        self._dimensions = dimensions
        self._max_retries = max_retries
        self._http_client = httpx.AsyncClient(
            base_url=base_url.rstrip("/") + "/",
            headers={
                "Authorization": f"Bearer {api_key.get_secret_value()}",
                "Content-Type": "application/json",
            },
            timeout=timeout,
            transport=transport,
        )

    async def embed(self, texts: list[str]) -> tuple[tuple[float, ...], ...]:
        """返回与输入顺序一致且已经 L2 归一化的向量。"""

        if not texts:
            return ()
        payload = {
            "model": self._model,
            "input": list(texts),
            "dimensions": self._dimensions,
            "encoding_format": "float",
        }
        response_payload = await self._post_with_retries(payload)
        return self._response_vectors(response_payload, expected_count=len(texts))

    async def aclose(self) -> None:
        """关闭共享异步 HTTP 客户端。"""

        await self._http_client.aclose()

    async def _post_with_retries(self, payload: dict[str, Any]) -> dict[str, Any]:
        for attempt in range(self._max_retries + 1):
            try:
                response = await self._http_client.post("embeddings", json=payload)
                retryable = (
                    response.status_code in self._RETRYABLE_STATUS_CODES
                    or response.status_code >= 500
                )
                if retryable and attempt < self._max_retries:
                    await asyncio.sleep(min(0.5 * (2**attempt), 4.0))
                    continue
                response.raise_for_status()
                response_payload = response.json()
                if not isinstance(response_payload, dict):
                    raise ValueError("Embedding 响应顶层必须是对象")
                return response_payload
            except (httpx.TimeoutException, httpx.TransportError):
                if attempt >= self._max_retries:
                    raise
                await asyncio.sleep(min(0.5 * (2**attempt), 4.0))
        raise RuntimeError("Embedding 请求重试状态无效")

    def _response_vectors(
        self,
        response_payload: dict[str, Any],
        *,
        expected_count: int,
    ) -> tuple[tuple[float, ...], ...]:
        data = response_payload.get("data")
        if not isinstance(data, list) or len(data) != expected_count:
            raise ValueError("Embedding 响应数量与输入不一致")
        ordered: list[tuple[float, ...] | None] = [None] * expected_count
        for item in data:
            if not isinstance(item, dict):
                raise ValueError("Embedding 响应条目结构无效")
            index = item.get("index")
            vector = item.get("embedding")
            if (
                not isinstance(index, int)
                or isinstance(index, bool)
                or not 0 <= index < expected_count
                or ordered[index] is not None
                or not isinstance(vector, list)
            ):
                raise ValueError("Embedding 响应条目索引或向量无效")
            ordered[index] = normalize_vector(
                vector,
                expected_dimension=self._dimensions,
            )
        if any(vector is None for vector in ordered):
            raise ValueError("Embedding 响应缺少输入对应向量")
        return tuple(vector for vector in ordered if vector is not None)


def is_embedding_configured(settings: Settings | None = None) -> bool:
    """判断是否同时配置了非空 Embedding Key 和 API 地址。"""

    current = settings or get_settings()
    return bool(
        current.embedding_api_key.get_secret_value().strip()
        and current.embedding_base_url.strip()
    )


def create_embedding_client(
    settings: Settings | None = None,
) -> OpenAICompatibleEmbeddingClient | None:
    """按独立配置创建 Embedding 客户端，配置不完整时返回 ``None``。"""

    current = settings or get_settings()
    if not is_embedding_configured(current):
        return None
    return OpenAICompatibleEmbeddingClient(
        api_key=current.embedding_api_key,
        base_url=current.embedding_base_url,
        model=current.embedding_model,
        dimensions=current.embedding_dimensions,
        timeout=current.embedding_request_timeout_seconds,
        max_retries=current.embedding_max_retries,
    )


def public_embedding_config(settings: Settings | None = None) -> dict[str, Any]:
    """返回不包含密钥和内部地址的 Embedding 公开诊断。"""

    current = settings or get_settings()
    return {
        "model": current.embedding_model,
        "dimensions": current.embedding_dimensions,
        "configured": is_embedding_configured(current),
        "request_timeout_seconds": current.embedding_request_timeout_seconds,
        "max_retries": current.embedding_max_retries,
    }


def normalize_vector(
    vector: Sequence[float],
    *,
    expected_dimension: int,
) -> tuple[float, ...]:
    """校验向量维度与数值，并返回 L2 归一化结果。"""

    if len(vector) != expected_dimension:
        raise ValueError("Embedding 向量维度无效")
    if not all(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        for value in vector
    ):
        raise ValueError("Embedding 向量必须只包含有限数值")
    values = tuple(float(value) for value in vector)
    norm = math.sqrt(sum(value * value for value in values))
    if norm <= 0.0:
        raise ValueError("Embedding 响应包含零向量")
    return tuple(value / norm for value in values)


__all__ = [
    "EmbeddingClient",
    "OpenAICompatibleEmbeddingClient",
    "create_embedding_client",
    "is_embedding_configured",
    "normalize_vector",
    "public_embedding_config",
]
