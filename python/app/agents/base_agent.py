"""Agent 公共重试、超时、日志和安全失败运行时。"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from typing import Any

import structlog
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from app.models.schemas import AgentResult

logger = structlog.get_logger()


class BaseAgent(ABC):
    """为 Agent 提供统一的重试、超时、日志和失败结果保护。"""

    def __init__(self, name: str, timeout: float = 10.0, max_retries: int = 2):
        if timeout <= 0:
            raise ValueError("timeout 必须大于 0")
        if max_retries < 0:
            raise ValueError("max_retries 不能小于 0")
        self.name = name
        self.timeout = timeout
        self.max_retries = max_retries
        self._call_count = 0
        self._error_count = 0

    @abstractmethod
    async def _execute(self, **kwargs: Any) -> AgentResult:
        """由具体 Agent 实现核心业务逻辑。"""

    async def run(self, **kwargs: Any) -> AgentResult:
        """执行 Agent，并统一处理计时、重试和失败降级。"""
        start = time.perf_counter()
        self._call_count += 1

        try:
            async with asyncio.timeout(self.timeout):
                result = await self._retry_execute(**kwargs)
            result.latency_ms = (time.perf_counter() - start) * 1000
            logger.info(
                "Agent 执行成功",
                agent=self.name,
                latency_ms=round(result.latency_ms, 1),
            )
            return result
        except Exception as exc:
            self._error_count += 1
            latency_ms = (time.perf_counter() - start) * 1000
            logger.error(
                "Agent 执行失败",
                agent=self.name,
                exception_type=type(exc).__name__,
            )
            return self._fallback(latency_ms, exc)

    async def _retry_execute(self, **kwargs: Any) -> AgentResult:
        @retry(
            retry=retry_if_exception(self._is_retryable_exception),
            stop=stop_after_attempt(self.max_retries + 1),
            wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
            reraise=True,
        )
        async def _inner():
            return await self._execute(**kwargs)

        return await _inner()

    @staticmethod
    def _is_retryable_exception(exc: BaseException) -> bool:
        """只重试超时或连接中断等可能恢复的基础设施异常。"""

        return isinstance(exc, (TimeoutError, ConnectionError))

    def _fallback(self, latency_ms: float, exc: Exception) -> AgentResult:
        """Agent 失败时返回不暴露底层异常详情的合法结果。"""

        return AgentResult(
            agent_name=self.name,
            success=False,
            latency_ms=latency_ms,
            error=type(exc).__name__,
            confidence=0.0,
        )

    @property
    def error_rate(self) -> float:
        if self._call_count == 0:
            return 0.0
        return self._error_count / self._call_count
