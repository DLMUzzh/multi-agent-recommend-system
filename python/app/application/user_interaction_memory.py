"""异步处理会话反馈并维护用户交互习惯。"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
import logging
from typing import Protocol

from app.agents.conversation_feedback_agent import ConversationFeedbackAgent
from app.domain.services.user_interaction_memory import (
    UserInteractionMemoryService,
)
from app.models.interaction_memory import (
    ConversationFeedbackEvent,
    UserInteractionMemory,
)


logger = logging.getLogger(__name__)


class UserInteractionMemoryRepository(Protocol):
    """Worker 所需的同步持久化最小契约。"""

    def append_event(self, event: ConversationFeedbackEvent) -> None: ...

    def list_pending(
        self,
        *,
        now: datetime,
        limit: int,
    ) -> tuple[ConversationFeedbackEvent, ...]: ...

    def get_memory(self, user_id: str) -> UserInteractionMemory | None: ...

    def save_memory(self, memory: UserInteractionMemory) -> None: ...

    def save_event(self, event: ConversationFeedbackEvent) -> None: ...

    def purge_raw_before(
        self,
        cutoff: datetime,
        *,
        purged_at: datetime,
    ) -> int: ...


class UserInteractionMemoryWorker:
    """串行领取待分析事件，失败退避，成功后清理过期原文。"""

    _RAW_RETENTION_DAYS = 30
    _MAX_RETRY_DELAY_SECONDS = 86400

    def __init__(
        self,
        *,
        repository: UserInteractionMemoryRepository,
        memory_service: UserInteractionMemoryService,
        feedback_agent: ConversationFeedbackAgent,
        clock: Callable[[], datetime] | None = None,
        scan_interval_seconds: float = 86400.0,
        batch_size: int = 20,
    ) -> None:
        if scan_interval_seconds <= 0:
            raise ValueError("交互记忆扫描间隔必须大于零")
        if batch_size < 1:
            raise ValueError("交互记忆批量大小必须大于零")
        self._repository = repository
        self._memory_service = memory_service
        self._feedback_agent = feedback_agent
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._scan_interval_seconds = scan_interval_seconds
        self._batch_size = batch_size
        self._wake_event = asyncio.Event()
        self._stopping = False
        self._closed = False
        self._task: asyncio.Task[None] | None = None

    def start(self) -> None:
        """启动单进程后台循环，并立即扫描一次遗留事件。"""

        if self._closed:
            raise RuntimeError("交互记忆 Worker 已关闭")
        if self._task is not None:
            return
        self._stopping = False
        self._task = asyncio.create_task(
            self._run(),
            name="user-interaction-memory-worker",
        )
        self.wake()

    def wake(self) -> None:
        """通知后台循环尽快处理新事件。"""

        if not self._closed:
            self._wake_event.set()

    async def stop(self) -> None:
        """等待当前批次结束，停止循环并关闭反馈 Agent。"""

        if self._closed:
            return
        self._stopping = True
        self._wake_event.set()
        task = self._task
        if task is not None:
            await task
            self._task = None
        await self._feedback_agent.aclose()
        self._closed = True

    async def run_once(self) -> int:
        """处理一批到期事件，返回本批成功分析的事件数量。"""

        now = self._now()
        events = await asyncio.to_thread(
            self._repository.list_pending,
            now=now,
            limit=self._batch_size,
        )
        processed = 0
        for event in events:
            try:
                analysis = await self._feedback_agent.analyze(event)
                if analysis is None:
                    continue
                memory = await asyncio.to_thread(
                    self._repository.get_memory,
                    event.user_id,
                )
                if memory is None:
                    memory = self._memory_service.empty(event.user_id)
                updated_memory = self._memory_service.apply_analysis(
                    memory,
                    event=event,
                    analysis=analysis,
                )
                await asyncio.to_thread(
                    self._repository.save_memory,
                    updated_memory,
                )
                analyzed_event = ConversationFeedbackEvent.model_validate(
                    event.model_copy(
                        update={
                            "status": "analyzed",
                            "analysis": analysis,
                            "analysis_attempts": event.analysis_attempts + 1,
                            "next_attempt_at": None,
                            "last_error_type": None,
                            "analyzed_at": now,
                        }
                    )
                )
                await asyncio.to_thread(
                    self._repository.save_event,
                    analyzed_event,
                )
                processed += 1
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                await self._schedule_retry(event, exc=exc, now=now)
        await self._purge_expired_raw(now)
        return processed

    async def _run(self) -> None:
        while True:
            try:
                await asyncio.wait_for(
                    self._wake_event.wait(),
                    timeout=self._scan_interval_seconds,
                )
            except TimeoutError:
                pass
            self._wake_event.clear()
            if self._stopping:
                return
            try:
                await self.run_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(
                    "用户交互记忆后台扫描失败，等待下次触发",
                    extra={"exception_type": type(exc).__name__},
                )

    async def _schedule_retry(
        self,
        event: ConversationFeedbackEvent,
        *,
        exc: Exception,
        now: datetime,
    ) -> None:
        attempts = event.analysis_attempts + 1
        retry_event = ConversationFeedbackEvent.model_validate(
            event.model_copy(
                update={
                    "analysis_attempts": attempts,
                    "next_attempt_at": now
                    + timedelta(seconds=self._retry_delay_seconds(attempts)),
                    "last_error_type": type(exc).__name__[:100],
                }
            )
        )
        try:
            await asyncio.to_thread(self._repository.save_event, retry_event)
        except Exception as save_exc:
            logger.warning(
                "用户交互反馈重试状态保存失败",
                extra={"exception_type": type(save_exc).__name__},
            )

    async def _purge_expired_raw(self, now: datetime) -> None:
        try:
            await asyncio.to_thread(
                self._repository.purge_raw_before,
                now - timedelta(days=self._RAW_RETENTION_DAYS),
                purged_at=now,
            )
        except Exception as exc:
            logger.warning(
                "用户交互反馈原文清理失败，保留数据等待下次扫描",
                extra={"exception_type": type(exc).__name__},
            )

    def _now(self) -> datetime:
        now = self._clock()
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("交互记忆 Worker 时钟必须包含时区")
        return now

    @classmethod
    def _retry_delay_seconds(cls, attempts: int) -> int:
        return min(300 * (2 ** max(0, attempts - 1)), cls._MAX_RETRY_DELAY_SECONDS)


__all__ = [
    "UserInteractionMemoryRepository",
    "UserInteractionMemoryWorker",
]
