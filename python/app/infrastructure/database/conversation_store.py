"""会话应用服务依赖的共享持久化契约。"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Protocol

from app.models.personal_feedback import (
    ConversationFeedbackContext,
    ConversationResultSnapshot,
    PersonalFeedbackEvent,
)
from app.models.schemas import ConversationSession


class ConversationStoreError(RuntimeError):
    """表示会话状态无法安全读取或提交。"""


class ConversationStore(Protocol):
    """会话服务依赖的异步持久化契约。"""

    async def load(
        self,
        user_id: str,
        session_id: str,
    ) -> ConversationSession | None:
        """读取指定用户会话，不存在时返回空。"""

        ...

    async def save(self, session: ConversationSession) -> None:
        """原子保存一个会话。"""

        ...

    async def save_many(self, sessions: Sequence[ConversationSession]) -> None:
        """在一个事务中原子保存多个有关联的会话。"""

        ...

    async def load_feedback_context(
        self,
        user_id: str,
        session_id: str,
    ) -> ConversationFeedbackContext:
        """读取同用户、同 Session 的最新结果和待处理反馈。"""

        ...

    async def get_result_snapshot(
        self,
        user_id: str,
        result_id: str,
    ) -> ConversationResultSnapshot | None:
        """按用户隔离读取一条结果快照。"""

        ...

    async def list_feedback_events(
        self,
        user_id: str,
        *,
        limit: int = 100,
    ) -> tuple[PersonalFeedbackEvent, ...]:
        """按时间倒序读取个人结构化反馈事件。"""

        ...

    async def commit_recovery(
        self,
        *,
        sessions: Sequence[ConversationSession],
        snapshots: Sequence[ConversationResultSnapshot] = (),
        feedback_events: Sequence[PersonalFeedbackEvent] = (),
    ) -> None:
        """原子保存会话、消息、结果快照和反馈事件。"""

        ...

    async def purge_feedback_raw_before(
        self,
        cutoff: datetime,
        *,
        purged_at: datetime,
    ) -> int:
        """清空已完成且超过保留期的反馈与查询原文。"""

        ...

    async def delete(self, user_id: str, session_id: str) -> None:
        """幂等删除一个会话。"""

        ...


__all__ = ["ConversationStore", "ConversationStoreError"]
