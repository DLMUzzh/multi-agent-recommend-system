"""使用独立 SQLite 表持久化交互反馈和回答偏好。"""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import sqlite3

from app.config.paths import USER_PROFILE_DATABASE_PATH
from app.models.interaction_memory import (
    ConversationFeedbackEvent,
    UserInteractionMemory,
)


class SQLiteUserInteractionMemoryRepository:
    """与推荐画像共库，但不混用画像模型、缓存或生命周期。"""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else USER_PROFILE_DATABASE_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_schema()
        self.path.chmod(0o600)

    def append_event(self, event: ConversationFeedbackEvent) -> None:
        """追加一条不可变反馈身份，重复事件 ID 明确失败。"""

        validated = ConversationFeedbackEvent.model_validate(event)
        try:
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO conversation_feedback_events (
                        event_id,
                        user_id,
                        session_id,
                        event_json,
                        status,
                        occurred_at,
                        next_attempt_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    self._event_values(validated),
                )
        except sqlite3.IntegrityError as exc:
            if "event_id" in str(exc) or "UNIQUE constraint failed" in str(exc):
                raise ValueError("feedback event_id 重复") from None
            raise ValueError("用户不存在或反馈事件违反数据库约束") from None

    def get_event(self, event_id: str) -> ConversationFeedbackEvent | None:
        """按事件 ID 读取完整生命周期快照。"""

        normalized = self._required_text(event_id, "event_id")
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT event_id, user_id, session_id, event_json, status,
                       occurred_at, next_attempt_at
                FROM conversation_feedback_events
                WHERE event_id = ?
                """,
                (normalized,),
            ).fetchone()
        return self._event_from_row(row) if row is not None else None

    def list_pending(
        self,
        *,
        now: datetime,
        limit: int,
    ) -> tuple[ConversationFeedbackEvent, ...]:
        """稳定返回当前已到重试时间的待分析事件。"""

        if limit < 1:
            raise ValueError("limit 必须大于零")
        timestamp = self._datetime_text(now)
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT event_id, user_id, session_id, event_json, status,
                       occurred_at, next_attempt_at
                FROM conversation_feedback_events
                WHERE status = 'pending'
                  AND (next_attempt_at IS NULL OR next_attempt_at <= ?)
                ORDER BY occurred_at, event_id
                LIMIT ?
                """,
                (timestamp, limit),
            ).fetchall()
        return tuple(self._event_from_row(row) for row in rows)

    def save_event(self, event: ConversationFeedbackEvent) -> None:
        """更新事件分析、重试或清理状态，不允许隐式创建。"""

        validated = ConversationFeedbackEvent.model_validate(event)
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE conversation_feedback_events
                SET event_json = ?, status = ?, occurred_at = ?,
                    next_attempt_at = ?
                WHERE event_id = ? AND user_id = ? AND session_id = ?
                """,
                (
                    self._json_object(validated.model_dump(mode="json")),
                    validated.status,
                    validated.occurred_at.isoformat(),
                    (
                        validated.next_attempt_at.isoformat()
                        if validated.next_attempt_at is not None
                        else None
                    ),
                    validated.event_id,
                    validated.user_id,
                    validated.session_id,
                ),
            )
            if cursor.rowcount != 1:
                raise ValueError("待更新反馈事件不存在")

    def get_memory(self, user_id: str) -> UserInteractionMemory | None:
        """读取单个用户的交互记忆；尚无数据时返回空。"""

        normalized = self._required_text(user_id, "user_id")
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT user_id, memory_json, memory_version, updated_at
                FROM user_interaction_memories
                WHERE user_id = ?
                """,
                (normalized,),
            ).fetchone()
        if row is None:
            return None
        try:
            memory = UserInteractionMemory.model_validate_json(row["memory_json"])
        except ValueError:
            raise ValueError("数据库中的用户交互记忆无效") from None
        if (
            memory.user_id != row["user_id"]
            or memory.memory_version != row["memory_version"]
            or memory.updated_at.isoformat() != row["updated_at"]
        ):
            raise ValueError("数据库中的用户交互记忆无效")
        return memory

    def save_memory(self, memory: UserInteractionMemory) -> None:
        """原子新增或替换一份严格校验的用户交互记忆。"""

        validated = UserInteractionMemory.model_validate(memory)
        try:
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO user_interaction_memories (
                        user_id, memory_json, memory_version, updated_at
                    ) VALUES (?, ?, ?, ?)
                    ON CONFLICT(user_id) DO UPDATE SET
                        memory_json = excluded.memory_json,
                        memory_version = excluded.memory_version,
                        updated_at = excluded.updated_at
                    """,
                    (
                        validated.user_id,
                        self._json_object(validated.model_dump(mode="json")),
                        validated.memory_version,
                        validated.updated_at.isoformat(),
                    ),
                )
        except sqlite3.IntegrityError:
            raise ValueError("用户不存在或交互记忆违反数据库约束") from None

    def purge_raw_before(
        self,
        cutoff: datetime,
        *,
        purged_at: datetime,
    ) -> int:
        """只清空已分析过期事件的原文，保留结构化结论。"""

        cutoff_text = self._datetime_text(cutoff)
        self._datetime_text(purged_at)
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT event_id, user_id, session_id, event_json, status,
                       occurred_at, next_attempt_at
                FROM conversation_feedback_events
                WHERE status = 'analyzed' AND occurred_at < ?
                ORDER BY occurred_at, event_id
                """,
                (cutoff_text,),
            ).fetchall()
            changed = 0
            for row in rows:
                event = self._event_from_row(row)
                if event.raw_purged_at is not None:
                    continue
                purged = event.model_copy(
                    update={
                        "previous_user_message": None,
                        "previous_assistant_message": None,
                        "feedback_message": None,
                        "raw_purged_at": purged_at,
                    }
                )
                validated = ConversationFeedbackEvent.model_validate(purged)
                connection.execute(
                    """
                    UPDATE conversation_feedback_events
                    SET event_json = ?
                    WHERE event_id = ?
                    """,
                    (
                        self._json_object(validated.model_dump(mode="json")),
                        validated.event_id,
                    ),
                )
                changed += 1
        return changed

    def _initialize_schema(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS conversation_feedback_events (
                    event_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    event_json TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (status IN ('pending', 'analyzed')),
                    occurred_at TEXT NOT NULL,
                    next_attempt_at TEXT,
                    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
                    CHECK (length(trim(event_id)) > 0),
                    CHECK (length(trim(session_id)) > 0),
                    CHECK (json_valid(event_json) AND json_type(event_json) = 'object')
                );

                CREATE INDEX IF NOT EXISTS idx_feedback_pending
                ON conversation_feedback_events(status, next_attempt_at, occurred_at, event_id);

                CREATE INDEX IF NOT EXISTS idx_feedback_user_time
                ON conversation_feedback_events(user_id, occurred_at, event_id);

                CREATE TABLE IF NOT EXISTS user_interaction_memories (
                    user_id TEXT PRIMARY KEY,
                    memory_json TEXT NOT NULL,
                    memory_version TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
                    CHECK (length(trim(user_id)) > 0),
                    CHECK (json_valid(memory_json) AND json_type(memory_json) = 'object')
                );
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    @classmethod
    def _event_values(cls, event: ConversationFeedbackEvent) -> tuple[object, ...]:
        return (
            event.event_id,
            event.user_id,
            event.session_id,
            cls._json_object(event.model_dump(mode="json")),
            event.status,
            event.occurred_at.isoformat(),
            event.next_attempt_at.isoformat() if event.next_attempt_at else None,
        )

    @staticmethod
    def _event_from_row(row: sqlite3.Row) -> ConversationFeedbackEvent:
        try:
            event = ConversationFeedbackEvent.model_validate_json(row["event_json"])
        except ValueError:
            raise ValueError("数据库中的会话反馈事件无效") from None
        if (
            event.event_id != row["event_id"]
            or event.user_id != row["user_id"]
            or event.session_id != row["session_id"]
            or event.status != row["status"]
            or event.occurred_at.isoformat() != row["occurred_at"]
            or (
                event.next_attempt_at.isoformat()
                if event.next_attempt_at is not None
                else None
            )
            != row["next_attempt_at"]
        ):
            raise ValueError("数据库中的会话反馈事件无效")
        return event

    @staticmethod
    def _json_object(value: object) -> str:
        return json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )

    @staticmethod
    def _datetime_text(value: datetime) -> str:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("时间必须包含时区")
        return value.isoformat()

    @staticmethod
    def _required_text(value: object, field_name: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field_name} 不能为空")
        return value.strip()


__all__ = ["SQLiteUserInteractionMemoryRepository"]
