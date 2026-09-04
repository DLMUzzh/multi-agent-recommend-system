"""用户基础偏好和行为事实的 SQLite 仓储。"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

from app.config.paths import USER_PROFILE_DATABASE_PATH
from app.infrastructure.database.json.feature_store_models import (
    ALLOWED_EVENT_TYPES,
    BehaviorEvent,
    UserBaseProfile,
)


class SQLiteUserProfileRepository:
    """使用短连接和事务维护用户偏好与不可变行为事实。"""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = (
            Path(path) if path is not None else USER_PROFILE_DATABASE_PATH
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_schema()
        self.path.chmod(0o600)

    def replace_user(self, user: UserBaseProfile) -> None:
        """原子写入或替换一条用户显式偏好。"""

        validated = UserBaseProfile.model_validate(user)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO users (
                    user_id,
                    topics,
                    blocked_topics,
                    preferred_content_types,
                    preferred_difficulty,
                    preferred_reading_length,
                    followed_author_ids,
                    blocked_author_ids,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    topics = excluded.topics,
                    blocked_topics = excluded.blocked_topics,
                    preferred_content_types = excluded.preferred_content_types,
                    preferred_difficulty = excluded.preferred_difficulty,
                    preferred_reading_length = excluded.preferred_reading_length,
                    followed_author_ids = excluded.followed_author_ids,
                    blocked_author_ids = excluded.blocked_author_ids
                """,
                (
                    validated.user_id,
                    self._json_array(validated.topics),
                    self._json_array(validated.blocked_topics),
                    self._json_array(validated.preferred_content_types),
                    validated.preferred_difficulty or None,
                    validated.preferred_reading_length or None,
                    self._json_array(validated.followed_author_ids),
                    self._json_array(validated.blocked_author_ids),
                    self._datetime_text(validated.created_at),
                ),
            )

    def get_user(self, user_id: str) -> UserBaseProfile | None:
        """按用户 ID 读取显式偏好，不存在时返回空。"""

        normalized_id = self._required_text(user_id, "user_id")
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM users WHERE user_id = ?",
                (normalized_id,),
            ).fetchone()
        if row is None:
            return None
        return UserBaseProfile.model_validate(
            {
                "user_id": row["user_id"],
                "topics": self._json_list(row["topics"], "topics"),
                "blocked_topics": self._json_list(
                    row["blocked_topics"], "blocked_topics"
                ),
                "preferred_content_types": self._json_list(
                    row["preferred_content_types"], "preferred_content_types"
                ),
                "preferred_difficulty": row["preferred_difficulty"] or "",
                "preferred_reading_length": row["preferred_reading_length"] or "",
                "followed_author_ids": self._json_list(
                    row["followed_author_ids"], "followed_author_ids"
                ),
                "blocked_author_ids": self._json_list(
                    row["blocked_author_ids"], "blocked_author_ids"
                ),
                "created_at": row["created_at"],
            }
        )

    def list_users(self) -> tuple[UserBaseProfile, ...]:
        """按用户 ID 稳定返回全部显式偏好。"""

        with self._connect() as connection:
            user_ids = [
                row["user_id"]
                for row in connection.execute(
                    "SELECT user_id FROM users ORDER BY user_id"
                ).fetchall()
            ]
        return tuple(
            user
            for user_id in user_ids
            if (user := self.get_user(user_id)) is not None
        )

    def append_event(self, event: BehaviorEvent) -> None:
        """追加一条不可变行为，重复事件 ID 明确失败。"""

        validated = BehaviorEvent.model_validate(event)
        try:
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO user_behavior_events (
                        event_id,
                        user_id,
                        event_type,
                        occurred_at,
                        document_id,
                        metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        validated.event_id,
                        validated.user_id,
                        validated.event_type,
                        self._datetime_text(validated.occurred_at),
                        validated.document_id,
                        json.dumps(
                            validated.metadata,
                            ensure_ascii=False,
                            separators=(",", ":"),
                            sort_keys=True,
                        ),
                    ),
                )
        except sqlite3.IntegrityError as exc:
            if "event_id" in str(exc) or "UNIQUE constraint failed" in str(exc):
                raise ValueError("event_id 重复") from None
            raise ValueError("行为事件违反数据库约束") from None

    def list_events(
        self,
        user_id: str,
        *,
        since: datetime,
        until: datetime,
        limit: int = 0,
    ) -> tuple[BehaviorEvent, ...]:
        """按时间和事件 ID 稳定返回指定窗口内的行为。"""

        normalized_id = self._required_text(user_id, "user_id")
        if since.tzinfo is None or since.utcoffset() is None:
            raise ValueError("since 必须包含时区")
        if until.tzinfo is None or until.utcoffset() is None:
            raise ValueError("until 必须包含时区")
        if since > until:
            raise ValueError("since 不能晚于 until")
        query = """
            SELECT event_id, user_id, event_type, occurred_at, document_id,
                   metadata_json
            FROM user_behavior_events
            WHERE user_id = ? AND occurred_at >= ? AND occurred_at <= ?
            ORDER BY occurred_at, event_id
        """
        parameters: list[object] = [
            normalized_id,
            self._datetime_text(since),
            self._datetime_text(until),
        ]
        if limit > 0:
            query += " LIMIT ?"
            parameters.append(limit)
        with self._connect() as connection:
            rows = connection.execute(query, tuple(parameters)).fetchall()
        return tuple(self._event_from_row(row) for row in rows)

    def list_all_events(self) -> tuple[BehaviorEvent, ...]:
        """按时间和事件 ID 稳定返回全部行为事实。"""

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT event_id, user_id, event_type, occurred_at, document_id,
                       metadata_json
                FROM user_behavior_events
                ORDER BY occurred_at, event_id
                """
            ).fetchall()
        return tuple(self._event_from_row(row) for row in rows)

    def _initialize_schema(self) -> None:
        event_values = ",".join(
            f"'{value}'" for value in sorted(ALLOWED_EVENT_TYPES)
        )
        with self._connect() as connection:
            connection.executescript(
                f"""
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    topics TEXT NOT NULL,
                    blocked_topics TEXT NOT NULL,
                    preferred_content_types TEXT NOT NULL,
                    preferred_difficulty TEXT,
                    preferred_reading_length TEXT,
                    followed_author_ids TEXT NOT NULL,
                    blocked_author_ids TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    CHECK (length(trim(user_id)) > 0),
                    CHECK (json_valid(topics) AND json_type(topics) = 'array'),
                    CHECK (json_valid(blocked_topics) AND json_type(blocked_topics) = 'array'),
                    CHECK (
                        json_valid(preferred_content_types)
                        AND json_type(preferred_content_types) = 'array'
                    ),
                    CHECK (
                        preferred_difficulty IS NULL
                        OR preferred_difficulty IN ('beginner', 'intermediate', 'advanced')
                    ),
                    CHECK (
                        preferred_reading_length IS NULL
                        OR preferred_reading_length IN ('short', 'medium', 'long', 'mixed')
                    ),
                    CHECK (
                        json_valid(followed_author_ids)
                        AND json_type(followed_author_ids) = 'array'
                    ),
                    CHECK (
                        json_valid(blocked_author_ids)
                        AND json_type(blocked_author_ids) = 'array'
                    )
                );

                CREATE TABLE IF NOT EXISTS user_behavior_events (
                    event_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    event_type TEXT NOT NULL CHECK (event_type IN ({event_values})),
                    occurred_at TEXT NOT NULL,
                    document_id TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
                    CHECK (length(trim(event_id)) > 0),
                    CHECK (length(trim(document_id)) > 0),
                    CHECK (
                        json_valid(metadata_json)
                        AND json_type(metadata_json) = 'object'
                    )
                );

                CREATE INDEX IF NOT EXISTS idx_behavior_user_time
                ON user_behavior_events(user_id, occurred_at, event_id);
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    @staticmethod
    def _json_array(values: Sequence[str]) -> str:
        return json.dumps(
            list(values),
            ensure_ascii=False,
            separators=(",", ":"),
        )

    @staticmethod
    def _json_list(raw: str, field_name: str) -> list[str]:
        try:
            values = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            raise ValueError(f"数据库中的 {field_name} 无效") from None
        if not isinstance(values, list) or any(
            not isinstance(value, str) for value in values
        ):
            raise ValueError(f"数据库中的 {field_name} 无效")
        return values

    @staticmethod
    def _datetime_text(value: datetime) -> str:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("时间必须包含时区")
        return value.astimezone(timezone.utc).isoformat()

    @staticmethod
    def _required_text(value: object, field_name: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field_name} 不能为空")
        return value.strip()

    @staticmethod
    def _event_from_row(row: sqlite3.Row) -> BehaviorEvent:
        try:
            metadata = json.loads(row["metadata_json"])
        except (TypeError, json.JSONDecodeError):
            raise ValueError("数据库中的 metadata_json 无效") from None
        return BehaviorEvent.model_validate(
            {
                "event_id": row["event_id"],
                "user_id": row["user_id"],
                "event_type": row["event_type"],
                "occurred_at": row["occurred_at"],
                "document_id": row["document_id"],
                "metadata": metadata,
            }
        )


__all__ = ["SQLiteUserProfileRepository"]
