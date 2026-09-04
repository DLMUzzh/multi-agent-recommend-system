"""使用独立 SQLite 表持久化跨会话用户意图记忆。"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from app.config.paths import USER_PROFILE_DATABASE_PATH
from app.models.intent_memory import UserIntentMemory


class SQLiteUserIntentMemoryRepository:
    """与推荐画像共用数据库文件，但不混用画像缓存或模型。"""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = (
            Path(path) if path is not None else USER_PROFILE_DATABASE_PATH
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_schema()
        self.path.chmod(0o600)

    def get(self, user_id: str) -> UserIntentMemory | None:
        """读取单个用户记忆；尚未形成记忆时返回空。"""

        normalized_id = self._required_text(user_id)
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT user_id, memory_json, memory_version, updated_at
                FROM user_intent_memories
                WHERE user_id = ?
                """,
                (normalized_id,),
            ).fetchone()
        if row is None:
            return None
        try:
            payload = json.loads(row["memory_json"])
            memory = UserIntentMemory.model_validate(payload)
        except (TypeError, json.JSONDecodeError, ValueError):
            raise ValueError("数据库中的用户意图记忆无效") from None
        if (
            memory.user_id != row["user_id"]
            or memory.memory_version != row["memory_version"]
            or memory.updated_at.isoformat() != row["updated_at"]
        ):
            raise ValueError("数据库中的用户意图记忆无效")
        return memory

    def save(self, memory: UserIntentMemory) -> None:
        """原子新增或替换一份通过严格模型校验的用户记忆。"""

        validated = UserIntentMemory.model_validate(memory)
        payload = json.dumps(
            validated.model_dump(mode="json"),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        try:
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO user_intent_memories (
                        user_id,
                        memory_json,
                        memory_version,
                        updated_at
                    ) VALUES (?, ?, ?, ?)
                    ON CONFLICT(user_id) DO UPDATE SET
                        memory_json = excluded.memory_json,
                        memory_version = excluded.memory_version,
                        updated_at = excluded.updated_at
                    """,
                    (
                        validated.user_id,
                        payload,
                        validated.memory_version,
                        validated.updated_at.isoformat(),
                    ),
                )
        except sqlite3.IntegrityError:
            raise ValueError("用户不存在或意图记忆违反数据库约束") from None

    def delete(self, user_id: str) -> None:
        """删除单个用户的长期意图记忆，不影响画像和行为事实。"""

        normalized_id = self._required_text(user_id)
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM user_intent_memories WHERE user_id = ?",
                (normalized_id,),
            )

    def _initialize_schema(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS user_intent_memories (
                    user_id TEXT PRIMARY KEY,
                    memory_json TEXT NOT NULL,
                    memory_version TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
                    CHECK (length(trim(user_id)) > 0),
                    CHECK (
                        json_valid(memory_json)
                        AND json_type(memory_json) = 'object'
                    )
                )
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    @staticmethod
    def _required_text(value: object) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("user_id 不能为空")
        return value.strip()


__all__ = ["SQLiteUserIntentMemoryRepository"]
