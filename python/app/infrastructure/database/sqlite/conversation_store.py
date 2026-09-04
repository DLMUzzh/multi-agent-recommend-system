"""使用独立 SQLite 数据库持久化唯一会话历史与业务状态。"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

from pydantic import ValidationError

from app.config.paths import CONVERSATION_DATABASE_PATH
from app.infrastructure.database.conversation_store import ConversationStoreError
from app.models.personal_feedback import (
    ConversationFeedbackContext,
    ConversationResultSnapshot,
    PersonalFeedbackEvent,
)
from app.models.schemas import (
    ConversationSession,
    ConversationTurn,
    IntentState,
    RecommendationContext,
)


class SQLiteConversationStore:
    """使用短连接、事务和外键维护会话级状态与有序消息。"""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = (
            Path(path) if path is not None else CONVERSATION_DATABASE_PATH
        )
        self._database_lock = asyncio.Lock()
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._initialize_schema()
            self.path.chmod(0o600)
        except (OSError, sqlite3.Error):
            raise ConversationStoreError("会话数据库无法安全初始化") from None

    async def load(
        self,
        user_id: str,
        session_id: str,
    ) -> ConversationSession | None:
        """在一致读事务中加载会话级状态和有序消息。"""

        normalized_user_id = str(user_id)
        async with self._database_lock:
            try:
                return await asyncio.to_thread(
                    self._load_sync,
                    normalized_user_id,
                    session_id,
                )
            except ConversationStoreError:
                raise
            except (
                OSError,
                sqlite3.Error,
                TypeError,
                ValueError,
                json.JSONDecodeError,
                ValidationError,
            ):
                raise ConversationStoreError("会话数据库无法安全读取") from None

    async def save(self, session: ConversationSession) -> None:
        """在一个事务中更新会话状态并稳定追加新消息。"""

        await self.save_many([session])

    async def save_many(
        self,
        sessions: Sequence[ConversationSession],
    ) -> None:
        """在一个事务中更新多个会话状态并稳定追加各自新消息。"""

        await self.commit_recovery(sessions=sessions)

    async def commit_recovery(
        self,
        *,
        sessions: Sequence[ConversationSession],
        snapshots: Sequence[ConversationResultSnapshot] = (),
        feedback_events: Sequence[PersonalFeedbackEvent] = (),
    ) -> None:
        """在短事务中原子保存会话、消息、快照和反馈事件。"""

        session_copies = [session.model_copy(deep=True) for session in sessions]
        snapshot_copies = [
            ConversationResultSnapshot.model_validate(item).model_copy(deep=True)
            for item in snapshots
        ]
        event_copies = [
            PersonalFeedbackEvent.model_validate(item).model_copy(deep=True)
            for item in feedback_events
        ]
        if not session_copies:
            raise ConversationStoreError("待保存会话不能为空")
        identities = [(item.user_id, item.session_id) for item in session_copies]
        if len(identities) != len(set(identities)):
            raise ConversationStoreError("同一事务不能重复保存同一会话")
        snapshot_ids = [item.result_id for item in snapshot_copies]
        if len(snapshot_ids) != len(set(snapshot_ids)):
            raise ConversationStoreError("同一事务不能重复保存结果快照")
        feedback_ids = [item.feedback_id for item in event_copies]
        if len(feedback_ids) != len(set(feedback_ids)):
            raise ConversationStoreError("同一事务不能重复保存反馈事件")
        for session in session_copies:
            self._validate_session(session)
        async with self._database_lock:
            try:
                await asyncio.to_thread(
                    self._commit_recovery_sync,
                    session_copies,
                    snapshot_copies,
                    event_copies,
                )
            except ConversationStoreError:
                raise
            except (
                OSError,
                sqlite3.Error,
                TypeError,
                ValueError,
                json.JSONDecodeError,
                ValidationError,
            ):
                raise ConversationStoreError("会话数据库无法安全写入") from None

    async def load_feedback_context(
        self,
        user_id: str,
        session_id: str,
    ) -> ConversationFeedbackContext:
        """读取同 Session 最新业务结果和合法待处理反馈。"""

        async with self._database_lock:
            try:
                return await asyncio.to_thread(
                    self._load_feedback_context_sync,
                    str(user_id),
                    str(session_id),
                )
            except ConversationStoreError:
                raise
            except (
                OSError,
                sqlite3.Error,
                TypeError,
                ValueError,
                json.JSONDecodeError,
                ValidationError,
            ):
                raise ConversationStoreError("会话反馈上下文无法安全读取") from None

    async def get_result_snapshot(
        self,
        user_id: str,
        result_id: str,
    ) -> ConversationResultSnapshot | None:
        """按用户读取结果快照，禁止跨用户命中。"""

        async with self._database_lock:
            try:
                return await asyncio.to_thread(
                    self._get_result_snapshot_sync,
                    str(user_id),
                    str(result_id),
                )
            except (
                OSError,
                sqlite3.Error,
                TypeError,
                ValueError,
                json.JSONDecodeError,
                ValidationError,
            ):
                raise ConversationStoreError("结果快照无法安全读取") from None

    async def list_feedback_events(
        self,
        user_id: str,
        *,
        limit: int = 100,
    ) -> tuple[PersonalFeedbackEvent, ...]:
        """按用户读取有界结构化反馈事件。"""

        if not 1 <= limit <= 100:
            raise ConversationStoreError("反馈事件读取数量必须在 1 到 100 之间")
        async with self._database_lock:
            try:
                return await asyncio.to_thread(
                    self._list_feedback_events_sync,
                    str(user_id),
                    limit,
                )
            except (
                OSError,
                sqlite3.Error,
                TypeError,
                ValueError,
                json.JSONDecodeError,
                ValidationError,
            ):
                raise ConversationStoreError("反馈事件无法安全读取") from None

    async def purge_feedback_raw_before(
        self,
        cutoff: datetime,
        *,
        purged_at: datetime,
    ) -> int:
        """清空终态反馈和过期结果查询原文，保留结构化身份。"""

        self._validate_aware_datetime(cutoff, field_name="反馈清理截止时间")
        self._validate_aware_datetime(purged_at, field_name="反馈清理执行时间")
        async with self._database_lock:
            try:
                return await asyncio.to_thread(
                    self._purge_feedback_raw_before_sync,
                    cutoff,
                    purged_at,
                )
            except (
                OSError,
                sqlite3.Error,
                TypeError,
                ValueError,
                json.JSONDecodeError,
                ValidationError,
            ):
                raise ConversationStoreError("反馈原文无法安全清理") from None

    async def delete(self, user_id: str, session_id: str) -> None:
        """在一个事务中幂等删除会话及其全部消息。"""

        normalized_user_id = str(user_id)
        async with self._database_lock:
            try:
                await asyncio.to_thread(
                    self._delete_sync,
                    normalized_user_id,
                    session_id,
                )
            except (OSError, sqlite3.Error):
                raise ConversationStoreError("会话数据库无法安全删除") from None

    def _initialize_schema(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    intent_state TEXT NOT NULL
                        CHECK (intent_state IN ('recommendation', 'knowledge_qa')),
                    active_context_json TEXT,
                    turn_count INTEGER NOT NULL CHECK (turn_count >= 0),
                    summary TEXT,
                    summary_watermark INTEGER NOT NULL DEFAULT -1
                        CHECK (summary_watermark >= -1),
                    summarized_turn_count INTEGER NOT NULL
                        CHECK (summarized_turn_count >= 0),
                    dropped_turn_count INTEGER NOT NULL
                        CHECK (dropped_turn_count >= 0),
                    pending_feedback_id TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (user_id, session_id),
                    CHECK (length(trim(user_id)) > 0),
                    CHECK (length(trim(session_id)) > 0),
                    CHECK (summary IS NULL OR length(summary) <= 2000)
                );

                CREATE TABLE IF NOT EXISTS conversation_messages (
                    message_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    sequence_no INTEGER NOT NULL CHECK (sequence_no >= 0),
                    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    message_type TEXT NOT NULL DEFAULT 'chat'
                        CHECK (message_type IN ('chat', 'child_handoff')),
                    related_session_id TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (user_id, session_id)
                        REFERENCES conversations(user_id, session_id)
                        ON DELETE CASCADE,
                    UNIQUE (user_id, session_id, sequence_no)
                );

                CREATE TABLE IF NOT EXISTS article_qa_sessions (
                    user_id TEXT NOT NULL,
                    child_session_id TEXT NOT NULL,
                    parent_session_id TEXT NOT NULL,
                    focus_document_id TEXT NOT NULL,
                    title_snapshot TEXT NOT NULL,
                    status TEXT NOT NULL
                        CHECK (status IN ('active', 'suspended', 'closed')),
                    unresolved_questions_json TEXT NOT NULL DEFAULT '[]',
                    cited_document_ids_json TEXT NOT NULL DEFAULT '[]',
                    handoff_summary TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (user_id, child_session_id),
                    FOREIGN KEY (user_id, child_session_id)
                        REFERENCES conversations(user_id, session_id)
                        ON DELETE CASCADE,
                    FOREIGN KEY (user_id, parent_session_id)
                        REFERENCES conversations(user_id, session_id)
                        ON DELETE CASCADE,
                    CHECK (child_session_id <> parent_session_id),
                    CHECK (length(trim(focus_document_id)) > 0),
                    CHECK (length(trim(title_snapshot)) > 0),
                    CHECK (
                        handoff_summary IS NULL
                        OR length(handoff_summary) <= 2000
                    )
                );

                CREATE UNIQUE INDEX IF NOT EXISTS
                    uq_article_qa_active_parent
                ON article_qa_sessions(user_id, parent_session_id)
                WHERE status = 'active';

                CREATE TABLE IF NOT EXISTS conversation_result_snapshots (
                    result_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    assistant_sequence_no INTEGER NOT NULL
                        CHECK (assistant_sequence_no >= 0),
                    result_type TEXT NOT NULL
                        CHECK (result_type IN (
                            'recommendation', 'knowledge_answer'
                        )),
                    snapshot_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    raw_purged_at TEXT,
                    FOREIGN KEY (user_id, session_id)
                        REFERENCES conversations(user_id, session_id)
                        ON DELETE CASCADE,
                    UNIQUE (user_id, session_id, assistant_sequence_no),
                    UNIQUE (result_id, user_id, session_id),
                    CHECK (
                        json_valid(snapshot_json)
                        AND json_type(snapshot_json) = 'object'
                    )
                );

                CREATE TABLE IF NOT EXISTS personal_feedback_events (
                    feedback_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    source_result_id TEXT NOT NULL,
                    feedback_json TEXT NOT NULL,
                    status TEXT NOT NULL
                        CHECK (status IN (
                            'classifying', 'awaiting_detail', 'recovering',
                            'recovered', 'recovery_failed', 'closed'
                        )),
                    next_action TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    raw_purged_at TEXT,
                    FOREIGN KEY (source_result_id, user_id, session_id)
                        REFERENCES conversation_result_snapshots(
                            result_id, user_id, session_id
                        ) ON DELETE CASCADE,
                    FOREIGN KEY (user_id, session_id)
                        REFERENCES conversations(user_id, session_id)
                        ON DELETE CASCADE,
                    CHECK (
                        json_valid(feedback_json)
                        AND json_type(feedback_json) = 'object'
                    )
                );

                CREATE INDEX IF NOT EXISTS idx_result_snapshot_session_sequence
                ON conversation_result_snapshots(
                    user_id, session_id, assistant_sequence_no DESC
                );

                CREATE INDEX IF NOT EXISTS idx_personal_feedback_session_status
                ON personal_feedback_events(
                    user_id, session_id, status, updated_at DESC
                );

                CREATE INDEX IF NOT EXISTS idx_personal_feedback_user_time
                ON personal_feedback_events(user_id, created_at, feedback_id);
                """
            )
            self._ensure_column(
                connection,
                "conversations",
                "summary_watermark",
                "INTEGER NOT NULL DEFAULT -1",
            )
            self._ensure_column(
                connection,
                "conversation_messages",
                "message_type",
                "TEXT NOT NULL DEFAULT 'chat'",
            )
            self._ensure_column(
                connection,
                "conversation_messages",
                "related_session_id",
                "TEXT",
            )
            self._ensure_column(
                connection,
                "conversations",
                "pending_feedback_id",
                "TEXT",
            )

    @staticmethod
    def _ensure_column(
        connection: sqlite3.Connection,
        table: str,
        column: str,
        definition: str,
    ) -> None:
        """只为既有 SQLite 表添加缺失列，不重建或覆盖已有数据。"""

        columns = {
            row[1]
            for row in connection.execute(f"PRAGMA table_info({table})")
        }
        if column not in columns:
            connection.execute(
                f"ALTER TABLE {table} ADD COLUMN {column} {definition}"
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _load_sync(
        self,
        user_id: str,
        session_id: str,
    ) -> ConversationSession | None:
        with self._connect() as connection:
            connection.execute("BEGIN")
            row = connection.execute(
                """
                SELECT
                    user_id,
                    session_id,
                    intent_state,
                    active_context_json,
                    turn_count,
                    summary,
                    summary_watermark,
                    summarized_turn_count,
                    dropped_turn_count
                    , pending_feedback_id
                FROM conversations
                WHERE user_id = ? AND session_id = ?
                """,
                (user_id, session_id),
            ).fetchone()
            if row is None:
                return None
            message_rows = connection.execute(
                """
                SELECT
                    message_id,
                    sequence_no,
                    role,
                    content,
                    message_type,
                    related_session_id,
                    created_at
                FROM conversation_messages
                WHERE user_id = ? AND session_id = ?
                ORDER BY sequence_no
                """,
                (user_id, session_id),
            ).fetchall()
            child_row = connection.execute(
                """
                SELECT
                    parent_session_id,
                    focus_document_id,
                    title_snapshot,
                    status,
                    unresolved_questions_json,
                    cited_document_ids_json,
                    handoff_summary
                FROM article_qa_sessions
                WHERE user_id = ? AND child_session_id = ?
                """,
                (user_id, session_id),
            ).fetchone()
            active_child_row = None
            if child_row is None:
                active_child_row = connection.execute(
                    """
                    SELECT child_session_id
                    FROM article_qa_sessions
                    WHERE user_id = ?
                      AND parent_session_id = ?
                      AND status = 'active'
                    """,
                    (user_id, session_id),
                ).fetchone()
        context_payload = row["active_context_json"]
        context = (
            RecommendationContext.model_validate(json.loads(context_payload))
            if context_payload is not None
            else None
        )
        return ConversationSession(
            session_id=row["session_id"],
            user_id=row["user_id"],
            session_type=("article_qa" if child_row is not None else "main"),
            parent_session_id=(
                child_row["parent_session_id"] if child_row is not None else None
            ),
            active_child_session_id=(
                active_child_row["child_session_id"]
                if active_child_row is not None
                else None
            ),
            focus_document_id=(
                child_row["focus_document_id"] if child_row is not None else None
            ),
            focus_document_title=(
                child_row["title_snapshot"] if child_row is not None else None
            ),
            session_status=(
                child_row["status"] if child_row is not None else "active"
            ),
            intent_state=IntentState(row["intent_state"]),
            active_context=context,
            history=[
                ConversationTurn(
                    message_id=item["message_id"],
                    sequence_no=item["sequence_no"],
                    role=item["role"],
                    content=item["content"],
                    message_type=item["message_type"],
                    related_session_id=item["related_session_id"],
                    created_at=datetime.fromisoformat(item["created_at"]),
                )
                for item in message_rows
            ],
            turn_count=row["turn_count"],
            summary=row["summary"],
            summary_watermark=row["summary_watermark"],
            summarized_turn_count=row["summarized_turn_count"],
            dropped_turn_count=row["dropped_turn_count"],
            unresolved_questions=(
                json.loads(child_row["unresolved_questions_json"])
                if child_row is not None
                else []
            ),
            cited_document_ids=(
                json.loads(child_row["cited_document_ids_json"])
                if child_row is not None
                else []
            ),
            handoff_summary=(
                child_row["handoff_summary"] if child_row is not None else None
            ),
            pending_feedback_id=row["pending_feedback_id"],
        )

    def _load_feedback_context_sync(
        self,
        user_id: str,
        session_id: str,
    ) -> ConversationFeedbackContext:
        with self._connect() as connection:
            session_row = connection.execute(
                "SELECT pending_feedback_id FROM conversations "
                "WHERE user_id = ? AND session_id = ?",
                (user_id, session_id),
            ).fetchone()
            if session_row is None:
                return ConversationFeedbackContext()
            pending_event = None
            latest_result = None
            pending_id = session_row["pending_feedback_id"]
            if pending_id is not None:
                event_row = connection.execute(
                    "SELECT feedback_json FROM personal_feedback_events "
                    "WHERE feedback_id = ? AND user_id = ? AND session_id = ? "
                    "AND status IN ('classifying', 'awaiting_detail', 'recovering')",
                    (pending_id, user_id, session_id),
                ).fetchone()
                if event_row is None:
                    raise ConversationStoreError("会话待处理反馈身份无效")
                pending_event = PersonalFeedbackEvent.model_validate_json(
                    event_row["feedback_json"]
                )
                snapshot_row = connection.execute(
                    "SELECT snapshot_json FROM conversation_result_snapshots "
                    "WHERE result_id = ? AND user_id = ? AND session_id = ?",
                    (pending_event.source_result_id, user_id, session_id),
                ).fetchone()
                if snapshot_row is None:
                    raise ConversationStoreError("待处理反馈缺少来源结果")
                latest_result = ConversationResultSnapshot.model_validate_json(
                    snapshot_row["snapshot_json"]
                )
            else:
                snapshot_row = connection.execute(
                    "SELECT snapshot_json FROM conversation_result_snapshots "
                    "WHERE user_id = ? AND session_id = ? "
                    "ORDER BY assistant_sequence_no DESC LIMIT 1",
                    (user_id, session_id),
                ).fetchone()
                if snapshot_row is not None:
                    latest_result = ConversationResultSnapshot.model_validate_json(
                        snapshot_row["snapshot_json"]
                    )
        return ConversationFeedbackContext(
            latest_result=latest_result,
            pending_feedback=pending_event,
        )

    def _get_result_snapshot_sync(
        self,
        user_id: str,
        result_id: str,
    ) -> ConversationResultSnapshot | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT snapshot_json FROM conversation_result_snapshots "
                "WHERE user_id = ? AND result_id = ?",
                (user_id, result_id),
            ).fetchone()
        if row is None:
            return None
        return ConversationResultSnapshot.model_validate_json(row["snapshot_json"])

    def _list_feedback_events_sync(
        self,
        user_id: str,
        limit: int,
    ) -> tuple[PersonalFeedbackEvent, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT feedback_json FROM personal_feedback_events "
                "WHERE user_id = ? ORDER BY created_at DESC, feedback_id DESC "
                "LIMIT ?",
                (user_id, limit),
            ).fetchall()
        return tuple(
            PersonalFeedbackEvent.model_validate_json(row["feedback_json"])
            for row in rows
        )

    def _commit_recovery_sync(
        self,
        sessions: Sequence[ConversationSession],
        snapshots: Sequence[ConversationResultSnapshot],
        feedback_events: Sequence[PersonalFeedbackEvent],
    ) -> None:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            for session in sessions:
                self._write_conversation_state(connection, session)
            for session in sessions:
                self._append_messages(connection, session)
            for session in sessions:
                self._write_child_state(connection, session)
            for snapshot in snapshots:
                self._write_result_snapshot(connection, snapshot)
            for event in feedback_events:
                self._write_feedback_event(connection, event)

    def _delete_sync(self, user_id: str, session_id: str) -> None:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            child_rows = connection.execute(
                """
                SELECT child_session_id
                FROM article_qa_sessions
                WHERE user_id = ? AND parent_session_id = ?
                """,
                (user_id, session_id),
            ).fetchall()
            for row in child_rows:
                connection.execute(
                    "DELETE FROM conversations "
                    "WHERE user_id = ? AND session_id = ?",
                    (user_id, row["child_session_id"]),
                )
            connection.execute(
                "DELETE FROM conversations WHERE user_id = ? AND session_id = ?",
                (user_id, session_id),
            )

    @staticmethod
    def _write_conversation_state(
        connection: sqlite3.Connection,
        session: ConversationSession,
    ) -> None:
        timestamp = datetime.now(timezone.utc).isoformat()
        context_json = (
            json.dumps(
                session.active_context.model_dump(mode="json"),
                ensure_ascii=False,
                separators=(",", ":"),
            )
            if session.active_context is not None
            else None
        )
        connection.execute(
            """
            INSERT INTO conversations (
                user_id,
                session_id,
                intent_state,
                active_context_json,
                turn_count,
                summary,
                summary_watermark,
                summarized_turn_count,
                dropped_turn_count,
                pending_feedback_id,
                created_at,
                updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id, session_id) DO UPDATE SET
                intent_state = excluded.intent_state,
                active_context_json = excluded.active_context_json,
                turn_count = excluded.turn_count,
                summary = excluded.summary,
                summary_watermark = excluded.summary_watermark,
                summarized_turn_count = excluded.summarized_turn_count,
                dropped_turn_count = excluded.dropped_turn_count,
                pending_feedback_id = excluded.pending_feedback_id,
                updated_at = excluded.updated_at
            """,
            (
                session.user_id,
                session.session_id,
                session.intent_state.value,
                context_json,
                session.turn_count,
                session.summary,
                session.summary_watermark,
                session.summarized_turn_count,
                session.dropped_turn_count,
                session.pending_feedback_id,
                timestamp,
                timestamp,
            ),
        )

    @classmethod
    def _write_result_snapshot(
        cls,
        connection: sqlite3.Connection,
        snapshot: ConversationResultSnapshot,
    ) -> None:
        """幂等写入结果快照，禁止覆盖既有结果身份。"""

        existing = connection.execute(
            "SELECT snapshot_json FROM conversation_result_snapshots "
            "WHERE result_id = ?",
            (snapshot.result_id,),
        ).fetchone()
        payload = cls._model_json(snapshot)
        if existing is not None:
            if existing["snapshot_json"] != payload:
                raise ConversationStoreError("结果快照不能覆盖既有身份")
            return
        connection.execute(
            """
            INSERT INTO conversation_result_snapshots (
                result_id,
                user_id,
                session_id,
                assistant_sequence_no,
                result_type,
                snapshot_json,
                created_at,
                raw_purged_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                snapshot.result_id,
                snapshot.user_id,
                snapshot.session_id,
                snapshot.assistant_sequence_no,
                snapshot.result_type,
                payload,
                snapshot.created_at.isoformat(),
                (
                    snapshot.raw_purged_at.isoformat()
                    if snapshot.raw_purged_at is not None
                    else None
                ),
            ),
        )

    @classmethod
    def _write_feedback_event(
        cls,
        connection: sqlite3.Connection,
        event: PersonalFeedbackEvent,
    ) -> None:
        """校验生命周期后新增或更新反馈事件。"""

        existing = connection.execute(
            "SELECT feedback_json FROM personal_feedback_events "
            "WHERE feedback_id = ?",
            (event.feedback_id,),
        ).fetchone()
        if existing is not None:
            previous = PersonalFeedbackEvent.model_validate_json(
                existing["feedback_json"]
            )
            cls._validate_event_transition(previous, event)
        payload = cls._model_json(event)
        connection.execute(
            """
            INSERT INTO personal_feedback_events (
                feedback_id,
                user_id,
                session_id,
                source_result_id,
                feedback_json,
                status,
                next_action,
                created_at,
                updated_at,
                raw_purged_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(feedback_id) DO UPDATE SET
                feedback_json = excluded.feedback_json,
                status = excluded.status,
                next_action = excluded.next_action,
                updated_at = excluded.updated_at,
                raw_purged_at = excluded.raw_purged_at
            """,
            (
                event.feedback_id,
                event.user_id,
                event.session_id,
                event.source_result_id,
                payload,
                event.status,
                event.next_action,
                event.created_at.isoformat(),
                event.updated_at.isoformat(),
                (
                    event.raw_purged_at.isoformat()
                    if event.raw_purged_at is not None
                    else None
                ),
            ),
        )

    def _purge_feedback_raw_before_sync(
        self,
        cutoff: datetime,
        purged_at: datetime,
    ) -> int:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            event_rows = connection.execute(
                "SELECT feedback_id, feedback_json FROM personal_feedback_events "
                "WHERE created_at < ? AND status IN ("
                "'recovered', 'recovery_failed', 'closed') "
                "AND raw_purged_at IS NULL",
                (cutoff.isoformat(),),
            ).fetchall()
            for row in event_rows:
                event = PersonalFeedbackEvent.model_validate_json(
                    row["feedback_json"]
                ).model_copy(
                    update={
                        "feedback_message": None,
                        "corrected_query": None,
                        "raw_purged_at": purged_at,
                        "updated_at": max(purged_at, cutoff),
                    }
                )
                event = PersonalFeedbackEvent.model_validate(event.model_dump())
                connection.execute(
                    "UPDATE personal_feedback_events SET feedback_json = ?, "
                    "updated_at = ?, raw_purged_at = ? WHERE feedback_id = ?",
                    (
                        self._model_json(event),
                        event.updated_at.isoformat(),
                        purged_at.isoformat(),
                        event.feedback_id,
                    ),
                )
            snapshot_rows = connection.execute(
                "SELECT result_id, snapshot_json "
                "FROM conversation_result_snapshots "
                "WHERE created_at < ? AND raw_purged_at IS NULL",
                (cutoff.isoformat(),),
            ).fetchall()
            for row in snapshot_rows:
                snapshot = ConversationResultSnapshot.model_validate_json(
                    row["snapshot_json"]
                ).model_copy(
                    update={"query": None, "raw_purged_at": purged_at}
                )
                snapshot = ConversationResultSnapshot.model_validate(
                    snapshot.model_dump()
                )
                connection.execute(
                    "UPDATE conversation_result_snapshots SET snapshot_json = ?, "
                    "raw_purged_at = ? WHERE result_id = ?",
                    (
                        self._model_json(snapshot),
                        purged_at.isoformat(),
                        snapshot.result_id,
                    ),
                )
        return len(event_rows) + len(snapshot_rows)

    @staticmethod
    def _model_json(model: object) -> str:
        """使用稳定 JSON 表示保存严格 Pydantic 模型。"""

        model_dump_json = getattr(model, "model_dump_json")
        return model_dump_json(exclude_none=False)

    @staticmethod
    def _validate_event_transition(
        previous: PersonalFeedbackEvent,
        current: PersonalFeedbackEvent,
    ) -> None:
        """禁止身份变化、计数回退和终态回退。"""

        if (
            previous.user_id != current.user_id
            or previous.session_id != current.session_id
            or previous.source_result_id != current.source_result_id
            or previous.created_at != current.created_at
        ):
            raise ConversationStoreError("反馈事件不能改换身份或来源结果")
        if (
            current.clarification_count < previous.clarification_count
            or current.recovery_count < previous.recovery_count
        ):
            raise ConversationStoreError("反馈事件计数不能回退")
        terminal = {"recovered", "recovery_failed", "closed"}
        if previous.status in terminal and current != previous:
            allowed = previous.model_copy(
                update={
                    "memory_statuses": current.memory_statuses,
                    "updated_at": current.updated_at,
                },
                deep=True,
            )
            if current != allowed:
                raise ConversationStoreError("反馈事件终态业务字段不能回退或覆盖")
            if set(previous.memory_statuses) != set(current.memory_statuses):
                raise ConversationStoreError("反馈事件终态不能增删记忆路由")
            for route, previous_status in previous.memory_statuses.items():
                current_status = current.memory_statuses[route]
                if previous_status != "pending" and current_status != previous_status:
                    raise ConversationStoreError("反馈记忆终态不能回退或改写")
        if current.updated_at < previous.updated_at:
            raise ConversationStoreError("反馈事件更新时间不能回退")

    @staticmethod
    def _validate_aware_datetime(value: datetime, *, field_name: str) -> None:
        """数据库时间边界必须显式包含时区。"""

        if value.tzinfo is None or value.utcoffset() is None:
            raise ConversationStoreError(f"{field_name}必须包含时区")

    @classmethod
    def _append_messages(
        cls,
        connection: sqlite3.Connection,
        session: ConversationSession,
    ) -> None:
        """校验持久前缀后只追加当前快照末尾的新消息。"""

        persisted = connection.execute(
            """
            SELECT
                message_id,
                sequence_no,
                role,
                content,
                message_type,
                related_session_id,
                created_at
            FROM conversation_messages
            WHERE user_id = ? AND session_id = ?
            ORDER BY sequence_no
            """,
            (session.user_id, session.session_id),
        ).fetchall()
        if len(session.history) < len(persisted):
            raise ConversationStoreError("会话历史不能删除已持久化消息")
        for index, row in enumerate(persisted):
            turn = session.history[index]
            if (
                turn.role != row["role"]
                or turn.content != row["content"]
                or turn.message_type != row["message_type"]
                or turn.related_session_id != row["related_session_id"]
                or (
                    turn.message_id is not None
                    and turn.message_id != row["message_id"]
                )
                or (
                    turn.sequence_no is not None
                    and turn.sequence_no != row["sequence_no"]
                )
            ):
                raise ConversationStoreError("会话历史不能覆盖已持久化消息")

        timestamp = datetime.now(timezone.utc).isoformat()
        new_turns = session.history[len(persisted) :]
        for offset, turn in enumerate(new_turns, start=len(persisted)):
            if turn.message_id is not None or turn.sequence_no is not None:
                raise ConversationStoreError("新增消息不能复用持久化身份")
            connection.execute(
                """
                INSERT INTO conversation_messages (
                    user_id,
                    session_id,
                    sequence_no,
                    role,
                    content,
                    message_type,
                    related_session_id,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session.user_id,
                    session.session_id,
                    offset,
                    turn.role,
                    turn.content,
                    turn.message_type,
                    turn.related_session_id,
                    (
                        turn.created_at.isoformat()
                        if turn.created_at is not None
                        else timestamp
                    ),
                ),
            )

    @staticmethod
    def _write_child_state(
        connection: sqlite3.Connection,
        session: ConversationSession,
    ) -> None:
        """把文章子会话专属事实写入独立关系表。"""

        if session.session_type != "article_qa":
            return
        timestamp = datetime.now(timezone.utc).isoformat()
        connection.execute(
            """
            INSERT INTO article_qa_sessions (
                user_id,
                child_session_id,
                parent_session_id,
                focus_document_id,
                title_snapshot,
                status,
                unresolved_questions_json,
                cited_document_ids_json,
                handoff_summary,
                created_at,
                updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id, child_session_id) DO UPDATE SET
                parent_session_id = excluded.parent_session_id,
                focus_document_id = excluded.focus_document_id,
                title_snapshot = excluded.title_snapshot,
                status = excluded.status,
                unresolved_questions_json = excluded.unresolved_questions_json,
                cited_document_ids_json = excluded.cited_document_ids_json,
                handoff_summary = excluded.handoff_summary,
                updated_at = excluded.updated_at
            """,
            (
                session.user_id,
                session.session_id,
                session.parent_session_id,
                session.focus_document_id,
                session.focus_document_title,
                session.session_status,
                json.dumps(
                    session.unresolved_questions,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                json.dumps(
                    session.cited_document_ids,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                session.handoff_summary,
                timestamp,
                timestamp,
            ),
        )

    @classmethod
    def _validate_session(cls, session: ConversationSession) -> None:
        if not session.user_id.strip() or not session.session_id.strip():
            raise ConversationStoreError("会话标识不能为空")
        if session.summary_watermark >= len(session.history):
            raise ConversationStoreError("摘要水位不能超过会话消息范围")


__all__ = [
    "SQLiteConversationStore",
]
