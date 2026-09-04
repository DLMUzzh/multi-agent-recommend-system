"""推荐与知识问答共享的 SQLite Chunk 读取与事实回查仓储。"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from app.infrastructure.database.sqlite.document_repository import (
    SQLiteDocumentRepository,
)
from app.models.document import (
    Document,
    DocumentChunk,
    DocumentChunkImageLink,
    DocumentFact,
    DocumentImage,
)
from app.models.knowledge_qa import KnowledgeChunkRecord, KnowledgeImageEvidence


class KnowledgeRepository(Protocol):
    """共享 Chunk 召回链依赖的最小持久化边界。"""

    def replace_document(
        self,
        document: Document,
        chunks: Sequence[DocumentChunk],
    ) -> None:
        """原子替换完整文档和全部派生 Chunk。"""

        ...

    def replace_document_bundle(
        self,
        document: Document,
        chunks: Sequence[DocumentChunk],
        images: Sequence[DocumentImage],
        links: Sequence[DocumentChunkImageLink],
    ) -> None:
        """单事务替换文档、Chunk、图片事实和关联。"""

        ...

    def get_document(self, document_id: str) -> Document | None:
        """读取已有文档，用于保留创建时间。"""

        ...

    def list_ready_chunks(
        self,
        document_ids: Sequence[str] = (),
    ) -> tuple[KnowledgeChunkRecord, ...]:
        """一致读取全部或指定 ready 文档的 Chunk。"""

        ...

    def get_chunks_by_ids(
        self,
        chunk_ids: Sequence[str],
    ) -> tuple[KnowledgeChunkRecord, ...]:
        """按请求顺序回查仍存在的 ready Chunk。"""

        ...

    def get_document_facts(
        self,
        document_ids: Sequence[str],
    ) -> dict[str, DocumentFact]:
        """批量读取推荐重排需要的文档事实。"""

        ...

    def get_image(self, image_id: str) -> DocumentImage | None:
        """读取图片事实，不存在时返回空。"""

        ...

    def mark_image_ready(
        self,
        *,
        image_id: str,
        content_hash: str,
        storage_key: str,
        mime_type: str,
        byte_size: int,
    ) -> DocumentImage:
        """只更新已存在图片的完整二进制事实。"""

        ...

    def list_ready_images_by_chunk_ids(
        self,
        chunk_ids: Sequence[str],
    ) -> tuple[KnowledgeImageEvidence, ...]:
        """按 Chunk 输入顺序返回 ready 图片及真实关联。"""

        ...

    def list_ready_image_storage_keys(self) -> tuple[str, ...]:
        """返回当前仍被 ready 图片事实引用的安全存储 Key。"""

        ...


class SQLiteKnowledgeRepository(SQLiteDocumentRepository):
    """在现有两张表上增加知识检索所需的只读查询。"""

    def __init__(self, path: str | Path | None = None) -> None:
        super().__init__(path)

    def list_ready_chunks(
        self,
        document_ids: Sequence[str] = (),
    ) -> tuple[KnowledgeChunkRecord, ...]:
        """按文档 ID 和 Chunk 位置稳定读取 ready 知识快照。"""

        normalized_ids = self._normalize_ids(document_ids)
        where_clause = "d.status = 'ready'"
        parameters: tuple[str, ...] = ()
        if normalized_ids:
            placeholders = ",".join("?" for _ in normalized_ids)
            where_clause += f" AND d.document_id IN ({placeholders})"
            parameters = normalized_ids
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT
                    c.chunk_id,
                    c.document_id,
                    d.title,
                    d.topics,
                    d.content_type,
                    d.difficulty,
                    d.author_id,
                    c.position,
                    c.heading_path,
                    c.content,
                    c.content_hash,
                    c.token_count
                FROM document_chunks AS c
                JOIN documents AS d ON d.document_id = c.document_id
                WHERE {where_clause}
                ORDER BY c.document_id, c.position
                """,
                parameters,
            ).fetchall()
        return tuple(self._record_from_row(row) for row in rows)

    def get_chunks_by_ids(
        self,
        chunk_ids: Sequence[str],
    ) -> tuple[KnowledgeChunkRecord, ...]:
        """按候选排名顺序回查 Chunk，静默丢弃已删除候选。"""

        normalized_ids = self._normalize_ids(chunk_ids)
        if not normalized_ids:
            return ()
        placeholders = ",".join("?" for _ in normalized_ids)
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT
                    c.chunk_id,
                    c.document_id,
                    d.title,
                    d.topics,
                    d.content_type,
                    d.difficulty,
                    d.author_id,
                    c.position,
                    c.heading_path,
                    c.content,
                    c.content_hash,
                    c.token_count
                FROM document_chunks AS c
                JOIN documents AS d ON d.document_id = c.document_id
                WHERE d.status = 'ready'
                  AND c.chunk_id IN ({placeholders})
                """,
                normalized_ids,
            ).fetchall()
        records = {
            row["chunk_id"]: self._record_from_row(row) for row in rows
        }
        return tuple(
            records[chunk_id]
            for chunk_id in normalized_ids
            if chunk_id in records
        )

    def list_ready_images_by_chunk_ids(
        self,
        chunk_ids: Sequence[str],
    ) -> tuple[KnowledgeImageEvidence, ...]:
        """按 Chunk 输入顺序、图片位置稳定返回 ready 图片。"""

        normalized_ids = self._normalize_ids(chunk_ids)
        if not normalized_ids:
            return ()
        placeholders = ",".join("?" for _ in normalized_ids)
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT
                    l.chunk_id,
                    i.image_id,
                    i.document_id,
                    d.title,
                    i.image_key,
                    i.position,
                    i.heading_path,
                    i.caption,
                    i.content_hash
                FROM document_chunk_images AS l
                JOIN document_images AS i ON i.image_id = l.image_id
                JOIN documents AS d ON d.document_id = i.document_id
                WHERE i.status = 'ready'
                  AND d.status = 'ready'
                  AND l.chunk_id IN ({placeholders})
                """,
                normalized_ids,
            ).fetchall()
        chunk_order = {chunk_id: index for index, chunk_id in enumerate(normalized_ids)}
        grouped: dict[str, dict[str, object]] = {}
        for row in sorted(
            rows,
            key=lambda item: (
                chunk_order[item["chunk_id"]],
                item["position"],
                item["image_id"],
            ),
        ):
            item = grouped.setdefault(
                row["image_id"],
                {
                    "row": row,
                    "linked_chunk_ids": [],
                },
            )
            linked = item["linked_chunk_ids"]
            assert isinstance(linked, list)
            if row["chunk_id"] not in linked:
                linked.append(row["chunk_id"])
        evidence: list[KnowledgeImageEvidence] = []
        for item in grouped.values():
            row = item["row"]
            linked = item["linked_chunk_ids"]
            assert isinstance(row, sqlite3.Row)
            assert isinstance(linked, list)
            try:
                heading_path = json.loads(row["heading_path"])
            except (TypeError, json.JSONDecodeError):
                raise ValueError("数据库中的图片 heading_path 无效") from None
            evidence.append(
                KnowledgeImageEvidence(
                    image_id=row["image_id"],
                    document_id=row["document_id"],
                    title=row["title"],
                    image_key=row["image_key"],
                    heading_path=heading_path,
                    caption=row["caption"],
                    content_hash=row["content_hash"],
                    linked_chunk_ids=tuple(linked),
                )
            )
        return tuple(evidence)

    def list_ready_image_storage_keys(self) -> tuple[str, ...]:
        """按 Key 稳定返回当前 ready 图片使用的内容对象。"""

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT DISTINCT storage_key
                FROM document_images
                WHERE status = 'ready'
                  AND storage_key IS NOT NULL
                ORDER BY storage_key
                """
            ).fetchall()
        return tuple(row["storage_key"] for row in rows)

    @staticmethod
    def _normalize_ids(values: Sequence[str]) -> tuple[str, ...]:
        normalized: list[str] = []
        seen: set[str] = set()
        for value in values:
            if not isinstance(value, str) or not value.strip():
                raise ValueError("知识 ID 不能为空")
            cleaned = value.strip()
            if cleaned not in seen:
                normalized.append(cleaned)
                seen.add(cleaned)
        return tuple(normalized)

    @staticmethod
    def _record_from_row(row: sqlite3.Row) -> KnowledgeChunkRecord:
        try:
            heading_path = json.loads(row["heading_path"])
        except (TypeError, json.JSONDecodeError):
            raise ValueError("数据库中的 heading_path 无效") from None
        if not isinstance(heading_path, list) or any(
            not isinstance(value, str) for value in heading_path
        ):
            raise ValueError("数据库中的 heading_path 无效")
        return KnowledgeChunkRecord.model_validate(
            {
                "chunk_id": row["chunk_id"],
                "document_id": row["document_id"],
                "title": row["title"],
                "topics": SQLiteDocumentRepository._json_list(
                    row["topics"],
                    "topics",
                ),
                "content_type": row["content_type"],
                "difficulty": row["difficulty"],
                "author_id": row["author_id"],
                "position": row["position"],
                "heading_path": heading_path,
                "content": row["content"],
                "content_hash": row["content_hash"],
                "token_count": row["token_count"],
            }
        )


__all__ = ["KnowledgeRepository", "SQLiteKnowledgeRepository"]
