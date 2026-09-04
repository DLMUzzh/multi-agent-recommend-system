"""知识问答文档与 Chunk 的 SQLite 仓储。"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from app.config.paths import DOCUMENT_DATABASE_PATH
from app.models.document import (
    Document,
    DocumentChunk,
    DocumentChunkImageLink,
    DocumentFact,
    DocumentImage,
)


class DocumentRepository(Protocol):
    """应用服务依赖的最小文档持久化边界。"""

    def replace_document(
        self,
        document: Document,
        chunks: Sequence[DocumentChunk],
    ) -> None:
        """在一个事务中写入文档并替换全部派生 Chunk。"""

        ...

    def get_document(self, document_id: str) -> Document | None:
        """按 ID 读取完整文档，不存在时返回空。"""

        ...

    def list_chunks(self, document_id: str) -> tuple[DocumentChunk, ...]:
        """按位置返回指定文档的全部 Chunk。"""

        ...

    def get_document_facts(
        self,
        document_ids: Sequence[str],
    ) -> dict[str, DocumentFact]:
        """批量读取推荐和画像需要的文档事实。"""

        ...


class SQLiteDocumentRepository:
    """使用独立短连接和事务维护本地 SQLite 文档事实。"""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else DOCUMENT_DATABASE_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_schema()
        self.path.chmod(0o600)

    def replace_document(
        self,
        document: Document,
        chunks: Sequence[DocumentChunk],
    ) -> None:
        """校验派生数据后，原子更新文档并完整替换 Chunk。"""

        self._replace_document_bundle(
            document,
            chunks,
            (),
            (),
            reject_existing_images=True,
        )

    def replace_document_bundle(
        self,
        document: Document,
        chunks: Sequence[DocumentChunk],
        images: Sequence[DocumentImage],
        links: Sequence[DocumentChunkImageLink],
    ) -> None:
        """单事务替换文档、Chunk、图片事实和关联。"""

        self._replace_document_bundle(
            document,
            chunks,
            images,
            links,
            reject_existing_images=False,
        )

    def _replace_document_bundle(
        self,
        document: Document,
        chunks: Sequence[DocumentChunk],
        images: Sequence[DocumentImage],
        links: Sequence[DocumentChunkImageLink],
        *,
        reject_existing_images: bool,
    ) -> None:
        """执行组合事务，并保护旧无图片入口不静默删除图片。"""

        self._validate_chunks(document.document_id, chunks)
        self._validate_images(document.document_id, images, chunks, links)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing_images = {
                row["image_id"]: self._image_from_row(row)
                for row in connection.execute(
                    """
                    SELECT *
                    FROM document_images
                    WHERE document_id = ?
                    """,
                    (document.document_id,),
                ).fetchall()
            }
            if reject_existing_images and existing_images:
                raise ValueError("已有图片的文档必须使用组合事务替换")
            connection.execute(
                """
                INSERT INTO documents (
                    document_id,
                    title,
                    content_markdown,
                    topics,
                    content_type,
                    difficulty,
                    author_id,
                    content_hash,
                    status,
                    created_at,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(document_id) DO UPDATE SET
                    title = excluded.title,
                    content_markdown = excluded.content_markdown,
                    topics = excluded.topics,
                    content_type = excluded.content_type,
                    difficulty = excluded.difficulty,
                    author_id = excluded.author_id,
                    content_hash = excluded.content_hash,
                    status = excluded.status,
                    updated_at = excluded.updated_at
                """,
                (
                    document.document_id,
                    document.title,
                    document.content_markdown,
                    self._json_array(document.topics),
                    document.content_type,
                    document.difficulty,
                    document.author_id,
                    document.content_hash,
                    document.status,
                    document.created_at.isoformat(),
                    document.updated_at.isoformat(),
                ),
            )
            connection.execute(
                "DELETE FROM document_chunks WHERE document_id = ?",
                (document.document_id,),
            )
            connection.executemany(
                """
                INSERT INTO document_chunks (
                    chunk_id,
                    document_id,
                    position,
                    heading_path,
                    content,
                    content_hash,
                    token_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    (
                        chunk.chunk_id,
                        chunk.document_id,
                        chunk.position,
                        json.dumps(
                            chunk.heading_path,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                        chunk.content,
                        chunk.content_hash,
                        chunk.token_count,
                    )
                    for chunk in chunks
                ),
            )
            connection.execute(
                "DELETE FROM document_images WHERE document_id = ?",
                (document.document_id,),
            )
            merged_images = tuple(
                self._preserve_ready_image(image, existing_images.get(image.image_id))
                for image in images
            )
            connection.executemany(
                """
                INSERT INTO document_images (
                    image_id,
                    document_id,
                    image_key,
                    position,
                    heading_path,
                    caption,
                    status,
                    content_hash,
                    storage_key,
                    mime_type,
                    byte_size
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    (
                        image.image_id,
                        image.document_id,
                        image.image_key,
                        image.position,
                        json.dumps(
                            image.heading_path,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                        image.caption,
                        image.status,
                        image.content_hash,
                        image.storage_key,
                        image.mime_type,
                        image.byte_size,
                    )
                    for image in merged_images
                ),
            )
            connection.executemany(
                """
                INSERT INTO document_chunk_images (chunk_id, image_id)
                VALUES (?, ?)
                """,
                ((link.chunk_id, link.image_id) for link in links),
            )

    def get_image(self, image_id: str) -> DocumentImage | None:
        """读取图片事实，不存在时返回空。"""

        normalized_id = self._required_id(image_id, "image_id")
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM document_images WHERE image_id = ?",
                (normalized_id,),
            ).fetchone()
        return self._image_from_row(row) if row is not None else None

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

        normalized_id = self._required_id(image_id, "image_id")
        candidate = DocumentImage(
            image_id=normalized_id,
            document_id="placeholder",
            image_key="placeholder",
            position=0,
            caption="placeholder",
            status="ready",
            content_hash=content_hash,
            storage_key=storage_key,
            mime_type=mime_type,
            byte_size=byte_size,
        )
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE document_images
                SET status = 'ready',
                    content_hash = ?,
                    storage_key = ?,
                    mime_type = ?,
                    byte_size = ?
                WHERE image_id = ?
                """,
                (
                    candidate.content_hash,
                    candidate.storage_key,
                    candidate.mime_type,
                    candidate.byte_size,
                    normalized_id,
                ),
            )
            if cursor.rowcount != 1:
                raise ValueError("图片不存在")
        image = self.get_image(normalized_id)
        if image is None:
            raise ValueError("图片不存在")
        return image

    def get_document(self, document_id: str) -> Document | None:
        """按 ID 读取一条完整 Markdown 文档。"""

        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT
                    document_id,
                    title,
                    content_markdown,
                    topics,
                    content_type,
                    difficulty,
                    author_id,
                    content_hash,
                    status,
                    created_at,
                    updated_at
                FROM documents
                WHERE document_id = ?
                """,
                (document_id,),
            ).fetchone()
        if row is None:
            return None
        payload = dict(row)
        payload["topics"] = self._json_list(payload["topics"], "topics")
        return Document.model_validate(payload)

    def list_chunks(self, document_id: str) -> tuple[DocumentChunk, ...]:
        """按连续位置读取指定文档的全部 Chunk。"""

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    chunk_id,
                    document_id,
                    position,
                    heading_path,
                    content,
                    content_hash,
                    token_count
                FROM document_chunks
                WHERE document_id = ?
                ORDER BY position
                """,
                (document_id,),
            ).fetchall()
        return tuple(self._chunk_from_row(row) for row in rows)

    def count_documents(self) -> int:
        """返回文档总数，供导入验证和运维探针使用。"""

        with self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS count FROM documents"
            ).fetchone()
        return int(row["count"])

    def get_document_facts(
        self,
        document_ids: Sequence[str],
    ) -> dict[str, DocumentFact]:
        """批量读取文档推荐元数据和派生 Chunk 总 token 数。"""

        normalized_ids = self._normalize_ids(document_ids)
        if not normalized_ids:
            return {}
        placeholders = ",".join("?" for _ in normalized_ids)
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT
                    d.document_id,
                    d.title,
                    d.topics,
                    d.content_type,
                    d.difficulty,
                    d.author_id,
                    SUM(c.token_count) AS total_token_count
                FROM documents AS d
                JOIN document_chunks AS c ON c.document_id = d.document_id
                WHERE d.status = 'ready'
                  AND d.document_id IN ({placeholders})
                GROUP BY d.document_id
                """,
                normalized_ids,
            ).fetchall()
        return {
            row["document_id"]: DocumentFact.model_validate(
                {
                    "document_id": row["document_id"],
                    "title": row["title"],
                    "topics": self._json_list(row["topics"], "topics"),
                    "content_type": row["content_type"],
                    "difficulty": row["difficulty"],
                    "author_id": row["author_id"],
                    "total_token_count": row["total_token_count"],
                }
            )
            for row in rows
        }

    def _initialize_schema(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    content_markdown TEXT NOT NULL,
                    topics TEXT NOT NULL,
                    content_type TEXT NOT NULL CHECK (
                        content_type IN (
                            'tutorial', 'analysis', 'case_study', 'technical_design'
                        )
                    ),
                    difficulty TEXT NOT NULL CHECK (
                        difficulty IN ('beginner', 'intermediate', 'advanced')
                    ),
                    author_id TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (status = 'ready'),
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    CHECK (length(trim(document_id)) > 0),
                    CHECK (length(trim(title)) > 0),
                    CHECK (length(trim(content_markdown)) > 0),
                    CHECK (json_valid(topics) AND json_type(topics) = 'array'),
                    CHECK (json_array_length(topics) > 0),
                    CHECK (length(trim(author_id)) > 0),
                    CHECK (length(content_hash) = 64)
                );

                CREATE TABLE IF NOT EXISTS document_chunks (
                    chunk_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    position INTEGER NOT NULL CHECK (position >= 0),
                    heading_path TEXT NOT NULL,
                    content TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    token_count INTEGER NOT NULL CHECK (token_count > 0),
                    FOREIGN KEY (document_id)
                        REFERENCES documents(document_id)
                        ON DELETE CASCADE,
                    UNIQUE (document_id, position),
                    CHECK (length(trim(chunk_id)) > 0),
                    CHECK (length(trim(content)) > 0),
                    CHECK (length(content_hash) = 64)
                );

                CREATE INDEX IF NOT EXISTS idx_document_chunks_document_position
                ON document_chunks(document_id, position);

                CREATE TABLE IF NOT EXISTS document_images (
                    image_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    image_key TEXT NOT NULL,
                    position INTEGER NOT NULL CHECK (position >= 0),
                    heading_path TEXT NOT NULL,
                    caption TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (status IN ('pending', 'ready')),
                    content_hash TEXT,
                    storage_key TEXT,
                    mime_type TEXT,
                    byte_size INTEGER,
                    FOREIGN KEY (document_id)
                        REFERENCES documents(document_id)
                        ON DELETE CASCADE,
                    UNIQUE (document_id, image_key),
                    UNIQUE (document_id, position),
                    CHECK (length(trim(caption)) > 0),
                    CHECK (
                        (
                            status = 'pending'
                            AND content_hash IS NULL
                            AND storage_key IS NULL
                            AND mime_type IS NULL
                            AND byte_size IS NULL
                        )
                        OR
                        (
                            status = 'ready'
                            AND length(content_hash) = 64
                            AND length(trim(storage_key)) > 0
                            AND mime_type IN (
                                'image/png',
                                'image/jpeg',
                                'image/webp',
                                'image/gif'
                            )
                            AND byte_size BETWEEN 1 AND 8388608
                        )
                    )
                );

                CREATE TABLE IF NOT EXISTS document_chunk_images (
                    chunk_id TEXT NOT NULL,
                    image_id TEXT NOT NULL,
                    PRIMARY KEY (chunk_id, image_id),
                    FOREIGN KEY (chunk_id)
                        REFERENCES document_chunks(chunk_id)
                        ON DELETE CASCADE,
                    FOREIGN KEY (image_id)
                        REFERENCES document_images(image_id)
                        ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_document_images_document_position
                ON document_images(document_id, position);

                CREATE INDEX IF NOT EXISTS idx_document_chunk_images_image
                ON document_chunk_images(image_id);
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    @staticmethod
    def _normalize_ids(values: Sequence[str]) -> tuple[str, ...]:
        normalized: list[str] = []
        seen: set[str] = set()
        for value in values:
            if not isinstance(value, str) or not value.strip():
                raise ValueError("文档 ID 不能为空")
            cleaned = value.strip()
            if cleaned not in seen:
                normalized.append(cleaned)
                seen.add(cleaned)
        return tuple(normalized)

    @staticmethod
    def _required_id(value: str, field_name: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field_name} 不能为空")
        return value.strip()

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
    def _validate_chunks(
        document_id: str,
        chunks: Sequence[DocumentChunk],
    ) -> None:
        positions = [chunk.position for chunk in chunks]
        if positions != list(range(len(chunks))):
            raise ValueError("Chunk position 必须从零连续递增")
        if any(chunk.document_id != document_id for chunk in chunks):
            raise ValueError("Chunk 必须属于当前文档")
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("Chunk ID 不能重复")

    @staticmethod
    def _validate_images(
        document_id: str,
        images: Sequence[DocumentImage],
        chunks: Sequence[DocumentChunk],
        links: Sequence[DocumentChunkImageLink],
    ) -> None:
        positions = [image.position for image in images]
        if positions != list(range(len(images))):
            raise ValueError("图片 position 必须从零连续递增")
        if any(image.document_id != document_id for image in images):
            raise ValueError("图片必须属于当前文档")
        image_ids = [image.image_id for image in images]
        if len(image_ids) != len(set(image_ids)):
            raise ValueError("图片 ID 不能重复")
        image_keys = [image.image_key for image in images]
        if len(image_keys) != len(set(image_keys)):
            raise ValueError("图片标识不能重复")
        chunk_ids = {chunk.chunk_id for chunk in chunks}
        image_id_set = set(image_ids)
        link_pairs = [(link.chunk_id, link.image_id) for link in links]
        if len(link_pairs) != len(set(link_pairs)):
            raise ValueError("图片关联不能重复")
        if any(link.chunk_id not in chunk_ids for link in links):
            raise ValueError("图片关联包含未知 Chunk")
        if any(link.image_id not in image_id_set for link in links):
            raise ValueError("图片关联包含未知图片")

    @staticmethod
    def _preserve_ready_image(
        image: DocumentImage,
        existing: DocumentImage | None,
    ) -> DocumentImage:
        if (
            image.status == "pending"
            and existing is not None
            and existing.status == "ready"
            and existing.document_id == image.document_id
            and existing.image_key == image.image_key
        ):
            return image.model_copy(
                update={
                    "status": "ready",
                    "content_hash": existing.content_hash,
                    "storage_key": existing.storage_key,
                    "mime_type": existing.mime_type,
                    "byte_size": existing.byte_size,
                }
            )
        return image

    @staticmethod
    def _chunk_from_row(row: sqlite3.Row) -> DocumentChunk:
        try:
            heading_path = json.loads(row["heading_path"])
        except (TypeError, json.JSONDecodeError):
            raise ValueError("数据库中的 heading_path 无效") from None
        if not isinstance(heading_path, list) or any(
            not isinstance(value, str) for value in heading_path
        ):
            raise ValueError("数据库中的 heading_path 无效")
        return DocumentChunk.model_validate(
            {
                "chunk_id": row["chunk_id"],
                "document_id": row["document_id"],
                "position": row["position"],
                "heading_path": heading_path,
                "content": row["content"],
                "content_hash": row["content_hash"],
                "token_count": row["token_count"],
            }
        )

    @staticmethod
    def _image_from_row(row: sqlite3.Row) -> DocumentImage:
        try:
            heading_path = json.loads(row["heading_path"])
        except (TypeError, json.JSONDecodeError):
            raise ValueError("数据库中的图片 heading_path 无效") from None
        if not isinstance(heading_path, list) or any(
            not isinstance(value, str) for value in heading_path
        ):
            raise ValueError("数据库中的图片 heading_path 无效")
        return DocumentImage.model_validate(
            {
                "image_id": row["image_id"],
                "document_id": row["document_id"],
                "image_key": row["image_key"],
                "position": row["position"],
                "heading_path": heading_path,
                "caption": row["caption"],
                "status": row["status"],
                "content_hash": row["content_hash"],
                "storage_key": row["storage_key"],
                "mime_type": row["mime_type"],
                "byte_size": row["byte_size"],
            }
        )


__all__ = ["DocumentRepository", "SQLiteDocumentRepository"]
