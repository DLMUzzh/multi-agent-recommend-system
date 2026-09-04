"""知识问答文档事实与切分结果契约。"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field, field_validator, model_validator

from app.models.common import _StrictModel


DocumentStatus = Literal["ready"]
DocumentContentType = Literal[
    "tutorial",
    "analysis",
    "case_study",
    "technical_design",
]
DocumentDifficulty = Literal["beginner", "intermediate", "advanced"]
ImageStatus = Literal["pending", "ready"]
ImageMimeType = Literal[
    "image/png",
    "image/jpeg",
    "image/webp",
    "image/gif",
]


class Document(_StrictModel):
    """SQLite 中保存的完整 Markdown 文档事实。"""

    document_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    content_markdown: str = Field(min_length=1)
    topics: list[str] = Field(min_length=1, max_length=20)
    content_type: DocumentContentType
    difficulty: DocumentDifficulty
    author_id: str = Field(min_length=1)
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: DocumentStatus = "ready"
    created_at: datetime
    updated_at: datetime

    @field_validator("created_at", "updated_at")
    @classmethod
    def validate_timezone(cls, value: datetime) -> datetime:
        """持久化时间必须带时区，避免不同时区运行时产生歧义。"""

        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("文档时间必须包含时区")
        return value

    @field_validator("topics")
    @classmethod
    def normalize_topics(cls, values: list[str]) -> list[str]:
        """清理主题并按大小写无关规则去重。"""

        normalized: list[str] = []
        seen: set[str] = set()
        for value in values:
            if not isinstance(value, str) or not value.strip():
                raise ValueError("文档主题不能为空")
            cleaned = " ".join(value.split())
            key = cleaned.casefold()
            if key not in seen:
                normalized.append(cleaned)
                seen.add(key)
        if not normalized:
            raise ValueError("文档主题不能为空")
        return normalized


class DocumentFact(_StrictModel):
    """推荐与画像批量读取的轻量文档事实。"""

    document_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    topics: list[str] = Field(min_length=1, max_length=20)
    content_type: DocumentContentType
    difficulty: DocumentDifficulty
    author_id: str = Field(min_length=1)
    total_token_count: int = Field(ge=1)


class DocumentChunk(_StrictModel):
    """由完整文档确定性派生、供后续检索使用的文本块。"""

    chunk_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    position: int = Field(ge=0)
    heading_path: tuple[str, ...] = ()
    content: str = Field(min_length=1)
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    token_count: int = Field(ge=1)


class DocumentImage(_StrictModel):
    """文档中的图片事实，二进制位置只对基础设施可见。"""

    image_id: str = Field(pattern=r"^img-[0-9a-f]{32}$")
    document_id: str = Field(min_length=1)
    image_key: str = Field(
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$"
    )
    position: int = Field(ge=0)
    heading_path: tuple[str, ...] = ()
    caption: str = Field(min_length=1, max_length=500)
    status: ImageStatus = "pending"
    content_hash: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    storage_key: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{2}/[0-9a-f]{64}\.(?:png|jpg|webp|gif)$",
    )
    mime_type: ImageMimeType | None = None
    byte_size: int | None = Field(default=None, ge=1, le=8 * 1024 * 1024)

    @model_validator(mode="after")
    def validate_binary_state(self) -> DocumentImage:
        """pending 不得泄露半完成二进制，ready 必须字段完整。"""

        binary_fields = (
            self.content_hash,
            self.storage_key,
            self.mime_type,
            self.byte_size,
        )
        if self.status == "pending" and any(
            value is not None for value in binary_fields
        ):
            raise ValueError("pending 图片不能携带二进制字段")
        if self.status == "ready" and any(
            value is None for value in binary_fields
        ):
            raise ValueError("ready 图片必须包含完整二进制字段")
        return self


class DocumentChunkImageLink(_StrictModel):
    """一张图片与包含其文本代理的真实 Chunk 关联。"""

    chunk_id: str = Field(min_length=1)
    image_id: str = Field(pattern=r"^img-[0-9a-f]{32}$")


__all__ = [
    "Document",
    "DocumentChunk",
    "DocumentChunkImageLink",
    "DocumentContentType",
    "DocumentDifficulty",
    "DocumentFact",
    "DocumentImage",
    "DocumentStatus",
    "ImageMimeType",
    "ImageStatus",
]
