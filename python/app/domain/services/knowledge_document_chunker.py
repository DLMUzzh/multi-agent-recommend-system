"""面向知识问答检索的独立 Markdown 文档切分服务。"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from itertools import groupby
from collections.abc import Sequence
from typing import Literal

from app.models.document import DocumentChunk


logger = logging.getLogger(__name__)

_HEADING_PATTERN = re.compile(
    r"^\s{0,3}(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$"
)
_FENCE_PATTERN = re.compile(r"^\s*(?P<marker>`{3,}|~{3,})")
_TOKEN_PATTERN = re.compile(
    r"[\u3400-\u4dbf\u4e00-\u9fff]|[A-Za-z0-9_]+|[^\s]"
)
_TABLE_SEPARATOR_CELL_PATTERN = re.compile(r"^:?-{3,}:?$")

_BlockKind = Literal["text", "heading", "code", "table"]


@dataclass(frozen=True, slots=True)
class _Block:
    heading_path: tuple[str, ...]
    content: str
    kind: _BlockKind


@dataclass(frozen=True, slots=True)
class _TextSegment:
    content: str
    overlap_only: bool = False
    has_leading_overlap: bool = False


@dataclass(frozen=True, slots=True)
class DocumentChunkDraft:
    """尚未计算文档相关身份的结构安全 Chunk 草稿。"""

    heading_path: tuple[str, ...]
    content: str


class KnowledgeDocumentChunker:
    """从标准化 Markdown 生成结构安全、局部稳定的知识 Chunk。"""

    def __init__(
        self,
        *,
        target_tokens: int = 280,
        max_tokens: int = 420,
        overlap_tokens: int = 40,
    ) -> None:
        if target_tokens <= 0:
            raise ValueError("target_tokens 必须大于零")
        if max_tokens < target_tokens:
            raise ValueError("max_tokens 不能小于 target_tokens")
        if overlap_tokens < 0 or overlap_tokens >= target_tokens:
            raise ValueError("overlap_tokens 必须位于零到 target_tokens 之间")
        self.target_tokens = target_tokens
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

    def chunk(
        self,
        document_id: str,
        content_markdown: str,
    ) -> tuple[DocumentChunk, ...]:
        """按原文顺序切分 Markdown，并返回现有 ``DocumentChunk`` 契约。"""

        drafts = self.split(content_markdown)
        return self.materialize(document_id, drafts)

    def split(self, content_markdown: str) -> tuple[DocumentChunkDraft, ...]:
        """只执行 Markdown 结构切分，不计算文档相关身份。"""

        if not isinstance(content_markdown, str) or not content_markdown.strip():
            raise ValueError("文档正文不能为空")
        blocks = self._parse_blocks(content_markdown)
        return tuple(
            DocumentChunkDraft(heading_path=heading_path, content=content)
            for heading_path, section_blocks in groupby(
                blocks,
                key=lambda block: block.heading_path,
            )
            for content in self._chunk_section(tuple(section_blocks))
        )

    def materialize(
        self,
        document_id: str,
        drafts: Sequence[DocumentChunkDraft],
    ) -> tuple[DocumentChunk, ...]:
        """对最终草稿计算位置、Hash、Token 和局部稳定身份。"""

        if not isinstance(document_id, str) or not document_id.strip():
            raise ValueError("document_id 不能为空")
        normalized_document_id = document_id.strip()
        occurrences: dict[tuple[tuple[str, ...], str], int] = {}
        chunks: list[DocumentChunk] = []
        for position, draft in enumerate(drafts):
            if not isinstance(draft, DocumentChunkDraft):
                raise ValueError("Chunk 草稿类型无效")
            content = draft.content.strip()
            if not content:
                raise ValueError("Chunk 草稿正文不能为空")
            content_hash = self._sha256(content)
            occurrence_key = (draft.heading_path, content_hash)
            occurrence = occurrences.get(occurrence_key, 0)
            occurrences[occurrence_key] = occurrence + 1
            chunks.append(
                DocumentChunk(
                    chunk_id=self._chunk_id(
                        normalized_document_id,
                        draft.heading_path,
                        content_hash,
                        occurrence,
                    ),
                    document_id=normalized_document_id,
                    position=position,
                    heading_path=draft.heading_path,
                    content=content,
                    content_hash=content_hash,
                    token_count=self.count_tokens(content),
                )
            )
        return tuple(chunks)

    @staticmethod
    def count_tokens(text: str) -> int:
        """用标准库近似统计中英文、数字和符号 Token 数。"""

        return sum(1 for _ in _TOKEN_PATTERN.finditer(text))

    def _parse_blocks(self, content_markdown: str) -> tuple[_Block, ...]:
        lines = content_markdown.splitlines()
        blocks: list[_Block] = []
        headings: dict[int, str] = {}
        index = 0

        while index < len(lines):
            line = lines[index]
            if not line.strip():
                index += 1
                continue

            heading_match = _HEADING_PATTERN.match(line)
            if heading_match:
                level = len(heading_match.group(1))
                title = heading_match.group(2).strip()
                headings = {
                    current_level: current_title
                    for current_level, current_title in headings.items()
                    if current_level < level
                }
                headings[level] = title
                if self._is_leaf_heading_only(lines, index, level):
                    blocks.append(
                        _Block(
                            tuple(
                                value for _, value in sorted(headings.items())
                            ),
                            title,
                            "heading",
                        )
                    )
                index += 1
                continue

            heading_path = tuple(
                title for _, title in sorted(headings.items())
            )
            fence_match = _FENCE_PATTERN.match(line)
            if fence_match:
                content, index = self._consume_fence(
                    lines,
                    index,
                    fence_match.group("marker"),
                )
                blocks.append(_Block(heading_path, content, "code"))
                continue

            if self._starts_table(lines, index):
                content, index = self._consume_table(lines, index)
                blocks.append(_Block(heading_path, content, "table"))
                continue

            content, index = self._consume_paragraph(lines, index)
            if content:
                blocks.append(_Block(heading_path, content, "text"))

        return tuple(blocks)

    @staticmethod
    def _is_leaf_heading_only(
        lines: list[str],
        heading_index: int,
        heading_level: int,
    ) -> bool:
        for line in lines[heading_index + 1 :]:
            if not line.strip():
                continue
            next_heading = _HEADING_PATTERN.match(line)
            if next_heading is None:
                return False
            return len(next_heading.group(1)) <= heading_level
        return True

    @classmethod
    def _consume_fence(
        cls,
        lines: list[str],
        start: int,
        marker: str,
    ) -> tuple[str, int]:
        block_lines = [lines[start]]
        closing_pattern = cls._closing_fence_pattern(marker)
        index = start + 1
        closed = False
        while index < len(lines):
            block_lines.append(lines[index])
            index += 1
            if closing_pattern.match(block_lines[-1]):
                closed = True
                break
        if not closed:
            block_lines.append(marker)
        return "\n".join(block_lines), index

    @staticmethod
    def _closing_fence_pattern(marker: str) -> re.Pattern[str]:
        return re.compile(
            rf"^\s*{re.escape(marker[0])}{{{len(marker)},}}\s*(?:\n)?$"
        )

    @classmethod
    def _starts_table(cls, lines: list[str], start: int) -> bool:
        return (
            start + 1 < len(lines)
            and cls._is_table_line(lines[start])
            and cls._is_table_separator(lines[start + 1])
        )

    @classmethod
    def _consume_table(
        cls,
        lines: list[str],
        start: int,
    ) -> tuple[str, int]:
        block_lines: list[str] = []
        index = start
        while index < len(lines) and cls._is_table_line(lines[index]):
            block_lines.append(lines[index])
            index += 1
        return "\n".join(block_lines), index

    @classmethod
    def _consume_paragraph(
        cls,
        lines: list[str],
        start: int,
    ) -> tuple[str, int]:
        block_lines: list[str] = []
        index = start
        while index < len(lines):
            line = lines[index]
            if not line.strip():
                break
            if _HEADING_PATTERN.match(line) or _FENCE_PATTERN.match(line):
                break
            if cls._starts_table(lines, index):
                break
            block_lines.append(line)
            index += 1
        return "\n".join(block_lines).strip(), index

    @staticmethod
    def _is_table_line(line: str) -> bool:
        stripped = line.strip()
        return stripped.startswith("|") and stripped.endswith("|")

    @classmethod
    def _is_table_separator(cls, line: str) -> bool:
        if not cls._is_table_line(line):
            return False
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        return bool(cells) and all(
            _TABLE_SEPARATOR_CELL_PATTERN.fullmatch(cell) for cell in cells
        )

    def _chunk_section(self, blocks: tuple[_Block, ...]) -> tuple[str, ...]:
        results: list[str] = []
        text_buffer: list[_TextSegment] = []

        for block in blocks:
            if block.kind in ("text", "heading"):
                for segment in self._split_text_block(block.content):
                    self._append_text_segment(results, text_buffer, segment)
                continue

            self._flush_text_buffer(results, text_buffer)
            if block.kind == "code":
                results.extend(self._split_code_block(block.content))
            else:
                results.extend(self._split_table_block(block.content))

        self._flush_text_buffer(results, text_buffer)
        return tuple(content for content in results if content.strip())

    def _split_text_block(self, content: str) -> tuple[_TextSegment, ...]:
        matches = tuple(_TOKEN_PATTERN.finditer(content))
        if len(matches) <= self.max_tokens:
            return (_TextSegment(content),)

        segments: list[_TextSegment] = []
        token_start = 0
        while token_start < len(matches):
            token_end = min(token_start + self.max_tokens, len(matches))
            piece = content[
                matches[token_start].start() : matches[token_end - 1].end()
            ].strip()
            if piece:
                segments.append(
                    _TextSegment(
                        piece,
                        has_leading_overlap=bool(segments),
                    )
                )
            if token_end == len(matches):
                break
            token_start = token_end - self.overlap_tokens
        return tuple(segments)

    def _append_text_segment(
        self,
        results: list[str],
        buffer: list[_TextSegment],
        segment: _TextSegment,
    ) -> None:
        if not buffer:
            buffer.append(segment)
            return

        current_tokens = self.count_tokens(self._join_text_segments(buffer))
        combined_tokens = current_tokens + self.count_tokens(segment.content)
        if current_tokens >= self.target_tokens or combined_tokens > self.max_tokens:
            results.append(self._join_text_segments(buffer))
            overlap = ""
            if not segment.has_leading_overlap:
                overlap = self._overlap_segment(
                    buffer[-1].content,
                    segment.content,
                )
            buffer.clear()
            if overlap:
                buffer.append(_TextSegment(overlap, overlap_only=True))
        buffer.append(segment)

    @staticmethod
    def _flush_text_buffer(
        results: list[str],
        buffer: list[_TextSegment],
    ) -> None:
        if buffer and any(not segment.overlap_only for segment in buffer):
            results.append(KnowledgeDocumentChunker._join_text_segments(buffer))
        buffer.clear()

    def _overlap_segment(self, previous: str, next_content: str) -> str:
        if self.overlap_tokens == 0:
            return ""
        available = self.max_tokens - self.count_tokens(next_content)
        limit = min(self.overlap_tokens, max(available, 0))
        if limit == 0:
            return ""
        matches = tuple(_TOKEN_PATTERN.finditer(previous))
        if not matches:
            return ""
        start = matches[max(len(matches) - limit, 0)].start()
        return previous[start:].strip()

    @staticmethod
    def _join_text_segments(segments: list[_TextSegment]) -> str:
        return "\n\n".join(segment.content for segment in segments).strip()

    def _split_code_block(self, content: str) -> tuple[str, ...]:
        if self.count_tokens(content) <= self.max_tokens:
            return (content,)

        lines = content.splitlines()
        opening = lines[0]
        closing = lines[-1]
        body_lines = lines[1:-1]
        empty_block = self._format_code_block(opening, (), closing)
        available = self.max_tokens - self.count_tokens(empty_block)
        if available <= 0:
            raise ValueError("代码围栏超过 max_tokens")

        safe_lines: list[str] = []
        for line in body_lines:
            if self.count_tokens(line) <= available:
                safe_lines.append(line)
                continue
            logger.warning("知识文档代码行超过 max_tokens，已按 Token 窗口拆分")
            safe_lines.extend(self._split_oversized_code_line(line, available))

        chunks: list[str] = []
        buffer: list[str] = []
        for line in safe_lines:
            candidate = self._format_code_block(opening, (*buffer, line), closing)
            current = self._format_code_block(opening, tuple(buffer), closing)
            if buffer and (
                self.count_tokens(current) >= self.target_tokens
                or self.count_tokens(candidate) > self.max_tokens
            ):
                chunks.append(current)
                buffer.clear()
            buffer.append(line)

        if buffer or not body_lines:
            chunks.append(self._format_code_block(opening, tuple(buffer), closing))
        return tuple(chunks)

    @staticmethod
    def _format_code_block(
        opening: str,
        body_lines: tuple[str, ...],
        closing: str,
    ) -> str:
        return "\n".join((opening, *body_lines, closing))

    @staticmethod
    def _split_oversized_code_line(line: str, limit: int) -> tuple[str, ...]:
        matches = tuple(_TOKEN_PATTERN.finditer(line))
        if not matches:
            return (line,)
        pieces: list[str] = []
        token_start = 0
        char_start = 0
        while token_start < len(matches):
            token_end = min(token_start + limit, len(matches))
            char_end = (
                matches[token_end].start()
                if token_end < len(matches)
                else len(line)
            )
            pieces.append(line[char_start:char_end])
            char_start = char_end
            token_start = token_end
        return tuple(pieces)

    def _split_table_block(self, content: str) -> tuple[str, ...]:
        if self.count_tokens(content) <= self.max_tokens:
            return (content,)

        lines = content.splitlines()
        header = lines[0]
        separator = lines[1]
        rows = lines[2:]
        table_prefix = "\n".join((header, separator))
        if self.count_tokens(table_prefix) > self.max_tokens:
            raise ValueError("表格表头超过 max_tokens")

        chunks: list[str] = []
        buffer: list[str] = []
        for row in rows:
            single_row = "\n".join((table_prefix, row))
            if self.count_tokens(single_row) > self.max_tokens:
                raise ValueError("表格行超过 max_tokens")
            candidate = "\n".join((table_prefix, *buffer, row))
            current = "\n".join((table_prefix, *buffer))
            if buffer and (
                self.count_tokens(current) >= self.target_tokens
                or self.count_tokens(candidate) > self.max_tokens
            ):
                chunks.append(current)
                buffer.clear()
            buffer.append(row)

        if buffer:
            chunks.append("\n".join((table_prefix, *buffer)))
        return tuple(chunks)

    @staticmethod
    def _sha256(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    @classmethod
    def _chunk_id(
        cls,
        document_id: str,
        heading_path: tuple[str, ...],
        content_hash: str,
        occurrence: int,
    ) -> str:
        serialized_path = json.dumps(
            heading_path,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        fingerprint = cls._sha256(
            f"{document_id}\0{serialized_path}\0{content_hash}\0{occurrence}"
        )
        return f"chk_{fingerprint[:32]}"


__all__ = ["DocumentChunkDraft", "KnowledgeDocumentChunker"]
