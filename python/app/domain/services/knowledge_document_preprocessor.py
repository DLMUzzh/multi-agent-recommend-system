"""知识文档进入 Chunk 切分前的确定性标准处理服务。"""

from __future__ import annotations

import re


_HEADING_PATTERN = re.compile(
    r"^\s{0,3}(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$"
)
_FENCE_PATTERN = re.compile(r"^\s*(?P<marker>`{3,}|~{3,})")
_RESOURCE_ELEMENT_PATTERNS = (
    re.compile(
        r"<readonly-block\b[^>]*(?:/\s*>|>.*?</readonly-block\s*>)",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"<figure\b[^>]*(?:/\s*>|>.*?</figure\s*>)",
        re.IGNORECASE | re.DOTALL,
    ),
)
_RESOURCE_TAG_PATTERN = re.compile(
    r"</?(?:readonly-block|figure|source)\b[^>]*>",
    re.IGNORECASE,
)
_LEADING_METADATA_PATTERN = re.compile(
    r"^\s*(?:作者|分类)\s*[：:]\s*\S.*$"
)
_MEANINGLESS_EMPTY_HEADING_PATTERN = re.compile(
    r"^(?:空章节|空白章节|待补充|待完善|暂无内容|无内容)$"
)


class KnowledgeDocumentPreprocessor:
    """保守清理头部展示元数据和资源占位，保留正文事实。"""

    def process(self, content_markdown: str) -> str:
        """返回只用于派生 Chunk 的标准化 Markdown 工作副本。"""

        if not isinstance(content_markdown, str):
            raise ValueError("文档正文必须是字符串")
        normalized = content_markdown.replace("\r\n", "\n").replace(
            "\r",
            "\n",
        )
        without_leading_metadata = self._remove_leading_metadata(normalized)
        native_heading_only_keys = self._native_heading_only_keys(
            without_leading_metadata
        )
        without_resources = self._clean_resources_outside_fences(
            without_leading_metadata
        )
        without_empty_sections = self._remove_empty_heading_sections(
            without_resources,
            protected_keys=native_heading_only_keys,
        )
        return self._collapse_blank_lines(without_empty_sections)

    @classmethod
    def _remove_leading_metadata(cls, markdown: str) -> str:
        """只移除首个 H1 后、正文开始前连续出现的展示元数据。"""

        lines = markdown.split("\n")
        heading_levels = cls._heading_levels(lines)
        try:
            first_h1_index = heading_levels.index(1)
        except ValueError:
            return markdown

        removed_indexes: set[int] = set()
        found_metadata = False
        for index in range(first_h1_index + 1, len(lines)):
            line = lines[index]
            if not line.strip():
                continue
            if _LEADING_METADATA_PATTERN.fullmatch(line):
                found_metadata = True
                removed_indexes.add(index)
                continue
            break
        if not found_metadata:
            return markdown
        return "\n".join(
            line
            for index, line in enumerate(lines)
            if index not in removed_indexes
        )

    @classmethod
    def _clean_resources_outside_fences(cls, markdown: str) -> str:
        cleaned_parts: list[str] = []
        text_parts: list[str] = []
        closing_pattern: re.Pattern[str] | None = None

        for line in markdown.splitlines(keepends=True):
            if closing_pattern is None:
                fence_match = _FENCE_PATTERN.match(line)
                if fence_match:
                    cleaned_parts.append(
                        cls._clean_text_resources("".join(text_parts))
                    )
                    text_parts.clear()
                    cleaned_parts.append(line)
                    closing_pattern = cls._closing_fence_pattern(
                        fence_match.group("marker")
                    )
                    continue
                text_parts.append(line)
                continue

            cleaned_parts.append(line)
            if closing_pattern.match(line):
                closing_pattern = None

        cleaned_parts.append(cls._clean_text_resources("".join(text_parts)))
        return "".join(cleaned_parts)

    @staticmethod
    def _clean_text_resources(text: str) -> str:
        cleaned = text
        for pattern in _RESOURCE_ELEMENT_PATTERNS:
            cleaned = pattern.sub("\n", cleaned)
        return _RESOURCE_TAG_PATTERN.sub("", cleaned)

    @classmethod
    def _remove_empty_heading_sections(
        cls,
        markdown: str,
        *,
        protected_keys: frozenset[tuple[tuple[str, ...], int]],
    ) -> str:
        lines = markdown.split("\n")
        entries = cls._heading_entries(lines)
        meaningful_keys: set[tuple[tuple[str, ...], int]] = set()
        removed_indexes: set[int] = set()

        for entry_index in range(len(entries) - 1, -1, -1):
            index, level, _title, key = entries[entry_index]
            next_index = (
                entries[entry_index + 1][0]
                if entry_index + 1 < len(entries)
                else len(lines)
            )
            has_direct_body = any(
                line.strip() for line in lines[index + 1 : next_index]
            )
            descendant_keys: list[tuple[tuple[str, ...], int]] = []
            for _, child_level, _, child_key in entries[entry_index + 1 :]:
                if child_level <= level:
                    break
                descendant_keys.append(child_key)
            has_meaningful_descendant = any(
                child_key in meaningful_keys for child_key in descendant_keys
            )
            if (
                has_direct_body
                or key in protected_keys
                or has_meaningful_descendant
            ):
                meaningful_keys.add(key)
            else:
                removed_indexes.add(index)

        return "\n".join(
            line for index, line in enumerate(lines) if index not in removed_indexes
        )

    @classmethod
    def _native_heading_only_keys(
        cls,
        markdown: str,
    ) -> frozenset[tuple[tuple[str, ...], int]]:
        lines = markdown.split("\n")
        entries = cls._heading_entries(lines)
        protected: set[tuple[tuple[str, ...], int]] = set()
        for entry_index, (index, level, title, key) in enumerate(entries):
            next_entry = (
                entries[entry_index + 1]
                if entry_index + 1 < len(entries)
                else None
            )
            next_index = next_entry[0] if next_entry is not None else len(lines)
            has_direct_body = any(
                line.strip() for line in lines[index + 1 : next_index]
            )
            is_leaf = next_entry is None or next_entry[1] <= level
            if (
                is_leaf
                and not has_direct_body
                and not _MEANINGLESS_EMPTY_HEADING_PATTERN.fullmatch(title)
            ):
                protected.add(key)
        return frozenset(protected)

    @classmethod
    def _heading_entries(
        cls,
        lines: list[str],
    ) -> tuple[
        tuple[int, int, str, tuple[tuple[str, ...], int]],
        ...,
    ]:
        entries: list[
            tuple[int, int, str, tuple[tuple[str, ...], int]]
        ] = []
        headings: dict[int, str] = {}
        occurrences: dict[tuple[str, ...], int] = {}
        closing_pattern: re.Pattern[str] | None = None
        for index, line in enumerate(lines):
            if closing_pattern is not None:
                if closing_pattern.match(line):
                    closing_pattern = None
                continue
            fence_match = _FENCE_PATTERN.match(line)
            if fence_match:
                closing_pattern = cls._closing_fence_pattern(
                    fence_match.group("marker")
                )
                continue
            heading_match = _HEADING_PATTERN.match(line)
            if heading_match is None:
                continue
            level = len(heading_match.group(1))
            title = heading_match.group(2).strip()
            headings = {
                current_level: current_title
                for current_level, current_title in headings.items()
                if current_level < level
            }
            headings[level] = title
            path = tuple(value for _, value in sorted(headings.items()))
            occurrence = occurrences.get(path, 0)
            occurrences[path] = occurrence + 1
            entries.append((index, level, title, (path, occurrence)))
        return tuple(entries)

    @classmethod
    def _heading_levels(cls, lines: list[str]) -> list[int | None]:
        levels: list[int | None] = [None] * len(lines)
        closing_pattern: re.Pattern[str] | None = None
        for index, line in enumerate(lines):
            if closing_pattern is not None:
                if closing_pattern.match(line):
                    closing_pattern = None
                continue
            fence_match = _FENCE_PATTERN.match(line)
            if fence_match:
                closing_pattern = cls._closing_fence_pattern(
                    fence_match.group("marker")
                )
                continue
            heading_match = _HEADING_PATTERN.match(line)
            if heading_match:
                levels[index] = len(heading_match.group(1))
        return levels

    @staticmethod
    def _closing_fence_pattern(marker: str) -> re.Pattern[str]:
        return re.compile(
            rf"^\s*{re.escape(marker[0])}{{{len(marker)},}}\s*$"
        )

    @classmethod
    def _collapse_blank_lines(cls, markdown: str) -> str:
        normalized_lines: list[str] = []
        blank_pending = False
        closing_pattern: re.Pattern[str] | None = None
        for line in markdown.split("\n"):
            if closing_pattern is not None:
                normalized_lines.append(line)
                if closing_pattern.match(line):
                    closing_pattern = None
                continue
            fence_match = _FENCE_PATTERN.match(line)
            if fence_match:
                if blank_pending and normalized_lines:
                    normalized_lines.append("")
                blank_pending = False
                normalized_lines.append(line)
                closing_pattern = cls._closing_fence_pattern(
                    fence_match.group("marker")
                )
                continue
            if not line.strip():
                blank_pending = bool(normalized_lines)
                continue
            if blank_pending:
                normalized_lines.append("")
                blank_pending = False
            normalized_lines.append(line)
        return "\n".join(normalized_lines)


__all__ = ["KnowledgeDocumentPreprocessor"]
