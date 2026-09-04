"""把规范图片锚点派生为可检索文本和可信 Chunk 关联。"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence

from app.domain.services.knowledge_document_chunker import (
    DocumentChunkDraft,
    KnowledgeDocumentChunker,
)
from app.domain.services.knowledge_document_preprocessor import (
    KnowledgeDocumentPreprocessor,
)
from app.models.common import _StrictModel
from app.models.document import (
    DocumentChunk,
    DocumentChunkImageLink,
    DocumentImage,
)


_FENCE_PATTERN = re.compile(r"^\s*(?P<marker>`{3,}|~{3,})")
_IMAGE_PATTERN = re.compile(
    r"!\[(?P<caption>[^\]\r\n]+)\]"
    r"\((?P<target>[^)\r\n]+)\)"
)
_KNOWLEDGE_TARGET_PATTERN = re.compile(
    r"knowledge-image://(?P<key>[A-Za-z0-9][A-Za-z0-9._-]{0,127})"
)
_INTERNAL_MARKER_PATTERN = re.compile(
    r"\[\[__KNOWLEDGE_IMAGE_(?P<image_id>img-[0-9a-f]{32})__\]\]"
)
_INTERNAL_MARKER_PREFIX = "[[__KNOWLEDGE_IMAGE_"


class KnowledgeDocumentDerivation(_StrictModel):
    """由原始 Markdown 派生出的 Chunk、图片事实和关联。"""

    chunks: tuple[DocumentChunk, ...]
    images: tuple[DocumentImage, ...] = ()
    links: tuple[DocumentChunkImageLink, ...] = ()


class KnowledgeDocumentMediaExtractor:
    """解析围栏外规范图片锚点并复用现有文本切分链。"""

    _MAX_IMAGES = 50

    def __init__(
        self,
        *,
        preprocessor: KnowledgeDocumentPreprocessor,
        chunker: KnowledgeDocumentChunker,
    ) -> None:
        self._preprocessor = preprocessor
        self._chunker = chunker

    def derive(
        self,
        *,
        document_id: str,
        content_markdown: str,
    ) -> KnowledgeDocumentDerivation:
        """生成不泄露内部标记的 Chunk、图片事实和关联。"""

        normalized_id = self._required_text(document_id, "document_id")
        if not isinstance(content_markdown, str) or not content_markdown.strip():
            raise ValueError("文档正文不能为空")
        if _INTERNAL_MARKER_PREFIX in content_markdown:
            raise ValueError("文档正文包含保留的图片关联标记")

        proxied_markdown, declarations = self._proxy_images(
            normalized_id,
            content_markdown,
        )
        processed = self._preprocessor.process(proxied_markdown)
        if not processed:
            return KnowledgeDocumentDerivation(chunks=())
        drafts = self._chunker.split(processed)
        return self._finalize(normalized_id, drafts, declarations)

    def _proxy_images(
        self,
        document_id: str,
        content_markdown: str,
    ) -> tuple[str, tuple[DocumentImage, ...]]:
        lines: list[str] = []
        images: list[DocumentImage] = []
        seen_keys: set[str] = set()
        closing_pattern: re.Pattern[str] | None = None

        for line in content_markdown.splitlines(keepends=True):
            if closing_pattern is not None:
                lines.append(line)
                if closing_pattern.match(line):
                    closing_pattern = None
                continue
            fence_match = _FENCE_PATTERN.match(line)
            if fence_match:
                lines.append(line)
                closing_pattern = self._closing_fence_pattern(
                    fence_match.group("marker")
                )
                continue

            def replace(match: re.Match[str]) -> str:
                target = match.group("target").strip()
                target_match = _KNOWLEDGE_TARGET_PATTERN.fullmatch(target)
                if target_match is None:
                    raise ValueError("文档图片只支持 knowledge-image 规范锚点")
                caption = " ".join(match.group("caption").split())
                if not caption or len(caption) > 500:
                    raise ValueError("图片说明长度必须位于 1 到 500 个字符")
                image_key = target_match.group("key")
                if image_key in seen_keys:
                    raise ValueError("图片标识不能重复")
                if len(images) >= self._MAX_IMAGES:
                    raise ValueError("单篇文档图片数量不能超过 50")
                seen_keys.add(image_key)
                image_id = self._image_id(document_id, image_key)
                images.append(
                    DocumentImage(
                        image_id=image_id,
                        document_id=document_id,
                        image_key=image_key,
                        position=len(images),
                        caption=caption,
                    )
                )
                safe_caption = caption.replace("\\", "\\\\").replace(
                    "|", "\\|"
                )
                return (
                    f"图片说明：{safe_caption} "
                    f"[[__KNOWLEDGE_IMAGE_{image_id}__]]"
                )

            lines.append(_IMAGE_PATTERN.sub(replace, line))

        return "".join(lines), tuple(images)

    def _finalize(
        self,
        document_id: str,
        drafts: Sequence[DocumentChunkDraft],
        declarations: Sequence[DocumentImage],
    ) -> KnowledgeDocumentDerivation:
        declaration_by_id = {image.image_id: image for image in declarations}
        cleaned_drafts: list[DocumentChunkDraft] = []
        draft_image_ids: list[tuple[str, ...]] = []
        image_heading_paths: dict[str, tuple[str, ...]] = {}

        for draft in drafts:
            image_ids = tuple(
                dict.fromkeys(
                    match.group("image_id")
                    for match in _INTERNAL_MARKER_PATTERN.finditer(draft.content)
                )
            )
            for image_id in image_ids:
                if image_id not in declaration_by_id:
                    raise ValueError("图片关联包含未知图片")
                image_heading_paths.setdefault(image_id, draft.heading_path)
            cleaned = _INTERNAL_MARKER_PATTERN.sub("", draft.content)
            cleaned = self._normalize_marker_whitespace(cleaned)
            if not cleaned:
                continue
            cleaned_drafts.append(
                DocumentChunkDraft(
                    heading_path=draft.heading_path,
                    content=cleaned,
                )
            )
            draft_image_ids.append(image_ids)

        chunks = self._chunker.materialize(document_id, cleaned_drafts)
        links = tuple(
            DocumentChunkImageLink(chunk_id=chunk.chunk_id, image_id=image_id)
            for chunk, image_ids in zip(chunks, draft_image_ids, strict=True)
            for image_id in image_ids
        )
        linked_ids = {link.image_id for link in links}
        if linked_ids != set(declaration_by_id):
            raise ValueError("图片说明未能关联到有效 Chunk")
        images = tuple(
            image.model_copy(
                update={"heading_path": image_heading_paths[image.image_id]}
            )
            for image in declarations
        )
        return KnowledgeDocumentDerivation(
            chunks=chunks,
            images=images,
            links=links,
        )

    @staticmethod
    def _normalize_marker_whitespace(content: str) -> str:
        lines = [line.rstrip() for line in content.splitlines()]
        return "\n".join(lines).strip()

    @staticmethod
    def _image_id(document_id: str, image_key: str) -> str:
        digest = hashlib.sha256(
            f"{document_id}\0{image_key}".encode("utf-8")
        ).hexdigest()
        return f"img-{digest[:32]}"

    @staticmethod
    def _closing_fence_pattern(marker: str) -> re.Pattern[str]:
        return re.compile(
            rf"^\s*{re.escape(marker[0])}{{{len(marker)},}}\s*(?:\n)?$"
        )

    @staticmethod
    def _required_text(value: str, field_name: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field_name} 不能为空")
        return value.strip()


__all__ = ["KnowledgeDocumentDerivation", "KnowledgeDocumentMediaExtractor"]
