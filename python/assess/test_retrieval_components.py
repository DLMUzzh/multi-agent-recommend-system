"""验证知识检索、统一意图路由和 SQLite 文档推荐组件。"""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import json
import sqlite3
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from app.agents.intent_recognition_agent import IntentRecognitionAgent
from app.agents.knowledge_answer_agent import KnowledgeAnswerAgent
from app.application.conversation_service import ServiceUnavailableError
from app.application.knowledge_qa import KnowledgeQaService
from app.api.routers.chat import _degraded_components as chat_degraded_components
from app.models.schemas import (
    ArbitrationAction,
    ChatResponse,
    ConversationReply,
    ConversationTurn,
    DocumentRerankResult,
    IntentName,
    IntentState,
    RecognitionSource,
    RecommendationContext,
    RelationHint,
    DocumentRecallResult,
)
from app.orchestration.conversation_graph import ConversationGraph
from app.infrastructure.retrieval.knowledge_search import InMemoryKnowledgeSearch
from app.domain.services.conversation_arbitrator import ConversationArbitrator
from app.domain.services.intent_decision_tree import IntentDecisionTree
from app.domain.services.knowledge_document_chunker import (
    KnowledgeDocumentChunker,
)
from app.infrastructure.database.sqlite.knowledge_repository import (
    SQLiteKnowledgeRepository,
)
from app.models.document import (
    Document,
    DocumentChunkImageLink,
    DocumentFact,
    DocumentImage,
)
from app.models.knowledge_qa import (
    KnowledgeAnswerResult,
    KnowledgeChunkRecord,
    KnowledgeCitation,
    KnowledgeExecutionInput,
    KnowledgeExecutionResult,
    KnowledgeExecutionTrace,
    KnowledgeExecutionChunk,
    KnowledgeExecutionDocument,
    KnowledgeGeneratedAnswer,
    KnowledgeImageEvidence,
    KnowledgeQueryAnalysis,
    KnowledgeRetrievalDiagnostics,
    KnowledgeSearchHit,
    KnowledgeSearchResult,
)
from app.models.interaction_memory import (
    ResponsePreferenceProjection,
    UserInteractionMemoryProjection,
)


_QUERY_ANALYSIS_CASES_PATH = (
    Path(__file__).parents[2]
    / "data"
    / "knowledge_query_analysis_evaluation_cases.json"
)


class _FixedLlm:
    def __init__(self, output: dict[str, Any]) -> None:
        self.output = output
        self.calls = 0
        self.messages: list[Any] = []

    async def ainvoke(self, messages: list[Any]) -> dict[str, Any]:
        self.calls += 1
        self.messages = list(messages)
        return self.output


def _knowledge_document_preprocessor() -> Any:
    try:
        module = importlib.import_module(
            "app.domain.services.knowledge_document_preprocessor"
        )
    except ModuleNotFoundError as exc:
        raise AssertionError("知识文档标准处理器尚未实现") from exc
    return module.KnowledgeDocumentPreprocessor()


def _knowledge_chunk_rerank_module() -> Any:
    try:
        return importlib.import_module(
            "app.agents.knowledge_chunk_rerank_agent"
        )
    except ModuleNotFoundError as exc:
        raise AssertionError("知识 Chunk 重排 Agent 尚未实现") from exc


def _knowledge_query_analysis_module() -> Any:
    try:
        return importlib.import_module(
            "app.agents.knowledge_query_analysis_agent"
        )
    except ModuleNotFoundError as exc:
        raise AssertionError("知识查询分析 Agent 尚未实现") from exc


def _evidence_routing_components() -> tuple[Any, Any, Any, Any]:
    try:
        models = importlib.import_module("app.models.evidence_routing")
        gate_module = importlib.import_module(
            "app.domain.services.knowledge_evidence_gate"
        )
    except ModuleNotFoundError as exc:
        raise AssertionError("知识证据五类门控尚未实现") from exc
    return (
        models.EvidenceOption,
        models.EvidenceSignals,
        models.KnowledgeEvidenceDecision,
        gate_module.KnowledgeEvidenceGate,
    )


def _runtime_skill_components() -> tuple[Any, Any, Any]:
    try:
        models = importlib.import_module("app.models.runtime_skill")
        catalog_module = importlib.import_module(
            "app.infrastructure.skills.file_skill_catalog"
        )
        matcher_module = importlib.import_module(
            "app.domain.services.runtime_skill_matcher"
        )
    except ModuleNotFoundError as exc:
        raise AssertionError("产品运行时 Skill 尚未实现") from exc
    return (
        models.RuntimeSkillManifest,
        catalog_module.FileSkillCatalog,
        matcher_module.RuntimeSkillMatcher,
    )


def _runtime_skill_registry() -> Any:
    try:
        module = importlib.import_module("app.application.runtime_skill_registry")
    except ModuleNotFoundError as exc:
        raise AssertionError("运行时 Skill Registry 尚未实现") from exc
    return module.RuntimeSkillRegistry


def _runtime_skill_manifest(
    skill_id: str,
    *,
    keywords: tuple[str, ...] = ("虚拟线程",),
    document_ids: tuple[str, ...] = (),
    topics: tuple[str, ...] = ("Java", "并发"),
    priority: int = 100,
    enabled: bool = True,
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "skill_id": skill_id,
        "version": "1.0.0",
        "enabled": enabled,
        "applies_to": ["knowledge_qa"],
        "activation": {"keywords": list(keywords)},
        "document_scope": {
            "topics": list(topics),
            "document_ids": list(document_ids),
        },
        "query_terms": ["Java", "并发"],
        "gate_profile": "strict_evidence",
        "response_policy": {
            "focus": ["mechanism", "constraints"],
            "organization": "conclusion_then_details",
        },
        "allowed_tools": [],
        "priority": priority,
    }
    canonical = json.dumps(
        manifest,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    manifest["content_hash"] = hashlib.sha256(canonical).hexdigest()
    return manifest


def _write_runtime_skill(root: Path, manifest: dict[str, Any]) -> None:
    skill_dir = root / str(manifest["skill_id"])
    skill_dir.mkdir(parents=True)
    (skill_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _knowledge_reasoning_planner_module() -> Any:
    try:
        return importlib.import_module(
            "app.agents.knowledge_reasoning_planner_agent"
        )
    except ModuleNotFoundError as exc:
        raise AssertionError("知识推理 Planner 尚未实现") from exc


def _knowledge_plan_coverage_module() -> Any:
    try:
        return importlib.import_module(
            "app.domain.services.knowledge_plan_coverage"
        )
    except ModuleNotFoundError as exc:
        raise AssertionError("知识计划覆盖检查器尚未实现") from exc


def _knowledge_plan_execution_module() -> Any:
    try:
        return importlib.import_module(
            "app.application.knowledge_plan_execution"
        )
    except ModuleNotFoundError as exc:
        raise AssertionError("知识计划执行器尚未实现") from exc


def _knowledge_reasoning_contracts() -> Any:
    module = importlib.import_module("app.models.knowledge_qa")
    required_names = (
        "KnowledgePlanCoverage",
        "KnowledgePlanStep",
        "KnowledgePlanStepResult",
        "KnowledgePlanTraceStep",
        "KnowledgeReasoningPlan",
    )
    missing_names = [
        name for name in required_names if not hasattr(module, name)
    ]
    if missing_names:
        raise AssertionError(
            "知识推理计划严格契约尚未实现：" + ", ".join(missing_names)
        )
    return module


def _document_image_model() -> Any:
    module = importlib.import_module("app.models.document")
    model = getattr(module, "DocumentImage", None)
    if model is None:
        raise AssertionError("文档图片事实契约尚未实现")
    return model


def _knowledge_document_media() -> Any:
    try:
        module = importlib.import_module(
            "app.domain.services.knowledge_document_media"
        )
    except ModuleNotFoundError as exc:
        raise AssertionError("知识文档媒体解析服务尚未实现") from exc
    return module.KnowledgeDocumentMediaExtractor(
        preprocessor=_knowledge_document_preprocessor(),
        chunker=KnowledgeDocumentChunker(),
    )


def _local_knowledge_image_store(root: Path) -> Any:
    try:
        module = importlib.import_module(
            "app.infrastructure.storage.local_knowledge_image_store"
        )
    except ModuleNotFoundError as exc:
        raise AssertionError("本地知识图片存储尚未实现") from exc
    return module.LocalKnowledgeImageStore(root)


class KnowledgeDocumentPreprocessorTests(unittest.TestCase):
    """验证文档在切分前执行保守、确定性的标准处理。"""

    def test_removes_resources_outside_fences_and_keeps_regular_content(
        self,
    ) -> None:
        markdown = """# 系统文档

## 项目定位

<readonly-block type="isv"></readonly-block>

核心结论。

<figure view-type="Preview"><source name="demo.zip"/></figure>

<source name="standalone.bin"/>

<div class="note">普通 HTML</div>

| 项目 | 说明 |
| --- | --- |
| A | 保留 |
"""

        processed = _knowledge_document_preprocessor().process(markdown)

        self.assertNotIn("readonly-block", processed)
        self.assertNotIn("<figure", processed)
        self.assertNotIn("<source", processed)
        self.assertIn("核心结论。", processed)
        self.assertIn('<div class="note">普通 HTML</div>', processed)
        self.assertIn("| A | 保留 |", processed)

    def test_keeps_resource_literals_inside_code_fences(self) -> None:
        markdown = """# 示例

```html
<readonly-block type="sample"></readonly-block>
<figure><source src="demo.png"/></figure>
```

<readonly-block type="isv"></readonly-block>
"""

        processed = _knowledge_document_preprocessor().process(markdown)

        self.assertIn(
            '<readonly-block type="sample"></readonly-block>',
            processed,
        )
        self.assertIn('<figure><source src="demo.png"/></figure>', processed)
        self.assertNotIn('type="isv"', processed)

    def test_keeps_blank_lines_inside_code_fences(self) -> None:
        markdown = """# 示例


```text
第一行


第二行
```


正文。
"""

        processed = _knowledge_document_preprocessor().process(markdown)

        self.assertEqual(
            processed,
            "# 示例\n\n```text\n第一行\n\n\n第二行\n```\n\n正文。",
        )

    def test_normalizes_line_endings_blank_lines_and_empty_sections(
        self,
    ) -> None:
        markdown = (
            "# 系统文档\r\n\r\n\r\n"
            "## 空章节\r\n\r\n\r\n"
            "## 正文章节\r\n\r\n\r\n核心内容。\r\n\r\n"
        )

        processed = _knowledge_document_preprocessor().process(markdown)

        self.assertEqual(
            processed,
            "# 系统文档\n\n## 正文章节\n\n核心内容。",
        )
        self.assertEqual(
            _knowledge_document_preprocessor().process(processed),
            processed,
        )

    def test_keeps_native_heading_only_and_removes_resource_emptied_section(
        self,
    ) -> None:
        markdown = """# 算法题目录

## 第 34 题：在排序数组中查找

## 资源图

<figure><source name="question.png"/></figure>
"""

        processed = _knowledge_document_preprocessor().process(markdown)

        self.assertIn("## 第 34 题：在排序数组中查找", processed)
        self.assertNotIn("## 资源图", processed)
        self.assertNotIn("figure", processed)

    def test_removes_only_leading_author_and_category_metadata(self) -> None:
        markdown = """# Spring Boot 部署指南

作者：林屿

分类：Spring Boot、部署

## 部署步骤

先构建可执行 Jar，再启动服务。

作者：正文中的负责人字段必须保留
分类：正文中的业务分类也必须保留
"""

        processed = _knowledge_document_preprocessor().process(markdown)

        self.assertNotIn("作者：林屿", processed)
        self.assertNotIn("分类：Spring Boot、部署", processed)
        self.assertIn("作者：正文中的负责人字段必须保留", processed)
        self.assertIn("分类：正文中的业务分类也必须保留", processed)
        self.assertTrue(processed.startswith("# Spring Boot 部署指南\n\n## 部署步骤"))


class KnowledgeDocumentChunkerTests(unittest.TestCase):
    """验证独立知识切分模块的结构保护和稳定身份。"""

    def test_defaults_match_ragflow_passage_window(self) -> None:
        chunker = KnowledgeDocumentChunker()

        self.assertEqual(chunker.target_tokens, 280)
        self.assertEqual(chunker.max_tokens, 420)
        self.assertEqual(chunker.overlap_tokens, 40)

    def test_leaf_heading_only_becomes_searchable_passage(self) -> None:
        chunks = KnowledgeDocumentChunker().chunk(
            "doc-heading",
            "# 算法\n\n## 第 34 题：在排序数组中查找",
        )

        self.assertEqual(len(chunks), 1)
        self.assertEqual(
            chunks[0].heading_path,
            ("算法", "第 34 题：在排序数组中查找"),
        )
        self.assertEqual(chunks[0].content, "第 34 题：在排序数组中查找")

    def test_does_not_hide_unprocessed_resource_placeholders(self) -> None:
        markdown = """
# 系统文档

## 流程图

<readonly-block type="isv"></readonly-block>
"""

        chunks = KnowledgeDocumentChunker().chunk("doc-clean", markdown)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].heading_path, ("系统文档", "流程图"))
        self.assertIn("readonly-block", chunks[0].content)

    def test_splits_long_code_by_line_and_repeats_fences(self) -> None:
        code_lines = [
            f"value_{index} = call(alpha, beta, gamma)"
            for index in range(14)
        ]
        markdown = (
            "# 系统文档\n\n"
            "## 代码\n\n"
            + chr(96) * 3
            + "python\n"
            + "\n".join(code_lines)
            + "\n"
            + chr(96) * 3
        )
        chunker = KnowledgeDocumentChunker(
            target_tokens=45,
            max_tokens=60,
            overlap_tokens=10,
        )

        chunks = chunker.chunk("doc-code", markdown)

        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(chunk.token_count, 60)
            self.assertTrue(chunk.content.startswith((chr(96) * 3) + "python"))
            self.assertTrue(chunk.content.endswith(chr(96) * 3))
        combined = "\n".join(chunk.content for chunk in chunks)
        for code_line in code_lines:
            self.assertEqual(combined.count(code_line), 1)

    def test_splits_long_table_by_row_and_repeats_header(self) -> None:
        header = "| 项目 | 说明 |"
        separator = "| --- | --- |"
        rows = [
            f"| item-{index} | alpha beta gamma delta {index} |"
            for index in range(12)
        ]
        markdown = "\n".join(
            ["# 系统文档", "", "## 参数", "", header, separator, *rows]
        )
        chunker = KnowledgeDocumentChunker(
            target_tokens=45,
            max_tokens=60,
            overlap_tokens=10,
        )

        chunks = chunker.chunk("doc-table", markdown)

        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            lines = chunk.content.splitlines()
            self.assertEqual(lines[:2], [header, separator])
            self.assertLessEqual(chunk.token_count, 60)
        combined = "\n".join(chunk.content for chunk in chunks)
        for row in rows:
            self.assertEqual(combined.count(row), 1)

    def test_rejects_table_row_that_cannot_fit_with_header(self) -> None:
        markdown = "\n".join(
            [
                "# 系统文档",
                "",
                "## 参数",
                "",
                "| 项目 | 说明 |",
                "| --- | --- |",
                f"| only | {'超' * 100} |",
            ]
        )

        with self.assertRaisesRegex(ValueError, "表格行超过 max_tokens"):
            KnowledgeDocumentChunker(
                target_tokens=30,
                max_tokens=40,
                overlap_tokens=5,
            ).chunk("doc-wide-row", markdown)

    def test_long_text_uses_single_controlled_overlap(self) -> None:
        words = [f"token{index}" for index in range(90)]
        markdown = "# 系统文档\n\n## 长段落\n\n" + " ".join(words)
        chunks = KnowledgeDocumentChunker(
            target_tokens=30,
            max_tokens=40,
            overlap_tokens=5,
        ).chunk("doc-long-text", markdown)

        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0].content.split(), words[:40])
        self.assertEqual(chunks[1].content.split(), words[35:75])
        self.assertEqual(chunks[2].content.split(), words[70:])
        self.assertTrue(all(chunk.token_count <= 40 for chunk in chunks))

    def test_early_edit_keeps_unchanged_later_section_chunk_ids(self) -> None:
        original = """
# 系统文档

## 前置章节

第一段。

## 稳定章节

这一节不会改变。
"""
        inserted = " ".join(f"新增词{index}" for index in range(60))
        updated = f"""
# 系统文档

## 前置章节

{inserted}

第一段。

## 稳定章节

这一节不会改变。
"""
        chunker = KnowledgeDocumentChunker(
            target_tokens=30,
            max_tokens=40,
            overlap_tokens=5,
        )

        original_chunks = chunker.chunk("doc-stable", original)
        updated_chunks = chunker.chunk("doc-stable", updated)
        original_later = next(
            chunk
            for chunk in original_chunks
            if chunk.heading_path[-1] == "稳定章节"
        )
        updated_later = next(
            chunk
            for chunk in updated_chunks
            if chunk.heading_path[-1] == "稳定章节"
        )

        self.assertEqual(original_later.content, updated_later.content)
        self.assertEqual(original_later.chunk_id, updated_later.chunk_id)
        self.assertNotEqual(original_later.position, updated_later.position)


class KnowledgeImageModelTests(unittest.TestCase):
    """验证图片事实状态不能出现半完成组合。"""

    def test_document_image_requires_complete_ready_payload(self) -> None:
        document_image = _document_image_model()

        with self.assertRaises(ValueError):
            document_image(
                image_id="img-" + "a" * 32,
                document_id="doc-1",
                image_key="architecture",
                position=0,
                caption="系统架构",
                status="ready",
            )

    def test_document_image_accepts_pending_without_binary_fields(self) -> None:
        document_image = _document_image_model()

        image = document_image(
            image_id="img-" + "b" * 32,
            document_id="doc-1",
            image_key="architecture",
            position=0,
            caption="系统架构",
        )

        self.assertEqual(image.status, "pending")
        self.assertIsNone(image.storage_key)


class KnowledgeDocumentMediaTests(unittest.TestCase):
    """验证规范图片锚点、文本代理和结构关联。"""

    def test_extracts_image_proxy_and_stable_link(self) -> None:
        media = _knowledge_document_media()

        derivation = media.derive(
            document_id="doc-1",
            content_markdown=(
                "# 架构\n\n调用链如下。\n\n"
                "![组件关系图](knowledge-image://architecture-overview)\n\n"
                "请求依次经过 API、Service 和 Search。"
            ),
        )

        self.assertEqual(len(derivation.images), 1)
        self.assertEqual(
            derivation.images[0].image_key,
            "architecture-overview",
        )
        self.assertTrue(
            any("图片说明：组件关系图" in chunk.content for chunk in derivation.chunks)
        )
        self.assertEqual(len(derivation.links), 1)
        self.assertFalse(
            any("__KNOWLEDGE_IMAGE" in chunk.content for chunk in derivation.chunks)
        )

    def test_does_not_extract_image_syntax_inside_code_fence(self) -> None:
        media = _knowledge_document_media()
        fence = "`" * 3

        derivation = media.derive(
            document_id="doc-code",
            content_markdown=(
                f"# 示例\n\n{fence}markdown\n"
                "![字面量](knowledge-image://literal)\n"
                f"{fence}"
            ),
        )

        self.assertEqual(derivation.images, ())
        self.assertIn("knowledge-image://literal", derivation.chunks[0].content)

    def test_image_inside_table_keeps_table_structure(self) -> None:
        media = _knowledge_document_media()

        derivation = media.derive(
            document_id="doc-table",
            content_markdown=(
                "# 组件\n\n"
                "| 组件 | 示意图 |\n"
                "| --- | --- |\n"
                "| Search | ![召回流程](knowledge-image://search-flow) |"
            ),
        )

        table_chunk = derivation.chunks[0]
        self.assertIn("| 组件 | 示意图 |", table_chunk.content)
        self.assertIn("图片说明：召回流程", table_chunk.content)
        self.assertEqual(derivation.links[0].chunk_id, table_chunk.chunk_id)

    def test_rejects_duplicate_or_external_image_reference(self) -> None:
        media = _knowledge_document_media()

        with self.assertRaisesRegex(ValueError, "图片标识不能重复"):
            media.derive(
                document_id="doc-duplicate",
                content_markdown=(
                    "![图一](knowledge-image://same)\n\n"
                    "![图二](knowledge-image://same)"
                ),
            )
        with self.assertRaisesRegex(ValueError, "只支持 knowledge-image"):
            media.derive(
                document_id="doc-external",
                content_markdown="![外链](https://example.com/a.png)",
            )


class LocalKnowledgeImageStoreTests(unittest.TestCase):
    """验证图片魔数、内容寻址和安全路径。"""

    _PNG_BYTES = (
        b"\x89PNG\r\n\x1a\n"
        b"\x00\x00\x00\rIHDR"
        b"\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x06\x00\x00\x00"
    )

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name) / "images"
        self.store = _local_knowledge_image_store(self.root)

    def test_store_rejects_declared_png_with_jpeg_bytes(self) -> None:
        with self.assertRaisesRegex(ValueError, "图片类型与内容不一致"):
            self.store.put(b"\xff\xd8\xff\xe0fake", "image/png")

    def test_store_uses_content_addressed_safe_path(self) -> None:
        stored = self.store.put(self._PNG_BYTES, "image/png")

        self.assertRegex(
            stored.storage_key,
            r"^[0-9a-f]{2}/[0-9a-f]{64}\.png$",
        )
        resolved = self.store.resolve(stored.storage_key)
        self.assertTrue(resolved.is_relative_to(self.root.resolve()))
        self.assertEqual(resolved.read_bytes(), self._PNG_BYTES)

    def test_store_rejects_path_traversal(self) -> None:
        with self.assertRaisesRegex(ValueError, "图片存储 Key 无效"):
            self.store.resolve("../outside.png")

    def test_store_rejects_unsupported_empty_and_oversized_content(self) -> None:
        with self.assertRaisesRegex(ValueError, "图片内容不能为空"):
            self.store.put(b"", "image/png")
        with self.assertRaisesRegex(ValueError, "不支持的图片类型"):
            self.store.put(self._PNG_BYTES, "image/svg+xml")
        with self.assertRaisesRegex(ValueError, "图片大小不能超过 8 MiB"):
            self.store.put(b"x" * (self.store.MAX_BYTES + 1), "image/png")

    def test_store_is_idempotent_and_garbage_collects_only_valid_files(
        self,
    ) -> None:
        first = self.store.put(self._PNG_BYTES, "image/png")
        second = self.store.put(self._PNG_BYTES, "image/png")
        self.assertEqual(first, second)
        self.assertEqual(self.store.resolve(first.storage_key).stat().st_mode & 0o777, 0o600)
        self.assertEqual(self.root.stat().st_mode & 0o777, 0o700)

        other_bytes = self._PNG_BYTES + b"other"
        other = self.store.put(other_bytes, "image/png")
        invalid = self.root / "zz" / "not-an-image.txt"
        invalid.parent.mkdir()
        invalid.write_text("keep", encoding="utf-8")

        deleted = self.store.delete_unreferenced((first.storage_key,))

        self.assertEqual(deleted, 1)
        self.assertTrue(self.store.resolve(first.storage_key).exists())
        self.assertFalse(self.store.resolve(other.storage_key).exists())
        self.assertTrue(invalid.exists())

    def test_garbage_collection_does_not_follow_symlinked_files(self) -> None:
        external = self.root / "external.png"
        external.write_bytes(self._PNG_BYTES)
        content_hash = hashlib.sha256(self._PNG_BYTES + b"link").hexdigest()
        shard = self.root / content_hash[:2]
        shard.mkdir()
        symlink = shard / f"{content_hash}.png"
        symlink.symlink_to(external)

        deleted = self.store.delete_unreferenced(())

        self.assertEqual(deleted, 0)
        self.assertTrue(symlink.is_symlink())
        self.assertEqual(external.read_bytes(), self._PNG_BYTES)

    def test_garbage_collection_does_not_follow_symlinked_directories(
        self,
    ) -> None:
        outside = Path(self.temporary_directory.name) / "outside"
        outside.mkdir()
        content_hash = hashlib.sha256(self._PNG_BYTES + b"directory").hexdigest()
        external = outside / f"{content_hash}.png"
        external.write_bytes(self._PNG_BYTES)
        shard_link = self.root / content_hash[:2]
        shard_link.symlink_to(outside, target_is_directory=True)

        deleted = self.store.delete_unreferenced(())

        self.assertEqual(deleted, 0)
        self.assertTrue(shard_link.is_symlink())
        self.assertEqual(external.read_bytes(), self._PNG_BYTES)

    def test_garbage_collection_skips_symlinked_directory_inside_root(
        self,
    ) -> None:
        target_shard = self.root / "zz"
        target_shard.mkdir()
        target_hash = "b" * 64
        target = target_shard / f"{target_hash}.png"
        target.write_bytes(self._PNG_BYTES)
        shard_link = self.root / "bb"
        shard_link.symlink_to(target_shard, target_is_directory=True)

        deleted = self.store.delete_unreferenced(())

        self.assertEqual(deleted, 0)
        self.assertTrue(shard_link.is_symlink())
        self.assertEqual(target.read_bytes(), self._PNG_BYTES)


class KnowledgeRepositoryTests(unittest.TestCase):
    """验证独立知识仓储复用现有 SQLite 表并提供稳定回查。"""

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.repository = SQLiteKnowledgeRepository(
            Path(self.temporary_directory.name) / "knowledge.sqlite3"
        )
        now = datetime(2026, 8, 6, tzinfo=timezone.utc)
        for document_id, title, content in (
            (
                "doc-python",
                "Python 异步编程",
                "# Python\n\n## 事件循环\n\n事件循环负责调度协程。",
            ),
            (
                "doc-java",
                "Java 并发编程",
                "# Java\n\n## 线程池\n\n线程池负责复用工作线程。",
            ),
        ):
            chunks = KnowledgeDocumentChunker().chunk(document_id, content)
            self.repository.replace_document(
                Document(
                    document_id=document_id,
                    title=title,
                    content_markdown=content,
                    topics=[title.split()[0]],
                    content_type="tutorial",
                    difficulty="intermediate",
                    author_id="author-fixture",
                    content_hash=hashlib.sha256(
                        content.encode("utf-8")
                    ).hexdigest(),
                    created_at=now,
                    updated_at=now,
                ),
                chunks,
            )

    def test_lists_ready_chunks_with_document_title_and_scope(self) -> None:
        all_chunks = self.repository.list_ready_chunks()
        scoped_chunks = self.repository.list_ready_chunks(("doc-java",))

        self.assertEqual(
            {chunk.document_id for chunk in all_chunks},
            {"doc-python", "doc-java"},
        )
        self.assertEqual(len(scoped_chunks), 1)
        self.assertEqual(scoped_chunks[0].document_id, "doc-java")
        self.assertEqual(scoped_chunks[0].title, "Java 并发编程")
        self.assertEqual(scoped_chunks[0].heading_path, ("Java", "线程池"))

    def test_chunk_recheck_preserves_requested_rank_order(self) -> None:
        chunks = self.repository.list_ready_chunks()
        requested_ids = (chunks[-1].chunk_id, "missing", chunks[0].chunk_id)

        rechecked = self.repository.get_chunks_by_ids(requested_ids)

        self.assertEqual(
            [chunk.chunk_id for chunk in rechecked],
            [chunks[-1].chunk_id, chunks[0].chunk_id],
        )

    def test_repository_initializes_empty_image_schema(self) -> None:
        with sqlite3.connect(self.repository.path) as connection:
            tables = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                ).fetchall()
            }

        self.assertIn("document_images", tables)
        self.assertIn("document_chunk_images", tables)
        self.assertEqual(self.repository.count_documents(), 2)

    def test_document_bundle_persists_image_and_ready_evidence(self) -> None:
        document = self.repository.get_document("doc-python")
        chunks = self.repository.list_chunks("doc-python")
        assert document is not None
        image_id = "img-" + "c" * 32
        image = DocumentImage(
            image_id=image_id,
            document_id=document.document_id,
            image_key="event-loop",
            position=0,
            heading_path=chunks[0].heading_path,
            caption="事件循环关系图",
        )
        link = DocumentChunkImageLink(
            chunk_id=chunks[0].chunk_id,
            image_id=image_id,
        )

        self.repository.replace_document_bundle(
            document,
            chunks,
            (image,),
            (link,),
        )
        ready = self.repository.mark_image_ready(
            image_id=image_id,
            content_hash="d" * 64,
            storage_key="dd/" + "d" * 64 + ".png",
            mime_type="image/png",
            byte_size=68,
        )
        evidence = self.repository.list_ready_images_by_chunk_ids(
            (chunks[0].chunk_id,)
        )

        self.assertEqual(ready.status, "ready")
        self.assertEqual(self.repository.get_image(image_id), ready)
        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0].title, "Python 异步编程")
        self.assertEqual(evidence[0].linked_chunk_ids, (chunks[0].chunk_id,))

    def test_replacing_same_image_identity_preserves_ready_binary(self) -> None:
        document = self.repository.get_document("doc-java")
        chunks = self.repository.list_chunks("doc-java")
        assert document is not None
        image_id = "img-" + "e" * 32
        original = DocumentImage(
            image_id=image_id,
            document_id=document.document_id,
            image_key="thread-pool",
            position=0,
            heading_path=chunks[0].heading_path,
            caption="旧线程池图",
        )
        link = DocumentChunkImageLink(
            chunk_id=chunks[0].chunk_id,
            image_id=image_id,
        )
        self.repository.replace_document_bundle(
            document,
            chunks,
            (original,),
            (link,),
        )
        self.repository.mark_image_ready(
            image_id=image_id,
            content_hash="f" * 64,
            storage_key="ff/" + "f" * 64 + ".png",
            mime_type="image/png",
            byte_size=68,
        )

        updated = original.model_copy(update={"caption": "新线程池图"})
        self.repository.replace_document_bundle(
            document,
            chunks,
            (updated,),
            (link,),
        )

        persisted = self.repository.get_image(image_id)
        assert persisted is not None
        self.assertEqual(persisted.caption, "新线程池图")
        self.assertEqual(persisted.status, "ready")
        self.assertEqual(persisted.content_hash, "f" * 64)

    def test_invalid_image_link_rolls_back_entire_document_bundle(self) -> None:
        document = self.repository.get_document("doc-python")
        chunks = self.repository.list_chunks("doc-python")
        assert document is not None
        original_hash = document.content_hash
        changed = document.model_copy(
            update={
                "title": "不应提交的标题",
                "content_hash": "9" * 64,
            }
        )
        image = DocumentImage(
            image_id="img-" + "8" * 32,
            document_id=document.document_id,
            image_key="rollback",
            position=0,
            caption="回滚图片",
        )
        invalid_link = DocumentChunkImageLink(
            chunk_id="unknown-chunk",
            image_id=image.image_id,
        )

        with self.assertRaisesRegex(ValueError, "未知 Chunk"):
            self.repository.replace_document_bundle(
                changed,
                chunks,
                (image,),
                (invalid_link,),
            )

        persisted = self.repository.get_document("doc-python")
        assert persisted is not None
        self.assertEqual(persisted.title, "Python 异步编程")
        self.assertEqual(persisted.content_hash, original_hash)
        self.assertIsNone(self.repository.get_image(image.image_id))


class _SemanticKnowledgeEmbeddingClient:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    async def embed(self, texts: list[str]) -> tuple[tuple[float, float], ...]:
        self.calls.append(list(texts))
        vectors: list[tuple[float, float]] = []
        for text in texts:
            if "线程池负责复用工作线程" in text or text == "资源调度策略":
                vectors.append((1.0, 0.0))
            else:
                vectors.append((0.0, 1.0))
        return tuple(vectors)


class _FailingKnowledgeEmbeddingClient:
    async def embed(self, texts: list[str]) -> tuple[tuple[float, float], ...]:
        del texts
        raise TimeoutError("向量服务超时")


def _knowledge_record(
    *,
    chunk_id: str,
    document_id: str,
    title: str,
    content: str,
    token_count: int | None = None,
    heading_path: tuple[str, ...] | None = None,
    position: int = 0,
) -> KnowledgeChunkRecord:
    return KnowledgeChunkRecord(
        chunk_id=chunk_id,
        document_id=document_id,
        title=title,
        topics=[title],
        content_type="tutorial",
        difficulty="intermediate",
        author_id="author-fixture",
        position=position,
        heading_path=heading_path or (title,),
        content=content,
        content_hash=hashlib.sha256(content.encode("utf-8")).hexdigest(),
        token_count=(
            token_count
            if token_count is not None
            else KnowledgeDocumentChunker.count_tokens(content)
        ),
    )


class KnowledgeSearchTests(unittest.IsolatedAsyncioTestCase):
    """验证独立 Chunk 检索的 BM25、Vector、RRF 和降级。"""

    def setUp(self) -> None:
        self.chunks = (
            _knowledge_record(
                chunk_id="chunk-python",
                document_id="doc-python",
                title="Python 异步编程",
                content="事件循环负责调度协程并处理异步任务。",
            ),
            _knowledge_record(
                chunk_id="chunk-java",
                document_id="doc-java",
                title="Java 并发编程",
                content="线程池负责复用工作线程。",
            ),
        )

    async def test_bm25_ranks_matching_chunk_without_embedding(self) -> None:
        search = InMemoryKnowledgeSearch()
        await search.refresh(self.chunks)

        result = await search.search("事件循环如何调度协程", limit=2)

        self.assertEqual(result.hits[0].chunk_id, "chunk-python")
        self.assertEqual(result.hits[0].bm25_rank, 1)
        self.assertIsNone(result.hits[0].vector_rank)
        self.assertEqual(result.diagnostics.bm25_status, "executed")
        self.assertEqual(result.diagnostics.vector_status, "skipped")

    async def test_bm25_matches_compact_ascii_query_to_spaced_title(self) -> None:
        search = InMemoryKnowledgeSearch()
        await search.refresh(
            (
                _knowledge_record(
                    chunk_id="chunk-spring-boot",
                    document_id="doc-spring-boot",
                    title="Spring Boot 运行指南",
                    content=(
                        "先构建可执行包，再配置生产环境参数、健康检查和滚动发布。"
                    ),
                ),
                _knowledge_record(
                    chunk_id="chunk-model",
                    document_id="doc-model",
                    title="模型部署",
                    content="部署流程需要关注部署、发布与回滚，部署前还要验证部署环境。",
                ),
            )
        )

        result = await search.search("springboot如何部署", limit=2)

        self.assertEqual(result.hits[0].chunk_id, "chunk-spring-boot")

    async def test_heading_bm25_has_an_independent_candidate_pool(self) -> None:
        heading_record = _knowledge_record(
            chunk_id="chunk-000-heading",
            document_id="doc-heading",
            title="第 34 题是什么题",
            heading_path=("算法", "第 34 题是什么题"),
            content=" ".join(f"无关词{index}" for index in range(500)),
        )
        body_records = tuple(
            _knowledge_record(
                chunk_id=f"chunk-body-{index:02d}",
                document_id=f"doc-body-{index:02d}",
                title=f"正文候选 {index}",
                heading_path=(f"正文候选 {index}",),
                content="第 34 题是什么题",
            )
            for index in range(25)
        )
        search = InMemoryKnowledgeSearch()
        await search.refresh((heading_record, *body_records))

        result = await search.search("第 34 题是什么题", limit=5)

        self.assertIn(
            "chunk-000-heading",
            [hit.chunk_id for hit in result.hits],
        )

    async def test_recall_quotas_keep_global_document_and_heading_caps(
        self,
    ) -> None:
        chunks = tuple(
            _knowledge_record(
                chunk_id=f"chunk-doc-{index:02d}",
                document_id="doc-dense",
                title="密集文档",
                heading_path=("密集文档", f"章节 {index}"),
                content="共同关键词",
            )
            for index in range(12)
        ) + tuple(
            _knowledge_record(
                chunk_id=f"chunk-heading-{index:02d}",
                document_id=f"doc-heading-{index:02d}",
                title=f"标题文档 {index}",
                heading_path=("重复标题路径",),
                content="共同关键词",
            )
            for index in range(5)
        )
        search = InMemoryKnowledgeSearch()
        await search.refresh(chunks)

        result = await search.search(
            "共同关键词",
            limit=20,
            max_chunks_per_document=10,
        )

        dense_hits = [
            hit
            for hit in result.hits
            if hit.chunk_id.startswith("chunk-doc-")
        ]
        repeated_heading_hits = [
            hit
            for hit in result.hits
            if hit.chunk_id.startswith("chunk-heading-")
        ]
        self.assertLessEqual(len(dense_hits), 8)
        self.assertLessEqual(len(repeated_heading_hits), 3)

    async def test_refresh_skips_only_resource_placeholder_chunks(self) -> None:
        search = InMemoryKnowledgeSearch()
        chunks = (
            _knowledge_record(
                chunk_id="legacy-readonly",
                document_id="doc-readonly",
                title="资源示例",
                content='<readonly-block type="isv"></readonly-block>',
            ),
            _knowledge_record(
                chunk_id="legacy-figure",
                document_id="doc-figure",
                title="资源示例",
                content='<figure><source src="demo.png"/></figure>',
            ),
            _knowledge_record(
                chunk_id="legacy-source",
                document_id="doc-source",
                title="资源示例",
                content='<source src="demo.png"/>',
            ),
            _knowledge_record(
                chunk_id="mixed-content",
                document_id="doc-mixed",
                title="资源示例",
                content=(
                    '<readonly-block type="isv"></readonly-block>\n保留正文。'
                ),
            ),
            _knowledge_record(
                chunk_id="code-literal",
                document_id="doc-code",
                title="资源示例",
                content=(
                    "```html\n"
                    '<readonly-block type="sample"></readonly-block>\n'
                    "```"
                ),
            ),
        )

        await search.refresh(chunks)
        result = await search.search("资源示例", limit=10)

        self.assertEqual(
            {hit.chunk_id for hit in result.hits},
            {"mixed-content", "code-literal"},
        )

    async def test_vector_adds_semantic_hit_and_rrf_rank(self) -> None:
        client = _SemanticKnowledgeEmbeddingClient()
        search = InMemoryKnowledgeSearch(
            embedding_client=client,
            embedding_dimensions=2,
        )
        await search.refresh(self.chunks)

        result = await search.search("资源调度策略", limit=2)

        self.assertEqual(result.hits[0].chunk_id, "chunk-java")
        self.assertEqual(result.hits[0].vector_rank, 1)
        self.assertEqual(result.diagnostics.vector_status, "executed")
        self.assertEqual(result.mode, "hybrid")
        self.assertEqual(len(client.calls), 2)

    async def test_vector_failure_keeps_bm25_results(self) -> None:
        search = InMemoryKnowledgeSearch(
            embedding_client=_FailingKnowledgeEmbeddingClient(),
            embedding_dimensions=2,
        )
        await search.refresh(self.chunks)

        result = await search.search("事件循环", limit=2)

        self.assertEqual(result.hits[0].chunk_id, "chunk-python")
        self.assertEqual(result.mode, "bm25")
        self.assertEqual(result.diagnostics.vector_status, "degraded")

    async def test_document_mode_excludes_seen_and_keeps_one_chunk_per_document(
        self,
    ) -> None:
        search = InMemoryKnowledgeSearch()
        chunks = (
            _knowledge_record(
                chunk_id="seen-1",
                document_id="doc-seen",
                title="Java 虚拟线程",
                content="Java virtual thread blocking task",
            ),
            _knowledge_record(
                chunk_id="seen-2",
                document_id="doc-seen",
                title="Java 虚拟线程",
                content="Java virtual thread concurrency",
            ),
            _knowledge_record(
                chunk_id="unseen-1",
                document_id="doc-unseen",
                title="Java 并发补充",
                content="Java virtual thread guide",
            ),
        )
        await search.refresh(chunks)

        result = await search.search(
            "Java virtual thread",
            limit=1,
            excluded_document_ids=("doc-seen",),
            max_chunks_per_document=1,
        )

        self.assertEqual([hit.chunk_id for hit in result.hits], ["unseen-1"])


class KnowledgeChunkRerankAgentTests(unittest.IsolatedAsyncioTestCase):
    """验证 QA Passage 的批量结构化重排和确定性降级。"""

    def test_support_level_rejects_inconsistent_score_band(self) -> None:
        module = _knowledge_chunk_rerank_module()

        with self.assertRaises(ValueError):
            module._KnowledgeChunkRerankItem.model_validate(
                {
                    "chunk_id": "chunk-1",
                    "llm_score": 0.1,
                    "support_level": "direct",
                    "reason": "分数与直接支持矛盾",
                }
            )
        with self.assertRaises(ValueError):
            module._KnowledgePlanRerankItem.model_validate(
                {
                    "step_id": "step-1",
                    "chunk_id": "chunk-1",
                    "llm_score": 0.9,
                    "support_level": "none",
                }
            )

    @staticmethod
    def _plan_batch(
        *,
        step_count: int = 5,
        relation_count: int = 30,
    ) -> tuple[tuple[Any, ...], tuple[KnowledgeChunkRecord, ...], tuple[Any, ...]]:
        contracts = _knowledge_reasoning_contracts()
        steps = tuple(
            contracts.KnowledgePlanStep(
                step_id=f"step-{index}",
                facet="definition" if index == 1 else "scenario",
                query=f"混合检索维度 {index}",
                target_subjects=("混合检索",),
                required=index == 1,
            )
            for index in range(1, step_count + 1)
        )
        chunk_count = max(1, min(20, relation_count))
        candidates = tuple(
            _knowledge_record(
                chunk_id=f"chunk-plan-{index}",
                document_id=f"doc-plan-{index % 4}",
                title=f"混合检索文档 {index}",
                content=f"混合检索维度 {index} 的可信内容。",
            )
            for index in range(1, chunk_count + 1)
        )
        relations = tuple(
            contracts.KnowledgePlanCandidateRelation(
                step_id=steps[
                    (index + index // len(candidates)) % len(steps)
                ].step_id,
                chunk_id=candidates[index % len(candidates)].chunk_id,
                deterministic_score=(relation_count - index) / relation_count,
            )
            for index in range(relation_count)
        )
        return steps, candidates, relations

    async def test_complete_batch_blends_deterministic_and_llm_scores(
        self,
    ) -> None:
        module = _knowledge_chunk_rerank_module()
        first = _knowledge_record(
            chunk_id="chunk-first",
            document_id="doc-first",
            title="第一候选",
            content="部分相关内容。",
        )
        second = _knowledge_record(
            chunk_id="chunk-second",
            document_id="doc-second",
            title="第二候选",
            content="直接回答问题的内容。",
        )
        llm = _FixedLlm(
            {
                "items": [
                    {
                        "chunk_id": "chunk-first",
                        "llm_score": 0.5,
                        "support_level": "partial",
                        "reason": "只支持部分问题",
                    },
                    {
                        "chunk_id": "chunk-second",
                        "llm_score": 1.0,
                        "support_level": "direct",
                        "reason": "直接支持问题",
                    },
                ]
            }
        )
        agent = module.KnowledgeChunkRerankAgent(llm=llm)

        outcome = await agent.rerank(
            query="哪个候选直接支持问题？",
            candidates=(first, second),
            deterministic_scores={
                "chunk-first": 0.40,
                "chunk-second": 0.50,
            },
        )

        self.assertEqual(llm.calls, 1)
        self.assertFalse(outcome.degraded)
        envelope = json.loads(str(llm.messages[1].content))
        self.assertEqual(set(envelope), {"contract", "input"})
        self.assertEqual(
            envelope["contract"]["name"],
            "knowledge_chunk_rerank",
        )
        self.assertEqual(envelope["contract"]["version"], 2)
        self.assertIsInstance(envelope["contract"]["output_schema"], dict)
        self.assertEqual(envelope["input"]["query"], "哪个候选直接支持问题？")
        self.assertEqual(
            [record.chunk_id for record in outcome.records],
            ["chunk-second", "chunk-first"],
        )
        self.assertAlmostEqual(outcome.scores["chunk-second"], 0.60)
        self.assertAlmostEqual(outcome.scores["chunk-first"], 0.42)

    async def test_duplicate_or_incomplete_batch_falls_back_as_a_whole(
        self,
    ) -> None:
        module = _knowledge_chunk_rerank_module()
        first = _knowledge_record(
            chunk_id="chunk-first",
            document_id="doc-first",
            title="第一候选",
            content="第一候选内容。",
        )
        second = _knowledge_record(
            chunk_id="chunk-second",
            document_id="doc-second",
            title="第二候选",
            content="第二候选内容。",
        )
        agent = module.KnowledgeChunkRerankAgent(
            llm=_FixedLlm(
                {
                    "items": [
                        {
                            "chunk_id": "chunk-first",
                            "llm_score": 0.1,
                            "support_level": "partial",
                            "reason": "部分支持",
                        },
                        {
                            "chunk_id": "chunk-first",
                            "llm_score": 0.9,
                            "support_level": "direct",
                            "reason": "重复 ID",
                        },
                    ]
                }
            )
        )

        outcome = await agent.rerank(
            query="测试整批保护",
            candidates=(first, second),
            deterministic_scores={
                "chunk-first": 0.8,
                "chunk-second": 0.6,
            },
        )

        self.assertTrue(outcome.degraded)
        self.assertEqual(
            [record.chunk_id for record in outcome.records],
            ["chunk-first", "chunk-second"],
        )
        self.assertEqual(
            outcome.scores,
            {"chunk-first": 0.8, "chunk-second": 0.6},
        )

    async def test_incomplete_small_batch_upgrades_to_large_once(self) -> None:
        from app.infrastructure.llm.client import llm_upgrade_scope

        module = _knowledge_chunk_rerank_module()
        first = _knowledge_record(
            chunk_id="chunk-first",
            document_id="doc-first",
            title="第一候选",
            content="第一候选内容。",
        )
        second = _knowledge_record(
            chunk_id="chunk-second",
            document_id="doc-second",
            title="第二候选",
            content="第二候选内容。",
        )
        small = _FixedLlm(
            {
                "items": [
                    {
                        "chunk_id": "chunk-first",
                        "llm_score": 0.5,
                        "support_level": "partial",
                        "reason": "输出不完整",
                    }
                ]
            }
        )
        large = _FixedLlm(
            {
                "items": [
                    {
                        "chunk_id": "chunk-first",
                        "llm_score": 0.5,
                        "support_level": "partial",
                        "reason": "部分支持",
                    },
                    {
                        "chunk_id": "chunk-second",
                        "llm_score": 0.9,
                        "support_level": "direct",
                        "reason": "直接支持",
                    },
                ]
            }
        )
        agent = module.KnowledgeChunkRerankAgent(
            llm=small,
            large_llm=large,
        )

        deadline = asyncio.get_running_loop().time() + 60.0
        with llm_upgrade_scope(deadline=deadline):
            outcome = await agent.rerank(
                query="哪个候选直接支持问题？",
                candidates=(first, second),
                deterministic_scores={
                    "chunk-first": 0.8,
                    "chunk-second": 0.6,
                },
            )

        self.assertEqual(small.calls, 1)
        self.assertEqual(large.calls, 1)
        self.assertFalse(outcome.degraded)
        self.assertAlmostEqual(outcome.scores["chunk-first"], 0.74)
        self.assertAlmostEqual(outcome.scores["chunk-second"], 0.66)

    async def test_plan_rerank_timeout_does_not_upgrade(self) -> None:
        from app.infrastructure.llm.client import llm_upgrade_scope

        module = _knowledge_chunk_rerank_module()
        steps, candidates, relations = self._plan_batch(
            step_count=2,
            relation_count=2,
        )
        small = _RecordingKnowledgeReasoningPlannerLlm([TimeoutError()])
        large = _FixedLlm(
            {
                "items": [
                    {
                        "step_id": relation.step_id,
                        "chunk_id": relation.chunk_id,
                        "llm_score": 0.9,
                        "support_level": "direct",
                    }
                    for relation in relations
                ]
            }
        )
        agent = module.KnowledgeChunkRerankAgent(
            llm=small,
            large_llm=large,
        )

        deadline = asyncio.get_running_loop().time() + 60.0
        with llm_upgrade_scope(deadline=deadline):
            outcome = await agent.rerank_plan(
                question="混合检索",
                steps=steps,
                candidates=candidates,
                relations=relations,
            )

        self.assertEqual(small.calls, 1)
        self.assertEqual(large.calls, 0)
        self.assertTrue(outcome.degraded)

    async def test_plan_rerank_calls_llm_once_for_five_steps_and_thirty_pairs(
        self,
    ) -> None:
        module = _knowledge_chunk_rerank_module()
        steps, candidates, relations = self._plan_batch()
        llm = _FixedLlm(
            {
                "items": [
                    {
                        "step_id": relation.step_id,
                        "chunk_id": relation.chunk_id,
                        "llm_score": 0.75,
                        "support_level": "direct",
                    }
                    for relation in relations
                ]
            }
        )
        agent = module.KnowledgeChunkRerankAgent(llm=llm)

        outcome = await agent.rerank_plan(
            question="全面了解混合检索",
            steps=steps,
            candidates=candidates,
            relations=relations,
        )

        self.assertEqual(llm.calls, 1)
        self.assertFalse(outcome.degraded)
        self.assertEqual(len(outcome.relations), 30)
        self.assertTrue(
            all(relation.support_level == "direct" for relation in outcome.relations)
        )

    async def test_plan_rerank_invalid_pairs_fall_back_as_a_whole(self) -> None:
        module = _knowledge_chunk_rerank_module()
        steps, candidates, relations = self._plan_batch(
            step_count=2,
            relation_count=2,
        )
        valid_items = [
            {
                "step_id": relation.step_id,
                "chunk_id": relation.chunk_id,
                "llm_score": 0.5,
                "support_level": "partial",
            }
            for relation in relations
        ]
        invalid_item_sets = (
            valid_items[:1],
            [valid_items[0], valid_items[0]],
            [
                valid_items[0],
                valid_items[1] | {"chunk_id": "chunk-unknown"},
            ],
        )
        for items in invalid_item_sets:
            with self.subTest(items=items):
                agent = module.KnowledgeChunkRerankAgent(
                    llm=_FixedLlm({"items": items})
                )

                outcome = await agent.rerank_plan(
                    question="混合检索",
                    steps=steps,
                    candidates=candidates,
                    relations=relations,
                )

                self.assertTrue(outcome.degraded)
                self.assertEqual(len(outcome.relations), 2)

    async def test_plan_rerank_rejects_bounds_before_calling_llm(self) -> None:
        module = _knowledge_chunk_rerank_module()
        contracts = _knowledge_reasoning_contracts()
        steps, candidates, relations = self._plan_batch()
        oversized_candidates = candidates + (
            _knowledge_record(
                chunk_id="chunk-plan-21",
                document_id="doc-plan-extra",
                title="混合检索额外候选",
                content="混合检索额外候选内容。",
            ),
        )
        oversized_relations = relations + (
            contracts.KnowledgePlanCandidateRelation(
                step_id="step-1",
                chunk_id="chunk-plan-1",
                deterministic_score=0.1,
            ),
        )
        for current_candidates, current_relations in (
            (oversized_candidates, relations),
            (candidates, oversized_relations),
        ):
            with self.subTest(
                candidate_count=len(current_candidates),
                relation_count=len(current_relations),
            ):
                llm = _FixedLlm({"items": []})
                agent = module.KnowledgeChunkRerankAgent(llm=llm)
                with self.assertRaises(ValueError):
                    await agent.rerank_plan(
                        question="混合检索",
                        steps=steps,
                        candidates=current_candidates,
                        relations=current_relations,
                    )
                self.assertEqual(llm.calls, 0)

    async def test_plan_rerank_llm_failure_returns_deterministic_relations(
        self,
    ) -> None:
        module = _knowledge_chunk_rerank_module()
        steps, candidates, relations = self._plan_batch(
            step_count=2,
            relation_count=2,
        )
        agent = module.KnowledgeChunkRerankAgent(
            llm=_RecordingKnowledgeReasoningPlannerLlm([TimeoutError()])
        )

        outcome = await agent.rerank_plan(
            question="混合检索",
            steps=steps,
            candidates=candidates,
            relations=relations,
        )

        self.assertTrue(outcome.degraded)
        self.assertEqual(
            {(item.step_id, item.chunk_id) for item in outcome.relations},
            {(item.step_id, item.chunk_id) for item in relations},
        )


class KnowledgeEvidenceGateTests(unittest.TestCase):
    """验证五类动作字段互斥、固定优先级和一次改写边界。"""

    @staticmethod
    def _components() -> tuple[Any, Any, Any, Any]:
        return _evidence_routing_components()

    def test_action_payload_rejects_non_answer_evidence(self) -> None:
        _, _, Decision, _ = self._components()

        with self.assertRaises(ValueError):
            Decision(
                action="ask",
                confidence=1.0,
                reason_code="missing_information",
                clarification_question="请补充目标版本。",
                approved_evidence_ids=("chunk-1",),
            )

        with self.assertRaises(ValueError):
            Decision(
                action="answer",
                confidence=1.0,
                reason_code="enough_evidence",
                approved_evidence_ids=(),
            )

    def test_precheck_uses_safety_selection_and_missing_information_priority(
        self,
    ) -> None:
        Option, _, _, Gate = self._components()
        gate = Gate()
        skill_options = (
            Option(option_id="java", label="Java 并发"),
            Option(option_id="python", label="Python 并发"),
        )

        unsafe = gate.precheck(
            safety_allowed=False,
            skill_candidates=skill_options,
        )
        selected = gate.precheck(skill_candidates=skill_options)
        asked = gate.precheck(
            missing_information=("目标版本",),
            clarification_question="请补充目标版本。",
        )

        self.assertEqual((unsafe.action, unsafe.reason_code), (
            "refuse",
            "unsafe_request",
        ))
        self.assertEqual((selected.action, selected.reason_code), (
            "select",
            "multiple_skill_candidates",
        ))
        self.assertEqual(
            tuple(option.option_id for option in selected.options),
            ("java", "python"),
        )
        self.assertEqual((asked.action, asked.reason_code), (
            "ask",
            "missing_information",
        ))
        self.assertEqual(asked.approved_evidence_ids, ())

    def test_precheck_distinguishes_document_selection_and_scope_conflict(
        self,
    ) -> None:
        Option, _, _, Gate = self._components()
        gate = Gate()
        document_options = (
            Option(option_id="doc-1", label="同名文档（一）"),
            Option(option_id="doc-2", label="同名文档（二）"),
        )

        selected = gate.precheck(scope_candidates=document_options)
        conflict = gate.precheck(skill_scope_conflict=True)
        unresolved = gate.precheck(
            scope_resolved=False,
            clarification_question="请直接提供文档标题。",
        )

        self.assertEqual(selected.action, "select")
        self.assertEqual(selected.reason_code, "multiple_document_candidates")
        self.assertEqual(conflict.action, "refuse")
        self.assertEqual(conflict.reason_code, "skill_scope_conflict")
        self.assertEqual(unresolved.action, "ask")
        self.assertEqual(unresolved.reason_code, "unresolved_reference")

    def test_after_retrieval_answers_only_with_approved_evidence(self) -> None:
        _, Signals, _, Gate = self._components()
        decision = Gate().decide_after_retrieval(
            Signals(
                relevance=0.4,
                answerability=1.0,
                ambiguity=0.0,
                gate_profile="default_evidence",
                selected_evidence_ids=("chunk-1", "chunk-2"),
            )
        )

        self.assertEqual(decision.action, "answer")
        self.assertEqual(decision.reason_code, "enough_evidence")
        self.assertEqual(
            decision.approved_evidence_ids,
            ("chunk-1", "chunk-2"),
        )

    def test_after_retrieval_rewrites_once_then_refuses(self) -> None:
        _, Signals, _, Gate = self._components()
        signals = Signals(
            relevance=0.0,
            answerability=0.0,
            ambiguity=0.0,
            gate_profile="default_evidence",
            selected_evidence_ids=(),
        )
        gate = Gate()

        first = gate.decide_after_retrieval(
            signals,
            retry_query="Java virtual thread 数据库连接池 限流",
            rewrite_attempted=False,
        )
        second = gate.decide_after_retrieval(
            signals,
            retry_query="Java virtual thread 数据库连接池 限流",
            rewrite_attempted=True,
        )

        self.assertEqual(first.action, "rewrite")
        self.assertEqual(first.reason_code, "low_relevance_retry_available")
        self.assertEqual(
            first.rewritten_query,
            "Java virtual thread 数据库连接池 限流",
        )
        self.assertEqual(second.action, "refuse")
        self.assertEqual(second.reason_code, "rewrite_exhausted")
        self.assertEqual(second.approved_evidence_ids, ())

    def test_strict_profile_only_tightens_answer_threshold(self) -> None:
        _, Signals, _, Gate = self._components()
        gate = Gate()
        default = gate.decide_after_retrieval(
            Signals(
                relevance=0.4,
                answerability=0.6,
                ambiguity=0.0,
                gate_profile="default_evidence",
                selected_evidence_ids=("chunk-1",),
            )
        )
        strict = gate.decide_after_retrieval(
            Signals(
                relevance=0.4,
                answerability=0.6,
                ambiguity=0.0,
                gate_profile="strict_evidence",
                selected_evidence_ids=("chunk-1",),
            )
        )

        self.assertEqual(default.action, "answer")
        self.assertEqual(strict.action, "refuse")
        self.assertEqual(strict.reason_code, "no_relevant_evidence")


class RuntimeSkillCatalogTests(unittest.TestCase):
    """验证 Manifest 只能从受控目录全量、安全编译。"""

    @staticmethod
    def _catalog() -> Any:
        _, Catalog, _ = _runtime_skill_components()
        return Catalog

    def test_empty_and_valid_catalog_load_compiled_skills(self) -> None:
        Catalog = self._catalog()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "skills"
            self.assertEqual(Catalog(root=root).load(), ())
            root.mkdir()
            manifest = _runtime_skill_manifest("java-concurrency")
            _write_runtime_skill(root, manifest)

            skills = Catalog(root=root).load()

        self.assertEqual(len(skills), 1)
        self.assertEqual(skills[0].skill_id, "java-concurrency")
        self.assertEqual(skills[0].activation_keywords, ("虚拟线程",))
        self.assertEqual(skills[0].response_policy.organization, (
            "conclusion_then_details"
        ))

    def test_catalog_rejects_unknown_hash_path_and_tool_payloads(self) -> None:
        Catalog = self._catalog()
        mutations = (
            ("unknown", lambda item: item.__setitem__("unexpected", True)),
            ("hash", lambda item: item.__setitem__("content_hash", "0" * 64)),
            ("tools", lambda item: item.__setitem__("allowed_tools", ["shell"])),
        )
        for label, mutate in mutations:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir) / "skills"
                root.mkdir()
                manifest = _runtime_skill_manifest("java-concurrency")
                mutate(manifest)
                _write_runtime_skill(root, manifest)
                with self.assertRaises(ValueError):
                    Catalog(root=root).load()

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "skills"
            root.mkdir()
            manifest = _runtime_skill_manifest("java-concurrency")
            _write_runtime_skill(root, manifest)
            (root / "java-concurrency" / "notes.md").write_text(
                "不允许的额外文件",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                Catalog(root=root).load()

    def test_catalog_rejects_directory_mismatch_and_symbolic_links(self) -> None:
        Catalog = self._catalog()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "skills"
            root.mkdir()
            manifest = _runtime_skill_manifest("java-concurrency")
            wrong_dir = root / "wrong-name"
            wrong_dir.mkdir()
            (wrong_dir / "manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                Catalog(root=root).load()

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            root = base / "skills"
            target = base / "target"
            root.mkdir()
            target.mkdir()
            manifest = _runtime_skill_manifest("java-concurrency")
            _write_runtime_skill(target, manifest)
            (root / "java-concurrency").symlink_to(
                target / "java-concurrency",
                target_is_directory=True,
            )
            with self.assertRaises(ValueError):
                Catalog(root=root).load()


class RuntimeSkillMatcherTests(unittest.TestCase):
    """验证匹配只读取原问题，并稳定处理唯一、并列和范围冲突。"""

    @staticmethod
    def _compiled_skills(manifests: tuple[dict[str, Any], ...]) -> tuple[Any, ...]:
        _, Catalog, _ = _runtime_skill_components()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "skills"
            root.mkdir()
            for manifest in manifests:
                _write_runtime_skill(root, manifest)
            return Catalog(root=root).load()

    def test_matcher_selects_unique_best_keyword_match(self) -> None:
        _, _, Matcher = _runtime_skill_components()
        skills = self._compiled_skills(
            (
                _runtime_skill_manifest(
                    "java-concurrency",
                    keywords=("虚拟线程", "并发"),
                    priority=100,
                ),
                _runtime_skill_manifest(
                    "python-async",
                    keywords=("异步",),
                    topics=("Python",),
                    priority=200,
                ),
            )
        )

        result = Matcher().match(
            "虚拟线程适合哪些并发任务？",
            skills=skills,
            document_topics_by_id={
                "doc-java": ("Java", "并发"),
                "doc-python": ("Python",),
            },
        )

        self.assertEqual(result.primary.skill.skill_id, "java-concurrency")
        self.assertEqual(result.primary.matched_keywords, ("虚拟线程", "并发"))
        self.assertEqual(result.primary.resolved_document_ids, ("doc-java",))
        self.assertEqual(result.candidates, ())

    def test_matcher_returns_stable_candidates_for_equal_scores(self) -> None:
        _, _, Matcher = _runtime_skill_components()
        skills = self._compiled_skills(
            (
                _runtime_skill_manifest(
                    "java-advanced",
                    keywords=("并发",),
                    priority=100,
                ),
                _runtime_skill_manifest(
                    "java-basic",
                    keywords=("并发",),
                    priority=100,
                ),
            )
        )

        result = Matcher().match("并发如何限流？", skills=skills)

        self.assertIsNone(result.primary)
        self.assertEqual(
            tuple(item.skill.skill_id for item in result.candidates),
            ("java-advanced", "java-basic"),
        )
        self.assertFalse(result.too_many_candidates)

    def test_matcher_reports_scope_conflict_without_falling_back(self) -> None:
        _, _, Matcher = _runtime_skill_components()
        skills = self._compiled_skills(
            (
                _runtime_skill_manifest(
                    "java-concurrency",
                    document_ids=("doc-java",),
                    topics=(),
                ),
            )
        )

        result = Matcher().match(
            "虚拟线程有哪些限制？",
            skills=skills,
            document_ids=("doc-python",),
        )

        self.assertIsNone(result.primary)
        self.assertEqual(result.candidates, ())
        self.assertTrue(result.scope_conflict)


class _FakeRuntimeSkillCatalog:
    def __init__(self, payload: tuple[Any, ...] = ()) -> None:
        self.payload = payload
        self.error: Exception | None = None
        self.calls = 0

    def load(self) -> tuple[Any, ...]:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.payload


class RuntimeSkillRegistryTests(unittest.TestCase):
    """验证 reload 原子替换且既有请求保留捕获代。"""

    @staticmethod
    def _compiled(version: str) -> Any:
        _, Catalog, _ = _runtime_skill_components()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "skills"
            root.mkdir()
            manifest = _runtime_skill_manifest("java-concurrency")
            manifest["version"] = version
            canonical = json.dumps(
                {
                    key: value
                    for key, value in manifest.items()
                    if key != "content_hash"
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            manifest["content_hash"] = hashlib.sha256(canonical).hexdigest()
            _write_runtime_skill(root, manifest)
            return Catalog(root=root).load()[0]

    def test_reload_advances_generation_without_mutating_captured_snapshot(
        self,
    ) -> None:
        Registry = _runtime_skill_registry()
        catalog = _FakeRuntimeSkillCatalog((self._compiled("1.0.0"),))
        moments = iter(
            (
                datetime(2026, 8, 14, 9, 0, tzinfo=timezone.utc),
                datetime(2026, 8, 14, 9, 1, tzinfo=timezone.utc),
                datetime(2026, 8, 14, 9, 2, tzinfo=timezone.utc),
            )
        )
        registry = Registry(catalog=catalog, clock=lambda: next(moments))

        first_result = registry.reload()
        captured = registry.capture_snapshot()
        catalog.payload = (self._compiled("2.0.0"),)
        second_result = registry.reload()

        self.assertTrue(first_result.reloaded)
        self.assertTrue(second_result.reloaded)
        self.assertEqual(first_result.snapshot.generation, 1)
        self.assertEqual(second_result.snapshot.generation, 2)
        self.assertEqual(captured.skills["java-concurrency"].version, "1.0.0")
        self.assertEqual(
            registry.capture_snapshot().skills["java-concurrency"].version,
            "2.0.0",
        )
        with self.assertRaises(TypeError):
            captured.skills["other"] = self._compiled("3.0.0")

    def test_failed_reload_keeps_previous_snapshot_identity(self) -> None:
        Registry = _runtime_skill_registry()
        catalog = _FakeRuntimeSkillCatalog((self._compiled("1.0.0"),))
        registry = Registry(
            catalog=catalog,
            clock=lambda: datetime(2026, 8, 14, tzinfo=timezone.utc),
        )
        registry.reload()
        previous = registry.capture_snapshot()
        catalog.error = ValueError("invalid manifest")

        result = registry.reload()

        self.assertFalse(result.reloaded)
        self.assertEqual(result.error_code, "catalog_invalid")
        self.assertIs(result.snapshot, previous)
        self.assertIs(registry.capture_snapshot(), previous)

    def test_empty_catalog_is_a_valid_new_snapshot(self) -> None:
        Registry = _runtime_skill_registry()
        catalog = _FakeRuntimeSkillCatalog()
        registry = Registry(
            catalog=catalog,
            clock=lambda: datetime(2026, 8, 14, tzinfo=timezone.utc),
        )

        result = registry.reload()

        self.assertTrue(result.reloaded)
        self.assertEqual(result.snapshot.generation, 1)
        self.assertEqual(dict(result.snapshot.skills), {})
        self.assertEqual(len(result.snapshot.catalog_hash), 64)


class KnowledgeQueryAnalysisContractTests(unittest.TestCase):
    """验证统一查询分析不会携带矛盾策略或无界子查询。"""

    def test_valid_comparative_plan_accepts_three_unique_queries(self) -> None:
        plan = KnowledgeQueryAnalysis(
            standalone_query="BM25 和向量检索有什么区别？",
            question_type="comparative",
            strategy="decomposed",
            sub_queries=("BM25", "向量检索", "BM25 和向量检索的区别"),
            confidence=1.0,
        )

        self.assertEqual(plan.strategy, "decomposed")
        self.assertEqual(len(plan.sub_queries), 3)

    def test_retry_query_and_missing_information_are_mutually_protected(
        self,
    ) -> None:
        retry = KnowledgeQueryAnalysis(
            standalone_query="Java 虚拟线程限制",
            retry_query="Java virtual thread 数据库连接池 限流",
            question_type="factual",
            strategy="direct",
            confidence=0.9,
        )
        missing = KnowledgeQueryAnalysis(
            standalone_query="分析部署方案",
            missing_information=("目标版本",),
            clarification_question="请补充目标版本。",
            question_type="analytical",
            strategy="direct",
            confidence=0.9,
        )

        self.assertEqual(
            retry.retry_query,
            "Java virtual thread 数据库连接池 限流",
        )
        self.assertEqual(missing.missing_information, ("目标版本",))
        for invalid in (
            {
                "retry_query": "Java 虚拟线程限制",
            },
            {
                "missing_information": ("目标版本",),
            },
            {
                "clarification_question": "请补充目标版本。",
            },
            {
                "retry_query": "Java virtual thread 限流",
                "missing_information": ("目标版本",),
                "clarification_question": "请补充目标版本。",
            },
        ):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    KnowledgeQueryAnalysis(
                        standalone_query="Java 虚拟线程限制",
                        question_type="factual",
                        strategy="direct",
                        confidence=0.9,
                        **invalid,
                    )

    def test_direct_plan_rejects_sub_queries(self) -> None:
        with self.assertRaises(ValueError):
            KnowledgeQueryAnalysis(
                standalone_query="RRF 是什么？",
                question_type="factual",
                strategy="direct",
                sub_queries=("RRF",),
                confidence=1.0,
            )

    def test_comparative_plan_rejects_direct_strategy(self) -> None:
        with self.assertRaises(ValueError):
            KnowledgeQueryAnalysis(
                standalone_query="比较这两种方案",
                question_type="comparative",
                strategy="direct",
                confidence=0.9,
            )

    def test_simple_question_types_reject_decomposed_strategy(self) -> None:
        for question_type in (
            "factual",
            "procedural",
            "verification",
            "summarization",
        ):
            with self.subTest(question_type=question_type):
                with self.assertRaises(ValueError):
                    KnowledgeQueryAnalysis(
                        standalone_query="受保护查询",
                        question_type=question_type,
                        strategy="decomposed",
                        sub_queries=("查询一", "查询二"),
                        confidence=0.9,
                    )

    def test_decomposed_plan_rejects_too_few_or_duplicate_queries(self) -> None:
        for sub_queries in (("BM25",), ("BM25", "BM25")):
            with self.subTest(sub_queries=sub_queries):
                with self.assertRaises(ValueError):
                    KnowledgeQueryAnalysis(
                        standalone_query="BM25 和向量检索有什么区别？",
                        question_type="comparative",
                        strategy="decomposed",
                        sub_queries=sub_queries,
                        confidence=1.0,
                    )

    def test_decomposed_plan_rejects_more_than_three_queries(self) -> None:
        with self.assertRaises(ValueError):
            KnowledgeQueryAnalysis(
                standalone_query="分析多种检索方案",
                question_type="analytical",
                strategy="decomposed",
                sub_queries=("一", "二", "三", "四"),
                confidence=0.9,
            )


class KnowledgeReasoningPlanContractTests(unittest.TestCase):
    """验证复杂问题计划、覆盖结果和公开轨迹保持严格有界。"""

    @staticmethod
    def _comparison_steps() -> tuple[Any, ...]:
        contracts = _knowledge_reasoning_contracts()
        return (
            contracts.KnowledgePlanStep(
                step_id="step-1",
                facet="subject",
                query="BM25 的检索机制",
                target_subjects=("BM25",),
                required=True,
            ),
            contracts.KnowledgePlanStep(
                step_id="step-2",
                facet="subject",
                query="向量检索的检索机制",
                target_subjects=("向量检索",),
                required=True,
            ),
            contracts.KnowledgePlanStep(
                step_id="step-3",
                facet="comparison",
                query="BM25 和向量检索的召回差异",
                target_subjects=("BM25", "向量检索"),
                required=True,
            ),
        )

    def test_comparison_plan_requires_two_subjects_and_common_dimension(
        self,
    ) -> None:
        contracts = _knowledge_reasoning_contracts()
        steps = self._comparison_steps()

        plan = contracts.KnowledgeReasoningPlan(
            revision=1,
            question_type="comparative",
            strategy="comparison_matrix",
            steps=steps,
            confidence=0.9,
        )

        self.assertEqual(len(plan.steps), 3)
        with self.assertRaises(ValueError):
            contracts.KnowledgeReasoningPlan(
                revision=1,
                question_type="comparative",
                strategy="comparison_matrix",
                steps=(steps[0], steps[2]),
                confidence=0.9,
            )
        with self.assertRaises(ValueError):
            contracts.KnowledgeReasoningPlan(
                revision=1,
                question_type="comparative",
                strategy="comparison_matrix",
                steps=steps[:2],
                confidence=0.9,
            )

    def test_first_revision_rejects_keep_replace_ids(self) -> None:
        contracts = _knowledge_reasoning_contracts()
        for field_name in ("kept_step_ids", "replaced_step_ids"):
            with self.subTest(field_name=field_name):
                with self.assertRaises(ValueError):
                    contracts.KnowledgeReasoningPlan(
                        revision=1,
                        question_type="comparative",
                        strategy="comparison_matrix",
                        steps=self._comparison_steps(),
                        confidence=0.9,
                        **{field_name: ("step-1",)},
                    )

    def test_second_revision_requires_disjoint_current_step_ids(self) -> None:
        contracts = _knowledge_reasoning_contracts()
        steps = self._comparison_steps()
        valid = contracts.KnowledgeReasoningPlan(
            revision=2,
            question_type="comparative",
            strategy="comparison_matrix",
            steps=steps,
            kept_step_ids=("step-1",),
            replaced_step_ids=("step-4",),
            confidence=0.9,
        )

        self.assertEqual(valid.kept_step_ids, ("step-1",))
        with self.assertRaises(ValueError):
            contracts.KnowledgeReasoningPlan(
                revision=2,
                question_type="comparative",
                strategy="comparison_matrix",
                steps=steps,
                kept_step_ids=("step-1",),
                replaced_step_ids=("step-1",),
                confidence=0.9,
            )
        with self.assertRaises(ValueError):
            contracts.KnowledgeReasoningPlan(
                revision=2,
                question_type="comparative",
                strategy="comparison_matrix",
                steps=steps,
                kept_step_ids=("step-9",),
                confidence=0.9,
            )

    def test_plan_rejects_duplicate_steps_unsafe_query_and_invalid_subject(
        self,
    ) -> None:
        contracts = _knowledge_reasoning_contracts()
        valid_steps = self._comparison_steps()
        invalid_steps = (
            (valid_steps[0], valid_steps[0]),
            (
                valid_steps[0].model_copy(update={"query": "   "}),
                valid_steps[1],
            ),
            (
                valid_steps[0].model_copy(update={"query": "x" * 501}),
                valid_steps[1],
            ),
            (
                valid_steps[0].model_copy(update={"query": "<script>"}),
                valid_steps[1],
            ),
            (
                valid_steps[0].model_copy(
                    update={"target_subjects": ("BM25", "bm25")}
                ),
                valid_steps[1],
            ),
            (
                valid_steps[0].model_copy(update={"target_subjects": (" ",)}),
                valid_steps[1],
            ),
        )
        for steps in invalid_steps:
            with self.subTest(steps=steps):
                with self.assertRaises(ValueError):
                    contracts.KnowledgeReasoningPlan(
                        revision=1,
                        question_type="comparative",
                        strategy="comparison_matrix",
                        steps=steps,
                        confidence=0.9,
                    )
        with self.assertRaises(ValueError):
            contracts.KnowledgePlanStep(
                step_id="step-1",
                facet="subject",
                query="BM25",
                target_subjects=("BM25",),
                required=True,
                reasoning="不允许的额外字段",
            )

    def test_coverage_ratio_counts_only_covered_steps(self) -> None:
        contracts = _knowledge_reasoning_contracts()
        results = (
            contracts.KnowledgePlanStepResult(
                step_id="step-1",
                status="covered",
                search_query="查询一",
                selected_chunk_ids=("chunk-1",),
                selected_document_ids=("doc-1",),
                reason_code="enough_evidence",
            ),
            contracts.KnowledgePlanStepResult(
                step_id="step-2",
                status="covered",
                search_query="查询二",
                selected_chunk_ids=("chunk-2",),
                selected_document_ids=("doc-2",),
                reason_code="enough_evidence",
            ),
            contracts.KnowledgePlanStepResult(
                step_id="step-3",
                status="weak",
                search_query="查询三",
                reason_code="insufficient_subject_coverage",
            ),
            contracts.KnowledgePlanStepResult(
                step_id="step-4",
                status="failed",
                search_query="查询四",
                reason_code="search_failed",
            ),
        )

        coverage = contracts.KnowledgePlanCoverage(
            step_results=results,
            required_steps=3,
            covered_required_steps=2,
            covered_steps=2,
            coverage_ratio=0.5,
            decision="replan",
        )

        self.assertEqual(coverage.coverage_ratio, 0.5)
        with self.assertRaises(ValueError):
            contracts.KnowledgePlanCoverage.model_validate(
                coverage.model_dump() | {"coverage_ratio": 0.75}
            )

    def test_execution_trace_keeps_plan_fields_bounded(self) -> None:
        contracts = _knowledge_reasoning_contracts()
        step_result = contracts.KnowledgePlanStepResult(
            step_id="step-1",
            status="covered",
            search_query="BM25",
            selected_chunk_ids=("chunk-1",),
            selected_document_ids=("doc-1",),
            reason_code="enough_evidence",
        )
        coverage = contracts.KnowledgePlanCoverage(
            step_results=(step_result, step_result.model_copy(update={"step_id": "step-2"})),
            required_steps=2,
            covered_required_steps=2,
            covered_steps=2,
            coverage_ratio=1.0,
            decision="answer",
        )
        trace_step = contracts.KnowledgePlanTraceStep(
            revision=1,
            step_id="step-1",
            facet="subject",
            query="BM25",
            required=True,
            status="covered",
            reason_code="enough_evidence",
            selected_chunk_ids=("chunk-1",),
        )
        trace_fields = {
            "trace_id": "a" * 32,
            "request_route": "/api/v1/knowledge/ask",
            "question": "比较 BM25 和向量检索",
            "standalone_query": "比较 BM25 和向量检索",
            "reasoning_strategy": "comparison_matrix",
            "plan_revision_count": 1,
            "coverage": coverage,
            "result": KnowledgeExecutionResult(
                status="success",
                elapsed_ms=1.0,
            ),
        }

        with self.assertRaises(ValueError):
            KnowledgeExecutionTrace(
                **trace_fields,
                plan_steps=tuple(trace_step for _ in range(11)),
            )
        with self.assertRaises(ValueError):
            KnowledgeExecutionTrace(
                **trace_fields,
                plan_steps=(
                    trace_step.model_copy(
                        update={
                            "selected_chunk_ids": tuple(
                                f"chunk-{index}" for index in range(7)
                            )
                        }
                    ),
                ),
            )


class _RecordingKnowledgeReasoningPlannerLlm:
    """按顺序返回完整计划对象并记录真实消息边界。"""

    def __init__(self, outputs: list[object]) -> None:
        self.outputs = list(outputs)
        self.calls = 0
        self.close_calls = 0
        self.messages: list[list[Any]] = []

    async def ainvoke(self, messages: list[Any]) -> object:
        self.messages.append(list(messages))
        output = self.outputs[self.calls]
        self.calls += 1
        if isinstance(output, BaseException):
            raise output
        return output

    async def aclose(self) -> None:
        self.close_calls += 1


class KnowledgeReasoningPlannerAgentTests(unittest.IsolatedAsyncioTestCase):
    """验证 Planner 只生成受控计划且不接触证据正文或答案。"""

    @staticmethod
    def _comparison_output(
        *,
        confidence: float = 0.9,
        first_subject: str = "BM25",
        query_suffix: str = "",
    ) -> dict[str, object]:
        return {
            "revision": 1,
            "question_type": "comparative",
            "strategy": "comparison_matrix",
            "steps": [
                {
                    "step_id": "step-1",
                    "facet": "subject",
                    "query": f"{first_subject} 的检索机制{query_suffix}",
                    "target_subjects": [first_subject],
                    "required": True,
                },
                {
                    "step_id": "step-2",
                    "facet": "subject",
                    "query": "向量检索的检索机制",
                    "target_subjects": ["向量检索"],
                    "required": True,
                },
                {
                    "step_id": "step-3",
                    "facet": "comparison",
                    "query": "BM25 和向量检索的召回差异",
                    "target_subjects": ["BM25", "向量检索"],
                    "required": True,
                },
            ],
            "kept_step_ids": [],
            "replaced_step_ids": [],
            "confidence": confidence,
        }

    @staticmethod
    def _analytical_output() -> dict[str, object]:
        return {
            "revision": 1,
            "question_type": "analytical",
            "strategy": "facet_analysis",
            "steps": [
                {
                    "step_id": "step-1",
                    "facet": "subject",
                    "query": "RRF 的事实基础",
                    "target_subjects": ["RRF"],
                    "required": True,
                },
                {
                    "step_id": "step-2",
                    "facet": "mechanism",
                    "query": "RRF 的机制",
                    "target_subjects": ["RRF"],
                    "required": True,
                },
                {
                    "step_id": "step-3",
                    "facet": "impact",
                    "query": "RRF 的影响",
                    "target_subjects": ["RRF"],
                    "required": True,
                },
            ],
            "kept_step_ids": [],
            "replaced_step_ids": [],
            "confidence": 0.9,
        }

    @staticmethod
    def _exploratory_output() -> dict[str, object]:
        return {
            "revision": 1,
            "question_type": "exploratory",
            "strategy": "coverage_synthesis",
            "steps": [
                {
                    "step_id": "step-1",
                    "facet": "definition",
                    "query": "混合检索的定义",
                    "target_subjects": ["混合检索"],
                    "required": True,
                },
                {
                    "step_id": "step-2",
                    "facet": "scenario",
                    "query": "混合检索的场景",
                    "target_subjects": ["混合检索"],
                    "required": False,
                },
            ],
            "kept_step_ids": [],
            "replaced_step_ids": [],
            "confidence": 0.9,
        }

    @staticmethod
    def _comparison_replan_output() -> dict[str, object]:
        return {
            "revision": 2,
            "question_type": "comparative",
            "strategy": "comparison_matrix",
            "steps": [
                KnowledgeReasoningPlannerAgentTests._comparison_output()[
                    "steps"
                ][0],
                KnowledgeReasoningPlannerAgentTests._comparison_output()[
                    "steps"
                ][1],
                {
                    "step_id": "step-4",
                    "facet": "comparison",
                    "query": "BM25 和向量检索的排序差异",
                    "target_subjects": ["BM25", "向量检索"],
                    "required": True,
                },
            ],
            "kept_step_ids": ["step-1", "step-2"],
            "replaced_step_ids": ["step-3"],
            "confidence": 0.9,
        }

    async def test_valid_plan_maps_question_type_to_strategy(self) -> None:
        module = _knowledge_reasoning_planner_module()
        cases = (
            (
                "comparative",
                "比较 BM25 和向量检索",
                ("BM25", "向量检索"),
                self._comparison_output(),
                "comparison_matrix",
            ),
            (
                "analytical",
                "分析 RRF 的机制和影响",
                ("RRF 的机制", "RRF 的影响"),
                self._analytical_output(),
                "facet_analysis",
            ),
            (
                "exploratory",
                "全面了解混合检索",
                ("混合检索",),
                self._exploratory_output(),
                "coverage_synthesis",
            ),
        )
        for question_type, query, sub_queries, output, strategy in cases:
            with self.subTest(question_type=question_type):
                llm = _RecordingKnowledgeReasoningPlannerLlm([output])
                agent = module.KnowledgeReasoningPlannerAgent(llm=llm)

                plan = await agent.plan(
                    standalone_query=query,
                    question_type=question_type,
                    sub_queries=sub_queries,
                )

                self.assertEqual(plan.strategy, strategy)
                self.assertEqual(plan.question_type, question_type)
                self.assertEqual(plan.revision, 1)

    async def test_invalid_small_plan_upgrades_to_large_once(self) -> None:
        from app.infrastructure.llm.client import llm_upgrade_scope

        module = _knowledge_reasoning_planner_module()
        small = _RecordingKnowledgeReasoningPlannerLlm(
            [self._comparison_output(confidence=0.5)]
        )
        large = _RecordingKnowledgeReasoningPlannerLlm(
            [self._comparison_output(confidence=0.94)]
        )
        agent = module.KnowledgeReasoningPlannerAgent(
            llm=small,
            large_llm=large,
        )

        deadline = asyncio.get_running_loop().time() + 60.0
        with llm_upgrade_scope(deadline=deadline):
            plan = await agent.plan(
                standalone_query="比较 BM25 和向量检索",
                question_type="comparative",
                sub_queries=("BM25", "向量检索"),
            )

        self.assertEqual(small.calls, 1)
        self.assertEqual(large.calls, 1)
        self.assertEqual(plan.confidence, 0.94)

    async def test_planner_timeout_does_not_upgrade(self) -> None:
        from app.infrastructure.llm.client import llm_upgrade_scope

        module = _knowledge_reasoning_planner_module()
        small = _RecordingKnowledgeReasoningPlannerLlm([TimeoutError()])
        large = _RecordingKnowledgeReasoningPlannerLlm(
            [self._comparison_output(confidence=0.94)]
        )
        agent = module.KnowledgeReasoningPlannerAgent(
            llm=small,
            large_llm=large,
        )

        deadline = asyncio.get_running_loop().time() + 60.0
        with llm_upgrade_scope(deadline=deadline):
            with self.assertRaises(RuntimeError):
                await agent.plan(
                    standalone_query="比较 BM25 和向量检索",
                    question_type="comparative",
                    sub_queries=("BM25", "向量检索"),
                )

        self.assertEqual(small.calls, 1)
        self.assertEqual(large.calls, 0)

    async def test_plan_rejects_low_confidence_new_subject_and_unsafe_text(
        self,
    ) -> None:
        module = _knowledge_reasoning_planner_module()
        invalid_outputs = (
            self._comparison_output(confidence=0.59),
            self._comparison_output(first_subject="Elasticsearch"),
            self._comparison_output(query_suffix="<script>"),
            self._comparison_output(query_suffix=" document_id=doc-secret"),
            self._comparison_output(query_suffix=" chunk_id=chunk-secret"),
            self._comparison_output(query_suffix=" Search"),
            self._comparison_output() | {"reasoning": "隐藏推理"},
        )
        for output in invalid_outputs:
            with self.subTest(output=output):
                agent = module.KnowledgeReasoningPlannerAgent(
                    llm=_RecordingKnowledgeReasoningPlannerLlm([output])
                )
                with self.assertRaises(ValueError):
                    await agent.plan(
                        standalone_query="比较 BM25 和向量检索",
                        question_type="comparative",
                        sub_queries=("BM25", "向量检索"),
                    )

    async def test_replan_receives_only_safe_statuses_and_preserves_lineage(
        self,
    ) -> None:
        contracts = _knowledge_reasoning_contracts()
        module = _knowledge_reasoning_planner_module()
        first_plan = contracts.KnowledgeReasoningPlan.model_validate(
            self._comparison_output()
        )
        step_results = (
            contracts.KnowledgePlanStepResult(
                step_id="step-1",
                status="covered",
                search_query="BM25 的检索机制",
                selected_chunk_ids=("chunk-secret-content",),
                selected_document_ids=("doc-secret-content",),
                reason_code="enough_evidence",
            ),
            contracts.KnowledgePlanStepResult(
                step_id="step-2",
                status="covered",
                search_query="向量检索的检索机制",
                selected_chunk_ids=("chunk-vector",),
                selected_document_ids=("doc-vector",),
                reason_code="enough_evidence",
            ),
            contracts.KnowledgePlanStepResult(
                step_id="step-3",
                status="weak",
                search_query="BM25 和向量检索的召回差异",
                reason_code="insufficient_subject_coverage",
            ),
        )
        llm = _RecordingKnowledgeReasoningPlannerLlm(
            [self._comparison_replan_output()]
        )
        agent = module.KnowledgeReasoningPlannerAgent(llm=llm)

        revised = await agent.replan(
            standalone_query="比较 BM25 和向量检索",
            question_type="comparative",
            previous_plan=first_plan,
            step_results=step_results,
            remaining_step_limit=3,
        )

        previous_by_id = {step.step_id: step for step in first_plan.steps}
        revised_by_id = {step.step_id: step for step in revised.steps}
        for step_id in revised.kept_step_ids:
            self.assertEqual(revised_by_id[step_id], previous_by_id[step_id])
        self.assertEqual(revised.replaced_step_ids, ("step-3",))
        message_text = "\n".join(
            str(message.content) for message in llm.messages[0]
        )
        self.assertIn("covered", message_text)
        self.assertIn("insufficient_subject_coverage", message_text)
        self.assertIn("selected_chunk_count", message_text)
        self.assertNotIn("chunk-secret-content", message_text)
        self.assertNotIn("doc-secret-content", message_text)

    async def test_prompt_marks_input_untrusted_and_omits_content_and_answer(
        self,
    ) -> None:
        module = _knowledge_reasoning_planner_module()
        llm = _RecordingKnowledgeReasoningPlannerLlm(
            [self._analytical_output()]
        )
        agent = module.KnowledgeReasoningPlannerAgent(llm=llm)

        await agent.plan(
            standalone_query="分析 RRF 的机制和影响",
            question_type="analytical",
            sub_queries=("RRF 的机制", "RRF 的影响"),
        )

        system_text = str(llm.messages[0][0].content)
        human_text = str(llm.messages[0][1].content)
        self.assertIn("不可信", system_text)
        envelope = json.loads(human_text)
        self.assertEqual(set(envelope), {"contract", "input"})
        self.assertEqual(
            envelope["contract"]["name"],
            "knowledge_reasoning_plan",
        )
        self.assertEqual(envelope["contract"]["version"], 2)
        self.assertIsInstance(envelope["contract"]["output_schema"], dict)
        self.assertEqual(envelope["input"]["mode"], "plan")
        for unsafe_marker in (
            "Chunk 正文秘密",
            "doc-secret",
            "答案草稿秘密",
            "历史全文秘密",
            "外部 Prompt 原文秘密",
            '"reasoning"',
        ):
            self.assertNotIn(unsafe_marker, human_text)
        self.assertIn("standalone_query", human_text)
        self.assertIn("question_type", human_text)
        self.assertIn("sub_queries", human_text)
        schema_text = json.dumps(
            envelope["contract"]["output_schema"],
            ensure_ascii=False,
            sort_keys=True,
        )
        self.assertIn('"const": 1', schema_text)
        self.assertIn('"const": 2', schema_text)

    async def test_cancellation_is_not_converted_to_planner_failure(self) -> None:
        module = _knowledge_reasoning_planner_module()
        agent = module.KnowledgeReasoningPlannerAgent(
            llm=_RecordingKnowledgeReasoningPlannerLlm(
                [asyncio.CancelledError()]
            )
        )

        with self.assertRaises(asyncio.CancelledError):
            await agent.plan(
                standalone_query="比较 BM25 和向量检索",
                question_type="comparative",
                sub_queries=("BM25", "向量检索"),
            )

    async def test_aclose_closes_owned_llm_once(self) -> None:
        module = _knowledge_reasoning_planner_module()
        llm = _RecordingKnowledgeReasoningPlannerLlm([])
        agent = module.KnowledgeReasoningPlannerAgent(llm=llm)

        await agent.aclose()
        await agent.aclose()

        self.assertEqual(llm.close_calls, 1)


class KnowledgePlanCoverageCheckerTests(unittest.TestCase):
    """验证覆盖决策只依据严格关系和当前快照正文。"""

    @staticmethod
    def _plan(
        *,
        question_type: str,
        strategy: str,
        steps: tuple[dict[str, object], ...],
    ) -> Any:
        contracts = _knowledge_reasoning_contracts()
        return contracts.KnowledgeReasoningPlan(
            revision=1,
            question_type=question_type,
            strategy=strategy,
            steps=steps,
            confidence=0.9,
        )

    @staticmethod
    def _relation(
        step_id: str,
        chunk_id: str,
        support_level: str,
        score: float,
    ) -> Any:
        contracts = _knowledge_reasoning_contracts()
        return contracts.KnowledgePlanEvidenceRelation(
            step_id=step_id,
            chunk_id=chunk_id,
            support_level=support_level,
            score=score,
        )

    @staticmethod
    def _comparison_plan() -> Any:
        return KnowledgePlanCoverageCheckerTests._plan(
            question_type="comparative",
            strategy="comparison_matrix",
            steps=(
                {
                    "step_id": "step-1",
                    "facet": "subject",
                    "query": "BM25 检索机制",
                    "target_subjects": ("BM25",),
                    "required": True,
                },
                {
                    "step_id": "step-2",
                    "facet": "subject",
                    "query": "向量检索机制",
                    "target_subjects": ("向量检索",),
                    "required": True,
                },
                {
                    "step_id": "step-3",
                    "facet": "comparison",
                    "query": "BM25 和向量检索差异",
                    "target_subjects": ("BM25", "向量检索"),
                    "required": True,
                },
            ),
        )

    @staticmethod
    def _comparison_records() -> tuple[KnowledgeChunkRecord, ...]:
        return (
            _knowledge_record(
                chunk_id="chunk-bm25",
                document_id="doc-bm25",
                title="BM25 检索",
                content="BM25 使用词频和逆文档频率计算相关性。",
            ),
            _knowledge_record(
                chunk_id="chunk-vector",
                document_id="doc-vector",
                title="向量检索",
                content="向量检索使用语义向量计算相似度。",
            ),
            _knowledge_record(
                chunk_id="chunk-comparison",
                document_id="doc-comparison",
                title="BM25 与向量检索比较",
                content="BM25 偏关键词匹配，向量检索偏语义召回。",
            ),
        )

    def test_required_weak_returns_replan_and_does_not_count_coverage(
        self,
    ) -> None:
        module = _knowledge_plan_coverage_module()
        checker = module.KnowledgePlanCoverageChecker()
        records = self._comparison_records()

        coverage = checker.evaluate(
            self._comparison_plan(),
            relations=(
                self._relation("step-1", "chunk-bm25", "direct", 0.9),
                self._relation("step-2", "chunk-vector", "direct", 0.9),
                self._relation(
                    "step-3", "chunk-comparison", "partial", 0.8
                ),
            ),
            records=records,
            empty_reason_by_step={},
            replanned=False,
            allow_replan=True,
        )

        self.assertEqual(coverage.decision, "replan")
        self.assertEqual(coverage.covered_steps, 2)
        self.assertAlmostEqual(coverage.coverage_ratio, 2 / 3)
        self.assertEqual(coverage.step_results[2].status, "weak")

    def test_comparison_requires_both_subjects_and_common_dimension(
        self,
    ) -> None:
        module = _knowledge_plan_coverage_module()
        checker = module.KnowledgePlanCoverageChecker()
        plan = self._comparison_plan()
        records = self._comparison_records()
        base_relations = (
            self._relation("step-1", "chunk-bm25", "direct", 0.9),
            self._relation("step-2", "chunk-vector", "direct", 0.9),
            self._relation("step-3", "chunk-comparison", "direct", 0.9),
        )
        cases = (
            base_relations[1:],
            base_relations[:2],
        )
        for relations in cases:
            with self.subTest(relations=relations):
                missing_step_ids = {
                    step.step_id for step in plan.steps
                } - {relation.step_id for relation in relations}
                coverage = checker.evaluate(
                    plan,
                    relations=relations,
                    records=records,
                    empty_reason_by_step={
                        step_id: "no_hits" for step_id in missing_step_ids
                    },
                    replanned=True,
                    allow_replan=False,
                )
                self.assertEqual(
                    coverage.decision,
                    "insufficient_evidence",
                )

    def test_analytical_first_round_replans_but_final_allows_optional_gaps(
        self,
    ) -> None:
        module = _knowledge_plan_coverage_module()
        checker = module.KnowledgePlanCoverageChecker()
        plan = self._plan(
            question_type="analytical",
            strategy="facet_analysis",
            steps=(
                {
                    "step_id": "step-1",
                    "facet": "subject",
                    "query": "RRF 事实基础",
                    "target_subjects": ("RRF",),
                    "required": True,
                },
                {
                    "step_id": "step-2",
                    "facet": "mechanism",
                    "query": "RRF 机制",
                    "target_subjects": ("RRF",),
                    "required": True,
                },
                {
                    "step_id": "step-3",
                    "facet": "impact",
                    "query": "RRF 影响",
                    "target_subjects": ("RRF",),
                    "required": False,
                },
                {
                    "step_id": "step-4",
                    "facet": "tradeoff",
                    "query": "RRF 权衡",
                    "target_subjects": ("RRF",),
                    "required": False,
                },
            ),
        )
        records = (
            _knowledge_record(
                chunk_id="chunk-rrf-fact",
                document_id="doc-rrf",
                title="RRF 事实基础",
                content="RRF 是一种排序融合方法。",
            ),
            _knowledge_record(
                chunk_id="chunk-rrf-mechanism",
                document_id="doc-rrf",
                title="RRF 机制",
                content="RRF 按各列表名次的倒数累加得分。",
            ),
        )
        relations = (
            self._relation("step-1", "chunk-rrf-fact", "direct", 0.9),
            self._relation(
                "step-2", "chunk-rrf-mechanism", "direct", 0.8
            ),
        )
        empty_reasons = {"step-3": "no_hits", "step-4": "no_hits"}

        first = checker.evaluate(
            plan,
            relations=relations,
            records=records,
            empty_reason_by_step=empty_reasons,
            replanned=False,
            allow_replan=True,
        )
        final = checker.evaluate(
            plan,
            relations=relations,
            records=records,
            empty_reason_by_step=empty_reasons,
            replanned=True,
            allow_replan=False,
        )

        self.assertEqual(first.decision, "replan")
        self.assertEqual(final.decision, "answer")
        self.assertEqual(final.coverage_ratio, 0.5)

    def test_exploratory_final_requires_three_quarters_coverage(self) -> None:
        module = _knowledge_plan_coverage_module()
        checker = module.KnowledgePlanCoverageChecker()
        plan = self._plan(
            question_type="exploratory",
            strategy="coverage_synthesis",
            steps=tuple(
                {
                    "step_id": f"step-{index}",
                    "facet": facet,
                    "query": f"混合检索{facet}",
                    "target_subjects": ("混合检索",),
                    "required": index == 1,
                }
                for index, facet in enumerate(
                    ("definition", "scenario", "constraint", "example"),
                    start=1,
                )
            ),
        )
        records = tuple(
            _knowledge_record(
                chunk_id=f"chunk-{index}",
                document_id="doc-hybrid",
                title=f"混合检索维度 {index}",
                content=f"混合检索维度 {index} 的可信说明。",
            )
            for index in range(1, 4)
        )
        three_relations = tuple(
            self._relation(
                f"step-{index}", f"chunk-{index}", "direct", 0.9
            )
            for index in range(1, 4)
        )

        enough = checker.evaluate(
            plan,
            relations=three_relations,
            records=records,
            empty_reason_by_step={"step-4": "no_hits"},
            replanned=True,
            allow_replan=False,
        )
        insufficient = checker.evaluate(
            plan,
            relations=three_relations[:2],
            records=records,
            empty_reason_by_step={
                "step-3": "no_hits",
                "step-4": "no_hits",
            },
            replanned=True,
            allow_replan=False,
        )

        self.assertEqual(enough.decision, "answer")
        self.assertEqual(enough.coverage_ratio, 0.75)
        self.assertEqual(insufficient.decision, "insufficient_evidence")

    def test_failed_and_empty_steps_preserve_safe_reason_codes(self) -> None:
        module = _knowledge_plan_coverage_module()
        checker = module.KnowledgePlanCoverageChecker()
        plan = self._plan(
            question_type="exploratory",
            strategy="coverage_synthesis",
            steps=(
                {
                    "step_id": "step-1",
                    "facet": "definition",
                    "query": "混合检索定义",
                    "target_subjects": ("混合检索",),
                    "required": True,
                },
                {
                    "step_id": "step-2",
                    "facet": "scenario",
                    "query": "混合检索场景",
                    "target_subjects": ("混合检索",),
                    "required": False,
                },
            ),
        )

        coverage = checker.evaluate(
            plan,
            relations=(),
            records=(),
            empty_reason_by_step={
                "step-1": "search_failed",
                "step-2": "scope_filtered",
            },
            replanned=False,
            allow_replan=True,
        )

        self.assertEqual(coverage.step_results[0].status, "failed")
        self.assertEqual(coverage.step_results[0].reason_code, "search_failed")
        self.assertEqual(coverage.step_results[1].status, "uncovered")
        self.assertEqual(
            coverage.step_results[1].reason_code,
            "scope_filtered",
        )

    def test_rejects_unknown_duplicate_and_out_of_snapshot_relations(
        self,
    ) -> None:
        module = _knowledge_plan_coverage_module()
        checker = module.KnowledgePlanCoverageChecker()
        plan = self._comparison_plan()
        records = self._comparison_records()
        invalid_relation_sets = (
            (self._relation("step-9", "chunk-bm25", "direct", 0.9),),
            (self._relation("step-1", "chunk-outside", "direct", 0.9),),
            (
                self._relation("step-1", "chunk-bm25", "direct", 0.9),
                self._relation("step-1", "chunk-bm25", "partial", 0.5),
            ),
        )
        for relations in invalid_relation_sets:
            with self.subTest(relations=relations):
                with self.assertRaises(ValueError):
                    checker.evaluate(
                        plan,
                        relations=relations,
                        records=records,
                        empty_reason_by_step={
                            "step-2": "no_hits",
                            "step-3": "no_hits",
                        },
                        replanned=False,
                        allow_replan=True,
                    )


class KnowledgeQueryAnalysisAgentTests(unittest.IsolatedAsyncioTestCase):
    """验证改写、分类和检索规划由一次结构化分析完成。"""

    async def test_procedural_rule_skips_llm(self) -> None:
        module = _knowledge_query_analysis_module()
        llm = _FixedLlm({})
        agent = module.KnowledgeQueryAnalysisAgent(llm=llm)

        result = await agent.analyze("如何配置 Docker 日志轮转？")

        self.assertEqual(llm.calls, 0)
        self.assertEqual(result.standalone_query, "如何配置 Docker 日志轮转？")
        self.assertEqual(result.question_type, "procedural")
        self.assertEqual(result.strategy, "direct")
        self.assertFalse(result.uses_history)
        self.assertFalse(result.degraded)

    async def test_follow_up_rewrite_and_classification_share_one_llm_call(
        self,
    ) -> None:
        module = _knowledge_query_analysis_module()
        llm = _FixedLlm(
            {
                "standalone_query": "Spring 事务传播机制有哪些限制？",
                "uses_history": True,
                "question_type": "analytical",
                "requires_decomposition": True,
                "sub_queries": ["Spring 事务传播代理边界", "Spring 事务自调用限制"],
                "retry_query": "Spring 事务传播 代理边界 自调用 限制",
                "missing_information": [],
                "clarification_question": None,
                "confidence": 0.92,
            }
        )
        agent = module.KnowledgeQueryAnalysisAgent(llm=llm)

        result = await agent.analyze(
            "它有哪些限制？",
            history=(
                ConversationTurn(
                    role="user",
                    content="我们正在讨论 Spring 事务传播机制。",
                ),
            ),
            conversation_summary="用户希望了解事务传播机制。",
        )

        self.assertEqual(llm.calls, 1)
        self.assertEqual(
            result.standalone_query,
            "Spring 事务传播机制有哪些限制？",
        )
        self.assertTrue(result.uses_history)
        self.assertEqual(result.question_type, "analytical")
        self.assertEqual(result.strategy, "decomposed")
        self.assertEqual(len(result.sub_queries), 2)
        self.assertEqual(
            result.retry_query,
            "Spring 事务传播 代理边界 自调用 限制",
        )
        prompt = "\n".join(str(message.content) for message in llm.messages)
        self.assertIn("Output JSON Schema", prompt)
        self.assertIn("不得回答", prompt)
        self.assertIn("history", prompt)

    async def test_missing_information_uses_same_structured_llm_call(self) -> None:
        module = _knowledge_query_analysis_module()
        llm = _FixedLlm(
            {
                "standalone_query": "深入分析部署方案",
                "uses_history": False,
                "question_type": "analytical",
                "requires_decomposition": False,
                "sub_queries": [],
                "retry_query": None,
                "missing_information": ["目标版本"],
                "clarification_question": "请补充目标版本。",
                "confidence": 0.9,
            }
        )

        result = await module.KnowledgeQueryAnalysisAgent(llm=llm).analyze(
            "深入分析部署方案"
        )

        self.assertEqual(llm.calls, 1)
        self.assertEqual(result.missing_information, ("目标版本",))
        self.assertEqual(result.clarification_question, "请补充目标版本。")
        self.assertIsNone(result.retry_query)

    async def test_invalid_output_falls_back_to_original_direct_query(self) -> None:
        module = _knowledge_query_analysis_module()
        llm = _FixedLlm(
            {
                "standalone_query": "伪造改写",
                "uses_history": True,
                "question_type": "comparative",
                "requires_decomposition": False,
                "sub_queries": [],
                "confidence": 0.95,
            }
        )

        result = await module.KnowledgeQueryAnalysisAgent(llm=llm).analyze(
            "它有什么区别？"
        )

        self.assertEqual(result.standalone_query, "它有什么区别？")
        self.assertEqual(result.question_type, "factual")
        self.assertEqual(result.strategy, "direct")
        self.assertTrue(result.degraded)

    async def test_comparative_rule_builds_bounded_unique_queries(self) -> None:
        module = _knowledge_query_analysis_module()
        llm = _FixedLlm({})

        result = await module.KnowledgeQueryAnalysisAgent(llm=llm).analyze(
            "BM25 和向量检索有什么区别？"
        )

        self.assertEqual(llm.calls, 0)
        self.assertEqual(result.question_type, "comparative")
        self.assertEqual(result.strategy, "decomposed")
        self.assertGreaterEqual(len(result.sub_queries), 2)
        self.assertLessEqual(len(result.sub_queries), 3)
        self.assertEqual(len(result.sub_queries), len(set(result.sub_queries)))

    async def test_complex_analysis_uses_one_structured_llm_call(self) -> None:
        module = _knowledge_query_analysis_module()
        llm = _FixedLlm(
            {
                "standalone_query": "为什么虚拟线程适合高并发阻塞任务？",
                "uses_history": False,
                "question_type": "analytical",
                "requires_decomposition": True,
                "sub_queries": [
                    "虚拟线程的调度机制",
                    "高并发阻塞任务的资源开销",
                    "虚拟线程适合高并发阻塞任务的原因",
                ],
                "confidence": 0.92,
            }
        )

        result = await module.KnowledgeQueryAnalysisAgent(llm=llm).analyze(
            "为什么虚拟线程适合高并发阻塞任务？"
        )

        self.assertEqual(llm.calls, 1)
        self.assertEqual(result.question_type, "analytical")
        self.assertEqual(result.strategy, "decomposed")
        self.assertFalse(result.degraded)
        self.assertEqual(len(result.sub_queries), 3)

    async def test_discriminated_retrieval_plan_maps_to_existing_analysis(
        self,
    ) -> None:
        module = _knowledge_query_analysis_module()
        llm = _FixedLlm(
            {
                "standalone_query": "分析 RRF 的机制和影响",
                "uses_history": False,
                "question_type": "analytical",
                "retrieval": {
                    "kind": "decomposed",
                    "sub_queries": ["RRF 的机制", "RRF 的影响"],
                    "retry_query": None,
                    "clarification": None,
                },
                "confidence": 0.93,
            }
        )

        result = await module.KnowledgeQueryAnalysisAgent(llm=llm).analyze(
            "分析 RRF 的机制和影响"
        )

        self.assertFalse(result.degraded)
        self.assertEqual(result.strategy, "decomposed")
        self.assertEqual(result.sub_queries, ("RRF 的机制", "RRF 的影响"))

    async def test_low_confidence_small_model_upgrades_to_large_once(self) -> None:
        from app.infrastructure.llm.client import llm_upgrade_scope

        module = _knowledge_query_analysis_module()
        small = _FixedLlm(
            {
                "standalone_query": "为什么虚拟线程适合高并发阻塞任务？",
                "uses_history": False,
                "question_type": "analytical",
                "requires_decomposition": True,
                "sub_queries": ["虚拟线程机制", "阻塞任务开销"],
                "confidence": 0.70,
            }
        )
        large = _FixedLlm(
            {
                "standalone_query": "为什么虚拟线程适合高并发阻塞任务？",
                "uses_history": False,
                "question_type": "analytical",
                "requires_decomposition": True,
                "sub_queries": [
                    "虚拟线程的调度机制",
                    "高并发阻塞任务的资源开销",
                ],
                "confidence": 0.94,
            }
        )
        agent = module.KnowledgeQueryAnalysisAgent(
            llm=small,
            large_llm=large,
        )

        deadline = asyncio.get_running_loop().time() + 60.0
        with llm_upgrade_scope(deadline=deadline):
            result = await agent.analyze(
                "为什么虚拟线程适合高并发阻塞任务？"
            )

        self.assertEqual(small.calls, 1)
        self.assertEqual(large.calls, 1)
        self.assertEqual(result.confidence, 0.94)
        self.assertFalse(result.degraded)

    async def test_small_model_timeout_does_not_call_large_model(self) -> None:
        from app.infrastructure.llm.client import llm_upgrade_scope

        module = _knowledge_query_analysis_module()

        class _TimeoutLlm:
            calls = 0

            async def ainvoke(self, _: list[Any]) -> Any:
                self.calls += 1
                raise TimeoutError("provider timeout")

        small = _TimeoutLlm()
        large = _FixedLlm(
            {
                "standalone_query": "不应被调用",
                "uses_history": False,
                "question_type": "analytical",
                "requires_decomposition": True,
                "sub_queries": ["不应", "调用"],
                "confidence": 0.99,
            }
        )
        agent = module.KnowledgeQueryAnalysisAgent(
            llm=small,
            large_llm=large,
        )

        deadline = asyncio.get_running_loop().time() + 60.0
        with llm_upgrade_scope(deadline=deadline):
            result = await agent.analyze("深入分析这个架构的权衡")

        self.assertEqual(small.calls, 1)
        self.assertEqual(large.calls, 0)
        self.assertTrue(result.degraded)

    async def test_low_confidence_or_failed_llm_falls_back_without_guessing(
        self,
    ) -> None:
        module = _knowledge_query_analysis_module()

        class _FailingLlm:
            async def ainvoke(self, _: list[Any]) -> Any:
                raise TimeoutError("query analysis timeout")

        low_confidence = await module.KnowledgeQueryAnalysisAgent(
            llm=_FixedLlm(
                {
                    "standalone_query": "伪造改写",
                    "uses_history": False,
                    "question_type": "analytical",
                    "requires_decomposition": True,
                    "sub_queries": ["原因", "影响"],
                    "confidence": 0.4,
                }
            )
        ).analyze("为什么这个方案会失效？")
        failed = await module.KnowledgeQueryAnalysisAgent(
            llm=_FailingLlm()
        ).analyze("深入分析这个架构的权衡")
        missing = await module.KnowledgeQueryAnalysisAgent(llm=None).analyze(
            "深入分析这个架构的权衡"
        )

        for result in (low_confidence, failed, missing):
            self.assertEqual(result.question_type, "factual")
            self.assertEqual(result.strategy, "direct")
            self.assertTrue(result.degraded)

    async def test_cancellation_is_not_converted_to_fallback(self) -> None:
        module = _knowledge_query_analysis_module()

        class _CanceledLlm:
            async def ainvoke(self, _: list[Any]) -> Any:
                raise asyncio.CancelledError

        with self.assertRaises(asyncio.CancelledError):
            await module.KnowledgeQueryAnalysisAgent(
                llm=_CanceledLlm()
            ).analyze("全面分析这个架构")


class KnowledgeQueryAnalysisEvaluationTests(unittest.IsolatedAsyncioTestCase):
    """验证固定分析样本覆盖改写、七类问题、两档策略和降级。"""

    @staticmethod
    def _cases() -> list[dict[str, Any]]:
        payload = json.loads(
            _QUERY_ANALYSIS_CASES_PATH.read_text(encoding="utf-8")
        )
        if payload.get("version") != 1 or not isinstance(payload.get("cases"), list):
            raise AssertionError("知识查询分析评估数据格式无效")
        return payload["cases"]

    def test_dataset_has_bounded_complete_contract(self) -> None:
        cases = self._cases()
        ids = [case.get("id") for case in cases]

        self.assertGreaterEqual(len(cases), 18)
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(
            {case["expected_question_type"] for case in cases},
            {
                "factual",
                "comparative",
                "procedural",
                "analytical",
                "exploratory",
                "verification",
                "summarization",
            },
        )
        self.assertEqual(
            {case["expected_strategy"] for case in cases},
            {"direct", "decomposed"},
        )
        self.assertTrue(any(case["expected_degraded"] for case in cases))
        for case in cases:
            self.assertIsInstance(case["query"], str)
            self.assertTrue(case["query"].strip())
            self.assertIn(case["expected_llm_calls"], (0, 1))
            self.assertLessEqual(len(case.get("expected_sub_queries", [])), 3)

    async def test_dataset_executes_real_rules_and_fake_llm_protocol(self) -> None:
        module = _knowledge_query_analysis_module()
        for case in self._cases():
            with self.subTest(case_id=case["id"]):
                llm = (
                    _FixedLlm(case.get("llm_output", {}))
                    if case.get("llm_available", True)
                    else None
                )
                result = await module.KnowledgeQueryAnalysisAgent(llm=llm).analyze(
                    case["query"],
                    history=tuple(
                        ConversationTurn.model_validate(turn)
                        for turn in case.get("history", ())
                    ),
                    conversation_summary=case.get("conversation_summary"),
                )

                self.assertEqual(
                    result.standalone_query,
                    case.get("expected_standalone_query", case["query"]),
                )
                self.assertEqual(
                    result.uses_history,
                    case.get("expected_uses_history", False),
                )
                self.assertEqual(result.question_type, case["expected_question_type"])
                self.assertEqual(result.strategy, case["expected_strategy"])
                self.assertEqual(result.degraded, case["expected_degraded"])
                self.assertEqual(
                    tuple(result.sub_queries),
                    tuple(case.get("expected_sub_queries", ())),
                )
                self.assertEqual(
                    0 if llm is None else llm.calls,
                    case["expected_llm_calls"],
                )


class _KnowledgeAnswerLlm:
    def __init__(self, output: dict[str, Any] | Exception) -> None:
        self.output = output
        self.calls = 0

    async def ainvoke(self, messages: list[Any]) -> dict[str, Any]:
        self.calls += 1
        self.messages = list(messages)
        if isinstance(self.output, Exception):
            raise self.output
        return self.output


class KnowledgeQueryAnalysisPromptContractTests(unittest.IsolatedAsyncioTestCase):
    """验证查询分析向 JSON Mode 提供完整、受保护的输出契约。"""

    async def test_prompt_includes_json_schema_and_untrusted_input_boundary(
        self,
    ) -> None:
        llm = _FixedLlm(
            {
                "standalone_query": "Spring 事务传播机制有哪些限制？",
                "uses_history": True,
                "question_type": "analytical",
                "requires_decomposition": True,
                "sub_queries": [
                    "Spring 事务传播代理边界",
                    "Spring 事务自调用限制",
                ],
                "confidence": 0.9,
            }
        )
        module = _knowledge_query_analysis_module()
        agent = module.KnowledgeQueryAnalysisAgent(llm=llm)

        result = await agent.analyze(
            "它有哪些限制？",
            history=(
                ConversationTurn(
                    role="user",
                    content="我们正在讨论 Spring 事务传播机制。",
                ),
            ),
            conversation_summary="用户希望得到简洁的中文说明。",
        )

        self.assertFalse(result.degraded)
        system_prompt = str(llm.messages[0].content)
        self.assertIn("JSON", system_prompt)
        self.assertIn("待处理数据", system_prompt)
        self.assertIn("不得生成答案", system_prompt)
        self.assertIn("文档 ID", system_prompt)
        envelope = json.loads(str(llm.messages[1].content))
        self.assertEqual(set(envelope), {"contract", "input"})
        self.assertEqual(
            envelope["contract"]["name"],
            "knowledge_query_analysis",
        )
        self.assertEqual(envelope["contract"]["version"], 2)
        schema = envelope["contract"]["output_schema"]
        payload = envelope["input"]
        self.assertEqual(
            set(schema["properties"]),
            {
                "standalone_query",
                "uses_history",
                "question_type",
                "retrieval",
                "confidence",
            },
        )
        self.assertEqual(payload["question"], "它有哪些限制？")
        self.assertEqual(len(payload["history"]), 1)


class KnowledgeAnswerAgentTests(unittest.IsolatedAsyncioTestCase):
    """验证回答只使用提供证据，非法模型引用整批降级。"""

    def setUp(self) -> None:
        self.evidence = (
            _knowledge_record(
                chunk_id="chunk-event-loop",
                document_id="doc-python",
                title="Python 异步编程",
                content="事件循环负责调度协程。",
            ),
            _knowledge_record(
                chunk_id="chunk-task",
                document_id="doc-python",
                title="Python 异步编程",
                content="任务用于包装协程并跟踪执行状态。",
            ),
        )

    async def test_accepts_answer_with_candidate_citations_only(self) -> None:
        llm = _KnowledgeAnswerLlm(
            {
                "outcome": "answer",
                "claims": [
                    {
                        "text": "事件循环调度协程。",
                        "evidence_ids": ["chunk-event-loop"],
                    },
                    {
                        "text": "任务负责跟踪执行状态。",
                        "evidence_ids": ["chunk-task"],
                    },
                ],
            }
        )

        generated = await KnowledgeAnswerAgent(llm=llm).generate(
            question="事件循环和任务分别做什么？",
            evidence=self.evidence,
        )

        self.assertFalse(generated.degraded)
        self.assertEqual(
            generated.cited_chunk_ids,
            ("chunk-event-loop", "chunk-task"),
        )
        self.assertEqual(
            generated.answer,
            "事件循环调度协程。[1]\n任务负责跟踪执行状态。[1]",
        )
        self.assertEqual(llm.calls, 1)

    async def test_accepts_structured_abstain_without_fake_citations(self) -> None:
        llm = _KnowledgeAnswerLlm(
            {
                "outcome": "abstain",
                "reason": "insufficient_evidence",
            }
        )

        generated = await KnowledgeAnswerAgent(llm=llm).generate(
            question="事件循环是否保证任务一定成功？",
            evidence=self.evidence,
        )

        self.assertEqual(generated.outcome, "abstain")
        self.assertEqual(generated.abstain_reason, "insufficient_evidence")
        self.assertEqual(generated.cited_chunk_ids, ())
        self.assertEqual(generated.cited_image_ids, ())
        self.assertFalse(generated.degraded)

    def test_generated_answer_rejects_contradictory_outcome_payloads(self) -> None:
        invalid_payloads = (
            {
                "outcome": "answer",
                "answer": "没有引用的正式答案。",
            },
            {
                "outcome": "abstain",
                "answer": "当前证据不足。",
                "abstain_reason": "insufficient_evidence",
                "cited_chunk_ids": ("chunk-event-loop",),
            },
            {
                "outcome": "abstain",
                "answer": "当前证据不足。",
            },
            {
                "outcome": "abstain",
                "answer": "当前证据不足。",
                "abstain_reason": "insufficient_evidence",
                "degraded": True,
            },
        )

        for payload in invalid_payloads:
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                KnowledgeGeneratedAnswer.model_validate(payload)

    async def test_prompt_includes_json_contract_and_conversation_context(
        self,
    ) -> None:
        llm = _KnowledgeAnswerLlm(
            {
                "outcome": "answer",
                "claims": [
                    {
                        "text": "事件循环调度协程。",
                        "evidence_ids": ["chunk-event-loop"],
                    }
                ],
            }
        )
        history = (
            ConversationTurn(role="user", content="我们正在讨论 Python 异步。"),
            ConversationTurn(role="assistant", content="可以继续比较两个组件。"),
        )

        await KnowledgeAnswerAgent(llm=llm).generate(
            question="它们分别做什么？",
            standalone_query="事件循环和任务分别做什么？",
            history=history,
            conversation_summary="用户希望使用中文简要比较。",
            evidence=self.evidence,
        )

        self.assertEqual(llm.calls, 1)
        self.assertIn("JSON", llm.messages[0].content)
        self.assertIn("outcome=answer", llm.messages[0].content)
        self.assertIn("outcome=abstain", llm.messages[0].content)
        envelope = json.loads(llm.messages[1].content)
        self.assertEqual(set(envelope), {"contract", "input"})
        self.assertEqual(envelope["contract"]["name"], "knowledge_answer")
        self.assertEqual(envelope["contract"]["version"], 2)
        schema = envelope["contract"]["output_schema"]
        self.assertEqual(schema["discriminator"]["propertyName"], "outcome")
        payload = envelope["input"]
        self.assertEqual(payload["question"], "它们分别做什么？")
        self.assertEqual(
            payload["standalone_query"],
            "事件循环和任务分别做什么？",
        )
        self.assertEqual(
            payload["recent_history"],
            [
                {"role": "user", "content": "我们正在讨论 Python 异步。"},
                {"role": "assistant", "content": "可以继续比较两个组件。"},
            ],
        )
        self.assertEqual(
            payload["conversation_summary"],
            "用户希望使用中文简要比较。",
        )
        self.assertEqual(
            [
                chunk["chunk_id"]
                for document in payload["documents"]
                for chunk in document["chunks"]
            ],
            ["chunk-event-loop", "chunk-task"],
        )
        self.assertNotIn(
            "用户希望使用中文简要比较",
            json.dumps(payload["documents"], ensure_ascii=False),
        )

    async def test_prompt_includes_only_low_priority_interaction_preferences(
        self,
    ) -> None:
        llm = _KnowledgeAnswerLlm(
            {
                "outcome": "answer",
                "claims": [
                    {
                        "text": "系统背景说明了建设动机。",
                        "evidence_ids": ["chunk-event-loop"],
                    }
                ],
            }
        )
        projection = UserInteractionMemoryProjection(
            preferences=[
                ResponsePreferenceProjection(
                    scope="system_explanation",
                    preferred_focus=["project_background", "architecture"],
                    answer_structure="overview_first",
                    evidence_count=2,
                    confidence=0.83,
                )
            ]
        )

        await KnowledgeAnswerAgent(llm=llm).generate(
            question="给我讲一下这个系统。",
            evidence=self.evidence,
            interaction_memory=projection,
        )

        system_prompt = llm.messages[0].content
        payload = json.loads(llm.messages[1].content)["input"]
        self.assertIn("低优先级", system_prompt)
        self.assertIn("不能作为事实证据", system_prompt)
        self.assertIn("当前明确要求", system_prompt)
        self.assertEqual(
            payload["interaction_preferences"],
            projection.model_dump(mode="json")["preferences"],
        )
        self.assertNotIn("user_id", json.dumps(payload, ensure_ascii=False))

    async def test_prompt_includes_only_enumerated_runtime_skill_policy(
        self,
    ) -> None:
        llm = _KnowledgeAnswerLlm(
            {
                "outcome": "answer",
                "claims": [
                    {
                        "text": "事件循环调度协程。",
                        "evidence_ids": ["chunk-event-loop"],
                    }
                ],
            }
        )
        policy_type = importlib.import_module(
            "app.models.runtime_skill"
        ).RuntimeSkillResponsePolicy

        await KnowledgeAnswerAgent(llm=llm).generate(
            question="事件循环做什么？",
            evidence=self.evidence,
            response_policy=policy_type(
                focus=("mechanism", "constraints"),
                organization="conclusion_then_details",
            ),
        )

        system_prompt = llm.messages[0].content
        payload = json.loads(llm.messages[1].content)["input"]
        self.assertIn("runtime_skill_policy", payload)
        self.assertEqual(
            payload["runtime_skill_policy"],
            {
                "focus": ["mechanism", "constraints"],
                "organization": "conclusion_then_details",
            },
        )
        self.assertIn("低于事实、范围、证据", system_prompt)
        serialized = json.dumps(payload, ensure_ascii=False)
        self.assertNotIn("skill_id", serialized)
        self.assertNotIn("content_hash", serialized)

    async def test_out_of_candidate_citation_discards_model_answer(self) -> None:
        llm = _KnowledgeAnswerLlm(
            {
                "outcome": "answer",
                "claims": [
                    {
                        "text": "这是一个没有证据约束的回答。",
                        "evidence_ids": ["unknown-chunk"],
                    }
                ],
            }
        )

        generated = await KnowledgeAnswerAgent(llm=llm).generate(
            question="事件循环做什么？",
            evidence=self.evidence,
        )

        self.assertTrue(generated.degraded)
        self.assertEqual(generated.answer, "回答模型暂时不可用，请稍后重试。")
        self.assertNotIn("没有证据约束", generated.answer)
        self.assertNotIn("事件循环负责调度协程", generated.answer)
        self.assertEqual(
            generated.cited_chunk_ids,
            ("chunk-event-loop", "chunk-task"),
        )

    async def test_missing_llm_returns_safe_message_without_chunk_excerpt(
        self,
    ) -> None:
        generated = await KnowledgeAnswerAgent(llm=None).generate(
            question="事件循环做什么？",
            evidence=self.evidence,
        )

        self.assertTrue(generated.degraded)
        self.assertEqual(generated.answer, "回答模型暂时不可用，请稍后重试。")
        self.assertNotIn("事件循环负责调度协程", generated.answer)
        self.assertEqual(
            generated.cited_chunk_ids,
            ("chunk-event-loop", "chunk-task"),
        )

    async def test_accepts_only_linked_candidate_image_ids(self) -> None:
        image = KnowledgeImageEvidence(
            image_id="img-" + "a" * 32,
            document_id="doc-python",
            title="Python 异步编程",
            image_key="event-loop",
            heading_path=("Python", "事件循环"),
            caption="事件循环关系图",
            content_hash="b" * 64,
            linked_chunk_ids=("chunk-event-loop",),
        )
        llm = _KnowledgeAnswerLlm(
            {
                "outcome": "answer",
                "claims": [
                    {
                        "text": "事件循环关系如图所示。",
                        "evidence_ids": ["chunk-event-loop"],
                        "image_ids": [image.image_id],
                    }
                ]
            }
        )

        generated = await KnowledgeAnswerAgent(llm=llm).generate(
            question="事件循环关系是什么？",
            evidence=self.evidence,
            images=(image,),
        )

        self.assertFalse(generated.degraded)
        self.assertEqual(generated.cited_image_ids, (image.image_id,))
        self.assertEqual(generated.answer, "事件循环关系如图所示。[1][图1]")
        payload = json.loads(llm.messages[1].content)["input"]
        self.assertEqual(payload["images"][0]["image_id"], image.image_id)
        self.assertNotIn("content_hash", payload["images"][0])
        self.assertNotIn("storage_key", json.dumps(payload, ensure_ascii=False))

    async def test_prompt_groups_chunks_by_document_for_answering(self) -> None:
        second_document = _knowledge_record(
            chunk_id="chunk-java",
            document_id="doc-java",
            title="Java 并发编程",
            heading_path=("Java", "线程池"),
            content="线程池负责复用工作线程。",
        )
        llm = _KnowledgeAnswerLlm(
            {
                "outcome": "answer",
                "claims": [
                    {
                        "text": "事件循环负责调度协程。",
                        "evidence_ids": ["chunk-event-loop"],
                        "image_ids": [],
                    },
                    {
                        "text": "线程池负责复用线程。",
                        "evidence_ids": ["chunk-java"],
                        "image_ids": [],
                    },
                ]
            }
        )

        generated = await KnowledgeAnswerAgent(llm=llm).generate(
            question="分别说明两篇文档的结论",
            evidence=(*self.evidence, second_document),
        )

        payload = json.loads(llm.messages[1].content)["input"]
        self.assertNotIn("evidence", payload)
        self.assertEqual(
            [document["document_id"] for document in payload["documents"]],
            ["doc-python", "doc-java"],
        )
        self.assertEqual(
            [chunk["chunk_id"] for chunk in payload["documents"][0]["chunks"]],
            ["chunk-event-loop", "chunk-task"],
        )
        self.assertEqual(
            generated.cited_chunk_ids,
            ("chunk-event-loop", "chunk-java"),
        )

    async def test_cross_document_claim_backtracking_is_rejected(self) -> None:
        second_document = _knowledge_record(
            chunk_id="chunk-java",
            document_id="doc-java",
            title="Java 并发编程",
            content="线程池负责复用工作线程。",
        )
        llm = _KnowledgeAnswerLlm(
            {
                "outcome": "answer",
                "claims": [
                    {
                        "text": "Python 结论一。",
                        "evidence_ids": ["chunk-event-loop"],
                        "image_ids": [],
                    },
                    {
                        "text": "Java 结论。",
                        "evidence_ids": ["chunk-java"],
                        "image_ids": [],
                    },
                    {
                        "text": "Python 结论二。",
                        "evidence_ids": ["chunk-task"],
                        "image_ids": [],
                    },
                ]
            }
        )

        generated = await KnowledgeAnswerAgent(llm=llm).generate(
            question="分别说明两篇文档",
            evidence=(*self.evidence, second_document),
        )

        self.assertTrue(generated.degraded)
        self.assertNotIn("Python 结论一", generated.answer)

    async def test_unknown_image_id_discards_whole_model_answer(self) -> None:
        image = KnowledgeImageEvidence(
            image_id="img-" + "c" * 32,
            document_id="doc-python",
            title="Python 异步编程",
            image_key="event-loop",
            caption="事件循环关系图",
            content_hash="d" * 64,
            linked_chunk_ids=("chunk-event-loop",),
        )
        llm = _KnowledgeAnswerLlm(
            {
                "outcome": "answer",
                "claims": [
                    {
                        "text": "伪造图片引用。",
                        "evidence_ids": ["chunk-event-loop"],
                        "image_ids": ["img-" + "f" * 32],
                    }
                ]
            }
        )

        generated = await KnowledgeAnswerAgent(llm=llm).generate(
            question="事件循环关系是什么？",
            evidence=self.evidence,
            images=(image,),
        )

        self.assertTrue(generated.degraded)
        self.assertEqual(generated.cited_image_ids, ())
        self.assertNotIn("伪造图片引用", generated.answer)

    async def test_missing_llm_returns_at_most_three_linked_images(self) -> None:
        images = tuple(
            KnowledgeImageEvidence(
                image_id="img-" + f"{index:x}" * 32,
                document_id="doc-python",
                title="Python 异步编程",
                image_key=f"image-{index}",
                caption=f"关系图 {index}",
                content_hash=f"{index + 5:x}" * 64,
                linked_chunk_ids=("chunk-event-loop",),
            )
            for index in range(4)
        )

        generated = await KnowledgeAnswerAgent(llm=None).generate(
            question="请给出关系图",
            evidence=self.evidence,
            images=images,
        )

        self.assertTrue(generated.degraded)
        self.assertEqual(
            generated.answer,
            "回答模型暂时不可用，已返回检索到的相关图片。",
        )
        self.assertEqual(
            generated.cited_image_ids,
            tuple(image.image_id for image in images[:3]),
        )


class _StaleKnowledgeSearch:
    async def refresh(self, chunks: tuple[KnowledgeChunkRecord, ...]) -> None:
        self.chunks = chunks

    async def search(
        self,
        question: str,
        *,
        limit: int = 5,
        document_ids: tuple[str, ...] = (),
    ) -> KnowledgeSearchResult:
        del question, limit, document_ids
        chunk = self.chunks[0]
        return KnowledgeSearchResult(
            hits=(
                KnowledgeSearchHit(
                    chunk_id=chunk.chunk_id,
                    content_hash="0" * 64,
                    score=1.0,
                    bm25_rank=1,
                ),
            ),
            diagnostics=KnowledgeRetrievalDiagnostics(),
        )

    async def aclose(self) -> None:
        return None


class _RecordingKnowledgeSearch(InMemoryKnowledgeSearch):
    def __init__(self) -> None:
        super().__init__()
        self.questions: list[str] = []
        self.scopes: list[tuple[str, ...]] = []
        self.refresh_calls = 0

    async def refresh(self, chunks: tuple[KnowledgeChunkRecord, ...]) -> None:
        self.refresh_calls += 1
        await super().refresh(chunks)

    async def search(
        self,
        question: str,
        *,
        limit: int = 5,
        document_ids: tuple[str, ...] = (),
    ) -> KnowledgeSearchResult:
        self.questions.append(question)
        self.scopes.append(document_ids)
        return await super().search(
            question,
            limit=limit,
            document_ids=document_ids,
        )


class _FixedKnowledgeQueryAnalyzer:
    def __init__(self, standalone_query: str) -> None:
        self.standalone_query = standalone_query
        self.calls: list[dict[str, Any]] = []

    async def analyze(
        self,
        question: str,
        *,
        history: tuple[ConversationTurn, ...] | list[ConversationTurn],
        conversation_summary: str | None,
    ) -> Any:
        self.calls.append(
            {
                "question": question,
                "history": list(history),
                "conversation_summary": conversation_summary,
            }
        )
        return KnowledgeQueryAnalysis(
            standalone_query=self.standalone_query,
            uses_history=True,
            question_type="analytical",
            strategy="direct",
            sub_queries=(),
            confidence=0.9,
            degraded=False,
        )

    async def aclose(self) -> None:
        return None


class _FixedKnowledgeQueryAnalysisAgent:
    def __init__(self, analysis: KnowledgeQueryAnalysis) -> None:
        self.fixed_analysis = analysis
        self.calls: list[str] = []
        self.closed = False

    async def analyze(self, query: str, **_: Any) -> Any:
        self.calls.append(query)
        return self.fixed_analysis

    async def aclose(self) -> None:
        self.closed = True


class _PlannedKnowledgeSearch:
    def __init__(
        self,
        responses: dict[str, KnowledgeSearchResult | BaseException],
    ) -> None:
        self.responses = responses
        self.calls: list[tuple[str, int, tuple[str, ...]]] = []
        self.active_calls = 0
        self.max_active_calls = 0

    async def refresh(self, chunks: tuple[KnowledgeChunkRecord, ...]) -> None:
        self.chunks = chunks

    async def search(
        self,
        question: str,
        *,
        limit: int = 5,
        document_ids: tuple[str, ...] = (),
    ) -> KnowledgeSearchResult:
        self.calls.append((question, limit, document_ids))
        self.active_calls += 1
        self.max_active_calls = max(self.max_active_calls, self.active_calls)
        try:
            await asyncio.sleep(0.01)
            response = self.responses[question]
            if isinstance(response, BaseException):
                raise response
            return response
        finally:
            self.active_calls -= 1

    async def aclose(self) -> None:
        return None


class _ClarifyingKnowledgeScopeResolver:
    def resolve(self, *args: Any, **kwargs: Any) -> Any:
        _ = args, kwargs
        return SimpleNamespace(
            document_ids=(),
            needs_clarification=True,
            clarification_question="请说明要询问的知识文档标题。",
        )


class _RecordingKnowledgeDocumentChunker:
    def __init__(self) -> None:
        self.inputs: list[tuple[str, str]] = []
        self._delegate = KnowledgeDocumentChunker()
        self._pending_markdown: str | None = None

    def chunk(self, document_id: str, content_markdown: str) -> Any:
        self.inputs.append((document_id, content_markdown))
        return self._delegate.chunk(document_id, content_markdown)

    def split(self, content_markdown: str) -> Any:
        self._pending_markdown = content_markdown
        return self._delegate.split(content_markdown)

    def materialize(self, document_id: str, drafts: Any) -> Any:
        if self._pending_markdown is not None:
            self.inputs.append((document_id, self._pending_markdown))
            self._pending_markdown = None
        return self._delegate.materialize(document_id, drafts)


class _RecordingKnowledgeAnswerGenerator:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self._delegate = KnowledgeAnswerAgent(llm=None)

    async def generate(
        self,
        *,
        question: str,
        standalone_query: str | None = None,
        history: tuple[ConversationTurn, ...] = (),
        conversation_summary: str | None = None,
        evidence: tuple[KnowledgeChunkRecord, ...],
        interaction_memory: UserInteractionMemoryProjection | None = None,
        images: tuple[KnowledgeImageEvidence, ...] = (),
        response_policy: Any | None = None,
    ) -> Any:
        self.calls.append(
            {
                "question": question,
                "standalone_query": standalone_query,
                "history": tuple(history),
                "conversation_summary": conversation_summary,
                "evidence": tuple(evidence),
                "images": tuple(images),
                "interaction_memory": interaction_memory,
                "response_policy": response_policy,
            }
        )
        return await self._delegate.generate(
            question=question,
            evidence=evidence,
            images=images,
            response_policy=response_policy,
        )

    async def aclose(self) -> None:
        return None


class _FixedKnowledgeSearchResult:
    def __init__(self, result: KnowledgeSearchResult) -> None:
        self.result = result
        self.limits: list[int] = []

    async def refresh(self, chunks: tuple[KnowledgeChunkRecord, ...]) -> None:
        self.chunks = chunks

    async def search(
        self,
        question: str,
        *,
        limit: int = 5,
        document_ids: tuple[str, ...] = (),
    ) -> KnowledgeSearchResult:
        del question, document_ids
        self.limits.append(limit)
        return self.result

    async def aclose(self) -> None:
        return None


class _RecordingKnowledgeChunkReranker:
    def __init__(self) -> None:
        self.received_ids: list[str] = []
        self.received_scores: dict[str, float] = {}

    async def rerank(
        self,
        *,
        query: str,
        candidates: tuple[KnowledgeChunkRecord, ...],
        deterministic_scores: dict[str, float],
    ) -> Any:
        del query
        self.received_ids = [record.chunk_id for record in candidates]
        self.received_scores = dict(deterministic_scores)
        return SimpleNamespace(
            records=tuple(candidates),
            scores=dict(deterministic_scores),
            degraded=False,
        )

    async def aclose(self) -> None:
        return None


class _RecordingKnowledgePlanReranker:
    """记录计划批次上限并返回确定性 direct 关系。"""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def rerank_plan(
        self,
        *,
        question: str,
        steps: tuple[Any, ...],
        candidates: tuple[KnowledgeChunkRecord, ...],
        relations: tuple[Any, ...],
    ) -> Any:
        contracts = _knowledge_reasoning_contracts()
        self.calls.append(
            {
                "question": question,
                "steps": tuple(steps),
                "candidates": tuple(candidates),
                "relations": tuple(relations),
            }
        )
        return SimpleNamespace(
            relations=tuple(
                contracts.KnowledgePlanEvidenceRelation(
                    step_id=relation.step_id,
                    chunk_id=relation.chunk_id,
                    support_level="direct",
                    score=relation.deterministic_score,
                )
                for relation in relations
            ),
            degraded=False,
        )


class _FixedKnowledgeReasoningPlanner:
    """按顺序返回计划或异常，并记录首版与修订调用。"""

    def __init__(
        self,
        *,
        plans: list[Any] | None = None,
        plan_error: BaseException | None = None,
        replan_error: BaseException | None = None,
    ) -> None:
        self.plans = list(plans or [])
        self.plan_error = plan_error
        self.replan_error = replan_error
        self.plan_calls: list[dict[str, Any]] = []
        self.replan_calls: list[dict[str, Any]] = []
        self.closed = False

    async def plan(self, **kwargs: Any) -> Any:
        self.plan_calls.append(dict(kwargs))
        if self.plan_error is not None:
            raise self.plan_error
        return self.plans[0]

    async def replan(self, **kwargs: Any) -> Any:
        self.replan_calls.append(dict(kwargs))
        if self.replan_error is not None:
            raise self.replan_error
        return self.plans[1]

    async def aclose(self) -> None:
        self.closed = True


class _FixedKnowledgePlanExecutor:
    """按调用顺序返回轮次结果，并记录快照、缓存和步骤复用。"""

    def __init__(
        self,
        outcomes: list[Any] | None = None,
        *,
        error: BaseException | None = None,
    ) -> None:
        self.outcomes = list(outcomes or [])
        self.error = error
        self.calls: list[dict[str, Any]] = []

    async def execute_round(self, plan: Any, **kwargs: Any) -> Any:
        self.calls.append({"plan": plan, **kwargs})
        if self.error is not None:
            raise self.error
        return self.outcomes[len(self.calls) - 1]


class _SequenceKnowledgePlanCoverageChecker:
    """按顺序返回严格覆盖结果或抛出确定性异常。"""

    def __init__(
        self,
        coverages: list[Any] | None = None,
        *,
        error: BaseException | None = None,
    ) -> None:
        self.coverages = list(coverages or [])
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def evaluate(self, plan: Any, **kwargs: Any) -> Any:
        self.calls.append({"plan": plan, **kwargs})
        if self.error is not None:
            raise self.error
        return self.coverages[len(self.calls) - 1]


class _FixedKnowledgeAnswerGenerator:
    """返回固定可信回答，并记录 Answer Agent 的最终白名单。"""

    def __init__(self, answer: str = "受控分析结论。") -> None:
        self.answer = answer
        self.calls: list[dict[str, Any]] = []
        self.closed = False

    async def generate(self, **kwargs: Any) -> KnowledgeGeneratedAnswer:
        self.calls.append(dict(kwargs))
        evidence = tuple(kwargs["evidence"])
        return KnowledgeGeneratedAnswer(
            answer=self.answer,
            cited_chunk_ids=tuple(record.chunk_id for record in evidence),
        )

    async def aclose(self) -> None:
        self.closed = True


class _CloseCountingDependency:
    """同时提供最小协议方法并记录服务生命周期关闭次数。"""

    def __init__(self) -> None:
        self.close_calls = 0

    async def aclose(self) -> None:
        self.close_calls += 1

    async def analyze(self, *_: Any, **__: Any) -> Any:
        raise AssertionError("关闭测试不应调用查询分析")

    async def generate(self, *_: Any, **__: Any) -> Any:
        raise AssertionError("关闭测试不应调用回答")

    async def rerank(self, *_: Any, **__: Any) -> Any:
        raise AssertionError("关闭测试不应调用重排")

    async def refresh(self, *_: Any, **__: Any) -> None:
        return None

    async def search(self, *_: Any, **__: Any) -> Any:
        raise AssertionError("关闭测试不应调用检索")

    async def plan(self, *_: Any, **__: Any) -> Any:
        raise AssertionError("关闭测试不应调用 Planner")

    async def replan(self, *_: Any, **__: Any) -> Any:
        raise AssertionError("关闭测试不应调用 Replanner")


class KnowledgePlanExecutorTests(unittest.IsolatedAsyncioTestCase):
    """验证单轮计划执行共享范围、并行检索、缓存和复用边界。"""

    @staticmethod
    def _plan(
        steps: tuple[dict[str, object], ...],
        *,
        revision: int = 1,
        kept_step_ids: tuple[str, ...] = (),
        replaced_step_ids: tuple[str, ...] = (),
    ) -> Any:
        contracts = _knowledge_reasoning_contracts()
        return contracts.KnowledgeReasoningPlan(
            revision=revision,
            question_type="exploratory",
            strategy="coverage_synthesis",
            steps=steps,
            kept_step_ids=kept_step_ids,
            replaced_step_ids=replaced_step_ids,
            confidence=0.9,
        )

    @staticmethod
    def _step(
        step_id: str,
        query: str,
        *,
        required: bool = False,
    ) -> dict[str, object]:
        return {
            "step_id": step_id,
            "facet": "definition" if required else "scenario",
            "query": query,
            "target_subjects": ("混合检索",),
            "required": required,
        }

    @staticmethod
    def _search_result(
        records: tuple[KnowledgeChunkRecord, ...],
        *,
        hash_overrides: dict[str, str] | None = None,
    ) -> KnowledgeSearchResult:
        overrides = hash_overrides or {}
        return KnowledgeSearchResult(
            hits=tuple(
                KnowledgeSearchHit(
                    chunk_id=record.chunk_id,
                    content_hash=overrides.get(
                        record.chunk_id,
                        record.content_hash,
                    ),
                    score=max(0.1, 1.0 - index * 0.02),
                    bm25_rank=index + 1,
                )
                for index, record in enumerate(records)
            ),
            diagnostics=KnowledgeRetrievalDiagnostics(),
        )

    async def test_round_searches_concurrently_with_same_scope_and_caps_batch(
        self,
    ) -> None:
        module = _knowledge_plan_execution_module()
        snapshot = tuple(
            _knowledge_record(
                chunk_id=f"chunk-executor-{index}",
                document_id=f"doc-{index % 4}",
                title=f"混合检索资料 {index}",
                content=f"混合检索维度 {index} 的完整内容。",
            )
            for index in range(20)
        )
        steps = tuple(
            self._step(
                f"step-{index}",
                f"混合检索查询 {index}",
                required=index == 1,
            )
            for index in range(1, 6)
        )
        responses = {
            step["query"]: self._search_result(
                snapshot[(index * 4) :] + snapshot[: (index * 4)]
            )
            for index, step in enumerate(steps)
        }
        search = _PlannedKnowledgeSearch(responses)
        reranker = _RecordingKnowledgePlanReranker()
        executor = module.KnowledgePlanExecutor(
            search=search,
            reranker=reranker,
        )
        cache = module.KnowledgePlanRequestCache.from_snapshot(snapshot)

        outcome = await executor.execute_round(
            self._plan(steps),
            document_ids=("doc-3", "doc-1", "doc-3", "doc-2", "doc-0"),
            snapshot=snapshot,
            cache=cache,
        )

        self.assertGreater(search.max_active_calls, 1)
        self.assertEqual(
            {scope for _, _, scope in search.calls},
            {("doc-3", "doc-1", "doc-2", "doc-0")},
        )
        self.assertTrue(all(limit == 20 for _, limit, _ in search.calls))
        self.assertLessEqual(len(outcome.records), 20)
        self.assertLessEqual(len(outcome.relations), 30)
        relation_counts: dict[str, int] = {}
        for relation in outcome.relations:
            relation_counts[relation.step_id] = (
                relation_counts.get(relation.step_id, 0) + 1
            )
        self.assertTrue(all(count <= 6 for count in relation_counts.values()))
        self.assertEqual(len(reranker.calls), 1)

    async def test_same_query_searches_once_and_replacement_hits_request_cache(
        self,
    ) -> None:
        module = _knowledge_plan_execution_module()
        snapshot = (
            _knowledge_record(
                chunk_id="chunk-shared",
                document_id="doc-shared",
                title="混合检索共享证据",
                content="混合检索共享查询的可信证据。",
            ),
            _knowledge_record(
                chunk_id="chunk-second",
                document_id="doc-second",
                title="混合检索第二证据",
                content="混合检索第二查询的可信证据。",
            ),
        )
        search = _PlannedKnowledgeSearch(
            {
                "混合检索共享查询": self._search_result(snapshot[:1]),
                "混合检索第二查询": self._search_result(snapshot[1:]),
            }
        )
        reranker = _RecordingKnowledgePlanReranker()
        executor = module.KnowledgePlanExecutor(
            search=search,
            reranker=reranker,
        )
        cache = module.KnowledgePlanRequestCache.from_snapshot(snapshot)
        first_plan = self._plan(
            (
                self._step(
                    "step-1",
                    "混合检索共享查询",
                    required=True,
                ),
                self._step("step-2", "混合检索共享查询"),
                self._step("step-3", "混合检索第二查询"),
            )
        )

        first = await executor.execute_round(
            first_plan,
            document_ids=("doc-shared", "doc-second"),
            snapshot=snapshot,
            cache=cache,
        )
        second_plan = self._plan(
            (
                self._step(
                    "step-1",
                    "混合检索共享查询",
                    required=True,
                ),
                self._step("step-4", "混合检索第二查询"),
            ),
            revision=2,
            kept_step_ids=("step-1",),
            replaced_step_ids=("step-3",),
        )
        second = await executor.execute_round(
            second_plan,
            document_ids=("doc-shared", "doc-second"),
            snapshot=snapshot,
            cache=cache,
            prior_outcome=first,
            reusable_step_ids=("step-1",),
        )

        self.assertEqual(
            [query for query, _, _ in search.calls],
            ["混合检索共享查询", "混合检索第二查询"],
        )
        self.assertEqual(len(reranker.calls), 2)
        self.assertEqual(
            {relation.step_id for relation in second.relations},
            {"step-1", "step-4"},
        )

    async def test_step_failures_and_rechecks_keep_specific_reason_codes(
        self,
    ) -> None:
        module = _knowledge_plan_execution_module()
        valid = _knowledge_record(
            chunk_id="chunk-valid",
            document_id="doc-in-scope",
            title="混合检索有效证据",
            content="混合检索有效证据正文。",
        )
        stale = _knowledge_record(
            chunk_id="chunk-stale",
            document_id="doc-in-scope",
            title="混合检索过期证据",
            content="混合检索过期证据正文。",
        )
        outside = _knowledge_record(
            chunk_id="chunk-outside",
            document_id="doc-outside",
            title="混合检索范围外证据",
            content="混合检索范围外证据正文。",
        )
        snapshot = (valid, stale, outside)
        search = _PlannedKnowledgeSearch(
            {
                "混合检索有效查询": self._search_result((valid,)),
                "混合检索失败查询": TimeoutError("search timeout"),
                "混合检索过期查询": self._search_result(
                    (stale,),
                    hash_overrides={"chunk-stale": "0" * 64},
                ),
                "混合检索范围查询": self._search_result((outside,)),
            }
        )
        reranker = _RecordingKnowledgePlanReranker()
        executor = module.KnowledgePlanExecutor(
            search=search,
            reranker=reranker,
        )
        plan = self._plan(
            (
                self._step(
                    "step-1",
                    "混合检索有效查询",
                    required=True,
                ),
                self._step("step-2", "混合检索失败查询"),
                self._step("step-3", "混合检索过期查询"),
                self._step("step-4", "混合检索范围查询"),
            )
        )

        outcome = await executor.execute_round(
            plan,
            document_ids=("doc-in-scope",),
            snapshot=snapshot,
            cache=module.KnowledgePlanRequestCache.from_snapshot(snapshot),
        )

        self.assertEqual(len(search.calls), 4)
        self.assertEqual(
            outcome.empty_reason_by_step,
            {
                "step-2": "search_failed",
                "step-3": "stale_candidates",
                "step-4": "scope_filtered",
            },
        )
        self.assertEqual(len(reranker.calls), 1)
        self.assertEqual(
            {relation.step_id for relation in outcome.relations},
            {"step-1"},
        )

    async def test_cancelled_search_cancels_the_round(self) -> None:
        module = _knowledge_plan_execution_module()
        snapshot = (
            _knowledge_record(
                chunk_id="chunk-cancel",
                document_id="doc-cancel",
                title="混合检索取消证据",
                content="混合检索取消证据正文。",
            ),
        )
        search = _PlannedKnowledgeSearch(
            {
                "混合检索取消查询": asyncio.CancelledError(),
                "混合检索继续查询": self._search_result(snapshot),
            }
        )
        executor = module.KnowledgePlanExecutor(
            search=search,
            reranker=_RecordingKnowledgePlanReranker(),
        )

        with self.assertRaises(asyncio.CancelledError):
            await executor.execute_round(
                self._plan(
                    (
                        self._step(
                            "step-1",
                            "混合检索取消查询",
                            required=True,
                        ),
                        self._step("step-2", "混合检索继续查询"),
                    )
                ),
                document_ids=("doc-cancel",),
                snapshot=snapshot,
                cache=module.KnowledgePlanRequestCache.from_snapshot(snapshot),
            )

    def test_merge_evidence_prioritizes_required_and_multi_step_support(
        self,
    ) -> None:
        module = _knowledge_plan_execution_module()
        contracts = _knowledge_reasoning_contracts()
        plan = self._plan(
            (
                self._step(
                    "step-1",
                    "混合检索事实",
                    required=True,
                ),
                self._step("step-2", "混合检索场景"),
                self._step("step-3", "混合检索限制"),
            )
        )
        shared = _knowledge_record(
            chunk_id="chunk-shared-support",
            document_id="doc-shared",
            title="混合检索共享证据",
            content="混合检索同时支持事实和场景。",
            position=1,
        )
        required_only = _knowledge_record(
            chunk_id="chunk-required-only",
            document_id="doc-required",
            title="混合检索事实证据",
            content="混合检索事实基础。",
            position=0,
        )
        optional_high_score = _knowledge_record(
            chunk_id="chunk-optional-high",
            document_id="doc-optional",
            title="混合检索限制证据",
            content="混合检索限制说明。",
            position=0,
        )
        relations = (
            contracts.KnowledgePlanEvidenceRelation(
                step_id="step-1",
                chunk_id=shared.chunk_id,
                support_level="direct",
                score=0.7,
            ),
            contracts.KnowledgePlanEvidenceRelation(
                step_id="step-2",
                chunk_id=shared.chunk_id,
                support_level="direct",
                score=0.6,
            ),
            contracts.KnowledgePlanEvidenceRelation(
                step_id="step-1",
                chunk_id=required_only.chunk_id,
                support_level="direct",
                score=0.65,
            ),
            contracts.KnowledgePlanEvidenceRelation(
                step_id="step-3",
                chunk_id=optional_high_score.chunk_id,
                support_level="direct",
                score=0.99,
            ),
        )
        coverage = contracts.KnowledgePlanCoverage(
            step_results=(
                contracts.KnowledgePlanStepResult(
                    step_id="step-1",
                    status="covered",
                    search_query="混合检索事实",
                    selected_chunk_ids=(
                        shared.chunk_id,
                        required_only.chunk_id,
                    ),
                    selected_document_ids=(
                        shared.document_id,
                        required_only.document_id,
                    ),
                    reason_code="enough_evidence",
                ),
                contracts.KnowledgePlanStepResult(
                    step_id="step-2",
                    status="covered",
                    search_query="混合检索场景",
                    selected_chunk_ids=(shared.chunk_id,),
                    selected_document_ids=(shared.document_id,),
                    reason_code="enough_evidence",
                ),
                contracts.KnowledgePlanStepResult(
                    step_id="step-3",
                    status="covered",
                    search_query="混合检索限制",
                    selected_chunk_ids=(optional_high_score.chunk_id,),
                    selected_document_ids=(optional_high_score.document_id,),
                    reason_code="enough_evidence",
                ),
            ),
            required_steps=1,
            covered_required_steps=1,
            covered_steps=3,
            coverage_ratio=1.0,
            decision="answer",
        )
        outcome = module.KnowledgePlanRoundOutcome(
            plan=plan,
            records=(shared, required_only, optional_high_score),
            relations=relations,
            empty_reason_by_step={},
            search_queries=(),
            retrieval_mode="bm25",
            diagnostics=KnowledgeRetrievalDiagnostics(),
            rerank_degraded=False,
        )

        merged = module.merge_evidence(plan, coverage, (outcome,))

        self.assertEqual(merged.records[0].chunk_id, shared.chunk_id)
        self.assertEqual(
            merged.supporting_step_ids[shared.chunk_id],
            ("step-1", "step-2"),
        )
        self.assertEqual(
            len({record.chunk_id for record in merged.records}),
            len(merged.records),
        )

    def test_merge_evidence_enforces_document_and_token_budgets(self) -> None:
        module = _knowledge_plan_execution_module()
        contracts = _knowledge_reasoning_contracts()
        plan = self._plan(
            (
                self._step(
                    "step-1",
                    "混合检索事实",
                    required=True,
                ),
                self._step("step-2", "混合检索场景"),
            )
        )
        records = tuple(
            _knowledge_record(
                chunk_id=f"chunk-budget-{index}",
                document_id="doc-budget" if index < 4 else "doc-extra",
                title="混合检索预算证据",
                content=f"混合检索预算证据 {index}。",
                token_count=800 if index < 4 else 3001,
                position=index,
            )
            for index in range(5)
        )
        relations = tuple(
            contracts.KnowledgePlanEvidenceRelation(
                step_id="step-1" if index < 4 else "step-2",
                chunk_id=record.chunk_id,
                support_level="direct",
                score=1.0 - index * 0.05,
            )
            for index, record in enumerate(records)
        )
        coverage = contracts.KnowledgePlanCoverage(
            step_results=(
                contracts.KnowledgePlanStepResult(
                    step_id="step-1",
                    status="covered",
                    search_query="混合检索事实",
                    selected_chunk_ids=tuple(
                        record.chunk_id for record in records[:4]
                    ),
                    selected_document_ids=("doc-budget",),
                    reason_code="enough_evidence",
                ),
                contracts.KnowledgePlanStepResult(
                    step_id="step-2",
                    status="covered",
                    search_query="混合检索场景",
                    selected_chunk_ids=(records[4].chunk_id,),
                    selected_document_ids=("doc-extra",),
                    reason_code="enough_evidence",
                ),
            ),
            required_steps=1,
            covered_required_steps=1,
            covered_steps=2,
            coverage_ratio=1.0,
            decision="answer",
        )
        outcome = module.KnowledgePlanRoundOutcome(
            plan=plan,
            records=records,
            relations=relations,
            empty_reason_by_step={},
            search_queries=(),
            retrieval_mode="bm25",
            diagnostics=KnowledgeRetrievalDiagnostics(),
            rerank_degraded=False,
        )

        merged = module.merge_evidence(plan, coverage, (outcome,))

        self.assertEqual(len(merged.records), 3)
        self.assertEqual(
            [record.chunk_id for record in merged.records],
            [record.chunk_id for record in records[:3]],
        )
        self.assertLessEqual(
            sum(record.token_count for record in merged.records),
            3000,
        )

    def test_merge_evidence_uses_only_covered_direct_relations_and_stable_order(
        self,
    ) -> None:
        module = _knowledge_plan_execution_module()
        contracts = _knowledge_reasoning_contracts()
        plan = self._plan(
            (
                self._step(
                    "step-1",
                    "混合检索事实",
                    required=True,
                ),
                self._step("step-2", "混合检索场景"),
                self._step("step-3", "混合检索限制"),
            )
        )
        doc_b_late = _knowledge_record(
            chunk_id="chunk-doc-b-late",
            document_id="doc-b",
            title="混合检索 B",
            content="混合检索 B 后段。",
            position=2,
        )
        doc_b_early = _knowledge_record(
            chunk_id="chunk-doc-b-early",
            document_id="doc-b",
            title="混合检索 B",
            content="混合检索 B 前段。",
            position=0,
        )
        doc_a = _knowledge_record(
            chunk_id="chunk-doc-a",
            document_id="doc-a",
            title="混合检索 A",
            content="混合检索 A 内容。",
            position=1,
        )
        weak = _knowledge_record(
            chunk_id="chunk-weak",
            document_id="doc-weak",
            title="混合检索弱证据",
            content="混合检索弱证据内容。",
        )
        relations = (
            contracts.KnowledgePlanEvidenceRelation(
                step_id="step-1",
                chunk_id=doc_b_late.chunk_id,
                support_level="direct",
                score=0.9,
            ),
            contracts.KnowledgePlanEvidenceRelation(
                step_id="step-1",
                chunk_id=doc_b_early.chunk_id,
                support_level="direct",
                score=0.8,
            ),
            contracts.KnowledgePlanEvidenceRelation(
                step_id="step-2",
                chunk_id=doc_a.chunk_id,
                support_level="direct",
                score=0.7,
            ),
            contracts.KnowledgePlanEvidenceRelation(
                step_id="step-2",
                chunk_id=weak.chunk_id,
                support_level="partial",
                score=1.0,
            ),
            contracts.KnowledgePlanEvidenceRelation(
                step_id="step-3",
                chunk_id=weak.chunk_id,
                support_level="direct",
                score=1.0,
            ),
        )
        coverage = contracts.KnowledgePlanCoverage(
            step_results=(
                contracts.KnowledgePlanStepResult(
                    step_id="step-1",
                    status="covered",
                    search_query="混合检索事实",
                    selected_chunk_ids=(
                        doc_b_late.chunk_id,
                        doc_b_early.chunk_id,
                    ),
                    selected_document_ids=("doc-b",),
                    reason_code="enough_evidence",
                ),
                contracts.KnowledgePlanStepResult(
                    step_id="step-2",
                    status="weak",
                    search_query="混合检索场景",
                    selected_chunk_ids=(weak.chunk_id,),
                    selected_document_ids=("doc-weak",),
                    reason_code="insufficient_subject_coverage",
                ),
                contracts.KnowledgePlanStepResult(
                    step_id="step-3",
                    status="failed",
                    search_query="混合检索限制",
                    reason_code="search_failed",
                ),
            ),
            required_steps=1,
            covered_required_steps=1,
            covered_steps=1,
            coverage_ratio=1 / 3,
            decision="answer",
        )
        outcome = module.KnowledgePlanRoundOutcome(
            plan=plan,
            records=(doc_b_late, doc_b_early, doc_a, weak),
            relations=relations,
            empty_reason_by_step={},
            search_queries=(),
            retrieval_mode="bm25",
            diagnostics=KnowledgeRetrievalDiagnostics(),
            rerank_degraded=False,
        )

        merged = module.merge_evidence(plan, coverage, (outcome,))

        self.assertEqual(
            [record.chunk_id for record in merged.records],
            [doc_b_early.chunk_id, doc_b_late.chunk_id],
        )
        self.assertNotIn(weak.chunk_id, merged.supporting_step_ids)


class _CountingKnowledgeRepository(SQLiteKnowledgeRepository):
    def __init__(self, path: Path) -> None:
        super().__init__(path)
        self.replace_calls = 0

    def replace_document(self, document: Document, chunks: Any) -> None:
        self.replace_calls += 1
        super().replace_document(document, chunks)

    def replace_document_bundle(
        self,
        document: Document,
        chunks: Any,
        images: Any,
        links: Any,
    ) -> None:
        self.replace_calls += 1
        super().replace_document_bundle(document, chunks, images, links)


class KnowledgeQaServiceTests(unittest.IsolatedAsyncioTestCase):
    """验证独立问答用例的导入、回查、回答和证据不足结果。"""

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.repository = SQLiteKnowledgeRepository(
            Path(self.temporary_directory.name) / "knowledge.sqlite3"
        )

    def _store_planning_records(
        self,
        records: tuple[KnowledgeChunkRecord, ...],
    ) -> None:
        markdown = "\n\n".join(record.content for record in records)
        now = datetime(2026, 8, 12, tzinfo=timezone.utc)
        self.repository.replace_document(
            Document(
                document_id="doc-plan",
                title="检索规划",
                content_markdown=markdown,
                topics=["检索"],
                content_type="technical_design",
                difficulty="advanced",
                author_id="author-plan",
                content_hash=hashlib.sha256(markdown.encode("utf-8")).hexdigest(),
                created_at=now,
                updated_at=now,
            ),
            records,
        )

    @staticmethod
    def _complex_plan_fixture(
        question_type: str,
        *,
        revision: int = 1,
    ) -> Any:
        contracts = _knowledge_reasoning_contracts()
        if question_type == "comparative":
            steps = (
                {
                    "step_id": "step-1",
                    "facet": "subject",
                    "query": "BM25 检索机制",
                    "target_subjects": ("BM25",),
                    "required": True,
                },
                {
                    "step_id": "step-2",
                    "facet": "subject",
                    "query": "向量检索机制",
                    "target_subjects": ("向量检索",),
                    "required": True,
                },
                {
                    "step_id": "step-3" if revision == 1 else "step-4",
                    "facet": "comparison",
                    "query": "BM25 和向量检索差异",
                    "target_subjects": ("BM25", "向量检索"),
                    "required": True,
                },
            )
            strategy = "comparison_matrix"
        else:
            subject = "RRF" if question_type == "analytical" else "混合检索"
            steps = (
                {
                    "step_id": "step-1",
                    "facet": "subject" if question_type == "analytical" else "definition",
                    "query": f"{subject} 事实基础",
                    "target_subjects": (subject,),
                    "required": True,
                },
                {
                    "step_id": "step-2",
                    "facet": "impact" if question_type == "analytical" else "scenario",
                    "query": f"{subject} 影响" if question_type == "analytical" else f"{subject} 场景",
                    "target_subjects": (subject,),
                    "required": question_type == "analytical",
                },
                {
                    "step_id": "step-3" if revision == 1 else "step-4",
                    "facet": "constraint",
                    "query": f"{subject} 限制",
                    "target_subjects": (subject,),
                    "required": False,
                },
                {
                    "step_id": "step-5",
                    "facet": "example",
                    "query": f"{subject} 示例",
                    "target_subjects": (subject,),
                    "required": False,
                },
            )
            strategy = (
                "facet_analysis"
                if question_type == "analytical"
                else "coverage_synthesis"
            )
        return contracts.KnowledgeReasoningPlan(
            revision=revision,
            question_type=question_type,
            strategy=strategy,
            steps=steps,
            kept_step_ids=("step-1", "step-2") if revision == 2 else (),
            replaced_step_ids=("step-3",) if revision == 2 else (),
            confidence=0.9,
        )

    @staticmethod
    def _round_outcome_fixture(
        plan: Any,
        records: tuple[KnowledgeChunkRecord, ...],
        *,
        support_by_step: dict[str, tuple[str, str, float]],
    ) -> Any:
        module = _knowledge_plan_execution_module()
        contracts = _knowledge_reasoning_contracts()
        return module.KnowledgePlanRoundOutcome(
            plan=plan,
            records=records,
            relations=tuple(
                contracts.KnowledgePlanEvidenceRelation(
                    step_id=step_id,
                    chunk_id=chunk_id,
                    support_level=support_level,
                    score=score,
                )
                for step_id, (
                    chunk_id,
                    support_level,
                    score,
                ) in support_by_step.items()
            ),
            empty_reason_by_step={},
            search_queries=tuple(step.query for step in plan.steps),
            retrieval_mode="bm25",
            diagnostics=KnowledgeRetrievalDiagnostics(),
            rerank_degraded=False,
        )

    @staticmethod
    def _coverage_fixture(
        plan: Any,
        statuses: dict[str, str],
        *,
        decision: str,
        replanned: bool = False,
    ) -> Any:
        contracts = _knowledge_reasoning_contracts()
        results = tuple(
            contracts.KnowledgePlanStepResult(
                step_id=step.step_id,
                status=statuses[step.step_id],
                search_query=step.query,
                selected_chunk_ids=(f"chunk-{step.step_id}",)
                if statuses[step.step_id] in {"covered", "weak"}
                else (),
                selected_document_ids=("doc-plan",)
                if statuses[step.step_id] in {"covered", "weak"}
                else (),
                reason_code={
                    "covered": "enough_evidence",
                    "weak": "insufficient_subject_coverage",
                    "uncovered": "no_hits",
                    "failed": "search_failed",
                }[statuses[step.step_id]],
            )
            for step in plan.steps
        )
        required_ids = {step.step_id for step in plan.steps if step.required}
        covered_ids = {
            result.step_id for result in results if result.status == "covered"
        }
        return contracts.KnowledgePlanCoverage(
            step_results=results,
            required_steps=len(required_ids),
            covered_required_steps=len(required_ids & covered_ids),
            covered_steps=len(covered_ids),
            coverage_ratio=len(covered_ids) / len(results),
            replanned=replanned,
            decision=decision,
        )

    async def test_simple_question_types_never_call_reasoning_planner(self) -> None:
        record = _knowledge_record(
            chunk_id="chunk-simple",
            document_id="doc-plan",
            title="简单问题证据",
            content="简单问题使用当前低延迟路径回答。",
        )
        self._store_planning_records((record,))
        for question_type in (
            "factual",
            "verification",
            "procedural",
            "summarization",
        ):
            with self.subTest(question_type=question_type):
                query = f"简单问题 {question_type}"
                planner = _FixedKnowledgeReasoningPlanner()
                search = _PlannedKnowledgeSearch(
                    {
                        query: KnowledgeSearchResult(
                            hits=(
                                KnowledgeSearchHit(
                                    chunk_id=record.chunk_id,
                                    content_hash=record.content_hash,
                                    score=1.0,
                                    bm25_rank=1,
                                ),
                            )
                        )
                    }
                )
                service = KnowledgeQaService(
                    repository=self.repository,
                    search=search,
                    answer_agent=_RecordingKnowledgeAnswerGenerator(),
                    query_analysis_agent=_FixedKnowledgeQueryAnalysisAgent(
                        KnowledgeQueryAnalysis(
                            standalone_query=query,
                            question_type=question_type,
                            strategy="direct",
                            confidence=1.0,
                        )
                    ),
                    reasoning_planner_agent=planner,
                    plan_executor=_FixedKnowledgePlanExecutor([]),
                    plan_coverage_checker=_SequenceKnowledgePlanCoverageChecker([]),
                )

                await service.ask(query, document_ids=("doc-plan",))

                self.assertEqual(planner.plan_calls, [])
                self.assertEqual(planner.replan_calls, [])

    async def test_complex_question_types_use_exact_strategy_and_answer_once(
        self,
    ) -> None:
        cases = (
            ("comparative", "比较 BM25 和向量检索"),
            ("analytical", "分析 RRF 的影响"),
            ("exploratory", "全面了解混合检索"),
        )
        for question_type, query in cases:
            with self.subTest(question_type=question_type):
                plan = self._complex_plan_fixture(question_type)
                records = tuple(
                    _knowledge_record(
                        chunk_id=f"chunk-{step.step_id}",
                        document_id="doc-plan",
                        title="检索规划",
                        content=f"{step.query}的可信正文。",
                        position=index,
                    )
                    for index, step in enumerate(plan.steps)
                )
                self._store_planning_records(records)
                outcome = self._round_outcome_fixture(
                    plan,
                    records,
                    support_by_step={
                        step.step_id: (
                            f"chunk-{step.step_id}",
                            "direct",
                            0.9,
                        )
                        for step in plan.steps
                    },
                )
                coverage = self._coverage_fixture(
                    plan,
                    {step.step_id: "covered" for step in plan.steps},
                    decision="answer",
                )
                planner = _FixedKnowledgeReasoningPlanner(plans=[plan])
                executor = _FixedKnowledgePlanExecutor([outcome])
                checker = _SequenceKnowledgePlanCoverageChecker([coverage])
                answer_agent = _FixedKnowledgeAnswerGenerator()
                service = KnowledgeQaService(
                    repository=self.repository,
                    search=_PlannedKnowledgeSearch({}),
                    answer_agent=answer_agent,
                    query_analysis_agent=_FixedKnowledgeQueryAnalysisAgent(
                        KnowledgeQueryAnalysis(
                            standalone_query=query,
                            question_type=question_type,
                            strategy="decomposed",
                            sub_queries=(
                                tuple(step.query for step in plan.steps[:2])
                            ),
                            confidence=0.9,
                        )
                    ),
                    reasoning_planner_agent=planner,
                    plan_executor=executor,
                    plan_coverage_checker=checker,
                )

                answer = await service.ask(query, document_ids=("doc-plan",))

                self.assertEqual(planner.plan_calls[0]["question_type"], question_type)
                self.assertEqual(plan.strategy, answer.execution_trace.reasoning_strategy)
                self.assertEqual(len(answer_agent.calls), 1)
                self.assertEqual(
                    {record.chunk_id for record in answer_agent.calls[0]["evidence"]},
                    {record.chunk_id for record in records[:3]},
                )

    async def test_complex_gate_approved_subset_is_the_only_answer_evidence(
        self,
    ) -> None:
        plan = self._complex_plan_fixture("analytical")
        records = tuple(
            _knowledge_record(
                chunk_id=f"chunk-{step.step_id}",
                document_id="doc-plan",
                title="检索规划",
                content=f"{step.query}的可信正文。",
                position=index,
            )
            for index, step in enumerate(plan.steps)
        )
        self._store_planning_records(records)
        outcome = self._round_outcome_fixture(
            plan,
            records,
            support_by_step={
                step.step_id: (
                    f"chunk-{step.step_id}",
                    "direct",
                    0.9,
                )
                for step in plan.steps
            },
        )
        coverage = self._coverage_fixture(
            plan,
            {step.step_id: "covered" for step in plan.steps},
            decision="answer",
        )
        _, _, Decision, Gate = _evidence_routing_components()

        class _SubsetGate:
            def __init__(self) -> None:
                self._delegate = Gate()
                self.signals: list[Any] = []

            def precheck(self, **kwargs: Any) -> Any:
                return self._delegate.precheck(**kwargs)

            def decide_after_retrieval(
                self,
                signals: Any,
                **_: Any,
            ) -> Any:
                self.signals.append(signals)
                return Decision(
                    action="answer",
                    confidence=1.0,
                    reason_code="enough_evidence",
                    approved_evidence_ids=(records[1].chunk_id,),
                )

        gate = _SubsetGate()
        answer_agent = _FixedKnowledgeAnswerGenerator()
        service = KnowledgeQaService(
            repository=self.repository,
            search=_PlannedKnowledgeSearch({}),
            answer_agent=answer_agent,
            query_analysis_agent=_FixedKnowledgeQueryAnalysisAgent(
                KnowledgeQueryAnalysis(
                    standalone_query="分析 RRF 的影响",
                    question_type="analytical",
                    strategy="decomposed",
                    sub_queries=("RRF 事实基础", "RRF 影响"),
                    confidence=0.9,
                )
            ),
            reasoning_planner_agent=_FixedKnowledgeReasoningPlanner(
                plans=[plan]
            ),
            plan_executor=_FixedKnowledgePlanExecutor([outcome]),
            plan_coverage_checker=_SequenceKnowledgePlanCoverageChecker(
                [coverage]
            ),
            evidence_gate=gate,
        )

        answer = await service.ask(
            "分析 RRF 的影响",
            document_ids=("doc-plan",),
        )

        self.assertIn(answer.status, {"success", "degraded"})
        self.assertEqual(len(gate.signals), 1)
        self.assertEqual(len(answer_agent.calls), 1)
        self.assertEqual(
            tuple(
                record.chunk_id
                for record in answer_agent.calls[0]["evidence"]
            ),
            (records[1].chunk_id,),
        )

    async def test_most_complex_request_never_exceeds_six_llm_calls(self) -> None:
        first_plan = self._complex_plan_fixture("analytical")
        second_plan = self._complex_plan_fixture("analytical", revision=2)
        records = tuple(
            _knowledge_record(
                chunk_id=f"chunk-{step.step_id}",
                document_id="doc-plan",
                title="检索规划",
                content=f"{step.query}可信正文。",
                position=index,
            )
            for index, step in enumerate(second_plan.steps)
        )
        self._store_planning_records(records)
        first_outcome = self._round_outcome_fixture(
            first_plan,
            records,
            support_by_step={
                "step-1": ("chunk-step-1", "direct", 0.9),
                "step-2": ("chunk-step-2", "direct", 0.9),
                "step-3": ("chunk-step-4", "partial", 0.4),
                "step-5": ("chunk-step-5", "none", 0.2),
            },
        )
        second_outcome = self._round_outcome_fixture(
            second_plan,
            records,
            support_by_step={
                step.step_id: (f"chunk-{step.step_id}", "direct", 0.9)
                for step in second_plan.steps
            },
        )
        first_coverage = self._coverage_fixture(
            first_plan,
            {
                "step-1": "covered",
                "step-2": "covered",
                "step-3": "weak",
                "step-5": "uncovered",
            },
            decision="replan",
        )
        second_coverage = self._coverage_fixture(
            second_plan,
            {step.step_id: "covered" for step in second_plan.steps},
            decision="answer",
            replanned=True,
        )
        analyzer = _FixedKnowledgeQueryAnalysisAgent(
            KnowledgeQueryAnalysis(
                standalone_query="分析 RRF 的影响",
                question_type="analytical",
                strategy="decomposed",
                sub_queries=("RRF 事实基础", "RRF 影响"),
                confidence=0.9,
            )
        )
        planner = _FixedKnowledgeReasoningPlanner(
            plans=[first_plan, second_plan]
        )
        executor = _FixedKnowledgePlanExecutor(
            [first_outcome, second_outcome]
        )
        answer_agent = _FixedKnowledgeAnswerGenerator()
        service = KnowledgeQaService(
            repository=self.repository,
            search=_PlannedKnowledgeSearch({}),
            answer_agent=answer_agent,
            query_analysis_agent=analyzer,
            reasoning_planner_agent=planner,
            plan_executor=executor,
            plan_coverage_checker=_SequenceKnowledgePlanCoverageChecker(
                [first_coverage, second_coverage]
            ),
        )

        await service.ask("分析 RRF 的影响", document_ids=("doc-plan",))

        llm_equivalent_calls = (
            len(analyzer.calls)
            + len(planner.plan_calls)
            + len(executor.calls)
            + len(planner.replan_calls)
            + len(answer_agent.calls)
        )
        self.assertEqual(llm_equivalent_calls, 6)
        self.assertEqual(len(planner.replan_calls), 1)
        self.assertEqual(len(executor.calls), 2)
        self.assertEqual(len(answer_agent.calls), 1)

    async def test_replanned_trace_and_stream_events_use_safe_plan_whitelist(
        self,
    ) -> None:
        from app.infrastructure.observability.conversation_trace import (
            ConversationStreamRecorder,
            conversation_stream_context,
        )

        first_plan = self._complex_plan_fixture("analytical")
        second_plan = self._complex_plan_fixture("analytical", revision=2)
        records = tuple(
            _knowledge_record(
                chunk_id=f"chunk-{step.step_id}",
                document_id="doc-plan",
                title="检索规划",
                content=f"{step.query}可信正文。",
                position=index,
            )
            for index, step in enumerate(second_plan.steps)
        )
        self._store_planning_records(records)
        first_outcome = self._round_outcome_fixture(
            first_plan,
            records,
            support_by_step={
                "step-1": ("chunk-step-1", "direct", 0.9),
                "step-2": ("chunk-step-2", "direct", 0.9),
                "step-3": ("chunk-step-4", "partial", 0.4),
                "step-5": ("chunk-step-5", "none", 0.2),
            },
        )
        second_outcome = self._round_outcome_fixture(
            second_plan,
            records,
            support_by_step={
                step.step_id: (f"chunk-{step.step_id}", "direct", 0.9)
                for step in second_plan.steps
            },
        )
        first_coverage = self._coverage_fixture(
            first_plan,
            {
                "step-1": "covered",
                "step-2": "covered",
                "step-3": "weak",
                "step-5": "uncovered",
            },
            decision="replan",
        )
        second_coverage = self._coverage_fixture(
            second_plan,
            {step.step_id: "covered" for step in second_plan.steps},
            decision="answer",
            replanned=True,
        )
        service = KnowledgeQaService(
            repository=self.repository,
            search=_PlannedKnowledgeSearch({}),
            answer_agent=_FixedKnowledgeAnswerGenerator(),
            query_analysis_agent=_FixedKnowledgeQueryAnalysisAgent(
                KnowledgeQueryAnalysis(
                    standalone_query="分析 RRF 的影响",
                    question_type="analytical",
                    strategy="decomposed",
                    sub_queries=("RRF 事实基础", "RRF 影响"),
                    confidence=0.9,
                )
            ),
            reasoning_planner_agent=_FixedKnowledgeReasoningPlanner(
                plans=[first_plan, second_plan]
            ),
            plan_executor=_FixedKnowledgePlanExecutor(
                [first_outcome, second_outcome]
            ),
            plan_coverage_checker=_SequenceKnowledgePlanCoverageChecker(
                [first_coverage, second_coverage]
            ),
        )

        recorder = ConversationStreamRecorder()
        with conversation_stream_context(recorder):
            answer = await service.ask(
                "分析 RRF 的影响",
                document_ids=("doc-plan",),
            )
        events = recorder.snapshot()

        trace = answer.execution_trace
        assert trace is not None
        self.assertEqual(trace.reasoning_strategy, "facet_analysis")
        self.assertEqual(trace.plan_revision_count, 2)
        self.assertEqual({step.revision for step in trace.plan_steps}, {1, 2})
        self.assertTrue(trace.coverage.replanned)
        payload = json.dumps(trace.model_dump(mode="json"), ensure_ascii=False)
        for forbidden in (
            '"prompt"',
            '"raw_output"',
            '"reasoning"',
            '"answer_draft"',
            "traceback",
            "/mnt/",
            "fake-secret",
        ):
            self.assertNotIn(forbidden, payload.casefold())
        plan_stages = [
            event["stage"]
            for event in events
            if event["stage"]
            in {"制定计划", "执行步骤", "检查覆盖", "调整计划", "合并证据"}
        ]
        self.assertEqual(
            plan_stages,
            ["制定计划", "执行步骤", "检查覆盖", "调整计划", "执行步骤", "检查覆盖", "合并证据"],
        )
        self.assertEqual(
            [event["sequence"] for event in events],
            list(range(1, len(events) + 1)),
        )

    async def test_replans_once_reuses_covered_step_and_never_answers_third_round(
        self,
    ) -> None:
        first_plan = self._complex_plan_fixture("analytical")
        second_plan = self._complex_plan_fixture("analytical", revision=2)
        first_records = tuple(
            _knowledge_record(
                chunk_id=f"chunk-{step.step_id}",
                document_id="doc-plan",
                title="检索规划",
                content=f"{step.query}可信正文。",
                position=index,
            )
            for index, step in enumerate(first_plan.steps)
        )
        second_extra = _knowledge_record(
            chunk_id="chunk-step-4",
            document_id="doc-plan",
            title="检索规划",
            content="RRF 限制可信正文。",
            position=4,
        )
        all_records = (*first_records, second_extra)
        self._store_planning_records(all_records)
        first_outcome = self._round_outcome_fixture(
            first_plan,
            first_records,
            support_by_step={
                "step-1": ("chunk-step-1", "direct", 0.9),
                "step-2": ("chunk-step-2", "direct", 0.9),
                "step-3": ("chunk-step-3", "partial", 0.4),
                "step-5": ("chunk-step-5", "none", 0.2),
            },
        )
        second_outcome = self._round_outcome_fixture(
            second_plan,
            all_records,
            support_by_step={
                "step-1": ("chunk-step-1", "direct", 0.9),
                "step-2": ("chunk-step-2", "direct", 0.9),
                "step-4": ("chunk-step-4", "direct", 0.8),
                "step-5": ("chunk-step-5", "direct", 0.7),
            },
        )
        first_coverage = self._coverage_fixture(
            first_plan,
            {
                "step-1": "covered",
                "step-2": "covered",
                "step-3": "weak",
                "step-5": "uncovered",
            },
            decision="replan",
        )
        second_coverage = self._coverage_fixture(
            second_plan,
            {step.step_id: "covered" for step in second_plan.steps},
            decision="answer",
            replanned=True,
        )
        planner = _FixedKnowledgeReasoningPlanner(
            plans=[first_plan, second_plan]
        )
        executor = _FixedKnowledgePlanExecutor(
            [first_outcome, second_outcome]
        )
        checker = _SequenceKnowledgePlanCoverageChecker(
            [first_coverage, second_coverage]
        )
        answer_agent = _FixedKnowledgeAnswerGenerator()
        service = KnowledgeQaService(
            repository=self.repository,
            search=_PlannedKnowledgeSearch({}),
            answer_agent=answer_agent,
            query_analysis_agent=_FixedKnowledgeQueryAnalysisAgent(
                KnowledgeQueryAnalysis(
                    standalone_query="分析 RRF 的影响",
                    question_type="analytical",
                    strategy="decomposed",
                    sub_queries=("RRF 事实基础", "RRF 影响"),
                    confidence=0.9,
                )
            ),
            reasoning_planner_agent=planner,
            plan_executor=executor,
            plan_coverage_checker=checker,
        )

        answer = await service.ask(
            "分析 RRF 的影响",
            document_ids=("doc-plan",),
        )

        self.assertEqual(len(planner.plan_calls), 1)
        self.assertEqual(len(planner.replan_calls), 1)
        self.assertEqual(len(executor.calls), 2)
        self.assertEqual(
            executor.calls[1]["reusable_step_ids"],
            ("step-1", "step-2"),
        )
        self.assertIs(executor.calls[0]["cache"], executor.calls[1]["cache"])
        self.assertEqual(len(checker.calls), 2)
        self.assertEqual(len(answer_agent.calls), 1)
        self.assertEqual(answer.execution_trace.plan_revision_count, 2)

    async def test_final_insufficient_and_component_failures_skip_answer(self) -> None:
        plan = self._complex_plan_fixture("exploratory")
        records = (
            _knowledge_record(
                chunk_id="chunk-step-1",
                document_id="doc-plan",
                title="混合检索事实基础",
                content="混合检索事实基础可信正文。",
            ),
        )
        self._store_planning_records(records)
        outcome = self._round_outcome_fixture(
            plan,
            records,
            support_by_step={
                "step-1": ("chunk-step-1", "direct", 0.9)
            },
        )
        insufficient = self._coverage_fixture(
            plan,
            {
                "step-1": "covered",
                "step-2": "uncovered",
                "step-3": "uncovered",
                "step-5": "uncovered",
            },
            decision="insufficient_evidence",
        )
        cases = (
            (
                _FixedKnowledgePlanExecutor([outcome]),
                _SequenceKnowledgePlanCoverageChecker([insufficient]),
                (),
            ),
            (
                _FixedKnowledgePlanExecutor(error=RuntimeError("executor")),
                _SequenceKnowledgePlanCoverageChecker([]),
                ("plan_execution",),
            ),
            (
                _FixedKnowledgePlanExecutor([outcome]),
                _SequenceKnowledgePlanCoverageChecker(error=RuntimeError("coverage")),
                ("coverage",),
            ),
        )
        for executor, checker, degraded_components in cases:
            with self.subTest(degraded_components=degraded_components):
                answer_agent = _FixedKnowledgeAnswerGenerator()
                service = KnowledgeQaService(
                    repository=self.repository,
                    search=_PlannedKnowledgeSearch({}),
                    answer_agent=answer_agent,
                    query_analysis_agent=_FixedKnowledgeQueryAnalysisAgent(
                        KnowledgeQueryAnalysis(
                            standalone_query="全面了解混合检索",
                            question_type="exploratory",
                            strategy="decomposed",
                            sub_queries=("混合检索事实", "混合检索场景"),
                            confidence=0.9,
                        )
                    ),
                    reasoning_planner_agent=_FixedKnowledgeReasoningPlanner(
                        plans=[plan]
                    ),
                    plan_executor=executor,
                    plan_coverage_checker=checker,
                )

                answer = await service.ask(
                    "全面了解混合检索",
                    document_ids=("doc-plan",),
                )

                self.assertEqual(answer.status, "insufficient_evidence")
                self.assertEqual(answer_agent.calls, [])
                for component in degraded_components:
                    self.assertIn(component, answer.degraded_components)

    async def test_planner_failure_uses_existing_decomposed_fallback(self) -> None:
        record = _knowledge_record(
            chunk_id="chunk-planner-fallback",
            document_id="doc-plan",
            title="混合检索降级证据",
            content="混合检索收益和成本的可信证据。",
        )
        self._store_planning_records((record,))
        hit = KnowledgeSearchHit(
            chunk_id=record.chunk_id,
            content_hash=record.content_hash,
            score=1.0,
            bm25_rank=1,
        )
        search = _PlannedKnowledgeSearch(
            {
                "混合检索收益": KnowledgeSearchResult(hits=(hit,)),
                "混合检索成本": KnowledgeSearchResult(hits=(hit,)),
            }
        )
        answer_agent = _RecordingKnowledgeAnswerGenerator()
        service = KnowledgeQaService(
            repository=self.repository,
            search=search,
            answer_agent=answer_agent,
            query_analysis_agent=_FixedKnowledgeQueryAnalysisAgent(
                KnowledgeQueryAnalysis(
                    standalone_query="分析混合检索的收益和成本",
                    question_type="analytical",
                    strategy="decomposed",
                    sub_queries=("混合检索收益", "混合检索成本"),
                    confidence=0.9,
                )
            ),
            reasoning_planner_agent=_FixedKnowledgeReasoningPlanner(
                plan_error=TimeoutError("planner timeout")
            ),
            plan_executor=_FixedKnowledgePlanExecutor([]),
            plan_coverage_checker=_SequenceKnowledgePlanCoverageChecker([]),
        )

        answer = await service.ask(
            "分析混合检索的收益和成本",
            document_ids=("doc-plan",),
        )

        self.assertEqual(search.max_active_calls, 2)
        self.assertIn("planner", answer.degraded_components)
        self.assertEqual(len(answer_agent.calls), 1)

    async def test_aclose_closes_each_owned_resource_once(self) -> None:
        search = _CloseCountingDependency()
        analyzer = _CloseCountingDependency()
        planner = _CloseCountingDependency()
        reranker = _CloseCountingDependency()
        answer_agent = _CloseCountingDependency()
        service = KnowledgeQaService(
            repository=self.repository,
            search=search,
            query_analysis_agent=analyzer,
            reasoning_planner_agent=planner,
            chunk_rerank_agent=reranker,
            answer_agent=answer_agent,
        )

        await service.aclose()
        await service.aclose()

        self.assertEqual(search.close_calls, 1)
        self.assertEqual(analyzer.close_calls, 1)
        self.assertEqual(planner.close_calls, 1)
        self.assertEqual(reranker.close_calls, 1)
        self.assertEqual(answer_agent.close_calls, 1)

    async def test_request_timeout_cancels_running_plan_step(self) -> None:
        plan = self._complex_plan_fixture("exploratory")
        record = _knowledge_record(
            chunk_id="chunk-timeout",
            document_id="doc-plan",
            title="检索规划",
            content="混合检索超时证据。",
        )
        self._store_planning_records((record,))
        cancelled = asyncio.Event()

        class _BlockingExecutor:
            async def execute_round(self, *_: Any, **__: Any) -> Any:
                try:
                    await asyncio.Future()
                except asyncio.CancelledError:
                    cancelled.set()
                    raise

        service = KnowledgeQaService(
            repository=self.repository,
            search=_PlannedKnowledgeSearch({}),
            answer_agent=_FixedKnowledgeAnswerGenerator(),
            query_analysis_agent=_FixedKnowledgeQueryAnalysisAgent(
                KnowledgeQueryAnalysis(
                    standalone_query="全面了解混合检索",
                    question_type="exploratory",
                    strategy="decomposed",
                    sub_queries=("混合检索事实", "混合检索场景"),
                    confidence=0.9,
                )
            ),
            reasoning_planner_agent=_FixedKnowledgeReasoningPlanner(plans=[plan]),
            plan_executor=_BlockingExecutor(),
            plan_coverage_checker=_SequenceKnowledgePlanCoverageChecker([]),
            request_timeout_seconds=0.02,
        )

        with self.assertRaises(asyncio.TimeoutError):
            await service.ask(
                "全面了解混合检索",
                document_ids=("doc-plan",),
            )

        self.assertTrue(cancelled.is_set())

    async def test_low_remaining_budget_skips_replan_but_runs_final_coverage(
        self,
    ) -> None:
        plan = self._complex_plan_fixture("analytical")
        record = _knowledge_record(
            chunk_id="chunk-step-1",
            document_id="doc-plan",
            title="检索规划",
            content="RRF 事实基础可信正文。",
        )
        self._store_planning_records((record,))
        outcome = self._round_outcome_fixture(
            plan,
            (record,),
            support_by_step={
                "step-1": ("chunk-step-1", "direct", 0.9)
            },
        )
        first_coverage = self._coverage_fixture(
            plan,
            {
                "step-1": "covered",
                "step-2": "uncovered",
                "step-3": "uncovered",
                "step-5": "uncovered",
            },
            decision="replan",
        )
        final_coverage = self._coverage_fixture(
            plan,
            {
                "step-1": "covered",
                "step-2": "uncovered",
                "step-3": "uncovered",
                "step-5": "uncovered",
            },
            decision="insufficient_evidence",
        )
        planner = _FixedKnowledgeReasoningPlanner(plans=[plan])
        checker = _SequenceKnowledgePlanCoverageChecker(
            [first_coverage, final_coverage]
        )
        service = KnowledgeQaService(
            repository=self.repository,
            search=_PlannedKnowledgeSearch({}),
            answer_agent=_FixedKnowledgeAnswerGenerator(),
            query_analysis_agent=_FixedKnowledgeQueryAnalysisAgent(
                KnowledgeQueryAnalysis(
                    standalone_query="分析 RRF 的影响",
                    question_type="analytical",
                    strategy="decomposed",
                    sub_queries=("RRF 事实基础", "RRF 影响"),
                    confidence=0.9,
                )
            ),
            reasoning_planner_agent=planner,
            plan_executor=_FixedKnowledgePlanExecutor([outcome]),
            plan_coverage_checker=checker,
            request_timeout_seconds=10.0,
        )

        answer = await service.ask(
            "分析 RRF 的影响",
            document_ids=("doc-plan",),
        )

        self.assertEqual(planner.replan_calls, [])
        self.assertEqual(len(checker.calls), 2)
        self.assertFalse(checker.calls[1]["allow_replan"])
        self.assertEqual(answer.status, "insufficient_evidence")

    async def test_direct_plan_searches_once_with_current_scope(self) -> None:
        record = _knowledge_record(
            chunk_id="chunk-direct",
            document_id="doc-plan",
            title="检索规划",
            heading_path=("检索规划", "RRF"),
            content="RRF 使用倒数排名融合多路召回结果。",
        )
        self._store_planning_records((record,))
        query = "RRF 是什么？"
        analyzer = _FixedKnowledgeQueryAnalysisAgent(
            KnowledgeQueryAnalysis(
                standalone_query=query,
                question_type="factual",
                strategy="direct",
                confidence=1.0,
            )
        )
        search = _PlannedKnowledgeSearch(
            {
                query: KnowledgeSearchResult(
                    hits=(
                        KnowledgeSearchHit(
                            chunk_id=record.chunk_id,
                            content_hash=record.content_hash,
                            score=1.0,
                            bm25_rank=1,
                        ),
                    )
                )
            }
        )
        service = KnowledgeQaService(
            repository=self.repository,
            search=search,
            answer_agent=_RecordingKnowledgeAnswerGenerator(),
            query_analysis_agent=analyzer,
        )

        await service.ask(query, document_ids=("doc-plan",))

        self.assertEqual(analyzer.calls, [query])
        self.assertEqual(search.calls, [(query, 20, ("doc-plan",))])

    async def test_decomposed_plan_searches_in_parallel_and_fuses_rrf(self) -> None:
        records = tuple(
            _knowledge_record(
                chunk_id=f"chunk-{name}",
                document_id="doc-plan",
                title="检索规划",
                heading_path=("检索规划", name.upper()),
                content=f"{name.upper()} 的检索证据。",
                position=index,
            )
            for index, name in enumerate(("a", "b", "c"))
        )
        self._store_planning_records(records)
        query = "BM25 和向量检索有什么区别？"
        sub_queries = ("BM25", "向量检索")
        analyzer = _FixedKnowledgeQueryAnalysisAgent(
            KnowledgeQueryAnalysis(
                standalone_query=query,
                question_type="comparative",
                strategy="decomposed",
                sub_queries=sub_queries,
                confidence=1.0,
            )
        )
        first, second, third = records
        search = _PlannedKnowledgeSearch(
            {
                "BM25": KnowledgeSearchResult(
                    hits=(
                        KnowledgeSearchHit(
                            chunk_id=first.chunk_id,
                            content_hash=first.content_hash,
                            score=0.9,
                            bm25_rank=1,
                        ),
                        KnowledgeSearchHit(
                            chunk_id=second.chunk_id,
                            content_hash=second.content_hash,
                            score=0.8,
                            bm25_rank=2,
                        ),
                    ),
                    mode="hybrid",
                    diagnostics=KnowledgeRetrievalDiagnostics(
                        vector_status="executed"
                    ),
                ),
                "向量检索": KnowledgeSearchResult(
                    hits=(
                        KnowledgeSearchHit(
                            chunk_id=second.chunk_id,
                            content_hash=second.content_hash,
                            score=0.95,
                            vector_rank=1,
                        ),
                        KnowledgeSearchHit(
                            chunk_id=third.chunk_id,
                            content_hash=third.content_hash,
                            score=0.85,
                            vector_rank=2,
                        ),
                    ),
                    mode="hybrid",
                    diagnostics=KnowledgeRetrievalDiagnostics(
                        vector_status="degraded"
                    ),
                ),
            }
        )
        reranker = _RecordingKnowledgeChunkReranker()
        service = KnowledgeQaService(
            repository=self.repository,
            search=search,
            answer_agent=_RecordingKnowledgeAnswerGenerator(),
            query_analysis_agent=analyzer,
            chunk_rerank_agent=reranker,
        )

        answer = await service.ask(query, document_ids=("doc-plan",))

        self.assertEqual(search.max_active_calls, 2)
        self.assertEqual(
            {call[0] for call in search.calls},
            set(sub_queries),
        )
        self.assertTrue(
            all(call[1:] == (20, ("doc-plan",)) for call in search.calls)
        )
        self.assertEqual(
            reranker.received_ids,
            [second.chunk_id, first.chunk_id, third.chunk_id],
        )
        self.assertGreater(
            reranker.received_scores[second.chunk_id],
            reranker.received_scores[first.chunk_id],
        )
        self.assertGreater(
            reranker.received_scores[first.chunk_id],
            reranker.received_scores[third.chunk_id],
        )
        self.assertLessEqual(max(reranker.received_scores.values()), 1.0)
        self.assertEqual(answer.diagnostics.vector_status, "degraded")

    async def test_failed_sub_search_falls_back_to_original_query(self) -> None:
        record = _knowledge_record(
            chunk_id="chunk-fallback",
            document_id="doc-plan",
            title="检索规划",
            heading_path=("检索规划", "降级"),
            content="原问题直接检索仍能返回受保护证据。",
        )
        self._store_planning_records((record,))
        query = "深入分析混合检索的权衡"
        analyzer = _FixedKnowledgeQueryAnalysisAgent(
            KnowledgeQueryAnalysis(
                standalone_query=query,
                question_type="analytical",
                strategy="decomposed",
                sub_queries=("混合检索收益", "混合检索成本"),
                confidence=0.9,
            )
        )
        direct_result = KnowledgeSearchResult(
            hits=(
                KnowledgeSearchHit(
                    chunk_id=record.chunk_id,
                    content_hash=record.content_hash,
                    score=1.0,
                    bm25_rank=1,
                ),
            )
        )
        search = _PlannedKnowledgeSearch(
            {
                "混合检索收益": direct_result,
                "混合检索成本": TimeoutError("sub query timeout"),
                query: direct_result,
            }
        )
        reranker = _RecordingKnowledgeChunkReranker()
        service = KnowledgeQaService(
            repository=self.repository,
            search=search,
            answer_agent=_RecordingKnowledgeAnswerGenerator(),
            query_analysis_agent=analyzer,
            chunk_rerank_agent=reranker,
        )

        answer = await service.ask(query, document_ids=("doc-plan",))

        self.assertEqual(search.calls[-1], (query, 20, ("doc-plan",)))
        self.assertEqual(
            {call[0] for call in search.calls[:-1]},
            {"混合检索收益", "混合检索成本"},
        )
        self.assertEqual(reranker.received_ids, [record.chunk_id])
        self.assertIn("query_analysis", answer.degraded_components)

    async def test_conflicting_hashes_in_sub_results_trigger_direct_fallback(
        self,
    ) -> None:
        record = _knowledge_record(
            chunk_id="chunk-hash",
            document_id="doc-plan",
            title="检索规划",
            heading_path=("检索规划", "Hash"),
            content="SQLite Hash 回查保护真实证据。",
        )
        self._store_planning_records((record,))
        query = "比较两种 Hash 校验方案"
        analyzer = _FixedKnowledgeQueryAnalysisAgent(
            KnowledgeQueryAnalysis(
                standalone_query=query,
                question_type="comparative",
                strategy="decomposed",
                sub_queries=("方案 A", "方案 B"),
                confidence=0.9,
            )
        )
        valid_hit = KnowledgeSearchHit(
            chunk_id=record.chunk_id,
            content_hash=record.content_hash,
            score=1.0,
            bm25_rank=1,
        )
        search = _PlannedKnowledgeSearch(
            {
                "方案 A": KnowledgeSearchResult(hits=(valid_hit,)),
                "方案 B": KnowledgeSearchResult(
                    hits=(
                        KnowledgeSearchHit(
                            chunk_id=record.chunk_id,
                            content_hash="0" * 64,
                            score=1.0,
                            bm25_rank=1,
                        ),
                    )
                ),
                query: KnowledgeSearchResult(hits=(valid_hit,)),
            }
        )
        service = KnowledgeQaService(
            repository=self.repository,
            search=search,
            answer_agent=_RecordingKnowledgeAnswerGenerator(),
            query_analysis_agent=analyzer,
        )

        answer = await service.ask(query, document_ids=("doc-plan",))

        self.assertEqual(search.calls[-1][0], query)
        self.assertIn("query_analysis", answer.degraded_components)

    async def test_comparative_plan_requires_evidence_for_both_sides(self) -> None:
        record = _knowledge_record(
            chunk_id="chunk-bm25-only",
            document_id="doc-plan",
            title="BM25 检索",
            heading_path=("检索规划", "BM25"),
            content="BM25 根据词频和文档长度计算相关性。",
        )
        self._store_planning_records((record,))
        query = "BM25 和向量检索有什么区别？"
        analyzer = _FixedKnowledgeQueryAnalysisAgent(
            KnowledgeQueryAnalysis(
                standalone_query=query,
                question_type="comparative",
                strategy="decomposed",
                sub_queries=("BM25", "向量检索"),
                confidence=1.0,
            )
        )
        hit = KnowledgeSearchHit(
            chunk_id=record.chunk_id,
            content_hash=record.content_hash,
            score=1.0,
            bm25_rank=1,
        )
        search = _PlannedKnowledgeSearch(
            {
                "BM25": KnowledgeSearchResult(hits=(hit,)),
                "向量检索": KnowledgeSearchResult(hits=(hit,)),
            }
        )
        answer_generator = _RecordingKnowledgeAnswerGenerator()
        service = KnowledgeQaService(
            repository=self.repository,
            search=search,
            answer_agent=answer_generator,
            query_analysis_agent=analyzer,
        )

        answer = await service.ask(query, document_ids=("doc-plan",))

        self.assertEqual(answer.status, "insufficient_evidence")
        self.assertEqual(answer_generator.calls, [])

    async def test_single_document_summary_uses_ordered_snapshot_budget(self) -> None:
        records = tuple(
            _knowledge_record(
                chunk_id=f"chunk-summary-{position}",
                document_id="doc-plan",
                title="完整总结文档",
                heading_path=("完整总结文档", f"第 {position + 1} 节"),
                content=f"第 {position + 1} 节正文。",
                token_count=1000,
                position=position,
            )
            for position in range(3)
        )
        self._store_planning_records(records)
        query = "总结《完整总结文档》"
        analyzer = _FixedKnowledgeQueryAnalysisAgent(
            KnowledgeQueryAnalysis(
                standalone_query=query,
                question_type="summarization",
                strategy="direct",
                confidence=1.0,
            )
        )
        middle = records[1]
        search = _PlannedKnowledgeSearch(
            {
                query: KnowledgeSearchResult(
                    hits=(
                        KnowledgeSearchHit(
                            chunk_id=middle.chunk_id,
                            content_hash=middle.content_hash,
                            score=1.0,
                            bm25_rank=1,
                        ),
                    )
                )
            }
        )
        answer_generator = _RecordingKnowledgeAnswerGenerator()
        service = KnowledgeQaService(
            repository=self.repository,
            search=search,
            answer_agent=answer_generator,
            query_analysis_agent=analyzer,
        )

        await service.ask(query, document_ids=("doc-plan",))

        self.assertEqual(
            tuple(
                record.chunk_id
                for record in answer_generator.calls[0]["evidence"]
            ),
            tuple(record.chunk_id for record in records),
        )

    async def test_ingests_with_new_chunker_and_answers_with_citation(self) -> None:
        service = KnowledgeQaService(
            repository=self.repository,
            search=InMemoryKnowledgeSearch(),
            answer_agent=KnowledgeAnswerAgent(llm=None),
        )

        ingested = await service.ingest_document(
            document_id="doc-python",
            title="Python 异步编程",
            content_markdown=(
                "# Python\n\n## 事件循环\n\n事件循环负责调度协程。"
            ),
            topics=["Python", "异步编程"],
            content_type="tutorial",
            difficulty="intermediate",
            author_id="author-python",
        )
        answer = await service.ask("事件循环负责什么？")
        document = self.repository.get_document("doc-python")
        facts = self.repository.get_document_facts(("doc-python",))

        self.assertEqual(ingested.document_id, "doc-python")
        self.assertEqual(ingested.chunk_count, 1)
        self.assertIsNotNone(document)
        assert document is not None
        self.assertEqual(document.topics, ["Python", "异步编程"])
        self.assertEqual(document.content_type, "tutorial")
        self.assertEqual(document.difficulty, "intermediate")
        self.assertEqual(document.author_id, "author-python")
        self.assertEqual(facts["doc-python"].topics, document.topics)
        self.assertEqual(
            facts["doc-python"].total_token_count,
            sum(chunk.token_count for chunk in self.repository.list_chunks("doc-python")),
        )
        self.assertEqual(answer.status, "degraded")
        self.assertEqual(answer.citations[0].document_id, "doc-python")
        self.assertEqual(answer.answer, "回答模型暂时不可用，请稍后重试。")
        self.assertNotIn("事件循环负责调度协程", answer.answer)

    async def test_answer_exposes_chunk_recall_and_document_group_trace(
        self,
    ) -> None:
        records = (
            _knowledge_record(
                chunk_id="chunk-b",
                document_id="doc-b",
                title="文档 B",
                content="文档 B 的证据。",
                position=0,
            ),
            _knowledge_record(
                chunk_id="chunk-a-early",
                document_id="doc-a",
                title="文档 A",
                content="文档 A 的前文证据。",
                position=0,
            ),
            _knowledge_record(
                chunk_id="chunk-a-late",
                document_id="doc-a",
                title="文档 A",
                content="文档 A 的后文证据。",
                position=1,
            ),
        )
        now = datetime(2026, 8, 13, tzinfo=timezone.utc)
        for document_id, title, document_records in (
            ("doc-a", "文档 A", records[1:]),
            ("doc-b", "文档 B", records[:1]),
        ):
            markdown = "\n\n".join(record.content for record in document_records)
            self.repository.replace_document(
                Document(
                    document_id=document_id,
                    title=title,
                    content_markdown=markdown,
                    topics=[title],
                    content_type="tutorial",
                    difficulty="intermediate",
                    author_id="author-trace",
                    content_hash=hashlib.sha256(markdown.encode()).hexdigest(),
                    created_at=now,
                    updated_at=now,
                ),
                document_records,
            )
        retrieval = KnowledgeSearchResult(
            hits=tuple(
                KnowledgeSearchHit(
                    chunk_id=record.chunk_id,
                    content_hash=record.content_hash,
                    score=score,
                    bm25_rank=rank,
                )
                for rank, (record, score) in enumerate(
                    zip(records, (1.0, 0.9, 0.8), strict=True),
                    start=1,
                )
            )
        )
        answer_generator = _RecordingKnowledgeAnswerGenerator()
        service = KnowledgeQaService(
            repository=self.repository,
            search=_FixedKnowledgeSearchResult(retrieval),
            answer_agent=answer_generator,
        )

        answer = await service.ask("证据是什么？")

        trace = answer.execution_trace
        self.assertIsNotNone(trace)
        assert trace is not None
        self.assertEqual(trace.route, "knowledge_qa")
        self.assertEqual(trace.request_route, "/api/v1/knowledge/ask")
        self.assertEqual(trace.question_type, "factual")
        self.assertEqual(trace.search_queries, ("证据是什么？",))
        self.assertEqual(
            [item.chunk_id for item in trace.retrieved_chunks],
            ["chunk-b", "chunk-a-early", "chunk-a-late"],
        )
        self.assertEqual(
            [item.document_id for item in trace.documents],
            ["doc-a", "doc-b"],
        )
        self.assertEqual(
            [record.chunk_id for record in answer_generator.calls[0]["evidence"]],
            ["chunk-a-early", "chunk-a-late", "chunk-b"],
        )
        self.assertEqual(
            [citation.citation_id for citation in answer.citations],
            ["1", "1", "2"],
        )
        serialized = json.dumps(trace.model_dump(mode="json"), ensure_ascii=False)
        for forbidden in ("Prompt", "embedding", "traceback", "/mnt/"):
            self.assertNotIn(forbidden, serialized)


    async def test_ingest_declares_image_and_persists_proxy_chunk(self) -> None:
        raw_markdown = (
            "# 架构\n\n## 组件\n\n"
            "调用链如下。\n\n"
            "![组件关系图](knowledge-image://architecture)\n\n"
            "请求经过 API、Service 和 Search。"
        )
        service = KnowledgeQaService(
            repository=self.repository,
            search=InMemoryKnowledgeSearch(),
            answer_agent=KnowledgeAnswerAgent(llm=None),
        )

        ingested = await service.ingest_document(
            document_id="doc-image",
            title="图片架构",
            content_markdown=raw_markdown,
            topics=["架构"],
            content_type="technical_design",
            difficulty="intermediate",
            author_id="author-image",
        )
        document = self.repository.get_document("doc-image")
        chunks = self.repository.list_chunks("doc-image")

        assert document is not None
        self.assertEqual(document.content_markdown, raw_markdown)
        self.assertEqual(len(ingested.images), 1)
        self.assertEqual(ingested.images[0].image_key, "architecture")
        self.assertEqual(ingested.images[0].status, "pending")
        self.assertTrue(any("图片说明：组件关系图" in c.content for c in chunks))
        self.assertFalse(any("__KNOWLEDGE_IMAGE" in c.content for c in chunks))

    async def test_uploads_declared_image_and_returns_safe_file(self) -> None:
        image_store = _local_knowledge_image_store(
            Path(self.temporary_directory.name) / "images"
        )
        service = KnowledgeQaService(
            repository=self.repository,
            search=InMemoryKnowledgeSearch(),
            answer_agent=KnowledgeAnswerAgent(llm=None),
            image_store=image_store,
        )
        ingested = await service.ingest_document(
            document_id="doc-upload",
            title="上传图片",
            content_markdown=(
                "# 架构\n\n"
                "![组件关系图](knowledge-image://architecture)"
            ),
            topics=["架构"],
            content_type="technical_design",
            difficulty="intermediate",
            author_id="author-image",
        )
        image_id = ingested.images[0].image_id

        uploaded = await service.upload_image(
            image_id=image_id,
            content=LocalKnowledgeImageStoreTests._PNG_BYTES,
            mime_type="image/png",
        )
        image_file = service.get_image_file(image_id)

        self.assertEqual(uploaded.image_id, image_id)
        self.assertEqual(uploaded.mime_type, "image/png")
        self.assertEqual(
            uploaded.content_hash,
            hashlib.sha256(LocalKnowledgeImageStoreTests._PNG_BYTES).hexdigest(),
        )
        self.assertIsNotNone(image_file)
        assert image_file is not None
        self.assertEqual(image_file.path.read_bytes(), LocalKnowledgeImageStoreTests._PNG_BYTES)
        self.assertEqual(image_file.mime_type, "image/png")
        self.assertEqual(image_file.content_hash, uploaded.content_hash)

    async def test_upload_rejects_undeclared_image_before_writing(self) -> None:
        root = Path(self.temporary_directory.name) / "images"
        service = KnowledgeQaService(
            repository=self.repository,
            image_store=_local_knowledge_image_store(root),
        )

        with self.assertRaisesRegex(ValueError, "图片不存在"):
            await service.upload_image(
                image_id="img-" + "f" * 32,
                content=LocalKnowledgeImageStoreTests._PNG_BYTES,
                mime_type="image/png",
            )

        self.assertEqual(tuple(path for path in root.rglob("*") if path.is_file()), ())

    async def test_pending_or_missing_binary_is_not_readable(self) -> None:
        image_store = _local_knowledge_image_store(
            Path(self.temporary_directory.name) / "images"
        )
        service = KnowledgeQaService(
            repository=self.repository,
            image_store=image_store,
        )
        ingested = await service.ingest_document(
            document_id="doc-missing-image",
            title="缺失图片",
            content_markdown="![说明](knowledge-image://missing)",
            topics=["图片"],
            content_type="technical_design",
            difficulty="intermediate",
            author_id="author-image",
        )
        image_id = ingested.images[0].image_id
        self.assertIsNone(service.get_image_file(image_id))

        await service.upload_image(
            image_id=image_id,
            content=LocalKnowledgeImageStoreTests._PNG_BYTES,
            mime_type="image/png",
        )
        ready = self.repository.get_image(image_id)
        assert ready is not None and ready.storage_key is not None
        image_store.resolve(ready.storage_key).unlink()

        self.assertIsNone(service.get_image_file(image_id))

    async def test_question_returns_ready_image_linked_to_final_evidence(
        self,
    ) -> None:
        image_store = _local_knowledge_image_store(
            Path(self.temporary_directory.name) / "images"
        )
        llm = _KnowledgeAnswerLlm({"outcome": "answer", "claims": []})
        service = KnowledgeQaService(
            repository=self.repository,
            search=InMemoryKnowledgeSearch(),
            answer_agent=KnowledgeAnswerAgent(llm=llm),
            image_store=image_store,
        )
        ingested = await service.ingest_document(
            document_id="doc-answer-image",
            title="系统架构",
            content_markdown=(
                "# 系统架构\n\n"
                "调用链如下。\n\n"
                "![组件关系图](knowledge-image://architecture)\n\n"
                "请求经过 API、Service 和 Search。"
            ),
            topics=["架构"],
            content_type="technical_design",
            difficulty="intermediate",
            author_id="author-image",
        )
        image_id = ingested.images[0].image_id
        await service.upload_image(
            image_id=image_id,
            content=LocalKnowledgeImageStoreTests._PNG_BYTES,
            mime_type="image/png",
        )
        chunk = self.repository.list_chunks("doc-answer-image")[0]
        llm.output = {
            "outcome": "answer",
            "claims": [
                {
                    "text": "请求依次经过三个组件。",
                    "evidence_ids": [chunk.chunk_id],
                    "image_ids": [image_id],
                }
            ]
        }

        answer = await service.ask("请给出组件关系图")

        self.assertEqual(answer.status, "success")
        self.assertEqual(answer.answer, "请求依次经过三个组件。[1][图1]")
        self.assertEqual(len(answer.images), 1)
        self.assertEqual(answer.images[0].citation_id, "图1")
        self.assertEqual(answer.images[0].image_id, image_id)
        self.assertEqual(answer.images[0].caption, "组件关系图")
        self.assertEqual(
            answer.images[0].url,
            (
                f"/api/v1/knowledge/images/{image_id}"
                f"?v={hashlib.sha256(LocalKnowledgeImageStoreTests._PNG_BYTES).hexdigest()[:12]}"
            ),
        )

    async def test_reingest_preserves_ready_image_and_gc_removes_deleted_one(
        self,
    ) -> None:
        image_store = _local_knowledge_image_store(
            Path(self.temporary_directory.name) / "images"
        )
        service = KnowledgeQaService(
            repository=self.repository,
            search=InMemoryKnowledgeSearch(),
            image_store=image_store,
        )
        first = await service.ingest_document(
            document_id="doc-reingest-image",
            title="架构",
            content_markdown="![旧说明](knowledge-image://architecture)",
            topics=["RAG"],
            content_type="technical_design",
            difficulty="intermediate",
            author_id="author-image",
        )
        image_id = first.images[0].image_id
        uploaded = await service.upload_image(
            image_id=image_id,
            content=LocalKnowledgeImageStoreTests._PNG_BYTES,
            mime_type="image/png",
        )
        second = await service.ingest_document(
            document_id="doc-reingest-image",
            title="架构",
            content_markdown="![新说明](knowledge-image://architecture)",
            topics=["RAG"],
            content_type="technical_design",
            difficulty="intermediate",
            author_id="author-image",
        )

        self.assertEqual(second.images[0].image_id, image_id)
        self.assertEqual(second.images[0].status, "ready")
        self.assertIsNotNone(service.get_image_file(image_id))

        await service.ingest_document(
            document_id="doc-reingest-image",
            title="架构",
            content_markdown="# 架构\n\n图片已经删除。",
            topics=["RAG"],
            content_type="technical_design",
            difficulty="intermediate",
            author_id="author-image",
        )
        stored_path = image_store.resolve(
            f"{uploaded.content_hash[:2]}/{uploaded.content_hash}.png"
        )
        self.assertIsNone(self.repository.get_image(image_id))
        self.assertIsNone(service.get_image_file(image_id))
        self.assertTrue(stored_path.exists())

        await service.refresh_index()

        self.assertFalse(stored_path.exists())

    async def test_missing_ready_binary_is_excluded_from_answer_candidates(
        self,
    ) -> None:
        image_store = _local_knowledge_image_store(
            Path(self.temporary_directory.name) / "images"
        )
        answer_generator = _RecordingKnowledgeAnswerGenerator()
        service = KnowledgeQaService(
            repository=self.repository,
            search=InMemoryKnowledgeSearch(),
            answer_agent=answer_generator,
            image_store=image_store,
        )
        ingested = await service.ingest_document(
            document_id="doc-unavailable-image",
            title="不可用图片",
            content_markdown=(
                "# 架构\n\n"
                "![组件关系图](knowledge-image://unavailable)\n\n"
                "请求经过 API 和 Service。"
            ),
            topics=["架构"],
            content_type="technical_design",
            difficulty="intermediate",
            author_id="author-image",
        )
        image_id = ingested.images[0].image_id
        await service.upload_image(
            image_id=image_id,
            content=LocalKnowledgeImageStoreTests._PNG_BYTES,
            mime_type="image/png",
        )
        ready = self.repository.get_image(image_id)
        assert ready is not None and ready.storage_key is not None
        image_store.resolve(ready.storage_key).unlink()

        answer = await service.ask("组件关系图是什么？")

        self.assertEqual(answer_generator.calls[0]["images"], ())
        self.assertEqual(answer.images, ())
        self.assertEqual(answer.answer, "回答模型暂时不可用，请稍后重试。")

    async def test_concurrent_uploads_leave_database_and_file_consistent(
        self,
    ) -> None:
        image_store = _local_knowledge_image_store(
            Path(self.temporary_directory.name) / "images"
        )
        service = KnowledgeQaService(
            repository=self.repository,
            image_store=image_store,
        )
        ingested = await service.ingest_document(
            document_id="doc-concurrent-image",
            title="并发图片",
            content_markdown="![并发图](knowledge-image://concurrent)",
            topics=["图片"],
            content_type="technical_design",
            difficulty="intermediate",
            author_id="author-image",
        )
        image_id = ingested.images[0].image_id
        contents = (
            LocalKnowledgeImageStoreTests._PNG_BYTES + b"first",
            LocalKnowledgeImageStoreTests._PNG_BYTES + b"second",
        )

        await asyncio.gather(
            *(
                service.upload_image(
                    image_id=image_id,
                    content=content,
                    mime_type="image/png",
                )
                for content in contents
            )
        )
        ready = self.repository.get_image(image_id)
        image_file = service.get_image_file(image_id)

        assert ready is not None and image_file is not None
        file_content = image_file.path.read_bytes()
        self.assertIn(file_content, contents)
        self.assertEqual(
            ready.content_hash,
            hashlib.sha256(file_content).hexdigest(),
        )
        self.assertEqual(ready.content_hash, image_file.content_hash)

    async def test_database_failure_leaves_image_pending_and_propagates_cancel(
        self,
    ) -> None:
        image_store = _local_knowledge_image_store(
            Path(self.temporary_directory.name) / "images"
        )
        service = KnowledgeQaService(
            repository=self.repository,
            image_store=image_store,
        )
        ingested = await service.ingest_document(
            document_id="doc-upload-failure",
            title="失败图片",
            content_markdown="![失败图](knowledge-image://failure)",
            topics=["图片"],
            content_type="technical_design",
            difficulty="intermediate",
            author_id="author-image",
        )
        image_id = ingested.images[0].image_id

        with patch.object(
            self.repository,
            "mark_image_ready",
            side_effect=RuntimeError("database unavailable"),
        ):
            with self.assertRaisesRegex(RuntimeError, "database unavailable"):
                await service.upload_image(
                    image_id=image_id,
                    content=LocalKnowledgeImageStoreTests._PNG_BYTES,
                    mime_type="image/png",
                )

        pending = self.repository.get_image(image_id)
        assert pending is not None
        self.assertEqual(pending.status, "pending")
        self.assertIsNone(service.get_image_file(image_id))
        self.assertEqual(
            len(tuple(path for path in image_store.root.rglob("*") if path.is_file())),
            1,
        )

        with patch.object(
            self.repository,
            "mark_image_ready",
            side_effect=asyncio.CancelledError,
        ):
            with self.assertRaises(asyncio.CancelledError):
                await service.upload_image(
                    image_id=image_id,
                    content=LocalKnowledgeImageStoreTests._PNG_BYTES,
                    mime_type="image/png",
                )

    async def test_image_gc_failure_does_not_break_text_index_refresh(
        self,
    ) -> None:
        delegate = _local_knowledge_image_store(
            Path(self.temporary_directory.name) / "images"
        )

        class _FailingGcStore:
            def put(self, content: bytes, mime_type: str) -> Any:
                return delegate.put(content, mime_type)

            def resolve(self, storage_key: str) -> Path:
                return delegate.resolve(storage_key)

            def delete_unreferenced(self, referenced_keys: Any) -> int:
                del referenced_keys
                raise RuntimeError("gc unavailable")

        search = InMemoryKnowledgeSearch()
        service = KnowledgeQaService(
            repository=self.repository,
            search=search,
            image_store=_FailingGcStore(),
        )
        await service.ingest_document(
            document_id="doc-gc-failure",
            title="GC 降级",
            content_markdown="# 检索\n\n文本索引必须继续可用。",
            topics=["检索"],
            content_type="technical_design",
            difficulty="intermediate",
            author_id="author-image",
        )

        await service.refresh_index()
        result = await search.search("文本索引", limit=1)

        self.assertEqual(len(result.hits), 1)

    async def test_rechecks_then_reranks_then_expands_complete_parent(
        self,
    ) -> None:
        markdown = (
            "# 发布指南\n\n## 完整流程\n\n"
            + " ".join(f"步骤{index}" for index in range(80))
        )
        chunks = KnowledgeDocumentChunker(
            target_tokens=20,
            max_tokens=30,
            overlap_tokens=5,
        ).chunk("doc-parent", markdown)
        now = datetime(2026, 8, 11, tzinfo=timezone.utc)
        self.repository.replace_document(
            Document(
                document_id="doc-parent",
                title="发布指南",
                content_markdown=markdown,
                topics=["发布"],
                content_type="tutorial",
                difficulty="intermediate",
                author_id="author-parent",
                content_hash=hashlib.sha256(markdown.encode("utf-8")).hexdigest(),
                created_at=now,
                updated_at=now,
            ),
            chunks,
        )
        snapshot = self.repository.list_ready_chunks()
        self.assertGreater(len(snapshot), 1)
        search = _FixedKnowledgeSearchResult(
            KnowledgeSearchResult(
                hits=(
                    KnowledgeSearchHit(
                        chunk_id=snapshot[0].chunk_id,
                        content_hash=snapshot[0].content_hash,
                        score=0.9,
                        bm25_rank=1,
                    ),
                    KnowledgeSearchHit(
                        chunk_id=snapshot[1].chunk_id,
                        content_hash="0" * 64,
                        score=0.8,
                        bm25_rank=2,
                    ),
                )
            )
        )
        reranker = _RecordingKnowledgeChunkReranker()
        answer_generator = _RecordingKnowledgeAnswerGenerator()
        service = KnowledgeQaService(
            repository=self.repository,
            search=search,
            answer_agent=answer_generator,
            chunk_rerank_agent=reranker,
        )

        await service.ask("发布流程包含哪些步骤？", limit=5)

        self.assertEqual(search.limits, [20])
        self.assertEqual(reranker.received_ids, [snapshot[0].chunk_id])
        self.assertEqual(
            tuple(
                record.chunk_id
                for record in answer_generator.calls[0]["evidence"]
            ),
            tuple(record.chunk_id for record in snapshot),
        )

    async def test_refresh_index_rechunks_old_passages_once(self) -> None:
        repository = _CountingKnowledgeRepository(
            Path(self.temporary_directory.name) / "rechunk.sqlite3"
        )
        markdown = (
            "# 长文档\n\n## 主章节\n\n"
            + " ".join(f"token{index}" for index in range(1000))
        )
        old_chunks = KnowledgeDocumentChunker(
            target_tokens=500,
            max_tokens=800,
            overlap_tokens=80,
        ).chunk("doc-old-window", markdown)
        now = datetime(2026, 8, 11, tzinfo=timezone.utc)
        original_document = Document(
            document_id="doc-old-window",
            title="长文档",
            content_markdown=markdown,
            topics=["长文"],
            content_type="technical_design",
            difficulty="advanced",
            author_id="author-long",
            content_hash=hashlib.sha256(markdown.encode("utf-8")).hexdigest(),
            created_at=now,
            updated_at=now,
        )
        repository.replace_document(original_document, old_chunks)
        repository.replace_calls = 0
        service = KnowledgeQaService(
            repository=repository,
            search=InMemoryKnowledgeSearch(),
            answer_agent=KnowledgeAnswerAgent(llm=None),
        )

        await service.refresh_index()
        refreshed_chunks = repository.list_chunks("doc-old-window")
        refreshed_document = repository.get_document("doc-old-window")
        await service.refresh_index()

        self.assertEqual(repository.replace_calls, 1)
        self.assertEqual(
            tuple(chunk.chunk_id for chunk in refreshed_chunks),
            tuple(
                chunk.chunk_id
                for chunk in KnowledgeDocumentChunker().chunk(
                    "doc-old-window",
                    markdown,
                )
            ),
        )
        self.assertEqual(refreshed_document, original_document)

    async def test_ingest_persists_raw_markdown_and_chunks_processed_copy(
        self,
    ) -> None:
        raw_markdown = (
            "# 系统文档\r\n\r\n"
            "作者：林屿\r\n\r\n"
            "分类：Spring Boot、部署\r\n\r\n"
            "## 资源\r\n\r\n"
            '<readonly-block type="isv"></readonly-block>\r\n\r\n'
            "## 正文\r\n\r\n核心结论。\r\n"
        )
        chunker = _RecordingKnowledgeDocumentChunker()
        service = KnowledgeQaService(
            repository=self.repository,
            search=InMemoryKnowledgeSearch(),
            answer_agent=KnowledgeAnswerAgent(llm=None),
            chunker=chunker,
            preprocessor=_knowledge_document_preprocessor(),
        )

        ingested = await service.ingest_document(
            document_id="doc-raw",
            title="原文保留",
            content_markdown=raw_markdown,
            topics=["文档处理"],
            content_type="technical_design",
            difficulty="intermediate",
            author_id="author-raw",
        )
        document = self.repository.get_document("doc-raw")
        chunks = self.repository.list_chunks("doc-raw")

        self.assertIsNotNone(document)
        assert document is not None
        self.assertEqual(document.content_markdown, raw_markdown)
        self.assertEqual(
            document.content_hash,
            hashlib.sha256(raw_markdown.encode("utf-8")).hexdigest(),
        )
        self.assertEqual(
            chunker.inputs,
            [("doc-raw", "# 系统文档\n\n## 正文\n\n核心结论。")],
        )
        self.assertEqual(ingested.chunk_count, 1)
        self.assertEqual(chunks[0].content, "核心结论。")

    async def test_ingest_rejects_resource_only_document_before_persistence(
        self,
    ) -> None:
        service = KnowledgeQaService(
            repository=self.repository,
            search=InMemoryKnowledgeSearch(),
            answer_agent=KnowledgeAnswerAgent(llm=None),
        )

        with self.assertRaisesRegex(ValueError, "文档正文没有可切分内容"):
            await service.ingest_document(
                document_id="doc-resource-only",
                title="纯资源文档",
                content_markdown=(
                    "# 资源\n\n"
                    '<readonly-block type="isv"></readonly-block>\n'
                ),
                topics=["资源"],
                content_type="technical_design",
                difficulty="intermediate",
                author_id="author-resource",
            )

        self.assertIsNone(
            self.repository.get_document("doc-resource-only")
        )

    async def test_stale_search_hit_is_not_used_as_answer_evidence(self) -> None:
        setup_service = KnowledgeQaService(
            repository=self.repository,
            search=InMemoryKnowledgeSearch(),
            answer_agent=KnowledgeAnswerAgent(llm=None),
        )
        await setup_service.ingest_document(
            document_id="doc-python",
            title="Python 异步编程",
            content_markdown="# Python\n\n事件循环负责调度协程。",
            topics=["Python"],
            content_type="tutorial",
            difficulty="intermediate",
            author_id="author-python",
        )
        stale_service = KnowledgeQaService(
            repository=self.repository,
            search=_StaleKnowledgeSearch(),
            answer_agent=KnowledgeAnswerAgent(llm=None),
        )
        await stale_service.refresh_index()

        answer = await stale_service.ask("事件循环负责什么？")

        self.assertEqual(answer.status, "insufficient_evidence")
        self.assertEqual(answer.citations, ())
        self.assertIn("没有找到足够依据", answer.answer)

    async def test_follow_up_query_is_rewritten_before_retrieval(self) -> None:
        search = _RecordingKnowledgeSearch()
        analyzer = _FixedKnowledgeQueryAnalyzer("Spring 事务传播机制有哪些限制？")
        answer_generator = _RecordingKnowledgeAnswerGenerator()
        service = KnowledgeQaService(
            repository=self.repository,
            search=search,
            answer_agent=answer_generator,
            query_analysis_agent=analyzer,
        )
        await service.ingest_document(
            document_id="doc-spring",
            title="Spring 事务实践",
            content_markdown=(
                "# Spring 事务\n\n传播机制的限制包括代理边界和自调用失效。"
            ),
            topics=["Spring", "事务"],
            content_type="technical_design",
            difficulty="advanced",
            author_id="author-spring",
        )
        history = [
            ConversationTurn(role="user", content="什么是 Spring 事务传播机制？"),
            ConversationTurn(role="assistant", content="它描述事务方法间的边界。"),
        ]

        answer = await service.ask(
            "它有哪些限制？",
            history=history,
            conversation_summary="用户正在了解 Spring 事务。",
            interaction_memory=UserInteractionMemoryProjection(
                preferences=[
                    ResponsePreferenceProjection(
                        scope="knowledge_qa",
                        detail_level="detailed",
                        answer_structure="overview_first",
                        evidence_count=2,
                        confidence=0.82,
                    )
                ]
            ),
        )

        self.assertEqual(search.questions[-1], "Spring 事务传播机制有哪些限制？")
        self.assertEqual(analyzer.calls[0]["history"], history)
        self.assertEqual(len(answer_generator.calls), 1)
        self.assertEqual(
            answer_generator.calls[0]["standalone_query"],
            "Spring 事务传播机制有哪些限制？",
        )
        self.assertEqual(answer_generator.calls[0]["history"], tuple(history))
        self.assertEqual(
            answer_generator.calls[0]["conversation_summary"],
            "用户正在了解 Spring 事务。",
        )
        self.assertEqual(
            answer_generator.calls[0]["interaction_memory"].preferences[0].detail_level,
            "detailed",
        )
        self.assertEqual(answer.citations[0].document_id, "doc-spring")

    async def test_scope_clarification_skips_retrieval_and_answer(self) -> None:
        service = KnowledgeQaService(
            repository=self.repository,
            search=_RecordingKnowledgeSearch(),
            answer_agent=KnowledgeAnswerAgent(llm=None),
            scope_resolver=_ClarifyingKnowledgeScopeResolver(),
        )

        answer = await service.ask(
            "第二篇文章的核心观点是什么？",
            history=[
                ConversationTurn(role="assistant", content="已找到 3 篇文章。")
            ],
        )

        self.assertEqual(answer.status, "needs_clarification")
        self.assertEqual(answer.citations, ())
        self.assertIn("请说明", answer.answer)

    async def test_ask_uses_explicit_scope_without_rebuilding_index(self) -> None:
        search = _RecordingKnowledgeSearch()
        service = KnowledgeQaService(
            repository=self.repository,
            search=search,
            answer_agent=KnowledgeAnswerAgent(llm=None),
        )
        await service.ingest_document(
            document_id="doc-spring",
            title="Spring 事务实践",
            content_markdown="# Spring\n\n事务代理需要经过代理边界。",
            topics=["Spring", "事务"],
            content_type="technical_design",
            difficulty="advanced",
            author_id="author-spring",
        )
        search.refresh_calls = 0

        answer = await service.ask("《Spring 事务实践》讲了什么？")

        self.assertEqual(search.refresh_calls, 0)
        self.assertEqual(search.scopes[-1], ("doc-spring",))
        self.assertEqual(answer.citations[0].document_id, "doc-spring")

    async def test_forced_document_scope_bypasses_history_scope_resolution(
        self,
    ) -> None:
        search = _RecordingKnowledgeSearch()
        service = KnowledgeQaService(
            repository=self.repository,
            search=search,
            answer_agent=KnowledgeAnswerAgent(llm=None),
            scope_resolver=_ClarifyingKnowledgeScopeResolver(),
        )
        await service.ingest_document(
            document_id="doc-spring",
            title="Spring Boot 部署",
            content_markdown="# 部署\n\n可以使用 Jar 或容器部署。",
            topics=["Spring Boot", "部署"],
            content_type="tutorial",
            difficulty="intermediate",
            author_id="author-spring",
        )

        answer = await service.ask(
            "它支持容器部署吗？",
            history=[
                ConversationTurn(role="user", content="它如何部署？"),
                ConversationTurn(role="assistant", content="可以使用 Jar。"),
            ],
            document_ids=("doc-spring",),
        )

        self.assertEqual(search.scopes[-1], ("doc-spring",))
        self.assertEqual(answer.resolved_document_ids, ("doc-spring",))
        self.assertEqual(answer.resolved_document_titles, ("Spring Boot 部署",))
        self.assertNotIn("resolved_document_ids", answer.model_dump(mode="json"))
        self.assertNotIn("resolved_document_titles", answer.model_dump(mode="json"))

    async def test_procedural_question_without_direct_steps_is_insufficient(
        self,
    ) -> None:
        answer_generator = _RecordingKnowledgeAnswerGenerator()
        service = KnowledgeQaService(
            repository=self.repository,
            search=InMemoryKnowledgeSearch(),
            answer_agent=answer_generator,
        )
        await service.ingest_document(
            document_id="doc-auto-config",
            title="Spring Boot 自动配置：理解条件装配与用户配置回退",
            content_markdown=(
                "# Spring Boot 自动配置\n\n"
                "应用创建上下文后加载外部配置，再导入候选自动配置并逐项判断条件，"
                "随后创建 Bean、启动嵌入式服务器并发布就绪事件。"
            ),
            topics=["Spring Boot", "自动配置"],
            content_type="tutorial",
            difficulty="intermediate",
            author_id="author-spring",
        )
        await service.ingest_document(
            document_id="doc-scheduled",
            title="Spring Boot 任务调度：Scheduled 与 Quartz 的选择",
            content_markdown=(
                "# Spring Boot 任务调度\n\n"
                "多实例部署时，每个实例都可能执行同一任务，造成重复发送或重复扣减。"
                "任务本身必须幂等，并通过数据库锁或单独调度平台控制唯一执行。"
            ),
            topics=["Spring Boot", "任务调度"],
            content_type="tutorial",
            difficulty="intermediate",
            author_id="author-spring",
        )

        answer = await service.ask("springboot如何部署")

        self.assertEqual(answer.status, "insufficient_evidence")
        self.assertEqual(answer.citations, ())
        self.assertIn("没有找到足够依据", answer.answer)
        self.assertEqual(answer_generator.calls, [])

    async def test_procedural_question_with_direct_steps_reaches_answer_agent(
        self,
    ) -> None:
        answer_generator = _RecordingKnowledgeAnswerGenerator()
        service = KnowledgeQaService(
            repository=self.repository,
            search=InMemoryKnowledgeSearch(),
            answer_agent=answer_generator,
        )
        await service.ingest_document(
            document_id="doc-spring-deploy",
            title="Spring Boot 部署指南",
            content_markdown=(
                "# Spring Boot 部署指南\n\n"
                "Spring Boot 部署时，先运行 mvn package 构建可执行 Jar，"
                "再执行 java -jar app.jar 启动服务，并检查健康状态。"
            ),
            topics=["Spring Boot", "部署"],
            content_type="tutorial",
            difficulty="intermediate",
            author_id="author-spring",
        )

        answer = await service.ask("springboot如何部署")

        self.assertEqual(answer.status, "degraded")
        self.assertEqual(len(answer_generator.calls), 1)
        self.assertEqual(answer.citations[0].document_id, "doc-spring-deploy")

    async def test_missing_forced_document_scope_never_falls_back_to_full_library(
        self,
    ) -> None:
        search = _RecordingKnowledgeSearch()
        service = KnowledgeQaService(
            repository=self.repository,
            search=search,
            answer_agent=KnowledgeAnswerAgent(llm=None),
        )

        answer = await service.ask(
            "这篇文章讲了什么？",
            document_ids=("missing-document",),
        )

        self.assertEqual(answer.status, "needs_clarification")
        self.assertEqual(answer.resolved_document_ids, ())
        self.assertEqual(search.scopes, [])


class KnowledgeQaEvidenceRoutingTests(unittest.IsolatedAsyncioTestCase):
    """验证 Skill、范围和必要条件在检索前统一短路。"""

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.repository = SQLiteKnowledgeRepository(
            Path(self.temporary_directory.name) / "knowledge.sqlite3"
        )

    def _store_records(self, records: tuple[KnowledgeChunkRecord, ...]) -> None:
        if not records:
            raise AssertionError("测试记录不能为空")
        document_id = records[0].document_id
        if any(record.document_id != document_id for record in records):
            raise AssertionError("测试记录必须来自同一文档")
        markdown = "\n\n".join(record.content for record in records)
        now = datetime(2026, 8, 14, tzinfo=timezone.utc)
        self.repository.replace_document(
            Document(
                document_id=document_id,
                title=records[0].title,
                content_markdown=markdown,
                topics=list(records[0].topics),
                content_type=records[0].content_type,
                difficulty=records[0].difficulty,
                author_id=records[0].author_id,
                content_hash=hashlib.sha256(markdown.encode("utf-8")).hexdigest(),
                created_at=now,
                updated_at=now,
            ),
            records,
        )

    @staticmethod
    def _registry(manifests: tuple[dict[str, Any], ...]) -> Any:
        Registry = _runtime_skill_registry()
        skills = RuntimeSkillMatcherTests._compiled_skills(manifests)
        registry = Registry(
            catalog=_FakeRuntimeSkillCatalog(skills),
            clock=lambda: datetime(2026, 8, 14, tzinfo=timezone.utc),
        )
        result = registry.reload()
        if not result.reloaded:
            raise AssertionError("测试 Skill Registry 加载失败")
        return registry

    async def test_equal_skill_candidates_select_before_retrieval(self) -> None:
        registry = self._registry(
            (
                _runtime_skill_manifest(
                    "java-advanced",
                    keywords=("并发",),
                    topics=(),
                ),
                _runtime_skill_manifest(
                    "java-basic",
                    keywords=("并发",),
                    topics=(),
                ),
            )
        )
        search = _RecordingKnowledgeSearch()
        answer_agent = _RecordingKnowledgeAnswerGenerator()
        service = KnowledgeQaService(
            repository=self.repository,
            search=search,
            answer_agent=answer_agent,
            runtime_skill_registry=registry,
        )

        answer = await service.ask("并发如何限流？")

        self.assertEqual(answer.status, "needs_clarification")
        self.assertEqual(answer.citations, ())
        self.assertEqual(answer.images, ())
        self.assertIn("java-advanced", answer.answer)
        self.assertIn("java-basic", answer.answer)
        self.assertEqual(search.questions, [])
        self.assertEqual(answer_agent.calls, [])

    async def test_duplicate_documents_select_before_query_analysis(self) -> None:
        search = _RecordingKnowledgeSearch()
        answer_agent = _RecordingKnowledgeAnswerGenerator()
        analyzer = _FixedKnowledgeQueryAnalysisAgent(
            KnowledgeQueryAnalysis(
                standalone_query="同名文档讲了什么？",
                question_type="factual",
                strategy="direct",
                confidence=1.0,
            )
        )
        service = KnowledgeQaService(
            repository=self.repository,
            search=search,
            answer_agent=answer_agent,
            query_analysis_agent=analyzer,
        )
        for document_id in ("doc-one", "doc-two"):
            await service.ingest_document(
                document_id=document_id,
                title="同名文档",
                content_markdown=f"# 同名文档\n\n{document_id} 的正文。",
                topics=["范围"],
                content_type="technical_design",
                difficulty="intermediate",
                author_id="author-scope",
            )
        search.questions.clear()

        answer = await service.ask("《同名文档》讲了什么？")

        self.assertEqual(answer.status, "needs_clarification")
        self.assertIn("doc-one", answer.answer)
        self.assertIn("doc-two", answer.answer)
        self.assertEqual(search.questions, [])
        self.assertEqual(analyzer.calls, [])
        self.assertEqual(answer_agent.calls, [])

    async def test_missing_information_asks_before_retrieval(self) -> None:
        search = _RecordingKnowledgeSearch()
        answer_agent = _RecordingKnowledgeAnswerGenerator()
        analyzer = _FixedKnowledgeQueryAnalysisAgent(
            KnowledgeQueryAnalysis(
                standalone_query="分析部署方案",
                missing_information=("目标版本",),
                clarification_question="请补充目标版本。",
                question_type="analytical",
                strategy="direct",
                confidence=0.9,
            )
        )
        service = KnowledgeQaService(
            repository=self.repository,
            search=search,
            answer_agent=answer_agent,
            query_analysis_agent=analyzer,
        )

        answer = await service.ask("分析部署方案")

        self.assertEqual(answer.status, "needs_clarification")
        self.assertEqual(answer.answer, "请补充目标版本。")
        self.assertEqual(answer.citations, ())
        self.assertEqual(search.questions, [])
        self.assertEqual(answer_agent.calls, [])

    async def test_skill_scope_conflict_refuses_without_full_library_fallback(
        self,
    ) -> None:
        registry = self._registry(
            (
                _runtime_skill_manifest(
                    "java-concurrency",
                    document_ids=("doc-java",),
                    topics=(),
                ),
            )
        )
        search = _RecordingKnowledgeSearch()
        answer_agent = _RecordingKnowledgeAnswerGenerator()
        service = KnowledgeQaService(
            repository=self.repository,
            search=search,
            answer_agent=answer_agent,
            runtime_skill_registry=registry,
        )
        await service.ingest_document(
            document_id="doc-python",
            title="Python 并发",
            content_markdown="# Python\n\nPython 并发说明。",
            topics=["Python"],
            content_type="technical_design",
            difficulty="intermediate",
            author_id="author-python",
        )
        search.questions.clear()

        answer = await service.ask(
            "虚拟线程有哪些限制？",
            document_ids=("doc-python",),
        )

        self.assertEqual(answer.status, "insufficient_evidence")
        self.assertEqual(answer.citations, ())
        self.assertEqual(answer.images, ())
        self.assertEqual(search.questions, [])
        self.assertEqual(answer_agent.calls, [])

    async def test_low_relevance_rewrites_once_and_answers_second_evidence(
        self,
    ) -> None:
        record = _knowledge_record(
            chunk_id="chunk-retry",
            document_id="doc-retry",
            title="Java 虚拟线程",
            content="虚拟线程仍需要限制数据库连接池和接口配额。",
        )
        self._store_records((record,))
        original_query = "线程很便宜还需要限流吗？"
        retry_query = "Java virtual thread 数据库连接池 接口配额 限流"
        search = _PlannedKnowledgeSearch(
            {
                original_query: KnowledgeSearchResult(),
                retry_query: KnowledgeSearchResult(
                    hits=(
                        KnowledgeSearchHit(
                            chunk_id=record.chunk_id,
                            content_hash=record.content_hash,
                            score=0.9,
                            bm25_rank=1,
                        ),
                    )
                ),
            }
        )
        answer_agent = _RecordingKnowledgeAnswerGenerator()
        service = KnowledgeQaService(
            repository=self.repository,
            search=search,
            answer_agent=answer_agent,
            query_analysis_agent=_FixedKnowledgeQueryAnalysisAgent(
                KnowledgeQueryAnalysis(
                    standalone_query=original_query,
                    retry_query=retry_query,
                    question_type="verification",
                    strategy="direct",
                    confidence=0.9,
                )
            ),
        )

        answer = await service.ask(original_query)

        self.assertIn(answer.status, {"success", "degraded"})
        self.assertEqual(
            [call[0] for call in search.calls],
            [original_query, retry_query],
        )
        self.assertEqual(len(answer_agent.calls), 1)
        self.assertEqual(
            tuple(
                item.chunk_id for item in answer_agent.calls[0]["evidence"]
            ),
            (record.chunk_id,),
        )

    async def test_rewrite_exhaustion_refuses_without_third_search(self) -> None:
        original_query = "线程很便宜还需要限流吗？"
        retry_query = "Java virtual thread 数据库连接池 接口配额 限流"
        search = _PlannedKnowledgeSearch(
            {
                original_query: KnowledgeSearchResult(),
                retry_query: KnowledgeSearchResult(),
            }
        )
        answer_agent = _RecordingKnowledgeAnswerGenerator()
        service = KnowledgeQaService(
            repository=self.repository,
            search=search,
            answer_agent=answer_agent,
            query_analysis_agent=_FixedKnowledgeQueryAnalysisAgent(
                KnowledgeQueryAnalysis(
                    standalone_query=original_query,
                    retry_query=retry_query,
                    question_type="verification",
                    strategy="direct",
                    confidence=0.9,
                )
            ),
        )

        answer = await service.ask(original_query)

        self.assertEqual(answer.status, "insufficient_evidence")
        self.assertEqual(len(search.calls), 2)
        self.assertEqual(answer_agent.calls, [])
        self.assertEqual(answer.citations, ())
        self.assertEqual(answer.images, ())

    async def test_gate_approved_subset_is_the_only_answer_evidence(self) -> None:
        first = _knowledge_record(
            chunk_id="chunk-first",
            document_id="doc-gate",
            title="证据门控",
            content="第一条候选证据。",
            position=0,
        )
        second = _knowledge_record(
            chunk_id="chunk-second",
            document_id="doc-gate",
            title="证据门控",
            content="第二条批准证据。",
            position=1,
        )
        self._store_records((first, second))
        search_result = KnowledgeSearchResult(
            hits=tuple(
                KnowledgeSearchHit(
                    chunk_id=record.chunk_id,
                    content_hash=record.content_hash,
                    score=score,
                    bm25_rank=index,
                )
                for index, (record, score) in enumerate(
                    ((first, 0.9), (second, 0.8)),
                    start=1,
                )
            )
        )
        _, _, Decision, Gate = _evidence_routing_components()

        class _SubsetGate:
            def __init__(self) -> None:
                self._delegate = Gate()

            def precheck(self, **kwargs: Any) -> Any:
                return self._delegate.precheck(**kwargs)

            def decide_after_retrieval(self, *_: Any, **__: Any) -> Any:
                return Decision(
                    action="answer",
                    confidence=1.0,
                    reason_code="enough_evidence",
                    approved_evidence_ids=(second.chunk_id,),
                )

        answer_agent = _RecordingKnowledgeAnswerGenerator()
        service = KnowledgeQaService(
            repository=self.repository,
            search=_FixedKnowledgeSearchResult(search_result),
            answer_agent=answer_agent,
            evidence_gate=_SubsetGate(),
        )

        answer = await service.ask("哪条证据被批准？")

        self.assertIn(answer.status, {"success", "degraded"})
        self.assertEqual(len(answer_agent.calls), 1)
        self.assertEqual(
            tuple(
                item.chunk_id for item in answer_agent.calls[0]["evidence"]
            ),
            (second.chunk_id,),
        )
        self.assertEqual(
            tuple(citation.chunk_id for citation in answer.citations),
            (second.chunk_id,),
        )

    async def test_runtime_skill_response_policy_reaches_answer_agent(self) -> None:
        record = _knowledge_record(
            chunk_id="chunk-skill-policy",
            document_id="doc-skill-policy",
            title="Java 虚拟线程",
            content="虚拟线程需要关注调度机制与外部资源约束。",
        )
        self._store_records((record,))
        registry = self._registry(
            (
                _runtime_skill_manifest(
                    "java-concurrency",
                    keywords=("虚拟线程",),
                    topics=(),
                ),
            )
        )
        search_result = KnowledgeSearchResult(
            hits=(
                KnowledgeSearchHit(
                    chunk_id=record.chunk_id,
                    content_hash=record.content_hash,
                    score=0.9,
                    bm25_rank=1,
                ),
            )
        )
        answer_agent = _RecordingKnowledgeAnswerGenerator()
        service = KnowledgeQaService(
            repository=self.repository,
            search=_FixedKnowledgeSearchResult(search_result),
            answer_agent=answer_agent,
            runtime_skill_registry=registry,
        )

        answer = await service.ask("虚拟线程有哪些限制？")

        self.assertIn(answer.status, {"success", "degraded"})
        self.assertEqual(len(answer_agent.calls), 1)
        response_policy = answer_agent.calls[0]["response_policy"]
        self.assertIsNotNone(response_policy)
        self.assertEqual(
            response_policy.focus,
            ("mechanism", "constraints"),
        )
        self.assertEqual(
            response_policy.organization,
            "conclusion_then_details",
        )


class KnowledgeTestRecordWriterTests(unittest.IsolatedAsyncioTestCase):
    """验证运行期 Markdown 记录只写安全诊断且并发追加完整。"""

    @staticmethod
    def _trace(trace_id: str) -> KnowledgeExecutionTrace:
        return KnowledgeExecutionTrace(
            trace_id=trace_id,
            route="knowledge_qa",
            request_route="/api/v1/knowledge/ask",
            question="HMI 刷机传输界面是什么？",
            input=KnowledgeExecutionInput(history_message_count=0),
            standalone_query="HMI 刷机传输到跳板机的进度界面",
            question_type="procedural",
            strategy="direct",
            search_queries=("HMI 刷机传输到跳板机的进度界面",),
            retrieved_chunks=(
                KnowledgeExecutionChunk(
                    rank=1,
                    chunk_id="chunk-hmi",
                    document_id="doc-hmi",
                    title="HMI 刷机指南",
                    heading_path=("传输跳板机",),
                    score=0.91,
                    bm25_rank=1,
                    selected=True,
                    excerpt="正在传输到跳板机（418.8 MB）。",
                ),
            ),
            documents=(
                KnowledgeExecutionDocument(
                    document_id="doc-hmi",
                    title="HMI 刷机指南",
                    score=0.91,
                    retrieved_chunk_ids=("chunk-hmi",),
                    selected_chunk_ids=("chunk-hmi",),
                ),
            ),
            result=KnowledgeExecutionResult(
                status="success",
                citation_count=1,
                image_count=1,
                elapsed_ms=12.5,
            ),
        )

    @staticmethod
    def _planned_trace(trace_id: str) -> KnowledgeExecutionTrace:
        contracts = _knowledge_reasoning_contracts()
        return KnowledgeExecutionTrace(
            trace_id=trace_id,
            route="knowledge_qa",
            request_route="/api/v1/knowledge/ask",
            question="分析 RRF 的影响与限制",
            input=KnowledgeExecutionInput(
                history_message_count=1,
                requested_document_ids=("doc-rrf",),
            ),
            standalone_query="分析 RRF 的影响与限制",
            uses_history=True,
            question_type="analytical",
            strategy="decomposed",
            sub_queries=("RRF 影响", "RRF 限制"),
            confidence=0.91,
            search_queries=("RRF 事实基础", "RRF 影响", "RRF 限制"),
            reasoning_strategy="facet_analysis",
            plan_revision_count=2,
            plan_steps=(
                contracts.KnowledgePlanTraceStep(
                    revision=1,
                    step_id="step-1",
                    facet="subject",
                    query="RRF 事实基础",
                    required=True,
                    status="covered",
                    reason_code="enough_evidence",
                    selected_chunk_ids=("chunk-subject",),
                ),
                contracts.KnowledgePlanTraceStep(
                    revision=1,
                    step_id="step-2",
                    facet="constraint",
                    query="RRF 限制",
                    required=True,
                    status="weak",
                    reason_code="insufficient_subject_coverage",
                    selected_chunk_ids=("chunk-constraint-old",),
                ),
                contracts.KnowledgePlanTraceStep(
                    revision=2,
                    step_id="step-1",
                    facet="subject",
                    query="RRF 事实基础",
                    required=True,
                    status="covered",
                    reason_code="enough_evidence",
                    selected_chunk_ids=("chunk-subject",),
                ),
                contracts.KnowledgePlanTraceStep(
                    revision=2,
                    step_id="step-3",
                    facet="constraint",
                    query="RRF 融合限制",
                    required=True,
                    status="covered",
                    reason_code="enough_evidence",
                    selected_chunk_ids=("chunk-constraint",),
                ),
            ),
            coverage=contracts.KnowledgePlanCoverage(
                step_results=(
                    contracts.KnowledgePlanStepResult(
                        step_id="step-1",
                        status="covered",
                        search_query="RRF 事实基础",
                        selected_chunk_ids=("chunk-subject",),
                        selected_document_ids=("doc-rrf",),
                        reason_code="enough_evidence",
                    ),
                    contracts.KnowledgePlanStepResult(
                        step_id="step-3",
                        status="covered",
                        search_query="RRF 融合限制",
                        selected_chunk_ids=("chunk-constraint",),
                        selected_document_ids=("doc-rrf",),
                        reason_code="enough_evidence",
                    ),
                ),
                required_steps=2,
                covered_required_steps=2,
                covered_steps=2,
                coverage_ratio=1.0,
                replanned=True,
                decision="answer",
            ),
            result=KnowledgeExecutionResult(
                status="success",
                citation_count=2,
                image_count=0,
                elapsed_ms=21.5,
            ),
        )

    async def test_writer_appends_concurrent_safe_markdown_records(self) -> None:
        module = importlib.import_module(
            "app.infrastructure.observability.knowledge_test_record"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "knowledge-qa.md"
            writer = module.KnowledgeTestRecordWriter(path)

            outcomes = await asyncio.gather(
                writer.append(self._trace("a" * 32)),
                writer.append(self._trace("b" * 32)),
            )
            content = path.read_text(encoding="utf-8")

        self.assertEqual(outcomes, [True, True])
        self.assertEqual(
            sum(line.startswith("## ") for line in content.splitlines()),
            2,
        )
        self.assertIn("### 1. 发送信息", content)
        self.assertIn("用户问题：HMI 刷机传输界面是什么？", content)
        self.assertIn("历史消息：0 条", content)
        self.assertIn("### 2. 路由与问题分析", content)
        self.assertIn("### 3. 计划与覆盖", content)
        self.assertIn("未启用", content)
        self.assertIn("### 4. 实际检索查询", content)
        self.assertIn("### 5. 召回 Chunk", content)
        self.assertIn("#1 · HMI 刷机指南 · doc-hmi · chunk-hmi", content)
        self.assertIn("分数 0.9100 · 进入最终证据", content)
        self.assertIn("正在传输到跳板机（418.8 MB）。", content)
        self.assertIn("### 6. 文档证据", content)
        self.assertIn("### 7. 最终结果", content)
        self.assertIn("路由：knowledge_qa", content)
        self.assertIn("chunk-hmi", content)
        self.assertIn("HMI 刷机指南", content)
        for forbidden in ("Prompt", "embedding", "traceback", "/mnt/"):
            self.assertNotIn(forbidden, content)

    async def test_writer_renders_two_plan_revisions_and_final_coverage(
        self,
    ) -> None:
        module = importlib.import_module(
            "app.infrastructure.observability.knowledge_test_record"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "knowledge-plan.md"
            writer = module.KnowledgeTestRecordWriter(path)

            recorded = await writer.append(self._planned_trace("d" * 32))
            content = path.read_text(encoding="utf-8")

        self.assertTrue(recorded)
        self.assertIn("### 3. 计划与覆盖", content)
        self.assertIn("推理策略：facet_analysis", content)
        self.assertIn("计划修订：2 版", content)
        self.assertIn("第 1 版", content)
        self.assertIn("第 2 版", content)
        self.assertIn("step-2", content)
        self.assertIn("insufficient_subject_coverage", content)
        self.assertIn("覆盖率：1.0000", content)
        self.assertIn("已重规划：是", content)
        self.assertIn("最终动作：answer", content)

    async def test_writer_appends_full_stream_in_sequence_with_final_result(
        self,
    ) -> None:
        module = importlib.import_module(
            "app.infrastructure.observability.knowledge_test_record"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stream.md"
            writer = module.KnowledgeTestRecordWriter(path)
            response = ChatResponse(
                session_id="stream-session",
                message="传输进度为 40%。[1][图1]",
                action=ArbitrationAction.KNOWLEDGE_ANSWER,
                intent_state=IntentState.KNOWLEDGE_QA,
            )
            events = (
                {
                    "event": "process",
                    "trace_id": "a" * 32,
                    "sequence": 1,
                    "elapsed_ms": 0.1,
                    "stage": "请求",
                    "component": "chat_controller",
                    "status": "started",
                    "title": "收到请求",
                    "summary": "开始处理",
                    "details": {"request_route": "/api/v1/chat/stream"},
                },
                {
                    "event": "process",
                    "trace_id": "a" * 32,
                    "sequence": 2,
                    "elapsed_ms": 8.2,
                    "stage": "召回",
                    "component": "knowledge_retrieval",
                    "status": "success",
                    "title": "召回 Chunk #1",
                    "summary": "chunk-hmi",
                    "details": {
                        "document_id": "doc-hmi",
                        "chunk_id": "chunk-hmi",
                        "score": 0.91,
                        "prompt": "Fake Prompt",
                        "raw_output": "模型原始输出",
                        "answer_draft": "答案草稿",
                        "internal_path": "/mnt/private/fake-secret",
                        "secret": "fake-secret",
                    },
                },
            )

            recorded = await writer.append_stream(
                trace_id="a" * 32,
                events=events,
                response=response,
            )

            self.assertTrue(recorded)
            content = path.read_text(encoding="utf-8")
            self.assertLess(
                content.index("收到请求"),
                content.index("召回 Chunk #1"),
            )
            self.assertIn("### 思考过程摘要", content)
            self.assertIn(
                "由安全执行轨迹确定性生成，不是模型隐藏思维链",
                content,
            )
            self.assertIn("理解问题：", content)
            self.assertIn("召回候选：已召回 1 个 Chunk", content)
            self.assertIn(
                "组织结果：形成知识回答，保留 0 条精确 Chunk 引用",
                content,
            )
            self.assertIn("### 详细执行链路", content)
            self.assertIn("最终结果：knowledge_answer", content)
            self.assertIn("Session：stream-session", content)
            self.assertIn("chunk-hmi", content)
            self.assertNotIn("Prompt", content)
            self.assertNotIn("模型原始输出", content)
            self.assertNotIn("答案草稿", content)
            self.assertNotIn("/mnt/", content)
            self.assertNotIn("fake-secret", content)

    async def test_writer_failure_returns_false_without_raising(self) -> None:
        module = importlib.import_module(
            "app.infrastructure.observability.knowledge_test_record"
        )
        with tempfile.TemporaryDirectory() as directory:
            blocker = Path(directory) / "blocker"
            blocker.write_text("not a directory", encoding="utf-8")
            writer = module.KnowledgeTestRecordWriter(
                blocker / "knowledge-qa.md"
            )

            outcome = await writer.append(self._trace("c" * 32))

        self.assertFalse(outcome)


class KnowledgeExecutionRecordIsolationTests(unittest.IsolatedAsyncioTestCase):
    """验证可替换记录器失败不会破坏已经生成的知识答案。"""

    async def test_record_writer_exception_does_not_fail_answer(self) -> None:
        class _FailingWriter:
            async def append(self, trace: KnowledgeExecutionTrace) -> bool:
                _ = trace
                raise OSError("受控记录器失败")

        with tempfile.TemporaryDirectory() as directory:
            service = KnowledgeQaService(
                repository=SQLiteKnowledgeRepository(
                    Path(directory) / "knowledge.sqlite3"
                ),
                search=InMemoryKnowledgeSearch(),
                answer_agent=KnowledgeAnswerAgent(llm=None),
                execution_record_writer=_FailingWriter(),
            )

            answer = await service.ask("没有文档时会怎样？")

        self.assertEqual(answer.status, "insufficient_evidence")
        self.assertIsNotNone(answer.execution_trace)


class KnowledgeEvidenceSelectorTests(unittest.TestCase):
    """验证证据经过 Hash、范围、文档配额和总预算保护。"""

    @staticmethod
    def _selector() -> Any:
        try:
            module = importlib.import_module(
                "app.domain.services.knowledge_evidence_selector"
            )
        except ModuleNotFoundError as exc:
            raise AssertionError("知识证据选择器尚未实现") from exc
        return module.KnowledgeEvidenceSelector()

    @staticmethod
    def _hit(record: KnowledgeChunkRecord, *, stale: bool = False) -> KnowledgeSearchHit:
        return KnowledgeSearchHit(
            chunk_id=record.chunk_id,
            content_hash="0" * 64 if stale else record.content_hash,
            score=1.0,
            bm25_rank=1,
        )

    def test_full_library_applies_hash_document_quota_and_budget(self) -> None:
        records = tuple(
            _knowledge_record(
                chunk_id=f"chunk-a-{index}",
                document_id="doc-a",
                title="文档 A",
                content=f"A 证据 {index}",
                token_count=900,
            )
            for index in range(4)
        ) + (
            _knowledge_record(
                chunk_id="chunk-b-1",
                document_id="doc-b",
                title="文档 B",
                content="B 证据 1",
                token_count=200,
            ),
            _knowledge_record(
                chunk_id="chunk-b-stale",
                document_id="doc-b",
                title="文档 B",
                content="失效证据",
                token_count=100,
            ),
        )
        retrieval = KnowledgeSearchResult(
            hits=tuple(
                self._hit(record, stale=record.chunk_id == "chunk-b-stale")
                for record in records
            )
        )

        selected = self._selector().select(
            retrieval,
            records,
            scope_document_ids=(),
        )

        self.assertEqual(
            tuple(record.chunk_id for record in selected),
            ("chunk-a-0", "chunk-a-1", "chunk-a-2", "chunk-b-1"),
        )
        self.assertLessEqual(sum(record.token_count for record in selected), 3000)

    def test_explicit_scope_rejects_out_of_scope_records(self) -> None:
        record_a = _knowledge_record(
            chunk_id="chunk-a",
            document_id="doc-a",
            title="文档 A",
            content="A 证据",
        )
        record_b = _knowledge_record(
            chunk_id="chunk-b",
            document_id="doc-b",
            title="文档 B",
            content="B 证据",
        )
        retrieval = KnowledgeSearchResult(
            hits=(self._hit(record_a), self._hit(record_b))
        )

        selected = self._selector().select(
            retrieval,
            (record_a, record_b),
            scope_document_ids=("doc-b",),
        )

        self.assertEqual(
            tuple(record.document_id for record in selected),
            ("doc-b",),
        )

    def test_full_parent_never_returns_a_partial_group_over_budget(self) -> None:
        first_parent = tuple(
            _knowledge_record(
                chunk_id=f"chunk-first-{index}",
                document_id="doc-parent",
                title="Parent 文档",
                heading_path=("Parent 文档", "第一节"),
                content=f"第一节内容 {index}",
                token_count=800,
                position=index,
            )
            for index in range(2)
        )
        second_parent = (
            _knowledge_record(
                chunk_id="chunk-second-0",
                document_id="doc-parent",
                title="Parent 文档",
                heading_path=("Parent 文档", "第二节"),
                content="第二节内容 0",
                token_count=800,
                position=2,
            ),
            _knowledge_record(
                chunk_id="chunk-second-1",
                document_id="doc-parent",
                title="Parent 文档",
                heading_path=("Parent 文档", "第二节"),
                content="第二节内容 1",
                token_count=700,
                position=3,
            ),
        )
        snapshot = first_parent + second_parent

        selected = self._selector().select_full_parent_context(
            ranked_records=(first_parent[0], second_parent[0]),
            scores={
                first_parent[0].chunk_id: 0.9,
                second_parent[0].chunk_id: 0.8,
            },
            snapshot=snapshot,
            seed_limit=5,
        )

        self.assertEqual(
            tuple(record.chunk_id for record in selected),
            ("chunk-first-0", "chunk-first-1"),
        )
        self.assertLessEqual(
            sum(record.token_count for record in selected),
            3000,
        )

    def test_comparative_support_requires_both_object_queries(self) -> None:
        bm25 = _knowledge_record(
            chunk_id="chunk-bm25",
            document_id="doc-compare",
            title="BM25 原理",
            heading_path=("检索", "BM25"),
            content="BM25 使用词频计算相关性。",
        )
        vector = _knowledge_record(
            chunk_id="chunk-vector",
            document_id="doc-compare",
            title="向量检索原理",
            heading_path=("检索", "向量检索"),
            content="向量检索使用语义距离召回。",
        )
        selector = self._selector()

        complete = selector.select_comparative_support(
            object_queries=("BM25", "向量检索"),
            records=(bm25, vector),
        )
        incomplete = selector.select_comparative_support(
            object_queries=("BM25", "向量检索"),
            records=(bm25,),
        )

        self.assertEqual(complete, (bm25, vector))
        self.assertEqual(incomplete, ())

    def test_document_summary_context_keeps_position_and_prefix_budget(
        self,
    ) -> None:
        records = tuple(
            _knowledge_record(
                chunk_id=f"chunk-summary-{position}",
                document_id="doc-summary",
                title="总结文档",
                content=f"第 {position + 1} 部分",
                token_count=token_count,
                position=position,
            )
            for position, token_count in enumerate((1400, 1400, 400))
        )

        selected = self._selector().select_document_summary_context(
            document_id="doc-summary",
            snapshot=(records[2], records[0], records[1]),
        )

        self.assertEqual(
            tuple(record.chunk_id for record in selected),
            ("chunk-summary-0", "chunk-summary-1"),
        )
        self.assertLessEqual(sum(record.token_count for record in selected), 3000)

    def test_document_evidence_groups_chunk_hits_before_answering(self) -> None:
        selector = self._selector()
        records = (
            _knowledge_record(
                chunk_id="chunk-b",
                document_id="doc-b",
                title="文档 B",
                heading_path=("文档 B",),
                content="文档 B 的单条高分证据。",
                position=0,
            ),
            _knowledge_record(
                chunk_id="chunk-a-late",
                document_id="doc-a",
                title="文档 A",
                heading_path=("文档 A", "后文"),
                content="文档 A 后文证据。",
                position=1,
            ),
            _knowledge_record(
                chunk_id="chunk-a-early",
                document_id="doc-a",
                title="文档 A",
                heading_path=("文档 A", "前文"),
                content="文档 A 前文证据。",
                position=0,
            ),
        )

        grouped = selector.group_by_document(
            records,
            scores={
                "chunk-b": 1.0,
                "chunk-a-late": 0.9,
                "chunk-a-early": 0.8,
            },
        )

        self.assertEqual(
            [bundle.document_id for bundle in grouped],
            ["doc-a", "doc-b"],
        )
        self.assertAlmostEqual(grouped[0].score, 1.3)
        self.assertEqual(
            [record.chunk_id for record in grouped[0].chunks],
            ["chunk-a-early", "chunk-a-late"],
        )

    def test_linked_images_follow_evidence_order_and_are_deduplicated(
        self,
    ) -> None:
        first = _knowledge_record(
            chunk_id="chunk-first-image",
            document_id="doc-image",
            title="图片文档",
            content="图片说明：第一张图",
        )
        second = _knowledge_record(
            chunk_id="chunk-second-image",
            document_id="doc-image",
            title="图片文档",
            content="图片说明：第二张图",
            position=1,
        )
        shared = KnowledgeImageEvidence(
            image_id="img-" + "1" * 32,
            document_id="doc-image",
            title="图片文档",
            image_key="shared",
            caption="共享图",
            content_hash="a" * 64,
            linked_chunk_ids=(first.chunk_id, second.chunk_id),
        )
        second_only = KnowledgeImageEvidence(
            image_id="img-" + "2" * 32,
            document_id="doc-image",
            title="图片文档",
            image_key="second",
            caption="第二张图",
            content_hash="b" * 64,
            linked_chunk_ids=(second.chunk_id,),
        )

        selected = self._selector().select_linked_images(
            evidence=(second, first),
            images=(shared, second_only),
            max_images=6,
        )

        self.assertEqual(
            tuple(image.image_id for image in selected),
            (shared.image_id, second_only.image_id),
        )


class KnowledgeScopeResolverTests(unittest.TestCase):
    """验证当前轮文档范围读取唯一历史，但不写入会话状态。"""

    @staticmethod
    def _resolver() -> Any:
        try:
            module = importlib.import_module(
                "app.domain.services.knowledge_scope_resolver"
            )
        except ModuleNotFoundError as exc:
            raise AssertionError("知识范围解析器尚未实现") from exc
        return module.KnowledgeScopeResolver()

    def test_explicit_title_and_recent_citation_resolve_request_scope(self) -> None:
        resolver = self._resolver()
        documents = (
            ("doc-spring", "Spring 事务实践"),
            ("doc-agent", "Agent 设计指南"),
        )

        explicit = resolver.resolve(
            "《Spring 事务实践》有哪些限制？",
            history=[],
            documents=documents,
        )
        recent = resolver.resolve(
            "这篇还有哪些限制？",
            history=[
                ConversationTurn(
                    role="assistant",
                    content=(
                        "事务代理需要经过代理边界。\n\n"
                        "参考资料：\n[1] Spring 事务实践（事务边界）"
                    ),
                )
            ],
            documents=documents,
        )

        self.assertEqual(explicit.document_ids, ("doc-spring",))
        self.assertEqual(recent.document_ids, ("doc-spring",))
        self.assertFalse(explicit.needs_clarification)
        self.assertFalse(recent.needs_clarification)

    def test_grouped_reference_marker_resolves_one_recent_document(self) -> None:
        """聚合后的多 Chunk 编号仍应支持“这篇”历史指代。"""

        recent = self._resolver().resolve(
            "这篇还有哪些限制？",
            history=[
                ConversationTurn(
                    role="assistant",
                    content=(
                        "事务代理需要经过代理边界。\n\n"
                        "参考资料：\n"
                        "[1, 2] Spring 事务实践（事务边界；传播规则）"
                    ),
                )
            ],
            documents=(("doc-spring", "Spring 事务实践"),),
        )

        self.assertEqual(recent.document_ids, ("doc-spring",))
        self.assertFalse(recent.needs_clarification)

    def test_unresolvable_position_requires_clarification(self) -> None:
        result = self._resolver().resolve(
            "第二篇文章的核心观点是什么？",
            history=[
                ConversationTurn(role="assistant", content="已找到 3 篇文章。")
            ],
            documents=(("doc-spring", "Spring 事务实践"),),
        )

        self.assertTrue(result.needs_clarification)
        self.assertEqual(result.document_ids, ())

    def test_recommendation_history_position_resolves_only_known_title(
        self,
    ) -> None:
        history = [
            ConversationTurn(
                role="assistant",
                content=(
                    "已按你的偏好找到 2 篇文章。 "
                    "推荐结果：Python 异步编程；Java 并发编程"
                ),
            )
        ]
        documents = (("doc-python", "Python 异步编程"),)

        known = self._resolver().resolve(
            "第一篇讲了什么？",
            history=history,
            documents=documents,
        )
        missing = self._resolver().resolve(
            "第二篇讲了什么？",
            history=history,
            documents=documents,
        )

        self.assertEqual(known.document_ids, ("doc-python",))
        self.assertFalse(known.needs_clarification)
        self.assertEqual(missing.document_ids, ())
        self.assertTrue(missing.needs_clarification)

    def test_it_resolves_only_from_one_recent_reference(self) -> None:
        documents = (
            ("doc-spring", "Spring 事务实践"),
            ("doc-agent", "Agent 设计指南"),
        )
        unique = self._resolver().resolve(
            "它还有哪些限制？",
            history=[
                ConversationTurn(
                    role="assistant",
                    content=(
                        "事务代理需要经过代理边界。\n\n"
                        "参考资料：\n[1] Spring 事务实践（事务边界）"
                    ),
                )
            ],
            documents=documents,
        )
        ambiguous = self._resolver().resolve(
            "它们分别讲了什么？",
            history=[
                ConversationTurn(
                    role="assistant",
                    content=(
                        "这里有两份资料。\n\n"
                        "参考资料：\n"
                        "[1] Spring 事务实践（事务边界）\n"
                        "[2] Agent 设计指南（职责边界）"
                    ),
                )
            ],
            documents=documents,
        )

        self.assertEqual(unique.document_ids, ("doc-spring",))
        self.assertFalse(unique.needs_clarification)
        self.assertEqual(ambiguous.document_ids, ())
        self.assertTrue(ambiguous.needs_clarification)
        self.assertEqual(
            ambiguous.candidate_document_ids,
            ("doc-spring", "doc-agent"),
        )

    def test_it_without_reference_history_keeps_full_database_scope(self) -> None:
        history = [
            ConversationTurn(role="user", content="什么是 Spring 事务传播机制？"),
            ConversationTurn(
                role="assistant",
                content="它描述事务方法之间的事务边界。",
            ),
        ]

        result = self._resolver().resolve(
            "它有哪些限制？",
            history=history,
            documents=(("doc-spring", "Spring 事务实践"),),
        )

        self.assertEqual(result.document_ids, ())
        self.assertFalse(result.needs_clarification)

    def test_duplicate_sqlite_title_requires_clarification_for_all_scopes(
        self,
    ) -> None:
        resolver = self._resolver()
        documents = (
            ("doc-spring-a", "Spring 事务实践"),
            ("doc-spring-b", "Spring 事务实践"),
        )
        cases = (
            (
                "明确标题",
                "《Spring 事务实践》有哪些限制？",
                [],
            ),
            (
                "推荐序号",
                "第一篇有哪些限制？",
                [
                    ConversationTurn(
                        role="assistant",
                        content="推荐结果：Spring 事务实践",
                    )
                ],
            ),
            (
                "历史指代",
                "它有哪些限制？",
                [
                    ConversationTurn(
                        role="assistant",
                        content="参考资料：\n[1] Spring 事务实践",
                    )
                ],
            ),
        )

        for label, question, history in cases:
            with self.subTest(label=label):
                result = resolver.resolve(
                    question,
                    history=history,
                    documents=documents,
                )

                self.assertEqual(result.document_ids, ())
                self.assertTrue(result.needs_clarification)
                self.assertEqual(
                    result.candidate_document_ids,
                    ("doc-spring-a", "doc-spring-b"),
                )

    def test_question_without_scope_words_keeps_full_database_scope(
        self,
    ) -> None:
        history = [
            ConversationTurn(
                role="assistant",
                content="参考资料：\n[1] Spring 事务实践（事务边界）",
            )
        ]
        original_history = list(history)

        result = self._resolver().resolve(
            "事务传播机制有哪些限制？",
            history=history,
            documents=(("doc-spring", "Spring 事务实践"),),
        )

        self.assertEqual(result.document_ids, ())
        self.assertFalse(result.needs_clarification)
        self.assertEqual(history, original_history)
        self.assertNotIn("doc-spring", history[0].content)


class IntentDecisionTreeTests(unittest.TestCase):
    """验证高置信规则只处理推荐、问答、无动作和推荐延续。"""

    @staticmethod
    def _context() -> RecommendationContext:
        return RecommendationContext(
            query="Java 虚拟线程 高并发",
            size=3,
            seen_article_ids=["30001"],
        )

    def test_greeting_returns_rule_no_action(self) -> None:
        result = IntentDecisionTree().decide("您好", active_context=None)

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.intent, IntentName.NO_ACTION)
        self.assertEqual(result.source, RecognitionSource.RULE)
        self.assertIsNone(result.rewritten_query)

    def test_clear_recommendation_keeps_natural_language_query(self) -> None:
        result = IntentDecisionTree().decide(
            "推荐 5 篇 Java 虚拟线程文章",
            active_context=None,
        )

        self.assertIsNotNone(result)
        assert result is not None and result.resolved_intent is not None
        self.assertEqual(result.intent, IntentName.RECOMMEND_ARTICLES)
        self.assertEqual(result.relation, RelationHint.NEW)
        self.assertEqual(result.rewritten_query, "推荐 5 篇 Java 虚拟线程文章")
        self.assertEqual(result.resolved_intent.size, 5)
        self.assertFalse(hasattr(result.resolved_intent, "primary_topics"))

    def test_clear_knowledge_question_keeps_original_query(self) -> None:
        result = IntentDecisionTree().decide(
            "虚拟线程是什么？",
            active_context=self._context(),
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.intent, IntentName.KNOWLEDGE_QA)
        self.assertEqual(result.rewritten_query, "虚拟线程是什么？")
        self.assertIsNone(result.resolved_intent)

    def test_repeat_and_quantity_reuse_previous_query(self) -> None:
        tree = IntentDecisionTree()
        context = self._context()

        repeat = tree.decide("换一批", active_context=context)
        quantity = tree.decide("再来 5 篇", active_context=context)

        assert repeat is not None and repeat.resolved_intent is not None
        assert quantity is not None and quantity.resolved_intent is not None
        self.assertEqual(repeat.relation, RelationHint.REPEAT)
        self.assertEqual(repeat.rewritten_query, context.query)
        self.assertEqual(quantity.relation, RelationHint.REPEAT)
        self.assertEqual(quantity.rewritten_query, context.query)
        self.assertEqual(quantity.resolved_intent.size, 5)

    def test_continue_recommendation_switches_from_knowledge_state(self) -> None:
        result = IntentDecisionTree().decide(
            "继续推荐",
            active_context=self._context(),
            intent_state=IntentState.KNOWLEDGE_QA,
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.intent, IntentName.RECOMMEND_ARTICLES)
        self.assertEqual(result.relation, RelationHint.REPEAT)

    def test_ambiguous_and_filter_only_inputs_defer_to_llm(self) -> None:
        context = self._context()
        for message in (
            "推荐 Java，但不要 Java",
            "推荐 Java，再解释第二篇",
            "为什么会失效？",
            "只要英文",
            "难一点",
            "近一年",
        ):
            with self.subTest(message=message):
                self.assertIsNone(
                    IntentDecisionTree().decide(
                        message,
                        active_context=context,
                    )
                )

    def test_repeat_uses_arbitrator_and_preserves_seen_documents(self) -> None:
        context = self._context()
        recognition = IntentDecisionTree().decide(
            "换一批",
            active_context=context,
        )

        assert recognition is not None
        decision = ConversationArbitrator().decide(recognition, context)
        self.assertEqual(decision.action, ArbitrationAction.REPEAT)
        assert decision.context is not None
        self.assertTrue(decision.context.avoid_seen)
        self.assertEqual(decision.context.seen_article_ids, ["30001"])
        self.assertEqual(decision.context.query, context.query)


class _FailingIntentDecisionTree:
    def decide(
        self,
        message: str,
        *,
        active_context: object | None,
        intent_state: IntentState | str,
    ) -> None:
        _ = message, active_context, intent_state
        raise RuntimeError("受控规则树失败")


class _FixedKnowledgeService:
    def __init__(
        self,
        *,
        error: Exception | None = None,
        result: KnowledgeAnswerResult | None = None,
    ) -> None:
        self.error = error
        self.result = result
        self.calls: list[dict[str, Any]] = []

    async def ask(
        self,
        question: str,
        *,
        limit: int = 5,
        history: list[ConversationTurn] | None = None,
        conversation_summary: str | None = None,
        document_ids: tuple[str, ...] = (),
        prepared_query: str | None = None,
        request_route: str = "/api/v1/knowledge/ask",
    ) -> KnowledgeAnswerResult:
        if self.error is not None:
            raise self.error
        self.calls.append(
            {
                "question": question,
                "limit": limit,
                "history": list(history or []),
                "conversation_summary": conversation_summary,
                "document_ids": document_ids,
                "prepared_query": prepared_query,
                "request_route": request_route,
            }
        )
        return self.result or KnowledgeAnswerResult(
            status="success",
            answer="Spring 事务通过代理边界生效。",
            citations=(
                KnowledgeCitation(
                    citation_id="1",
                    document_id="spring-transactions",
                    title="Spring 事务实践",
                    chunk_id="chunk-1",
                    heading_path=("事务边界",),
                    excerpt="Spring 事务通过代理边界生效。",
                ),
            ),
        )


class IntentDecisionTreeGraphTests(unittest.IsolatedAsyncioTestCase):
    """验证意图 Agent 内部先走规则树，Graph 只调用统一入口。"""

    @staticmethod
    def _graph(
        llm: _FixedLlm,
        knowledge_service: _FixedKnowledgeService | None = None,
    ) -> ConversationGraph:
        return ConversationGraph(
            intent_agent=IntentRecognitionAgent(llm=llm),
            profile_agent=SimpleNamespace(),
            arbitrator=ConversationArbitrator(),
            recall_agent=SimpleNamespace(),
            rerank_agent=SimpleNamespace(),
            aggregator=SimpleNamespace(),
            knowledge_qa_service=knowledge_service,
        )

    @staticmethod
    def _no_action_llm() -> _FixedLlm:
        return _FixedLlm(
            {
                "intent": "no_action",
                "relation": "unclear",
                "updated_intent": None,
                "confidence": 0.95,
            }
        )

    @staticmethod
    def _knowledge_llm() -> _FixedLlm:
        return _FixedLlm(
            {
                "intent": "knowledge_qa",
                "relation": "new",
                "rewritten_query": "Spring 事务为什么会失效？",
                "updated_intent": None,
                "confidence": 0.95,
            }
        )

    async def test_intent_agent_owns_rule_first_decision(self) -> None:
        llm = self._no_action_llm()
        agent = IntentRecognitionAgent(llm=llm)

        result = await agent.run("您好")

        self.assertEqual(result.source, RecognitionSource.RULE)
        self.assertEqual(llm.messages, [])

    async def test_rule_hit_skips_intent_llm(self) -> None:
        llm = self._no_action_llm()
        graph = self._graph(llm)
        result = await graph.run(
            user_id="10001",
            session_id="rule-tree",
            message="您好",
            history=[],
            previous_context=None,
        )

        self.assertEqual(llm.messages, [])
        self.assertEqual(result.reply.intent_source, RecognitionSource.RULE)
        self.assertIn("知识问答", result.reply.message)
        self.assertNotIn("只处理文章推荐", result.reply.message)
        self.assertFalse(hasattr(graph, "intent_tree"))
        self.assertNotIn("rule_intent_decision", result.trace)
        self.assertIn("recognize_intent", result.trace)

    async def test_rule_miss_calls_intent_llm_once(self) -> None:
        llm = self._no_action_llm()
        result = await self._graph(llm).run(
            user_id="10001",
            session_id="rule-miss",
            message="推荐 Java，再解释第二篇",
            history=[],
            previous_context=None,
        )

        self.assertEqual(len(llm.messages), 2)
        self.assertEqual(result.reply.intent_source, RecognitionSource.LLM)
        self.assertNotIn("rule_intent_decision", result.trace)
        self.assertIn("recognize_intent", result.trace)

    async def test_rule_failure_safely_falls_back_to_intent_llm(self) -> None:
        llm = self._no_action_llm()
        graph = self._graph(llm)
        graph.intent_agent._decision_tree = _FailingIntentDecisionTree()
        result = await graph.run(
            user_id="10001",
            session_id="rule-failure",
            message="您好",
            history=[],
            previous_context=None,
        )

        self.assertEqual(len(llm.messages), 2)
        self.assertEqual(result.reply.intent_source, RecognitionSource.LLM)
        self.assertIn("recognize_intent", result.trace)

    async def test_knowledge_intent_uses_dedicated_route_and_commits_state(
        self,
    ) -> None:
        llm = self._knowledge_llm()
        knowledge_service = _FixedKnowledgeService()
        history = [
            ConversationTurn(role="user", content="推荐 Spring 文章"),
            ConversationTurn(role="assistant", content="已完成推荐。"),
        ]

        result = await self._graph(llm, knowledge_service).run(
            user_id="10001",
            session_id="knowledge-route",
            message="为什么会失效？",
            history=history,
            previous_context=None,
            conversation_summary="用户正在了解 Spring。",
            intent_state=IntentState.RECOMMENDATION,
        )

        envelope = json.loads(llm.messages[1].content)
        self.assertEqual(set(envelope), {"contract", "input"})
        payload = envelope["input"]
        self.assertEqual(payload["current_intent_state"], "recommendation")
        self.assertEqual(result.reply.action.value, "knowledge_answer")
        self.assertTrue(result.commit_intent_state)
        self.assertEqual(result.pending_intent_state, IntentState.KNOWLEDGE_QA)
        self.assertEqual(result.reply.citations[0].chunk_id, "chunk-1")
        self.assertIn("run_knowledge_qa", result.trace)
        self.assertIn("respond_knowledge", result.trace)
        self.assertNotIn("arbitrate", result.trace)
        self.assertNotIn("user_profile_agent", result.trace)
        self.assertEqual(knowledge_service.calls[0]["history"], history)
        self.assertEqual(
            knowledge_service.calls[0]["conversation_summary"],
            "用户正在了解 Spring。",
        )
        self.assertEqual(knowledge_service.calls[0]["document_ids"], ())
        self.assertEqual(
            knowledge_service.calls[0]["prepared_query"],
            "Spring 事务为什么会失效？",
        )

    async def test_knowledge_history_groups_chunk_citations_by_document(
        self,
    ) -> None:
        """同一文档的多个 Chunk 引用只能生成一条历史参考资料。"""

        knowledge_service = _FixedKnowledgeService(
            result=KnowledgeAnswerResult(
                status="success",
                answer="HMI 刷机界面和前端实现分别来自两份文档。[1][2]",
                citations=(
                    KnowledgeCitation(
                        citation_id="1",
                        document_id="doc-hmi",
                        title="每日问题记录文档",
                        chunk_id="chunk-hmi-agent",
                        heading_path=("6.8刷包", "1.HMI", "1.1 agent刷hmi"),
                        excerpt="HMI 刷机进度界面。",
                    ),
                    KnowledgeCitation(
                        citation_id="1",
                        document_id="doc-hmi",
                        title="每日问题记录文档",
                        chunk_id="chunk-hmi-manual",
                        heading_path=("6.8刷包", "1.HMI", "1.2 手动刷pad的hmi"),
                        excerpt="手动刷 pad 的 HMI 操作。",
                    ),
                    KnowledgeCitation(
                        citation_id="2",
                        document_id="doc-system",
                        title="飞书知识问答系统 - 整体技术方案文档",
                        chunk_id="chunk-frontend",
                        heading_path=(
                            "飞书知识问答系统 - 整体技术方案文档",
                            "3. 核心模块详解",
                            "3.8 前端界面层（frontend）",
                        ),
                        excerpt="前端流式展示和图片处理。",
                    ),
                ),
            )
        )

        result = await self._graph(
            self._knowledge_llm(),
            knowledge_service,
        ).run(
            user_id="10001",
            session_id="knowledge-reference-grouping",
            message="HMI刷机正在传输到跳板机的界面是什么？",
            history=[],
            previous_context=None,
        )

        self.assertEqual(result.history_message.count("每日问题记录文档"), 1)
        self.assertEqual(
            result.history_message.count("飞书知识问答系统 - 整体技术方案文档"),
            1,
        )
        self.assertIn(
            "[1] 每日问题记录文档（"
            "6.8刷包 > 1.HMI > 1.1 agent刷hmi；"
            "6.8刷包 > 1.HMI > 1.2 手动刷pad的hmi）",
            result.history_message,
        )
        self.assertIn(
            "[2] 飞书知识问答系统 - 整体技术方案文档（"
            "3. 核心模块详解 > 3.8 前端界面层（frontend））",
            result.history_message,
        )

    async def test_knowledge_failure_returns_hard_failure_without_commit(
        self,
    ) -> None:
        llm = self._knowledge_llm()

        result = await self._graph(
            llm,
            _FixedKnowledgeService(error=RuntimeError("内部路径")),
        ).run(
            user_id="10001",
            session_id="knowledge-failure",
            message="Spring 事务为什么会失效？",
            history=[],
            previous_context=None,
        )

        self.assertEqual(result.error_stage, "knowledge_qa")
        self.assertFalse(result.commit_intent_state)
        self.assertIsNone(result.pending_intent_state)

    async def test_scope_clarification_keeps_existing_intent_state(self) -> None:
        llm = self._knowledge_llm()
        service = _FixedKnowledgeService(
            result=KnowledgeAnswerResult(
                status="needs_clarification",
                answer="请说明要询问的知识文档标题。",
            )
        )

        result = await self._graph(llm, service).run(
            user_id="10001",
            session_id="knowledge-clarification",
            message="第二篇文章讲了什么？",
            history=[],
            previous_context=None,
            intent_state=IntentState.RECOMMENDATION,
        )

        self.assertEqual(result.reply.action, ArbitrationAction.CLARIFY)
        self.assertTrue(result.reply.needs_clarification)
        self.assertFalse(result.commit_intent_state)
        self.assertIsNone(result.pending_intent_state)

    async def test_successful_recommendation_requests_state_commit(self) -> None:
        context = IntentDecisionTreeTests._context()
        recognition = IntentDecisionTree().decide(
            "继续推荐",
            active_context=context,
            intent_state=IntentState.KNOWLEDGE_QA,
        )
        assert recognition is not None
        decision = ConversationArbitrator().decide(recognition, context)
        graph = object.__new__(ConversationGraph)

        transition = await graph._respond_success(
            {
                "session_id": "recommendation-transition",
                "recognition": recognition,
                "decision": decision,
                "effective_context": decision.context,
                "final_documents": [],
                "document_rerank_result": DocumentRerankResult(
                    data={"llm_applied": False}
                ),
                "agent_statuses": {},
            }
        )

        self.assertTrue(transition.get("commit_intent_state"))
        self.assertEqual(
            transition["pending_intent_state"],
            IntentState.RECOMMENDATION,
        )


class RecallDegradationContractTests(unittest.TestCase):
    """验证 Vector 软失败从召回状态传递到公开响应。"""

    def test_vector_failure_marks_successful_recall_as_degraded(self) -> None:
        result = DocumentRecallResult(
            success=True,
            retrieval_diagnostics=KnowledgeRetrievalDiagnostics(
                bm25_status="executed",
                vector_status="degraded",
            ),
        )
        self.assertEqual(ConversationGraph._result_status(result), "degraded")

    def test_vector_skip_is_not_a_degradation(self) -> None:
        result = DocumentRecallResult(
            success=True,
            retrieval_diagnostics=KnowledgeRetrievalDiagnostics(
                bm25_status="executed",
                vector_status="skipped",
            ),
        )
        self.assertEqual(ConversationGraph._result_status(result), "success")

    def test_chat_response_exposes_recall_degradation(self) -> None:
        reply = ConversationReply(
            session_id="vector-degraded",
            message="已使用关键词召回完成推荐。",
            intent_source=RecognitionSource.LLM,
            action=ArbitrationAction.NEW,
            agent_statuses={"document_recall": "degraded"},
        )

        self.assertEqual(chat_degraded_components(reply), ["document_recall"])

    def test_rule_intent_source_is_not_a_degradation(self) -> None:
        reply = ConversationReply(
            session_id="rule-source",
            message="已按规则更新推荐条件。",
            intent_source=RecognitionSource.RULE,
            action=ArbitrationAction.REFINE,
        )
        self.assertEqual(chat_degraded_components(reply), [])


class SQLiteDocumentRecommendationTests(unittest.IsolatedAsyncioTestCase):
    """验证推荐直接消费 SQLite Chunk 召回并按文档聚合。"""

    async def test_recall_uses_best_chunk_and_skips_seen_document(self) -> None:
        module = importlib.import_module("app.agents.document_recall_agent")
        self.assertTrue(
            hasattr(module, "DocumentRecallAgent"),
            "缺少基于知识 Chunk 的 DocumentRecallAgent",
        )
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        repository = SQLiteKnowledgeRepository(
            Path(temporary_directory.name) / "documents.sqlite3"
        )
        search = InMemoryKnowledgeSearch()
        service = KnowledgeQaService(repository=repository, search=search)
        await service.ingest_document(
            document_id="doc-java",
            title="Java 虚拟线程实践",
            content_markdown=(
                "# Java 虚拟线程\n\n虚拟线程适合大量阻塞式高并发任务。"
            ),
            topics=["Java", "虚拟线程"],
            content_type="tutorial",
            difficulty="intermediate",
            author_id="author-java",
        )
        await service.ingest_document(
            document_id="doc-agent",
            title="多智能体编排",
            content_markdown="# 多智能体\n\n终止条件用于避免智能体无限循环。",
            topics=["多智能体"],
            content_type="technical_design",
            difficulty="advanced",
            author_id="author-agent",
        )
        agent = module.DocumentRecallAgent(
            repository=repository,
            search=search,
        )

        result = await agent.run(
            query="Java 虚拟线程如何处理高并发阻塞任务？",
            size=2,
            seen_document_ids=("doc-agent",),
        )

        self.assertTrue(result.success)
        self.assertEqual(
            [candidate.document_id for candidate in result.candidates],
            ["doc-java"],
        )
        self.assertIn("大量阻塞式高并发任务", result.candidates[0].excerpt)
        self.assertEqual(len(result.candidates[0].matched_chunk_ids), 1)
        self.assertEqual(result.candidates[0].topics, ["Java", "虚拟线程"])
        self.assertEqual(result.candidates[0].content_type, "tutorial")
        self.assertEqual(result.candidates[0].difficulty, "intermediate")
        self.assertEqual(result.candidates[0].author_id, "author-java")
        self.assertGreater(result.candidates[0].total_token_count, 0)
        await service.aclose()

    async def test_multiple_strong_chunks_can_raise_document_a_above_document_b(
        self,
    ) -> None:
        records = {
            "chunk-b-1": _knowledge_record(
                chunk_id="chunk-b-1",
                document_id="doc-b",
                title="文档 B",
                content="文档 B 的第一名 Chunk。",
            ),
            "chunk-a-1": _knowledge_record(
                chunk_id="chunk-a-1",
                document_id="doc-a",
                title="文档 A",
                content="文档 A 的第二名 Chunk。",
            ),
            "chunk-a-2": _knowledge_record(
                chunk_id="chunk-a-2",
                document_id="doc-a",
                title="文档 A",
                content="文档 A 的第三名 Chunk。",
            ),
            "chunk-a-3": _knowledge_record(
                chunk_id="chunk-a-3",
                document_id="doc-a",
                title="文档 A",
                content="文档 A 的第四名 Chunk。",
            ),
            "chunk-a-4": _knowledge_record(
                chunk_id="chunk-a-4",
                document_id="doc-a",
                title="文档 A",
                content="文档 A 的第五名 Chunk，不应参与聚合。",
            ),
        }

        class _RankedChunkSearch:
            async def search(
                self,
                question: str,
                *,
                limit: int = 5,
                document_ids: tuple[str, ...] = (),
                excluded_document_ids: tuple[str, ...] = (),
                max_chunks_per_document: int | None = None,
            ) -> KnowledgeSearchResult:
                _ = question, document_ids, excluded_document_ids
                self.limit = limit
                self.max_chunks_per_document = max_chunks_per_document
                return KnowledgeSearchResult(
                    hits=tuple(
                        KnowledgeSearchHit(
                            chunk_id=chunk_id,
                            content_hash=records[chunk_id].content_hash,
                            score=score,
                            bm25_rank=rank,
                        )
                        for rank, (chunk_id, score) in enumerate(
                            (
                                ("chunk-b-1", 0.9),
                                ("chunk-a-1", 0.899),
                                ("chunk-a-2", 0.898),
                                ("chunk-a-3", 0.897),
                                ("chunk-a-4", 0.896),
                            ),
                            start=1,
                        )
                    )
                )

        search = _RankedChunkSearch()
        repository = SimpleNamespace(
            get_chunks_by_ids=lambda chunk_ids: tuple(
                records[chunk_id] for chunk_id in chunk_ids
            ),
            get_document_facts=lambda document_ids: {
                document_id: DocumentFact(
                    document_id=document_id,
                    title=next(
                        record.title
                        for record in records.values()
                        if record.document_id == document_id
                    ),
                    topics=["固定主题"],
                    content_type="tutorial",
                    difficulty="intermediate",
                    author_id="author-fixture",
                    total_token_count=sum(
                        record.token_count
                        for record in records.values()
                        if record.document_id == document_id
                    ),
                )
                for document_id in document_ids
            },
        )
        module = importlib.import_module("app.agents.document_recall_agent")
        agent = module.DocumentRecallAgent(repository=repository, search=search)

        result = await agent.run(query="固定查询", size=1)

        self.assertEqual(search.limit, 12)
        self.assertEqual(search.max_chunks_per_document, 3)
        self.assertEqual(
            [candidate.document_id for candidate in result.candidates],
            ["doc-a", "doc-b"],
        )
        self.assertEqual(
            result.candidates[0].matched_chunk_ids,
            ["chunk-a-1", "chunk-a-2", "chunk-a-3"],
        )
        self.assertGreater(
            result.candidates[0].recall_score,
            result.candidates[1].recall_score,
        )

    async def test_recall_excludes_seen_document_before_chunk_window(self) -> None:
        module = importlib.import_module("app.agents.document_recall_agent")
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        repository = SQLiteKnowledgeRepository(
            Path(temporary_directory.name) / "documents.sqlite3"
        )
        search = InMemoryKnowledgeSearch()
        service = KnowledgeQaService(
            repository=repository,
            search=search,
            chunker=KnowledgeDocumentChunker(
                target_tokens=20,
                max_tokens=30,
                overlap_tokens=0,
            ),
        )
        seen_sections = "\n\n".join(
            f"## 场景 {index}\n\n"
            "Java virtual thread blocking task Java virtual thread"
            for index in range(1, 6)
        )
        await service.ingest_document(
            document_id="doc-seen",
            title="Java 虚拟线程主文档",
            content_markdown=f"# Java 虚拟线程\n\n{seen_sections}",
            topics=["Java", "虚拟线程"],
            content_type="tutorial",
            difficulty="intermediate",
            author_id="author-java",
        )
        await service.ingest_document(
            document_id="doc-unseen",
            title="Java 并发补充",
            content_markdown=(
                "# Java 并发补充\n\nJava virtual thread guide"
            ),
            topics=["Java", "并发"],
            content_type="analysis",
            difficulty="advanced",
            author_id="author-java",
        )
        agent = module.DocumentRecallAgent(
            repository=repository,
            search=search,
        )

        result = await agent.run(
            query="Java virtual thread",
            size=1,
            seen_document_ids=("doc-seen",),
        )

        self.assertEqual(
            [candidate.document_id for candidate in result.candidates],
            ["doc-unseen"],
        )
        await service.aclose()

    async def test_rerank_only_uses_chunk_evidence_and_guards_unknown_ids(
        self,
    ) -> None:
        article_models = importlib.import_module("app.models.article")
        rerank_module = importlib.import_module("app.agents.document_rerank_agent")
        self.assertTrue(
            hasattr(rerank_module, "DocumentRerankAgent"),
            "缺少只读取 Chunk 证据的 DocumentRerankAgent",
        )
        candidate_type = article_models.DocumentCandidate
        candidates = [
            candidate_type(
                document_id="doc-a",
                title="文档 A",
                topics=["主题 A"],
                content_type="tutorial",
                difficulty="intermediate",
                author_id="author-a",
                total_token_count=600,
                excerpt="A 的可信 Chunk 摘录",
                matched_chunk_ids=["chunk-a"],
                recall_score=0.8,
            ),
            candidate_type(
                document_id="doc-b",
                title="文档 B",
                topics=["主题 B"],
                content_type="analysis",
                difficulty="advanced",
                author_id="author-b",
                total_token_count=1200,
                excerpt="B 的可信 Chunk 摘录",
                matched_chunk_ids=["chunk-b"],
                recall_score=0.7,
            ),
        ]
        llm = _FixedLlm(
            {
                "items": [
                    {
                        "document_id": "doc-a",
                        "llm_score": 0.1,
                        "reason": "A 与查询部分相关。",
                    },
                    {
                        "document_id": "doc-b",
                        "llm_score": 0.9,
                        "reason": "B 与查询高度相关。",
                    },
                    {
                        "document_id": "unknown",
                        "llm_score": 1.0,
                        "reason": "越权文档。",
                    },
                ]
            }
        )
        agent = rerank_module.DocumentRerankAgent(llm=llm)

        result = await agent.run(query="需要 B 的内容", candidates=candidates)

        self.assertTrue(result.success)
        self.assertEqual(
            [item.document_id for item in result.ranked_documents],
            ["doc-b", "doc-a"],
        )
        self.assertEqual(result.data["guarded_reasons"], {"unknown_document_id": 1})
        envelope = json.loads(str(llm.messages[-1].content))
        self.assertEqual(set(envelope), {"contract", "input"})
        payload = envelope["input"]
        self.assertEqual(
            set(payload["candidates"][0]),
            {"document_id", "title", "excerpt", "recall_score"},
        )

    async def test_rerank_discards_duplicate_and_missing_document_ids(
        self,
    ) -> None:
        article_models = importlib.import_module("app.models.article")
        rerank_module = importlib.import_module("app.agents.document_rerank_agent")
        candidates = [
            article_models.DocumentCandidate(
                document_id=document_id,
                title=f"文档 {document_id}",
                topics=[f"主题 {document_id}"],
                content_type="tutorial",
                difficulty="intermediate",
                author_id=f"author-{document_id}",
                total_token_count=800,
                excerpt=f"{document_id} 的可信摘录",
                matched_chunk_ids=[f"chunk-{document_id}"],
                recall_score=score,
            )
            for document_id, score in (("doc-a", 0.8), ("doc-b", 0.7))
        ]
        llm = _FixedLlm(
            {
                "items": [
                    {
                        "document_id": "doc-a",
                        "llm_score": 0.9,
                        "reason": "重复一。",
                    },
                    {
                        "document_id": "doc-a",
                        "llm_score": 0.8,
                        "reason": "重复二。",
                    },
                ]
            }
        )
        agent = rerank_module.DocumentRerankAgent(llm=llm)

        result = await agent.run(query="测试重复和缺失", candidates=candidates)

        self.assertEqual(
            [item.document_id for item in result.ranked_documents],
            ["doc-a", "doc-b"],
        )
        self.assertEqual(result.data["llm_status"], "discarded_incomplete_batch")
        self.assertEqual(
            result.data["guarded_reasons"],
            {"duplicate_document_id": 2, "missing_document_id": 2},
        )

    async def test_rerank_excludes_llm_explicitly_irrelevant_candidates(
        self,
    ) -> None:
        article_models = importlib.import_module("app.models.article")
        rerank_module = importlib.import_module("app.agents.document_rerank_agent")
        candidates = [
            article_models.DocumentCandidate(
                document_id=document_id,
                title=title,
                topics=[topic],
                content_type="tutorial",
                difficulty="intermediate",
                author_id=f"author-{document_id}",
                total_token_count=800,
                excerpt=excerpt,
                matched_chunk_ids=[f"chunk-{document_id}"],
                recall_score=recall_score,
            )
            for document_id, title, topic, excerpt, recall_score in (
                (
                    "doc-java",
                    "Java 虚拟线程",
                    "Java",
                    "虚拟线程适合大量阻塞式并发任务。",
                    0.8,
                ),
                (
                    "doc-feishu",
                    "飞书同步方案",
                    "飞书",
                    "文档同步、元数据和调度器。",
                    0.95,
                ),
            )
        ]
        llm = _FixedLlm(
            {
                "items": [
                    {
                        "document_id": "doc-java",
                        "llm_score": 0.9,
                        "reason": "标题和摘录直接支持 Java 查询。",
                    },
                    {
                        "document_id": "doc-feishu",
                        "llm_score": 0.0,
                        "reason": "标题和摘录均未提及 Java，与查询不相关。",
                    },
                ]
            }
        )
        agent = rerank_module.DocumentRerankAgent(llm=llm)

        result = await agent.run(query="推荐 Java 文档", candidates=candidates)

        self.assertEqual(
            [item.document_id for item in result.ranked_documents],
            ["doc-java"],
        )
        self.assertEqual(result.data["absolute_irrelevant_count"], 1)

    async def test_empty_rerank_reports_zero_absolute_irrelevant_count(
        self,
    ) -> None:
        rerank_module = importlib.import_module("app.agents.document_rerank_agent")

        result = await rerank_module.DocumentRerankAgent(enable_llm=False).run(
            query="推荐 Java 文档",
            candidates=[],
        )

        self.assertTrue(result.success)
        self.assertEqual(result.ranked_documents, [])
        self.assertEqual(result.data["absolute_irrelevant_count"], 0)

    async def test_personal_negative_article_and_difficulty_only_zero_profile_score(
        self,
    ) -> None:
        from datetime import timedelta

        from app.models.profile import (
            ActivityProfile,
            BaseProfileSnapshot,
            BehaviorProfile,
            PreferenceConflict,
            ProfileEvidence,
            ReaderProfileAnalysis,
            ReadingPreferencesAnalysis,
            RecommendationStrategy,
            SemanticProfile,
            InterestAnalysis,
            ExplorationStrategy,
            UserProfile,
            ValuePreference,
        )

        article_models = importlib.import_module("app.models.article")
        rerank_module = importlib.import_module("app.agents.document_rerank_agent")
        now = datetime(2026, 8, 13, 8, 0, tzinfo=timezone.utc)
        activity = ActivityProfile(
            recency_score=0.5,
            frequency_score=0.5,
            engagement_score=0.5,
            level="active_reader",
        )
        profile = UserProfile(
            user_id="user-1",
            profile_status="ready",
            base_profile=BaseProfileSnapshot(
                topics=[],
                blocked_topics=[],
                preferred_content_types=["tutorial"],
                preferred_difficulty="intermediate",
                preferred_reading_length="medium",
                followed_author_ids=[],
                blocked_author_ids=[],
                created_at=now,
            ),
            behavior_profile=BehaviorProfile(
                negative_document_ids=["doc-negative"],
                negative_difficulty_preferences=[
                    ValuePreference(
                        value="beginner",
                        weight=-1.0,
                        evidence_count=1,
                    )
                ],
                activity=activity,
            ),
            semantic_profile=SemanticProfile(
                reader_profile=ReaderProfileAnalysis(
                    reader_type="测试画像",
                    activity_level="active_reader",
                    analysis_confidence=0.8,
                ),
                interest_analysis=InterestAnalysis(),
                reading_preferences=ReadingPreferencesAnalysis(
                    recommended_difficulty="intermediate",
                    content_depth="mixed",
                    preferred_reading_length="medium",
                    preferred_content_types=["tutorial"],
                    technical_density="mixed",
                ),
                exploration_strategy=ExplorationStrategy(
                    mode="balanced",
                    focus_ratio=0.8,
                    exploration_ratio=0.2,
                    diversity_level="medium",
                ),
                recommendation_strategy=RecommendationStrategy(),
                preference_conflicts=[
                    PreferenceConflict(
                        type="none",
                        description="无冲突",
                        resolution="无",
                    )
                ],
            ),
            profile_summary="测试画像",
            profile_confidence=1.0,
            profile_cold_start=False,
            evidence=ProfileEvidence(
                valid_event_count=1,
                invalid_event_count=0,
                strong_signal_count=0,
                latest_event_at=now,
                realtime_event_count=1,
            ),
            generated_at=now,
            expires_at=now + timedelta(minutes=30),
        )
        candidates = [
            article_models.DocumentCandidate(
                document_id="doc-negative",
                title="已明确不喜欢的文章",
                topics=["Java"],
                content_type="tutorial",
                difficulty="advanced",
                author_id="author-1",
                total_token_count=800,
                excerpt="仍与当前查询高度相关。",
                matched_chunk_ids=["chunk-1"],
                recall_score=0.95,
            ),
            article_models.DocumentCandidate(
                document_id="doc-beginner",
                title="入门文章",
                topics=["Java"],
                content_type="tutorial",
                difficulty="beginner",
                author_id="author-2",
                total_token_count=800,
                excerpt="仍与当前查询相关。",
                matched_chunk_ids=["chunk-2"],
                recall_score=0.9,
            ),
            article_models.DocumentCandidate(
                document_id="doc-neutral",
                title="中等难度文章",
                topics=["Java"],
                content_type="tutorial",
                difficulty="intermediate",
                author_id="author-3",
                total_token_count=800,
                excerpt="仍与当前查询相关。",
                matched_chunk_ids=["chunk-3"],
                recall_score=0.85,
            ),
        ]

        result = await rerank_module.DocumentRerankAgent(enable_llm=False).run(
            query="Java 并发",
            candidates=candidates,
            user_profile=profile,
        )

        self.assertTrue(result.success, result.model_dump())
        self.assertEqual(len(result.ranked_documents), 3)
        scores = {
            item.document_id: item.profile_score
            for item in result.ranked_documents
        }
        self.assertEqual(scores["doc-negative"], 0.0)
        self.assertLess(scores["doc-beginner"], scores["doc-neutral"])
        self.assertGreater(
            result.ranked_documents[0].relevance_score,
            0.0,
        )

    def test_aggregator_rejects_unknown_and_seen_documents(self) -> None:
        article_models = importlib.import_module("app.models.article")
        aggregator_module = importlib.import_module(
            "app.domain.services.document_result_aggregator"
        )
        self.assertTrue(
            hasattr(aggregator_module, "DocumentResultAggregator"),
            "缺少文档推荐白名单聚合器",
        )
        candidate = article_models.DocumentCandidate(
            document_id="doc-a",
            title="文档 A",
            topics=["主题 A"],
            content_type="tutorial",
            difficulty="intermediate",
            author_id="author-a",
            total_token_count=600,
            excerpt="可信摘录",
            matched_chunk_ids=["chunk-a"],
            recall_score=0.8,
        )
        ranked_type = article_models.RankedDocument
        ranked = [
            ranked_type(
                **candidate.model_dump(),
                llm_score=0.9,
                relevance_score=0.82,
                profile_score=0.0,
                length_level="short",
                final_score=0.82,
                rerank_reason="查询与摘录直接相关。",
            ),
            ranked_type(
                document_id="unknown",
                title="越权文档",
                topics=["越权主题"],
                content_type="analysis",
                difficulty="advanced",
                author_id="unknown-author",
                total_token_count=4000,
                excerpt="越权摘录",
                matched_chunk_ids=["unknown-chunk"],
                recall_score=1.0,
                llm_score=1.0,
                relevance_score=1.0,
                profile_score=0.0,
                length_level="long",
                final_score=1.0,
                rerank_reason="越权理由。",
            ),
        ]
        aggregator = aggregator_module.DocumentResultAggregator()

        visible = aggregator.aggregate(
            candidates=[candidate],
            ranked_documents=ranked,
            seen_document_ids=(),
            size=2,
        )
        hidden = aggregator.aggregate(
            candidates=[candidate],
            ranked_documents=ranked,
            seen_document_ids=("doc-a",),
            size=2,
        )

        self.assertEqual([item.document_id for item in visible], ["doc-a"])
        self.assertEqual(hidden, [])

    async def test_graph_uses_protected_query_when_profile_degrades(
        self,
    ) -> None:
        article_models = importlib.import_module("app.models.article")
        aggregator_module = importlib.import_module(
            "app.domain.services.document_result_aggregator"
        )
        candidate = article_models.DocumentCandidate(
            document_id="doc-java",
            title="Java 虚拟线程实践",
            topics=["Java", "虚拟线程"],
            content_type="tutorial",
            difficulty="intermediate",
            author_id="author-java",
            total_token_count=700,
            excerpt="虚拟线程适合大量阻塞式高并发任务。",
            matched_chunk_ids=["chunk-java"],
            recall_score=0.8,
        )
        recall_started = asyncio.Event()

        class _ParallelUnavailableProfileAgent:
            def __init__(self) -> None:
                self.observed_recall = False

            async def run(self, *, user_id: str) -> Any:
                _ = user_id
                await asyncio.wait_for(recall_started.wait(), timeout=0.5)
                self.observed_recall = True
                raise RuntimeError("受控画像不可用")

        class _DocumentRecall:
            def __init__(self) -> None:
                self.calls: list[dict[str, Any]] = []

            async def run(
                self,
                *,
                query: str,
                size: int,
                seen_document_ids: tuple[str, ...],
            ) -> Any:
                recall_started.set()
                self.calls.append(
                    {
                        "query": query,
                        "size": size,
                        "seen_document_ids": seen_document_ids,
                    }
                )
                return article_models.DocumentRecallResult(candidates=[candidate])

        class _DocumentRerank:
            async def run(
                self,
                *,
                query: str,
                candidates: list[Any],
                user_profile: Any | None,
                current_topics: tuple[str, ...],
            ) -> Any:
                self.query = query
                self.user_profile = user_profile
                self.current_topics = current_topics
                return article_models.DocumentRerankResult(
                    ranked_documents=[
                        article_models.RankedDocument(
                            **candidates[0].model_dump(),
                            llm_score=0.9,
                            relevance_score=0.82,
                            profile_score=0.0,
                            length_level="short",
                            final_score=0.82,
                            rerank_reason="查询与虚拟线程摘录直接相关。",
                        )
                    ],
                    data={"llm_applied": True},
                )

        intent_llm = _FixedLlm(
            {
                "intent": "unknown",
                "relation": "unclear",
                "rewritten_query": None,
                "updated_intent": None,
                "confidence": 0.0,
            }
        )
        recall = _DocumentRecall()
        profile_agent = _ParallelUnavailableProfileAgent()
        graph = ConversationGraph(
            intent_agent=IntentRecognitionAgent(llm=intent_llm),
            arbitrator=ConversationArbitrator(),
            recall_agent=recall,
            rerank_agent=_DocumentRerank(),
            aggregator=aggregator_module.DocumentResultAggregator(),
            profile_agent=profile_agent,
            knowledge_qa_service=None,
        )

        result = await graph.run(
            user_id="10001",
            session_id="sqlite-document-recommendation",
            message="推荐一篇 Java 虚拟线程文章",
            history=[],
            previous_context=None,
        )

        self.assertEqual(
            recall.calls[0]["query"],
            "推荐一篇 Java 虚拟线程文章",
        )
        self.assertEqual(
            [item.document_id for item in result.reply.recommendations],
            ["doc-java"],
        )
        self.assertNotIn("rewrite_recommendation_query", result.trace)
        self.assertIn("user_profile_agent", result.trace)
        self.assertTrue(profile_agent.observed_recall)
        self.assertEqual(result.reply.agent_statuses["user_profile"], "failed")
        self.assertNotIn("recommendation_reason_agent", result.trace)

    async def test_repeat_restores_previous_query_without_rewrite(
        self,
    ) -> None:
        """“继续推荐”应直接沿用上一轮受保护查询。"""

        article_models = importlib.import_module("app.models.article")
        aggregator_module = importlib.import_module(
            "app.domain.services.document_result_aggregator"
        )
        class _EmptyDocumentRecall:
            def __init__(self) -> None:
                self.queries: list[str] = []

            async def run(
                self,
                *,
                query: str,
                size: int,
                seen_document_ids: tuple[str, ...],
            ) -> Any:
                _ = size, seen_document_ids
                self.queries.append(query)
                return article_models.DocumentRecallResult()

        recall = _EmptyDocumentRecall()
        previous_query = "Java 虚拟线程 高并发 阻塞任务"
        graph = ConversationGraph(
            intent_agent=IntentRecognitionAgent(enable_llm=False),
            arbitrator=ConversationArbitrator(),
            recall_agent=recall,
            rerank_agent=importlib.import_module(
                "app.agents.document_rerank_agent"
            ).DocumentRerankAgent(enable_llm=False),
            aggregator=aggregator_module.DocumentResultAggregator(),
        )

        result = await graph.run(
            user_id="10001",
            session_id="repeat-query-restore",
            message="继续推荐",
            history=[
                ConversationTurn(role="user", content="推荐 Java 虚拟线程文章"),
                ConversationTurn(role="assistant", content="已完成推荐。"),
            ],
            previous_context=RecommendationContext(
                query=previous_query,
                size=1,
                seen_article_ids=["doc-java"],
            ),
            intent_state=IntentState.KNOWLEDGE_QA,
        )

        self.assertEqual(recall.queries, [previous_query])
        self.assertEqual(result.pending_intent_state, IntentState.RECOMMENDATION)

    def test_chat_response_exposes_only_minimal_document_fields(self) -> None:
        article_models = importlib.import_module("app.models.article")
        chat_module = importlib.import_module("app.api.routers.chat")
        reply = ConversationReply(
            session_id="minimal-document-response",
            message="找到 1 篇。",
            intent_source=RecognitionSource.RULE,
            action=ArbitrationAction.NEW,
            active_context=RecommendationContext(
                query="Java 虚拟线程",
                size=1,
            ),
            recommendations=[
                article_models.DocumentRecommendation(
                    document_id="doc-java",
                    title="Java 虚拟线程实践",
                    excerpt="虚拟线程适合大量阻塞式任务。",
                    score=0.82,
                    reason="查询与命中摘录直接相关。",
                )
            ],
        )

        response = chat_module._to_chat_response(reply)
        payload = response.model_dump(mode="json")

        self.assertEqual(
            set(payload["recommendations"][0]),
            {"document_id", "title", "excerpt", "score", "reason"},
        )
        self.assertEqual(payload["active_context"], {"query": "Java 虚拟线程", "size": 1})

    async def test_similar_service_projects_source_and_reuses_recommendation_chain(
        self,
    ) -> None:
        module_name = "app.application.similar_document_recommendation"
        self.assertIsNotNone(importlib.util.find_spec(module_name), module_name)
        module = importlib.import_module(module_name)
        article_models = importlib.import_module("app.models.article")
        aggregator_module = importlib.import_module(
            "app.domain.services.document_result_aggregator"
        )
        now = datetime(2026, 8, 8, tzinfo=timezone.utc)
        source = Document(
            document_id="doc-source",
            title="Java 虚拟线程与阻塞任务",
            content_markdown="# Java 虚拟线程\n\n源文档正文。",
            topics=["Java", "虚拟线程"],
            content_type="tutorial",
            difficulty="intermediate",
            author_id="author-source",
            content_hash="a" * 64,
            created_at=now,
            updated_at=now,
        )
        source_chunks = (
            _knowledge_record(
                chunk_id="chunk-source-1",
                document_id="doc-source",
                title=source.title,
                content="虚拟线程适合大量阻塞式并发任务。" + "扩展内容" * 100,
                heading_path=("Java 虚拟线程", "适用场景"),
            ),
        )
        candidates = [
            article_models.DocumentCandidate(
                document_id=document_id,
                title=f"候选文档 {index}",
                topics=["Java", "并发"],
                content_type="tutorial",
                difficulty="intermediate",
                author_id=f"author-{index}",
                total_token_count=600 + index,
                excerpt=f"候选 {index} 的可信 Chunk 摘录。",
                matched_chunk_ids=[f"chunk-{index}"],
                recall_score=max(0.95 - index * 0.05, 0.1),
            )
            for index, document_id in enumerate(
                (
                    "doc-source",
                    "doc-1",
                    "doc-2",
                    "doc-3",
                    "doc-4",
                    "doc-5",
                    "doc-6",
                )
            )
        ]
        recall_started = asyncio.Event()

        class _UserStore:
            async def get_user(self, user_id: str) -> dict[str, str] | None:
                return {"user_id": user_id}

        class _ProfileAgent:
            def __init__(self) -> None:
                self.observed_recall = False
                self.profile = object()

            async def run(self, *, user_id: str) -> Any:
                self.user_id = user_id
                await asyncio.wait_for(recall_started.wait(), timeout=0.5)
                self.observed_recall = True
                return SimpleNamespace(
                    success=True,
                    profile=self.profile,
                    data={},
                    degraded_reason=None,
                )

        class _RecallAgent:
            async def run(
                self,
                *,
                query: str,
                size: int,
                seen_document_ids: tuple[str, ...],
            ) -> Any:
                self.query = query
                self.size = size
                self.seen_document_ids = seen_document_ids
                recall_started.set()
                return article_models.DocumentRecallResult(
                    candidates=candidates,
                    retrieval_diagnostics=KnowledgeRetrievalDiagnostics(
                        bm25_status="executed",
                        vector_status="degraded",
                    ),
                )

        class _RerankAgent:
            async def run(
                self,
                *,
                query: str,
                candidates: list[Any],
                user_profile: Any | None,
                current_topics: list[str],
            ) -> Any:
                self.query = query
                self.user_profile = user_profile
                self.current_topics = current_topics
                return article_models.DocumentRerankResult(
                    ranked_documents=[
                        article_models.RankedDocument(
                            **candidate.model_dump(),
                            llm_score=candidate.recall_score,
                            relevance_score=candidate.recall_score,
                            profile_score=0.0,
                            length_level="short",
                            final_score=candidate.recall_score,
                            rerank_reason="源文档查询与候选摘录相关。",
                        )
                        for candidate in candidates
                    ],
                    degraded_reason="受控重排降级",
                )

        profile_agent = _ProfileAgent()
        recall_agent = _RecallAgent()
        rerank_agent = _RerankAgent()
        repository = SimpleNamespace(
            get_document=lambda document_id: (
                source if document_id == source.document_id else None
            ),
            list_ready_chunks=lambda document_ids: (
                source_chunks if tuple(document_ids) == (source.document_id,) else ()
            ),
        )
        service = module.SimilarDocumentRecommendationService(
            user_store=_UserStore(),
            repository=repository,
            profile_agent=profile_agent,
            recall_agent=recall_agent,
            rerank_agent=rerank_agent,
            aggregator=aggregator_module.DocumentResultAggregator(),
        )

        result = await service.recommend(
            user_id="10001",
            document_id=source.document_id,
        )

        self.assertTrue(profile_agent.observed_recall)
        self.assertEqual(recall_agent.size, 5)
        self.assertEqual(recall_agent.seen_document_ids, (source.document_id,))
        self.assertLessEqual(len(recall_agent.query), 500)
        self.assertIn(source.title, recall_agent.query)
        self.assertIn("适用场景", recall_agent.query)
        self.assertEqual(rerank_agent.query, recall_agent.query)
        self.assertIs(rerank_agent.user_profile, profile_agent.profile)
        self.assertEqual(rerank_agent.current_topics, source.topics)
        self.assertEqual(
            [item.document_id for item in result.recommendations],
            ["doc-1", "doc-2", "doc-3", "doc-4", "doc-5"],
        )
        self.assertEqual(
            result.agent_statuses,
            {
                "user_profile": "success",
                "document_recall": "degraded",
                "document_rerank": "degraded",
            },
        )

    async def test_similar_service_rejects_missing_user_before_document_lookup(
        self,
    ) -> None:
        module_name = "app.application.similar_document_recommendation"
        self.assertIsNotNone(importlib.util.find_spec(module_name), module_name)
        module = importlib.import_module(module_name)

        class _MissingUserStore:
            async def get_user(self, user_id: str) -> None:
                self.user_id = user_id
                return None

        class _Repository:
            def get_document(self, document_id: str) -> None:
                raise AssertionError("用户不存在时不应读取文档")

        service = module.SimilarDocumentRecommendationService(
            user_store=_MissingUserStore(),
            repository=_Repository(),
            profile_agent=SimpleNamespace(),
            recall_agent=SimpleNamespace(),
            rerank_agent=SimpleNamespace(),
            aggregator=SimpleNamespace(),
        )

        with self.assertRaisesRegex(Exception, "用户不存在"):
            await service.recommend(user_id="missing", document_id="doc-source")

    async def test_similar_service_maps_recall_failure_to_service_unavailable(
        self,
    ) -> None:
        module_name = "app.application.similar_document_recommendation"
        self.assertIsNotNone(importlib.util.find_spec(module_name), module_name)
        module = importlib.import_module(module_name)
        now = datetime(2026, 8, 8, tzinfo=timezone.utc)
        source = Document(
            document_id="doc-source",
            title="源文档",
            content_markdown="# 源文档\n\n正文。",
            topics=["源主题"],
            content_type="tutorial",
            difficulty="beginner",
            author_id="author-source",
            content_hash="b" * 64,
            created_at=now,
            updated_at=now,
        )

        class _UserStore:
            async def get_user(self, user_id: str) -> dict[str, str]:
                return {"user_id": user_id}

        class _ProfileAgent:
            async def run(self, *, user_id: str) -> Any:
                return SimpleNamespace(success=False, profile=None, data={})

        class _RecallAgent:
            async def run(self, **_: Any) -> Any:
                return DocumentRecallResult(success=False, error="RuntimeError")

        repository = SimpleNamespace(
            get_document=lambda document_id: source,
            list_ready_chunks=lambda document_ids: (),
        )
        service = module.SimilarDocumentRecommendationService(
            user_store=_UserStore(),
            repository=repository,
            profile_agent=_ProfileAgent(),
            recall_agent=_RecallAgent(),
            rerank_agent=SimpleNamespace(),
            aggregator=SimpleNamespace(),
        )

        with self.assertRaises(ServiceUnavailableError):
            await service.recommend(user_id="10001", document_id="doc-source")

    async def test_similar_service_rejects_missing_source_document(self) -> None:
        module = importlib.import_module(
            "app.application.similar_document_recommendation"
        )

        class _UserStore:
            async def get_user(self, user_id: str) -> dict[str, str]:
                return {"user_id": user_id}

        class _Repository:
            def get_document(self, document_id: str) -> None:
                self.document_id = document_id
                return None

            def list_ready_chunks(
                self,
                document_ids: tuple[str, ...],
            ) -> tuple[Any, ...]:
                raise AssertionError("源文档不存在时不应读取 Chunk")

        service = module.SimilarDocumentRecommendationService(
            user_store=_UserStore(),
            repository=_Repository(),
            profile_agent=SimpleNamespace(),
            recall_agent=SimpleNamespace(),
            rerank_agent=SimpleNamespace(),
            aggregator=SimpleNamespace(),
        )

        with self.assertRaises(module.DocumentNotFoundError):
            await service.recommend(user_id="10001", document_id="missing-doc")

    async def test_similar_service_degrades_profile_and_rerank_failures(self) -> None:
        module = importlib.import_module(
            "app.application.similar_document_recommendation"
        )
        article_models = importlib.import_module("app.models.article")
        aggregator_module = importlib.import_module(
            "app.domain.services.document_result_aggregator"
        )
        now = datetime(2026, 8, 8, tzinfo=timezone.utc)
        source = Document(
            document_id="doc-source",
            title="源文档",
            content_markdown="# 源文档\n\n正文。",
            topics=["源主题"],
            content_type="tutorial",
            difficulty="beginner",
            author_id="author-source",
            content_hash="c" * 64,
            created_at=now,
            updated_at=now,
        )
        candidate = article_models.DocumentCandidate(
            document_id="doc-related",
            title="相关文档",
            topics=["源主题"],
            content_type="analysis",
            difficulty="intermediate",
            author_id="author-related",
            total_token_count=900,
            excerpt="相关文档的可信 Chunk 摘录。",
            matched_chunk_ids=["chunk-related"],
            recall_score=0.75,
        )

        class _UserStore:
            async def get_user(self, user_id: str) -> dict[str, str]:
                return {"user_id": user_id}

        class _ProfileAgent:
            async def run(self, *, user_id: str) -> Any:
                raise TimeoutError("受控画像超时")

        class _RecallAgent:
            async def run(self, **_: Any) -> Any:
                return DocumentRecallResult(candidates=[candidate])

        class _RerankAgent:
            async def run(self, **_: Any) -> Any:
                raise TimeoutError("受控重排超时")

        repository = SimpleNamespace(
            get_document=lambda document_id: source,
            list_ready_chunks=lambda document_ids: (),
        )
        service = module.SimilarDocumentRecommendationService(
            user_store=_UserStore(),
            repository=repository,
            profile_agent=_ProfileAgent(),
            recall_agent=_RecallAgent(),
            rerank_agent=_RerankAgent(),
            aggregator=aggregator_module.DocumentResultAggregator(),
        )

        result = await service.recommend(
            user_id="10001",
            document_id="doc-source",
        )

        self.assertEqual(
            [item.document_id for item in result.recommendations],
            ["doc-related"],
        )
        self.assertEqual(result.recommendations[0].score, 0.75)
        self.assertEqual(result.agent_statuses["user_profile"], "failed")
        self.assertEqual(result.agent_statuses["document_rerank"], "degraded")

    async def test_bootstrap_shares_chunk_search_and_skips_json_catalog(
        self,
    ) -> None:
        bootstrap = importlib.import_module("app.bootstrap")
        fastapi = importlib.import_module("fastapi")
        application = fastapi.FastAPI()
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        isolated_root = Path(temporary_directory.name)
        isolated_document_path = isolated_root / "documents.sqlite3"
        document_repository = bootstrap.SQLiteKnowledgeRepository(
            isolated_document_path
        )
        user_repository = bootstrap.SQLiteUserProfileRepository(
            isolated_root / "user_profiles.sqlite3"
        )
        conversation_store = bootstrap.SQLiteConversationStore(
            isolated_root / "conversations.sqlite3"
        )
        isolated_image_root = isolated_root / "knowledge_images"
        isolated_skill_root = isolated_root / "skills"
        _write_runtime_skill(
            isolated_skill_root,
            _runtime_skill_manifest(
                "bootstrap-skill",
                keywords=("启动期",),
                topics=(),
            ),
        )
        image_store = bootstrap.LocalKnowledgeImageStore(isolated_image_root)
        trace_writer = bootstrap.ConversationTraceWriter(
            isolated_root / "log"
        )
        query_analyzer = _FixedKnowledgeQueryAnalysisAgent(
            KnowledgeQueryAnalysis(
                standalone_query="启动期规划",
                question_type="factual",
                strategy="direct",
                confidence=1.0,
            )
        )

        with (
            patch.object(
                bootstrap,
                "SQLiteKnowledgeRepository",
                return_value=document_repository,
            ),
            patch.object(
                bootstrap,
                "SQLiteUserProfileRepository",
                return_value=user_repository,
            ),
            patch.object(
                bootstrap,
                "SQLiteConversationStore",
                return_value=conversation_store,
            ),
            patch.object(
                bootstrap,
                "ConversationTraceWriter",
                return_value=trace_writer,
            ),
            patch.object(
                bootstrap,
                "LocalKnowledgeImageStore",
                return_value=image_store,
            ),
            patch.object(
                bootstrap.KnowledgeQueryAnalysisAgent,
                "from_settings",
                return_value=query_analyzer,
            ),
            patch.object(
                bootstrap,
                "RUNTIME_SKILL_ROOT",
                isolated_skill_root,
            ),
            patch.object(bootstrap, "create_embedding_client", return_value=None),
        ):
            async with bootstrap.lifespan(application):
                self.assertEqual(
                    application.state.document_repository.path,
                    isolated_document_path,
                )
                self.assertEqual(
                    application.state.knowledge_qa_service._image_store.root,
                    isolated_image_root,
                )
                self.assertFalse(hasattr(application.state, "article_catalog"))
                self.assertFalse(hasattr(application.state, "similar_article_service"))
                similar_service = (
                    application.state.similar_document_recommendation_service
                )
                self.assertIs(
                    application.state.document_search,
                    application.state.knowledge_qa_service._search,
                )
                self.assertIsInstance(
                    application.state.knowledge_qa_service._chunk_rerank_agent,
                    _knowledge_chunk_rerank_module().KnowledgeChunkRerankAgent,
                )
                self.assertIs(
                    application.state.knowledge_qa_service._query_analysis_agent,
                    query_analyzer,
                )
                registry = application.state.runtime_skill_registry
                self.assertEqual(registry.capture_snapshot().generation, 1)
                self.assertEqual(
                    tuple(registry.capture_snapshot().skills),
                    ("bootstrap-skill",),
                )
                self.assertIs(
                    application.state.knowledge_qa_service._runtime_skill_registry,
                    registry,
                )
                self.assertIs(
                    application.state.conversation_service.knowledge_qa_service,
                    application.state.knowledge_qa_service,
                )
                self.assertIs(similar_service._user_store, application.state.user_store)
                self.assertIs(
                    similar_service._repository,
                    application.state.document_repository,
                )
                self.assertIs(
                    similar_service._profile_agent,
                    application.state.conversation_service.profile_agent,
                )
                self.assertIs(
                    similar_service._recall_agent._search,
                    application.state.document_search,
                )
        self.assertTrue(query_analyzer.closed)

    def test_main_application_publishes_document_similar_route_only(self) -> None:
        main_module = importlib.import_module("app.main")

        paths = {route.path for route in main_module.app.routes}

        self.assertIn("/api/v1/documents/{document_id}/similar", paths)
        self.assertNotIn("/api/v1/articles/{article_id}/similar", paths)


class SQLiteDocumentMetadataContractTests(unittest.TestCase):
    """文档导入必须显式提供推荐需要的 SQLite 元数据。"""

    def test_ingest_request_requires_and_normalizes_document_metadata(self) -> None:
        from pydantic import ValidationError

        from app.models.knowledge_qa import KnowledgeDocumentIngestRequest

        with self.assertRaises(ValidationError):
            KnowledgeDocumentIngestRequest(
                document_id="doc-metadata",
                title="元数据文档",
                content_markdown="# 文档\n\n正文。",
            )

        request = KnowledgeDocumentIngestRequest(
            document_id="doc-metadata",
            title="元数据文档",
            content_markdown="# 文档\n\n正文。",
            topics=[" Spring Boot ", "spring boot", "部署"],
            content_type="tutorial",
            difficulty="intermediate",
            author_id="author-1",
        )

        self.assertEqual(request.topics, ["Spring Boot", "部署"])
        self.assertEqual(request.content_type, "tutorial")
        self.assertEqual(request.difficulty, "intermediate")
        self.assertEqual(request.author_id, "author-1")


if __name__ == "__main__":
    unittest.main()
