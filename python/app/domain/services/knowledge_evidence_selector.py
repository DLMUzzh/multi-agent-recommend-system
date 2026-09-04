"""从 SQLite 回查结果中选择满足事实、范围和预算约束的知识证据。"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence

from app.models.knowledge_qa import (
    KnowledgeChunkRecord,
    KnowledgeDocumentEvidence,
    KnowledgeImageEvidence,
    KnowledgeSearchResult,
)


_PROCEDURAL_MARKERS = ("如何", "怎么", "怎样")
_PROCEDURAL_ACTIONS = (
    "部署",
    "安装",
    "配置",
    "启动",
    "发布",
    "升级",
    "迁移",
    "接入",
    "搭建",
    "构建",
    "编译",
)
_QUESTION_NOISE = (
    "请问",
    "请告诉我",
    "告诉我",
    "通常",
    "一般",
    "应该",
    "可以",
    "需要",
    "一下",
    "呢",
    "吗",
    "在",
    "上",
    "中",
)
_PRONOUN_TOPICS = frozenset(("它", "这个", "这篇", "该文档", "该文章"))
_ORDERED_METHOD_MARKERS = (
    "步骤",
    "流程",
    "首先",
    "先",
    "然后",
    "接着",
    "随后",
    "再",
    "最后",
)
_OPERATION_VERBS = (
    "运行",
    "执行",
    "配置",
    "设置",
    "构建",
    "启动",
    "安装",
    "创建",
    "检查",
    "验证",
    "发布",
    "上传",
    "下载",
    "连接",
)
_COMMAND_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])(?:mvn|gradle|java|docker|docker-compose|"
    r"kubectl|helm|npm|pnpm|yarn|pip|python|systemctl|service|"
    r"sh|bash)(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_ASCII_TOPIC_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]*")
_CHINESE_TOPIC_PATTERN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]{2,}")


class KnowledgeEvidenceSelector:
    """执行事实预算保护，并拦截缺少正文方法的操作型问题。"""

    _PARENT_TARGET_TOKENS = 1200
    _PARENT_MAX_TOKENS = 1600
    _MAX_PARENT_SECTIONS = 6

    def __init__(
        self,
        *,
        max_chunks: int = 6,
        token_budget: int = 3000,
        max_chunks_per_document: int = 3,
    ) -> None:
        if max_chunks <= 0:
            raise ValueError("知识证据数量上限必须大于零")
        if token_budget <= 0:
            raise ValueError("知识证据 Token 预算必须大于零")
        if max_chunks_per_document <= 0:
            raise ValueError("单文档证据数量上限必须大于零")
        self._max_chunks = max_chunks
        self._token_budget = token_budget
        self._max_chunks_per_document = max_chunks_per_document

    def select(
        self,
        retrieval: KnowledgeSearchResult,
        records: Sequence[KnowledgeChunkRecord],
        *,
        scope_document_ids: Sequence[str] = (),
    ) -> tuple[KnowledgeChunkRecord, ...]:
        """只返回仍匹配当前检索快照且不越过请求范围的有限证据。"""

        records_by_id = {
            record.chunk_id: KnowledgeChunkRecord.model_validate(record)
            for record in records
        }
        scope = self._normalize_scope(scope_document_ids)
        per_document_limit = (
            self._max_chunks if len(scope) == 1 else self._max_chunks_per_document
        )
        selected: list[KnowledgeChunkRecord] = []
        selected_tokens = 0
        document_counts: Counter[str] = Counter()
        seen_chunk_ids: set[str] = set()
        seen_content: set[tuple[str, str]] = set()

        for hit in retrieval.hits:
            if len(selected) >= self._max_chunks:
                break
            if hit.chunk_id in seen_chunk_ids:
                continue
            record = records_by_id.get(hit.chunk_id)
            if record is None or record.content_hash != hit.content_hash:
                continue
            if scope and record.document_id not in scope:
                continue
            content_key = (record.document_id, record.content_hash)
            if content_key in seen_content:
                continue
            if document_counts[record.document_id] >= per_document_limit:
                continue
            if selected_tokens + record.token_count > self._token_budget:
                continue
            selected.append(record.model_copy(deep=True))
            selected_tokens += record.token_count
            document_counts[record.document_id] += 1
            seen_chunk_ids.add(record.chunk_id)
            seen_content.add(content_key)

        return tuple(selected)

    @staticmethod
    def group_by_document(
        records: Sequence[KnowledgeChunkRecord],
        *,
        scores: Mapping[str, float],
    ) -> tuple[KnowledgeDocumentEvidence, ...]:
        """按文档聚合最终证据，并用 Top Chunk 衰减分确定文档顺序。"""

        grouped: dict[str, list[KnowledgeChunkRecord]] = {}
        titles: dict[str, str] = {}
        for raw_record in records:
            record = KnowledgeChunkRecord.model_validate(raw_record).model_copy(
                deep=True
            )
            current_title = titles.setdefault(record.document_id, record.title)
            if current_title != record.title:
                raise ValueError("同一文档证据的标题不一致")
            grouped.setdefault(record.document_id, []).append(record)

        bundles: list[KnowledgeDocumentEvidence] = []
        weights = (1.0, 0.5, 0.25)
        for document_id, document_records in grouped.items():
            ranked_scores = sorted(
                (
                    float(scores.get(record.chunk_id, 0.0))
                    for record in document_records
                ),
                reverse=True,
            )
            if any(not math.isfinite(score) or score < 0.0 for score in ranked_scores):
                raise ValueError("文档证据分必须是非负有限数")
            document_score = sum(
                score * weight
                for score, weight in zip(ranked_scores, weights, strict=False)
            )
            bundles.append(
                KnowledgeDocumentEvidence(
                    document_id=document_id,
                    title=titles[document_id],
                    score=document_score,
                    chunks=tuple(
                        sorted(
                            document_records,
                            key=lambda record: (record.position, record.chunk_id),
                        )
                    ),
                )
            )
        bundles.sort(key=lambda bundle: (-bundle.score, bundle.document_id))
        return tuple(bundles)

    def select_full_parent_context(
        self,
        *,
        ranked_records: Sequence[KnowledgeChunkRecord],
        scores: Mapping[str, float],
        snapshot: Sequence[KnowledgeChunkRecord],
        seed_limit: int,
    ) -> tuple[KnowledgeChunkRecord, ...]:
        """按重排种子选择完整 Parent，超预算时整组跳过。"""

        if seed_limit <= 0:
            raise ValueError("Parent 命中种子数量必须大于零")
        ranked = tuple(
            KnowledgeChunkRecord.model_validate(record).model_copy(deep=True)
            for record in ranked_records
        )
        ranked_ids = [record.chunk_id for record in ranked]
        if len(ranked_ids) != len(set(ranked_ids)):
            raise ValueError("Parent 命中种子 ID 重复")
        normalized_scores = self._validate_parent_scores(ranked_ids, scores)
        records = tuple(
            sorted(
                (
                    KnowledgeChunkRecord.model_validate(record).model_copy(
                        deep=True
                    )
                    for record in snapshot
                ),
                key=lambda record: (
                    record.document_id,
                    record.position,
                    record.chunk_id,
                ),
            )
        )
        parents = self._build_parent_groups(records)
        parent_by_chunk_id = {
            record.chunk_id: parent
            for parent in parents
            for record in parent
        }
        score_lists: dict[tuple[str, ...], list[float]] = {}
        parent_records: dict[
            tuple[str, ...],
            tuple[KnowledgeChunkRecord, ...],
        ] = {}
        for record in ranked[:seed_limit]:
            parent = parent_by_chunk_id.get(record.chunk_id)
            if parent is None:
                continue
            parent_id = tuple(item.chunk_id for item in parent)
            parent_records[parent_id] = parent
            score_lists.setdefault(parent_id, []).append(
                normalized_scores[record.chunk_id]
            )

        ranked_parents: list[
            tuple[float, str, tuple[KnowledgeChunkRecord, ...]]
        ] = []
        for parent_id, parent in parent_records.items():
            top_scores = sorted(score_lists[parent_id], reverse=True)[:2]
            highest = top_scores[0]
            top2_average = sum(top_scores) / len(top_scores)
            parent_score = 0.70 * highest + 0.30 * top2_average
            ranked_parents.append((parent_score, parent_id[0], parent))
        ranked_parents.sort(key=lambda item: (-item[0], item[1]))

        selected: list[KnowledgeChunkRecord] = []
        used_tokens = 0
        for _, _, parent in ranked_parents[: self._MAX_PARENT_SECTIONS]:
            parent_tokens = sum(record.token_count for record in parent)
            if used_tokens + parent_tokens > self._token_budget:
                continue
            selected.extend(record.model_copy(deep=True) for record in parent)
            used_tokens += parent_tokens
        return tuple(selected)

    @classmethod
    def _build_parent_groups(
        cls,
        records: Sequence[KnowledgeChunkRecord],
    ) -> tuple[tuple[KnowledgeChunkRecord, ...], ...]:
        parents: list[tuple[KnowledgeChunkRecord, ...]] = []
        section: list[KnowledgeChunkRecord] = []
        section_key: tuple[str, tuple[str, ...]] | None = None

        def flush_section() -> None:
            if not section:
                return
            buffer: list[KnowledgeChunkRecord] = []
            current_tokens = 0
            for record in section:
                if buffer and (
                    current_tokens >= cls._PARENT_TARGET_TOKENS
                    or current_tokens + record.token_count
                    > cls._PARENT_MAX_TOKENS
                ):
                    parents.append(tuple(buffer))
                    buffer = []
                    current_tokens = 0
                buffer.append(record)
                current_tokens += record.token_count
            if buffer:
                parents.append(tuple(buffer))

        for record in records:
            current_key = (record.document_id, record.heading_path)
            if section_key is not None and current_key != section_key:
                flush_section()
                section = []
            section.append(record)
            section_key = current_key
        flush_section()
        return tuple(parents)

    @staticmethod
    def _validate_parent_scores(
        ranked_ids: Sequence[str],
        scores: Mapping[str, float],
    ) -> dict[str, float]:
        if set(ranked_ids) != set(scores):
            raise ValueError("Parent 排序分与命中种子不一致")
        normalized: dict[str, float] = {}
        for chunk_id in ranked_ids:
            value = scores[chunk_id]
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or not 0.0 <= float(value) <= 1.0
            ):
                raise ValueError("Parent 排序分必须位于 0 到 1")
            normalized[chunk_id] = float(value)
        return normalized

    def select_direct_support(
        self,
        question: str,
        records: Sequence[KnowledgeChunkRecord],
    ) -> tuple[KnowledgeChunkRecord, ...]:
        """对明确操作型问题只保留正文能够直接给出方法的文档证据。"""

        normalized_question = self._required_question(question)
        validated_records = tuple(
            KnowledgeChunkRecord.model_validate(record).model_copy(deep=True)
            for record in records
        )
        action = self._procedural_action(normalized_question)
        if action is None or not validated_records:
            return validated_records

        topic_anchors = self._topic_anchors(normalized_question, action)
        records_by_document: dict[str, list[KnowledgeChunkRecord]] = {}
        for record in validated_records:
            records_by_document.setdefault(record.document_id, []).append(record)

        supported_document_ids = {
            document_id
            for document_id, document_records in records_by_document.items()
            if self._document_supports_procedure(
                document_records,
                action=action,
                topic_anchors=topic_anchors,
            )
        }
        return tuple(
            record
            for record in validated_records
            if record.document_id in supported_document_ids
        )

    def select_comparative_support(
        self,
        *,
        object_queries: Sequence[str],
        records: Sequence[KnowledgeChunkRecord],
    ) -> tuple[KnowledgeChunkRecord, ...]:
        """仅在标题、标题路径或正文同时覆盖比较双方时保留证据。"""

        if len(object_queries) < 2:
            raise ValueError("比较证据检查必须提供两个对象查询")
        objects = tuple(
            self._compact_text(self._required_question(query))
            for query in object_queries[:2]
        )
        validated_records = tuple(
            KnowledgeChunkRecord.model_validate(record).model_copy(deep=True)
            for record in records
        )
        searchable_records = tuple(
            self._compact_text(
                "\n".join(
                    (
                        record.title,
                        *record.heading_path,
                        record.content,
                    )
                )
            )
            for record in validated_records
        )
        if all(
            any(object_query in searchable for searchable in searchable_records)
            for object_query in objects
        ):
            return validated_records
        return ()

    def select_document_summary_context(
        self,
        *,
        document_id: str,
        snapshot: Sequence[KnowledgeChunkRecord],
    ) -> tuple[KnowledgeChunkRecord, ...]:
        """按原始位置装载唯一文档的连续前缀，且不越过总 Token 预算。"""

        normalized_document_id = self._required_question(document_id)
        ordered_records = tuple(
            sorted(
                (
                    KnowledgeChunkRecord.model_validate(record).model_copy(
                        deep=True
                    )
                    for record in snapshot
                    if record.document_id == normalized_document_id
                ),
                key=lambda record: (record.position, record.chunk_id),
            )
        )
        selected: list[KnowledgeChunkRecord] = []
        used_tokens = 0
        for record in ordered_records:
            if used_tokens + record.token_count > self._token_budget:
                break
            selected.append(record)
            used_tokens += record.token_count
        return tuple(selected)

    @staticmethod
    def select_linked_images(
        *,
        evidence: Sequence[KnowledgeChunkRecord],
        images: Sequence[KnowledgeImageEvidence],
        max_images: int = 6,
    ) -> tuple[KnowledgeImageEvidence, ...]:
        """按证据顺序和仓储图片顺序选择有限、去重的关联图片。"""

        if max_images <= 0:
            raise ValueError("知识图片候选上限必须大于零")
        evidence_ids = tuple(
            KnowledgeChunkRecord.model_validate(record).chunk_id
            for record in evidence
        )
        images_by_chunk: dict[str, list[KnowledgeImageEvidence]] = {
            chunk_id: [] for chunk_id in evidence_ids
        }
        for value in images:
            image = KnowledgeImageEvidence.model_validate(value).model_copy(
                deep=True
            )
            for chunk_id in image.linked_chunk_ids:
                if chunk_id in images_by_chunk:
                    images_by_chunk[chunk_id].append(image)

        selected: list[KnowledgeImageEvidence] = []
        seen_ids: set[str] = set()
        for chunk_id in evidence_ids:
            for image in images_by_chunk[chunk_id]:
                if image.image_id in seen_ids:
                    continue
                selected.append(image)
                seen_ids.add(image.image_id)
                if len(selected) >= max_images:
                    return tuple(selected)
        return tuple(selected)

    @staticmethod
    def _normalize_scope(values: Sequence[str]) -> frozenset[str]:
        normalized: set[str] = set()
        for value in values:
            if not isinstance(value, str) or not value.strip():
                raise ValueError("知识文档范围 ID 不能为空")
            normalized.add(value.strip())
        return frozenset(normalized)

    @staticmethod
    def _required_question(question: str) -> str:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("知识问题不能为空")
        return " ".join(question.split())

    @staticmethod
    def _procedural_action(question: str) -> str | None:
        if not any(marker in question for marker in _PROCEDURAL_MARKERS):
            return None
        return next(
            (action for action in _PROCEDURAL_ACTIONS if action in question),
            None,
        )

    @classmethod
    def _topic_anchors(cls, question: str, action: str) -> tuple[str, ...]:
        remainder = question.casefold()
        for value in (*_PROCEDURAL_MARKERS, action, *_QUESTION_NOISE):
            remainder = remainder.replace(value, " ")
        ascii_anchors = tuple(
            dict.fromkeys(
                cls._compact_text(match.group())
                for match in _ASCII_TOPIC_PATTERN.finditer(remainder)
                if cls._compact_text(match.group())
            )
        )
        chinese_anchors = tuple(
            anchor
            for anchor in dict.fromkeys(
                cls._compact_text(match.group())
                for match in _CHINESE_TOPIC_PATTERN.finditer(remainder)
            )
            if anchor and anchor not in _PRONOUN_TOPICS
        )
        return ascii_anchors + chinese_anchors

    @classmethod
    def _document_supports_procedure(
        cls,
        records: Sequence[KnowledgeChunkRecord],
        *,
        action: str,
        topic_anchors: Sequence[str],
    ) -> bool:
        content = "\n".join(record.content for record in records)
        compact_content = cls._compact_text(content)
        if action not in content:
            return False
        if topic_anchors and any(
            anchor not in compact_content for anchor in topic_anchors
        ):
            return False
        return cls._contains_direct_method(content, action=action)

    @staticmethod
    def _contains_direct_method(content: str, *, action: str) -> bool:
        direct_clause_patterns = (
            re.compile(
                rf"(?:可以|可|使用|采用|通过)[^。！？；\n]{{0,30}}{re.escape(action)}"
            ),
            re.compile(
                rf"{re.escape(action)}[^。！？；\n]{{0,30}}(?:可以|可|使用|采用|通过)"
            ),
        )
        if any(pattern.search(content) for pattern in direct_clause_patterns):
            return True
        ordered_markers = sum(
            marker in content for marker in _ORDERED_METHOD_MARKERS
        )
        operation_verbs = sum(verb in content for verb in _OPERATION_VERBS)
        has_command = _COMMAND_PATTERN.search(content) is not None
        return (
            (has_command and (ordered_markers >= 1 or operation_verbs >= 1))
            or ordered_markers >= 2
            or operation_verbs >= 3
        )

    @staticmethod
    def _compact_text(value: str) -> str:
        return "".join(
            character.casefold()
            for character in value
            if character.isalnum()
        )


__all__ = ["KnowledgeEvidenceSelector"]
