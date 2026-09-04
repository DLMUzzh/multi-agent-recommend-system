"""通过共享知识 Chunk 检索构造受保护的文档推荐候选。"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Protocol

import structlog

from app.infrastructure.database.sqlite.knowledge_repository import (
    KnowledgeRepository,
)
from app.models.article import DocumentCandidate, DocumentRecallResult
from app.models.knowledge_qa import KnowledgeChunkRecord, KnowledgeSearchResult


logger = structlog.get_logger()


class DocumentSearch(Protocol):
    """文档推荐对共享 Chunk 检索器的最小依赖。"""

    async def search(
        self,
        question: str,
        *,
        limit: int = 5,
        document_ids: Sequence[str] = (),
        excluded_document_ids: Sequence[str] = (),
        max_chunks_per_document: int | None = None,
    ) -> KnowledgeSearchResult:
        """返回按相关性排序的 Chunk 命中。"""

        ...


class DocumentRecallAgent:
    """使用共享知识 Chunk 索引召回并执行文档级封顶聚合。"""

    _CANDIDATE_MULTIPLIER = 4
    _MAX_CHUNKS_PER_DOCUMENT = 3
    _MAX_EXCERPT_LENGTH = 1200
    _SUPPORT_BONUS_CAP = 0.10
    _SECOND_CHUNK_WEIGHT = 0.10
    _THIRD_CHUNK_WEIGHT = 0.05
    _MAX_DOCUMENT_RAW_SCORE = 1.10

    def __init__(
        self,
        *,
        repository: KnowledgeRepository,
        search: DocumentSearch,
    ) -> None:
        self._repository = repository
        self._search = search

    async def run(
        self,
        *,
        query: str,
        size: int,
        seen_document_ids: Sequence[str] = (),
    ) -> DocumentRecallResult:
        """召回 Chunk、回查 SQLite，并按每篇前三个命中聚合候选。"""

        normalized_query = self._required_text(query, "推荐检索查询")
        if not 1 <= size <= 10:
            raise ValueError("推荐数量必须位于 1 到 10 之间")
        seen = {
            self._required_text(value, "已展示文档 ID") for value in seen_document_ids
        }
        try:
            retrieval = await self._search.search(
                normalized_query,
                limit=(
                    size * self._CANDIDATE_MULTIPLIER * self._MAX_CHUNKS_PER_DOCUMENT
                ),
                excluded_document_ids=tuple(seen),
                max_chunks_per_document=self._MAX_CHUNKS_PER_DOCUMENT,
            )
            records = await asyncio.to_thread(
                self._repository.get_chunks_by_ids,
                tuple(hit.chunk_id for hit in retrieval.hits),
            )
            scores = {hit.chunk_id: hit.score for hit in retrieval.hits}
            ranks = {
                hit.chunk_id: rank for rank, hit in enumerate(retrieval.hits, start=1)
            }
            chunks_by_document: dict[str, list[KnowledgeChunkRecord]] = {}
            for record in records:
                if record.document_id in seen:
                    continue
                document_chunks = chunks_by_document.setdefault(
                    record.document_id,
                    [],
                )
                if len(document_chunks) >= self._MAX_CHUNKS_PER_DOCUMENT:
                    continue
                document_chunks.append(record)

            document_facts = await asyncio.to_thread(
                self._repository.get_document_facts,
                tuple(chunks_by_document),
            )

            query_max_chunk_score = max(
                (
                    max(scores[record.chunk_id], 0.0)
                    for document_chunks in chunks_by_document.values()
                    for record in document_chunks
                ),
                default=0.0,
            )
            scored_candidates: list[tuple[DocumentCandidate, int]] = []
            for document_chunks in chunks_by_document.values():
                first_chunk = document_chunks[0]
                document_fact = document_facts.get(first_chunk.document_id)
                if document_fact is None:
                    continue
                chunk_scores = [scores[record.chunk_id] for record in document_chunks]
                scored_candidates.append(
                    (
                        DocumentCandidate(
                            document_id=first_chunk.document_id,
                            title=first_chunk.title,
                            topics=document_fact.topics,
                            content_type=document_fact.content_type,
                            difficulty=document_fact.difficulty,
                            author_id=document_fact.author_id,
                            total_token_count=document_fact.total_token_count,
                            excerpt=self._excerpt(
                                [record.content for record in document_chunks]
                            ),
                            matched_chunk_ids=[
                                record.chunk_id for record in document_chunks
                            ],
                            recall_score=self._document_score(
                                chunk_scores,
                                query_max_chunk_score=query_max_chunk_score,
                            ),
                        ),
                        ranks[first_chunk.chunk_id],
                    )
                )
            scored_candidates.sort(
                key=lambda item: (
                    -item[0].recall_score,
                    item[1],
                    item[0].document_id,
                )
            )
            candidate_limit = size * self._CANDIDATE_MULTIPLIER
            candidates = [
                candidate for candidate, _ in scored_candidates[:candidate_limit]
            ]
        except Exception as exc:
            logger.error(
                "SQLite 文档召回失败",
                exception_type=type(exc).__name__,
            )
            return DocumentRecallResult(
                success=False,
                error=type(exc).__name__,
            )
        return DocumentRecallResult(
            candidates=candidates,
            retrieval_mode=retrieval.mode,
            retrieval_diagnostics=retrieval.diagnostics,
        )

    @classmethod
    def _excerpt(cls, contents: Sequence[str]) -> str:
        normalized_contents = [" ".join(content.split()) for content in contents]
        if not normalized_contents or any(
            not content for content in normalized_contents
        ):
            raise ValueError("召回 Chunk 正文不能为空")
        return "\n\n".join(normalized_contents)[: cls._MAX_EXCERPT_LENGTH]

    @classmethod
    def _document_score(
        cls,
        chunk_scores: Sequence[float],
        *,
        query_max_chunk_score: float,
    ) -> float:
        """以最佳 Chunk 为主，并限制额外命中最多增加最佳分的 10%。"""

        if not chunk_scores or query_max_chunk_score <= 0.0:
            return 0.0
        normalized = [
            max(score, 0.0) / query_max_chunk_score
            for score in chunk_scores[: cls._MAX_CHUNKS_PER_DOCUMENT]
        ]
        best_score = normalized[0]
        second_score = normalized[1] if len(normalized) > 1 else 0.0
        third_score = normalized[2] if len(normalized) > 2 else 0.0
        support_bonus = min(
            cls._SUPPORT_BONUS_CAP * best_score,
            cls._SECOND_CHUNK_WEIGHT * second_score
            + cls._THIRD_CHUNK_WEIGHT * third_score,
        )
        return min(
            max(
                (best_score + support_bonus) / cls._MAX_DOCUMENT_RAW_SCORE,
                0.0,
            ),
            1.0,
        )

    @staticmethod
    def _required_text(value: object, field_name: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field_name}不能为空")
        return " ".join(value.split())


__all__ = ["DocumentRecallAgent", "DocumentSearch"]
