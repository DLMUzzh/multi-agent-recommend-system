"""推荐与知识问答共享的内存 Chunk BM25、可选向量检索与 RRF。"""

from __future__ import annotations

import asyncio
import hashlib
import math
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol

from app.models.knowledge_qa import (
    KnowledgeChunkRecord,
    KnowledgeRetrievalDiagnostics,
    KnowledgeSearchHit,
    KnowledgeSearchResult,
    RetrievalChannelStatus,
)


_TOKEN_PATTERN = re.compile(
    r"[\u3400-\u4dbf\u4e00-\u9fff]+|[A-Za-z0-9_]+"
)
_IDENTIFIER_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])(?:[A-Za-z_][A-Za-z0-9_.:/+\-]*|"
    r"\d+(?:\.\d+)+)(?![A-Za-z0-9_])"
)
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


class EmbeddingClient(Protocol):
    """知识向量检索依赖的最小异步 Embedding 契约。"""

    async def embed(
        self,
        texts: list[str],
    ) -> Sequence[Sequence[float]]:
        """按输入顺序返回文本向量。"""

        ...


@dataclass(frozen=True, slots=True)
class _IndexedChunk:
    record: KnowledgeChunkRecord
    body_term_frequencies: Counter[str]
    body_length: int
    heading_term_frequencies: Counter[str]
    heading_length: int


@dataclass(frozen=True, slots=True)
class _ChannelHit:
    chunk_id: str
    score: float


@dataclass(frozen=True, slots=True)
class _FusedChunk:
    chunk_id: str
    body_rank: int | None
    heading_rank: int | None
    vector_rank: int | None
    vector_similarity: float
    rrf_score: float


class InMemoryKnowledgeSearch:
    """从 SQLite 快照构建可重建索引，并在请求期执行混合检索。"""

    _BODY_TOP_K = 40
    _HEADING_TOP_K = 30
    _VECTOR_TOP_K = 40
    _CANDIDATE_CAP = 80
    _MAX_CHUNKS_PER_DOCUMENT = 8
    _MAX_CHUNKS_PER_HEADING_PATH = 3
    _DETERMINISTIC_TOP_K = 20

    def __init__(
        self,
        *,
        embedding_client: EmbeddingClient | None = None,
        embedding_dimensions: int | None = None,
        embedding_batch_size: int = 10,
        rrf_k: int = 60,
        bm25_k1: float = 1.2,
        bm25_b: float = 0.75,
    ) -> None:
        if embedding_client is not None and (
            embedding_dimensions is None or embedding_dimensions <= 0
        ):
            raise ValueError("启用向量检索时 embedding_dimensions 必须大于零")
        if embedding_batch_size <= 0:
            raise ValueError("embedding_batch_size 必须大于零")
        if rrf_k <= 0:
            raise ValueError("rrf_k 必须大于零")
        if bm25_k1 <= 0.0:
            raise ValueError("bm25_k1 必须大于零")
        if not 0.0 <= bm25_b <= 1.0:
            raise ValueError("bm25_b 必须位于零到一之间")

        self._embedding_client = embedding_client
        self._embedding_dimensions = embedding_dimensions
        self._embedding_batch_size = embedding_batch_size
        self._rrf_k = rrf_k
        self._bm25_k1 = bm25_k1
        self._bm25_b = bm25_b
        self._chunks: tuple[_IndexedChunk, ...] = ()
        self._chunks_by_id: dict[str, _IndexedChunk] = {}
        self._body_document_frequencies: Counter[str] = Counter()
        self._heading_document_frequencies: Counter[str] = Counter()
        self._average_body_length = 1.0
        self._average_heading_length = 1.0
        self._ascii_compound_expansions: dict[str, tuple[str, ...]] = {}
        self._vectors: dict[str, tuple[float, ...]] = {}
        self._vector_status: RetrievalChannelStatus = (
            "skipped" if embedding_client is None else "degraded"
        )
        self._snapshot_fingerprint = ""

    async def refresh(self, chunks: Sequence[KnowledgeChunkRecord]) -> None:
        """按内容指纹刷新 BM25，并在可用时重建进程内 Chunk 向量。"""

        records = tuple(
            KnowledgeChunkRecord.model_validate(chunk).model_copy(deep=True)
            for chunk in chunks
            if not self._is_resource_placeholder_only(chunk.content)
        )
        fingerprint = self._fingerprint(records)
        if (
            fingerprint == self._snapshot_fingerprint
            and self._vector_status != "degraded"
        ):
            return

        indexed_chunks = tuple(self._index_chunk(record) for record in records)
        body_document_frequencies: Counter[str] = Counter()
        heading_document_frequencies: Counter[str] = Counter()
        for chunk in indexed_chunks:
            body_document_frequencies.update(
                chunk.body_term_frequencies.keys()
            )
            heading_document_frequencies.update(
                chunk.heading_term_frequencies.keys()
            )
        self._chunks = indexed_chunks
        self._chunks_by_id = {
            chunk.record.chunk_id: chunk for chunk in indexed_chunks
        }
        self._body_document_frequencies = body_document_frequencies
        self._heading_document_frequencies = heading_document_frequencies
        self._average_body_length = (
            sum(chunk.body_length for chunk in indexed_chunks)
            / len(indexed_chunks)
            if indexed_chunks
            else 1.0
        )
        self._average_heading_length = (
            sum(chunk.heading_length for chunk in indexed_chunks)
            / len(indexed_chunks)
            if indexed_chunks
            else 1.0
        )
        self._ascii_compound_expansions = (
            self._build_ascii_compound_expansions(records)
        )
        self._snapshot_fingerprint = fingerprint

        if self._embedding_client is None or not records:
            self._vectors = {}
            self._vector_status = "skipped"
            return
        try:
            self._vectors = await self._build_vectors(records)
        except asyncio.CancelledError:
            raise
        except Exception:
            self._vectors = {}
            self._vector_status = "degraded"
        else:
            self._vector_status = "executed"

    async def search(
        self,
        question: str,
        *,
        limit: int = 5,
        document_ids: Sequence[str] = (),
        excluded_document_ids: Sequence[str] = (),
        max_chunks_per_document: int | None = None,
    ) -> KnowledgeSearchResult:
        """执行 BM25 与可选 Vector，并按调用方文档约束返回 RRF 排名。"""

        if not isinstance(question, str) or not question.strip():
            raise ValueError("知识问题不能为空")
        if limit <= 0:
            return KnowledgeSearchResult(
                diagnostics=KnowledgeRetrievalDiagnostics(
                    vector_status=self._vector_status
                )
            )
        if max_chunks_per_document is not None and max_chunks_per_document <= 0:
            raise ValueError("单文档 Chunk 上限必须大于零")

        scope_document_ids = self._normalize_document_ids(document_ids)
        excluded_ids = self._normalize_document_ids(excluded_document_ids)
        body_hits = self._bm25_ranking(
            question.strip(),
            text_kind="body",
            limit=self._BODY_TOP_K,
            document_ids=scope_document_ids,
            excluded_document_ids=excluded_ids,
        )
        heading_hits = self._bm25_ranking(
            question.strip(),
            text_kind="heading",
            limit=self._HEADING_TOP_K,
            document_ids=scope_document_ids,
            excluded_document_ids=excluded_ids,
        )
        vector_hits: tuple[_ChannelHit, ...] = ()
        vector_status = self._vector_status
        if self._vectors and self._embedding_client is not None:
            try:
                vector_hits = await self._vector_ranking(
                    question.strip(),
                    limit=self._VECTOR_TOP_K,
                    document_ids=scope_document_ids,
                    excluded_document_ids=excluded_ids,
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                vector_status = "degraded"
            else:
                vector_status = "executed"

        fused = self._fuse_channels(body_hits, heading_hits, vector_hits)
        diversified = self._apply_recall_quotas(
            fused[: self._CANDIDATE_CAP],
            max_chunks_per_document=(
                min(
                    max_chunks_per_document,
                    self._MAX_CHUNKS_PER_DOCUMENT,
                )
                if max_chunks_per_document is not None
                else self._MAX_CHUNKS_PER_DOCUMENT
            ),
        )
        ranked = self._deterministic_ranking(
            question.strip(),
            diversified,
            vector_executed=vector_status == "executed",
        )
        hits = tuple(ranked[: min(limit, self._DETERMINISTIC_TOP_K)])
        mode = "hybrid" if vector_status == "executed" else "bm25"
        return KnowledgeSearchResult(
            hits=hits,
            mode=mode,
            diagnostics=KnowledgeRetrievalDiagnostics(
                bm25_status="executed",
                vector_status=vector_status,
            ),
        )

    async def aclose(self) -> None:
        """关闭当前实例拥有的可关闭 Embedding 客户端。"""

        close = getattr(self._embedding_client, "aclose", None)
        if close is not None:
            await close()

    def _index_chunk(self, record: KnowledgeChunkRecord) -> _IndexedChunk:
        body_frequencies = Counter(self._tokenize(record.content))
        heading_frequencies = Counter(
            self._tokenize(self._heading_text(record))
        )
        return _IndexedChunk(
            record=record,
            body_term_frequencies=body_frequencies,
            body_length=max(sum(body_frequencies.values()), 1),
            heading_term_frequencies=heading_frequencies,
            heading_length=max(sum(heading_frequencies.values()), 1),
        )

    def _bm25_ranking(
        self,
        question: str,
        *,
        text_kind: str,
        limit: int,
        document_ids: frozenset[str],
        excluded_document_ids: frozenset[str],
    ) -> tuple[_ChannelHit, ...]:
        tokenized_query = self._tokenize(question)
        expanded_query: list[str] = []
        for token in tokenized_query:
            expanded_query.append(token)
            expanded_query.extend(
                self._ascii_compound_expansions.get(token, ())
            )
        query_terms = tuple(dict.fromkeys(expanded_query))
        if not query_terms or not self._chunks:
            return ()
        if text_kind == "body":
            document_frequencies = self._body_document_frequencies
            average_length = self._average_body_length
        elif text_kind == "heading":
            document_frequencies = self._heading_document_frequencies
            average_length = self._average_heading_length
        else:
            raise ValueError("未知 BM25 索引类型")

        ranked: list[_ChannelHit] = []
        document_count = len(self._chunks)
        for chunk in self._chunks:
            if (
                chunk.record.document_id in excluded_document_ids
                or document_ids
                and chunk.record.document_id not in document_ids
            ):
                continue
            if text_kind == "body":
                term_frequencies = chunk.body_term_frequencies
                length = chunk.body_length
            else:
                term_frequencies = chunk.heading_term_frequencies
                length = chunk.heading_length
            length_ratio = length / average_length
            normalization = self._bm25_k1 * (
                1.0 - self._bm25_b + self._bm25_b * length_ratio
            )
            score = 0.0
            for token in query_terms:
                term_frequency = term_frequencies.get(token, 0)
                if term_frequency <= 0:
                    continue
                document_frequency = document_frequencies[token]
                inverse_document_frequency = math.log(
                    1.0
                    + (document_count - document_frequency + 0.5)
                    / (document_frequency + 0.5)
                )
                score += inverse_document_frequency * (
                    term_frequency * (self._bm25_k1 + 1.0)
                    / (term_frequency + normalization)
                )
            if score > 0.0:
                ranked.append(
                    _ChannelHit(
                        chunk_id=chunk.record.chunk_id,
                        score=score,
                    )
                )
        ranked.sort(key=lambda item: (-item.score, item.chunk_id))
        return tuple(ranked[:limit])

    async def _build_vectors(
        self,
        records: tuple[KnowledgeChunkRecord, ...],
    ) -> dict[str, tuple[float, ...]]:
        if self._embedding_client is None or self._embedding_dimensions is None:
            return {}
        vectors: dict[str, tuple[float, ...]] = {}
        for start in range(0, len(records), self._embedding_batch_size):
            batch = records[start : start + self._embedding_batch_size]
            raw_vectors = await self._embedding_client.embed(
                [self._index_text(record) for record in batch]
            )
            if len(raw_vectors) != len(batch):
                raise ValueError("知识 Chunk Embedding 数量与输入不一致")
            for record, vector in zip(batch, raw_vectors, strict=True):
                vectors[record.chunk_id] = self._normalize_vector(vector)
        return vectors

    async def _vector_ranking(
        self,
        question: str,
        *,
        limit: int,
        document_ids: frozenset[str],
        excluded_document_ids: frozenset[str],
    ) -> tuple[_ChannelHit, ...]:
        if self._embedding_client is None:
            return ()
        raw_vectors = await self._embedding_client.embed([question])
        if len(raw_vectors) != 1:
            raise ValueError("知识查询 Embedding 数量无效")
        query_vector = self._normalize_vector(raw_vectors[0])
        ranked = [
            _ChannelHit(
                chunk_id=chunk_id,
                score=sum(
                    query_value * chunk_value
                    for query_value, chunk_value in zip(
                        query_vector,
                        chunk_vector,
                        strict=True,
                    )
                ),
            )
            for chunk_id, chunk_vector in self._vectors.items()
            if self._chunks_by_id[chunk_id].record.document_id
            not in excluded_document_ids
            and (
                not document_ids
                or self._chunks_by_id[chunk_id].record.document_id
                in document_ids
            )
        ]
        ranked = [item for item in ranked if item.score > 0.0]
        ranked.sort(key=lambda item: (-item.score, item.chunk_id))
        return tuple(ranked[:limit])

    def _fuse_channels(
        self,
        body_hits: Sequence[_ChannelHit],
        heading_hits: Sequence[_ChannelHit],
        vector_hits: Sequence[_ChannelHit],
    ) -> tuple[_FusedChunk, ...]:
        body_ranks = {
            hit.chunk_id: rank
            for rank, hit in enumerate(body_hits, start=1)
        }
        heading_ranks = {
            hit.chunk_id: rank
            for rank, hit in enumerate(heading_hits, start=1)
        }
        vector_ranks = {
            hit.chunk_id: rank
            for rank, hit in enumerate(vector_hits, start=1)
        }
        vector_scores = {hit.chunk_id: hit.score for hit in vector_hits}
        chunk_ids = (
            body_ranks.keys() | heading_ranks.keys() | vector_ranks.keys()
        )
        fused = [
            _FusedChunk(
                chunk_id=chunk_id,
                body_rank=body_ranks.get(chunk_id),
                heading_rank=heading_ranks.get(chunk_id),
                vector_rank=vector_ranks.get(chunk_id),
                vector_similarity=vector_scores.get(chunk_id, 0.0),
                rrf_score=sum(
                    1.0 / (self._rrf_k + rank)
                    for rank in (
                        body_ranks.get(chunk_id),
                        heading_ranks.get(chunk_id),
                        vector_ranks.get(chunk_id),
                    )
                    if rank is not None
                ),
            )
            for chunk_id in chunk_ids
        ]
        fused.sort(key=lambda item: (-item.rrf_score, item.chunk_id))
        return tuple(fused)

    def _apply_recall_quotas(
        self,
        candidates: Sequence[_FusedChunk],
        *,
        max_chunks_per_document: int,
    ) -> tuple[_FusedChunk, ...]:
        selected: list[_FusedChunk] = []
        document_counts: Counter[str] = Counter()
        heading_counts: Counter[tuple[str, ...]] = Counter()
        for candidate in candidates:
            record = self._chunks_by_id[candidate.chunk_id].record
            heading_key = record.heading_path
            if (
                document_counts[record.document_id]
                >= max_chunks_per_document
                or heading_counts[heading_key]
                >= self._MAX_CHUNKS_PER_HEADING_PATH
            ):
                continue
            selected.append(candidate)
            document_counts[record.document_id] += 1
            heading_counts[heading_key] += 1
        return tuple(selected)

    def _deterministic_ranking(
        self,
        question: str,
        candidates: Sequence[_FusedChunk],
        *,
        vector_executed: bool,
    ) -> tuple[KnowledgeSearchHit, ...]:
        query_terms = frozenset(self._expanded_query_terms(question))
        query_identifiers = frozenset(
            match.group().casefold()
            for match in _IDENTIFIER_PATTERN.finditer(question)
        )
        max_vector_similarity = max(
            (
                max(candidate.vector_similarity, 0.0)
                for candidate in candidates
            ),
            default=0.0,
        )
        ranked: list[tuple[float, float, KnowledgeSearchHit]] = []
        for candidate in candidates:
            record = self._chunks_by_id[candidate.chunk_id].record
            body_coverage = self._coverage(query_terms, record.content)
            heading_coverage = self._coverage(
                query_terms,
                self._heading_text(record),
            )
            title_coverage = self._coverage(query_terms, record.title)
            identifier_coverage = self._identifier_coverage(
                query_identifiers,
                self._index_text(record),
            )
            keyword_score = (
                0.45 * body_coverage
                + 0.25 * heading_coverage
                + 0.20 * title_coverage
                + 0.10 * identifier_coverage
            )
            vector_score = (
                max(candidate.vector_similarity, 0.0)
                / max_vector_similarity
                if vector_executed and max_vector_similarity > 0.0
                else 0.0
            )
            score = (
                0.70 * keyword_score + 0.30 * vector_score
                if vector_executed
                else keyword_score
            )
            lexical_ranks = tuple(
                rank
                for rank in (candidate.body_rank, candidate.heading_rank)
                if rank is not None
            )
            ranked.append(
                (
                    score,
                    candidate.rrf_score,
                    KnowledgeSearchHit(
                        chunk_id=record.chunk_id,
                        content_hash=record.content_hash,
                        score=score,
                        bm25_rank=min(lexical_ranks) if lexical_ranks else None,
                        vector_rank=candidate.vector_rank,
                    ),
                )
            )
        ranked.sort(
            key=lambda item: (
                -item[0],
                -item[1],
                item[2].chunk_id,
            )
        )
        return tuple(item[2] for item in ranked)

    def _expanded_query_terms(self, question: str) -> tuple[str, ...]:
        expanded: list[str] = []
        for token in self._tokenize(question):
            expanded.append(token)
            expanded.extend(self._ascii_compound_expansions.get(token, ()))
        return tuple(dict.fromkeys(expanded))

    @classmethod
    def _coverage(cls, query_terms: frozenset[str], target: str) -> float:
        if not query_terms:
            return 0.0
        target_terms = frozenset(cls._tokenize(target))
        return len(query_terms & target_terms) / len(query_terms)

    @staticmethod
    def _identifier_coverage(
        query_identifiers: frozenset[str],
        target: str,
    ) -> float:
        if not query_identifiers:
            return 0.0
        normalized_target = target.casefold()
        matched = sum(
            identifier in normalized_target
            for identifier in query_identifiers
        )
        return matched / len(query_identifiers)

    def _limit_chunk_ids_by_document(
        self,
        chunk_ids: Sequence[str] | Iterable[str],
        *,
        limit: int,
        max_chunks_per_document: int | None,
    ) -> tuple[str, ...]:
        selected: list[str] = []
        counts: Counter[str] = Counter()
        for chunk_id in chunk_ids:
            document_id = self._chunks_by_id[chunk_id].record.document_id
            if (
                max_chunks_per_document is not None
                and counts[document_id] >= max_chunks_per_document
            ):
                continue
            selected.append(chunk_id)
            counts[document_id] += 1
            if len(selected) >= limit:
                break
        return tuple(selected)

    def _limit_hits_by_document(
        self,
        hits: Sequence[KnowledgeSearchHit],
        *,
        limit: int,
        max_chunks_per_document: int | None,
    ) -> list[KnowledgeSearchHit]:
        selected: list[KnowledgeSearchHit] = []
        counts: Counter[str] = Counter()
        for hit in hits:
            document_id = self._chunks_by_id[hit.chunk_id].record.document_id
            if (
                max_chunks_per_document is not None
                and counts[document_id] >= max_chunks_per_document
            ):
                continue
            selected.append(hit)
            counts[document_id] += 1
            if len(selected) >= limit:
                break
        return selected

    @staticmethod
    def _normalize_document_ids(values: Sequence[str]) -> frozenset[str]:
        normalized: set[str] = set()
        for value in values:
            if not isinstance(value, str) or not value.strip():
                raise ValueError("知识文档范围 ID 不能为空")
            normalized.add(value.strip())
        return frozenset(normalized)

    def _normalize_vector(
        self,
        vector: Sequence[float],
    ) -> tuple[float, ...]:
        if self._embedding_dimensions is None or (
            len(vector) != self._embedding_dimensions
        ):
            raise ValueError("知识 Embedding 向量维度无效")
        if not all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            for value in vector
        ):
            raise ValueError("知识 Embedding 向量必须只包含有限数值")
        values = tuple(float(value) for value in vector)
        norm = math.sqrt(sum(value * value for value in values))
        if norm <= 0.0:
            raise ValueError("知识 Embedding 响应包含零向量")
        return tuple(value / norm for value in values)

    @classmethod
    def _tokenize(cls, text: str) -> list[str]:
        normalized_text = text.casefold()
        tokens: list[str] = []
        previous_ascii: tuple[str, int] | None = None
        for match in _TOKEN_PATTERN.finditer(normalized_text):
            value = match.group(0)
            if cls._is_chinese(value):
                tokens.append(value)
                if len(value) > 1:
                    tokens.extend(
                        value[index : index + 2]
                        for index in range(len(value) - 1)
                    )
                previous_ascii = None
            else:
                tokens.append(value)
                if previous_ascii is not None:
                    previous_value, previous_end = previous_ascii
                    separator = normalized_text[previous_end : match.start()]
                    if (
                        separator
                        and separator.isspace()
                        and "\n" not in separator
                        and "\r" not in separator
                    ):
                        tokens.append(previous_value + value)
                previous_ascii = (value, match.end())
        return tokens

    @classmethod
    def _build_ascii_compound_expansions(
        cls,
        records: tuple[KnowledgeChunkRecord, ...],
    ) -> dict[str, tuple[str, ...]]:
        expansions: dict[str, list[str]] = {}
        for record in records:
            for compound, left, right in cls._ascii_compounds(record.title):
                parts = expansions.setdefault(compound, [])
                for value in (left, right):
                    if value not in parts:
                        parts.append(value)
        return {
            compound: tuple(parts)
            for compound, parts in expansions.items()
        }

    @classmethod
    def _ascii_compounds(cls, text: str) -> tuple[tuple[str, str, str], ...]:
        normalized_text = text.casefold()
        compounds: list[tuple[str, str, str]] = []
        previous_ascii: tuple[str, int] | None = None
        for match in _TOKEN_PATTERN.finditer(normalized_text):
            value = match.group(0)
            if cls._is_chinese(value):
                previous_ascii = None
                continue
            if previous_ascii is not None:
                previous_value, previous_end = previous_ascii
                separator = normalized_text[previous_end : match.start()]
                if (
                    separator
                    and separator.isspace()
                    and "\n" not in separator
                    and "\r" not in separator
                ):
                    compounds.append(
                        (previous_value + value, previous_value, value)
                    )
            previous_ascii = (value, match.end())
        return tuple(compounds)

    @staticmethod
    def _is_chinese(value: str) -> bool:
        return bool(value) and all("\u3400" <= char <= "\u9fff" for char in value)

    @staticmethod
    def _heading_text(record: KnowledgeChunkRecord) -> str:
        heading = " > ".join(record.heading_path)
        return "\n".join(
            value for value in (record.title, heading) if value
        )

    @classmethod
    def _index_text(cls, record: KnowledgeChunkRecord) -> str:
        return "\n".join(
            value
            for value in (cls._heading_text(record), record.content)
            if value
        )

    @staticmethod
    def _is_resource_placeholder_only(content: str) -> bool:
        cleaned = content
        for pattern in _RESOURCE_ELEMENT_PATTERNS:
            cleaned = pattern.sub("", cleaned)
        cleaned = _RESOURCE_TAG_PATTERN.sub("", cleaned)
        return not cleaned.strip()

    @staticmethod
    def _fingerprint(records: tuple[KnowledgeChunkRecord, ...]) -> str:
        digest = hashlib.sha256()
        for record in records:
            digest.update(record.chunk_id.encode("utf-8"))
            digest.update(b"\0")
            digest.update(record.content_hash.encode("ascii"))
            digest.update(b"\0")
        return digest.hexdigest()


__all__ = ["EmbeddingClient", "InMemoryKnowledgeSearch"]
