"""在固定文档范围与 Chunk 快照内并行执行一个知识推理计划版本。"""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from app.models.knowledge_qa import (
    KnowledgeChunkRecord,
    KnowledgePlanCandidateRelation,
    KnowledgePlanCoverage,
    KnowledgePlanEvidenceRelation,
    KnowledgePlanReasonCode,
    KnowledgePlanStep,
    KnowledgeReasoningPlan,
    KnowledgeRetrievalDiagnostics,
    KnowledgeRetrievalMode,
    KnowledgeSearchResult,
)


class KnowledgePlanSearch(Protocol):
    """Executor 依赖的最小异步知识检索协议。"""

    async def search(
        self,
        question: str,
        *,
        limit: int = 5,
        document_ids: Sequence[str] = (),
    ) -> KnowledgeSearchResult:
        """在调用方指定范围内返回当前快照的候选 ID。"""

        ...


class KnowledgePlanReranker(Protocol):
    """Executor 依赖的单轮计划批量重排协议。"""

    async def rerank_plan(
        self,
        *,
        question: str,
        steps: Sequence[KnowledgePlanStep],
        candidates: Sequence[KnowledgeChunkRecord],
        relations: Sequence[KnowledgePlanCandidateRelation],
    ) -> object:
        """返回带 relations 和 degraded 属性的受保护批量结果。"""

        ...


@dataclass(frozen=True, slots=True)
class KnowledgePlanCachedQuery:
    """请求内同一查询的检索结果和通过快照回查的 Chunk。"""

    retrieval: KnowledgeSearchResult
    records: tuple[KnowledgeChunkRecord, ...]


@dataclass(slots=True)
class KnowledgePlanRequestCache:
    """只在一次问答请求内复用同范围、同快照查询结果。"""

    snapshot_fingerprint: str
    query_results: dict[
        tuple[str, tuple[str, ...], str],
        KnowledgePlanCachedQuery,
    ] = field(default_factory=dict)

    @classmethod
    def from_snapshot(
        cls,
        snapshot: Sequence[KnowledgeChunkRecord],
    ) -> KnowledgePlanRequestCache:
        """从已校验快照创建空请求缓存。"""

        records = tuple(
            KnowledgeChunkRecord.model_validate(record) for record in snapshot
        )
        return cls(snapshot_fingerprint=_snapshot_fingerprint(records))


@dataclass(frozen=True, slots=True)
class KnowledgePlanRoundOutcome:
    """单版计划执行后供 Coverage 与最终合并消费的不可变结果。"""

    plan: KnowledgeReasoningPlan
    records: tuple[KnowledgeChunkRecord, ...]
    relations: tuple[KnowledgePlanEvidenceRelation, ...]
    empty_reason_by_step: Mapping[str, KnowledgePlanReasonCode]
    search_queries: tuple[str, ...]
    retrieval_mode: KnowledgeRetrievalMode
    diagnostics: KnowledgeRetrievalDiagnostics
    rerank_degraded: bool


@dataclass(frozen=True, slots=True)
class KnowledgePlanMergedEvidence:
    """完成价值选择和预算保护后的最终 Chunk 白名单。"""

    records: tuple[KnowledgeChunkRecord, ...]
    scores: dict[str, float]
    supporting_step_ids: dict[str, tuple[str, ...]]


@dataclass(frozen=True, slots=True)
class _StepCandidates:
    step: KnowledgePlanStep
    records: tuple[KnowledgeChunkRecord, ...]
    scores: Mapping[str, float]
    empty_reason: KnowledgePlanReasonCode | None = None


class KnowledgePlanExecutor:
    """在单次请求的固定范围与快照内执行一个计划版本。"""

    _SEARCH_LIMIT = 20
    _MAX_STEP_CANDIDATES = 6
    _MAX_ROUND_CHUNKS = 20
    _MAX_ROUND_RELATIONS = 30

    def __init__(
        self,
        *,
        search: KnowledgePlanSearch,
        reranker: KnowledgePlanReranker,
    ) -> None:
        self._search = search
        self._reranker = reranker

    async def execute_round(
        self,
        plan: KnowledgeReasoningPlan,
        *,
        document_ids: Sequence[str],
        snapshot: Sequence[KnowledgeChunkRecord],
        cache: KnowledgePlanRequestCache,
        prior_outcome: KnowledgePlanRoundOutcome | None = None,
        reusable_step_ids: Sequence[str] = (),
    ) -> KnowledgePlanRoundOutcome:
        """并行执行未复用步骤，并对本轮新关系至多批量重排一次。"""

        normalized_plan = KnowledgeReasoningPlan.model_validate(plan).model_copy(
            deep=True
        )
        normalized_snapshot = self._snapshot(snapshot)
        snapshot_fingerprint = _snapshot_fingerprint(normalized_snapshot)
        if cache.snapshot_fingerprint != snapshot_fingerprint:
            raise ValueError("知识计划请求缓存与当前 Chunk 快照不一致")
        scope = self._normalize_scope(document_ids)
        records_by_id = {record.chunk_id: record for record in normalized_snapshot}
        reusable_ids = self._validate_reuse(
            normalized_plan,
            prior_outcome=prior_outcome,
            reusable_step_ids=reusable_step_ids,
        )
        reused_relations, reused_records, reused_reasons = self._reuse_prior(
            reusable_ids,
            prior_outcome=prior_outcome,
        )
        pending_steps = tuple(
            step
            for step in normalized_plan.steps
            if step.step_id not in reusable_ids
        )
        query_by_key = {
            self._normalized_query(step.query): step.query
            for step in pending_steps
        }
        await self._populate_cache(
            query_by_key,
            scope=scope,
            records_by_id=records_by_id,
            cache=cache,
        )
        step_candidates = tuple(
            self._step_candidates(
                step,
                scope=scope,
                records_by_id=records_by_id,
                cache=cache,
            )
            for step in pending_steps
        )
        new_records, candidate_relations = self._bounded_round_batch(
            step_candidates
        )
        if candidate_relations:
            rerank_outcome = await self._reranker.rerank_plan(
                question="；".join(step.query for step in pending_steps),
                steps=pending_steps,
                candidates=new_records,
                relations=candidate_relations,
            )
            new_relations = tuple(
                KnowledgePlanEvidenceRelation.model_validate(relation).model_copy(
                    deep=True
                )
                for relation in getattr(rerank_outcome, "relations")
            )
            rerank_degraded = bool(
                getattr(rerank_outcome, "degraded", False)
            )
        else:
            new_relations = ()
            rerank_degraded = False
        combined_relations = (*reused_relations, *new_relations)
        self._validate_output_relations(
            normalized_plan,
            combined_relations,
            records_by_id,
        )
        combined_records = self._combined_records(
            combined_relations,
            records_by_id,
            preferred_records=(*reused_records, *new_records),
        )
        empty_reasons = dict(reused_reasons)
        empty_reasons.update(
            {
                item.step.step_id: item.empty_reason
                for item in step_candidates
                if item.empty_reason is not None
            }
        )
        retrievals = tuple(
            self._cached_query(step.query, scope=scope, cache=cache).retrieval
            for step in pending_steps
        )
        return KnowledgePlanRoundOutcome(
            plan=normalized_plan,
            records=combined_records,
            relations=tuple(combined_relations),
            empty_reason_by_step=empty_reasons,
            search_queries=tuple(query_by_key.values()),
            retrieval_mode=self._retrieval_mode(retrievals),
            diagnostics=self._diagnostics(retrievals),
            rerank_degraded=rerank_degraded,
        )

    async def _populate_cache(
        self,
        query_by_key: Mapping[str, str],
        *,
        scope: tuple[str, ...],
        records_by_id: Mapping[str, KnowledgeChunkRecord],
        cache: KnowledgePlanRequestCache,
    ) -> None:
        missing_items = tuple(
            (query_key, query)
            for query_key, query in query_by_key.items()
            if self._cache_key(query_key, scope, cache) not in cache.query_results
        )
        if not missing_items:
            return
        raw_results = await asyncio.gather(
            *(
                self._search.search(
                    query,
                    limit=self._SEARCH_LIMIT,
                    document_ids=scope,
                )
                for _, query in missing_items
            ),
            return_exceptions=True,
        )
        for (query_key, _), raw_result in zip(
            missing_items,
            raw_results,
            strict=True,
        ):
            if isinstance(raw_result, asyncio.CancelledError):
                raise raw_result
            cache_key = self._cache_key(query_key, scope, cache)
            if isinstance(raw_result, BaseException):
                cache.query_results[cache_key] = KnowledgePlanCachedQuery(
                    retrieval=KnowledgeSearchResult(
                        diagnostics=KnowledgeRetrievalDiagnostics(
                            bm25_status="skipped",
                            vector_status="skipped",
                        )
                    ),
                    records=(),
                )
                continue
            retrieval = KnowledgeSearchResult.model_validate(raw_result).model_copy(
                deep=True
            )
            valid_records = self._valid_records(
                retrieval,
                scope=scope,
                records_by_id=records_by_id,
            )
            cache.query_results[cache_key] = KnowledgePlanCachedQuery(
                retrieval=retrieval,
                records=valid_records,
            )

    def _step_candidates(
        self,
        step: KnowledgePlanStep,
        *,
        scope: tuple[str, ...],
        records_by_id: Mapping[str, KnowledgeChunkRecord],
        cache: KnowledgePlanRequestCache,
    ) -> _StepCandidates:
        cached = self._cached_query(step.query, scope=scope, cache=cache)
        hit_scores = {hit.chunk_id: hit.score for hit in cached.retrieval.hits}
        records = tuple(
            sorted(
                cached.records,
                key=lambda record: (
                    -hit_scores.get(record.chunk_id, 0.0),
                    record.chunk_id,
                ),
            )[: self._MAX_STEP_CANDIDATES]
        )
        scores = {
            record.chunk_id: min(
                1.0,
                max(0.0, float(hit_scores[record.chunk_id])),
            )
            for record in records
        }
        empty_reason = None
        if not records:
            empty_reason = self._empty_reason(
                cached.retrieval,
                scope=scope,
                records_by_id=records_by_id,
            )
        return _StepCandidates(
            step=step,
            records=records,
            scores=scores,
            empty_reason=empty_reason,
        )

    @classmethod
    def _bounded_round_batch(
        cls,
        step_candidates: Sequence[_StepCandidates],
    ) -> tuple[
        tuple[KnowledgeChunkRecord, ...],
        tuple[KnowledgePlanCandidateRelation, ...],
    ]:
        records_by_id = {
            record.chunk_id: record
            for item in step_candidates
            for record in item.records
        }
        ranked_pairs = sorted(
            (
                (
                    item.step,
                    record,
                    float(item.scores[record.chunk_id]),
                )
                for item in step_candidates
                for record in item.records
            ),
            key=lambda item: (
                -int(item[0].required),
                -item[2],
                item[0].step_id,
                item[1].chunk_id,
            ),
        )
        selected_chunk_ids: list[str] = []
        selected_chunk_set: set[str] = set()
        relations: list[KnowledgePlanCandidateRelation] = []
        for step, record, score in ranked_pairs:
            if len(relations) >= cls._MAX_ROUND_RELATIONS:
                break
            if (
                record.chunk_id not in selected_chunk_set
                and len(selected_chunk_ids) >= cls._MAX_ROUND_CHUNKS
            ):
                continue
            if record.chunk_id not in selected_chunk_set:
                selected_chunk_ids.append(record.chunk_id)
                selected_chunk_set.add(record.chunk_id)
            relations.append(
                KnowledgePlanCandidateRelation(
                    step_id=step.step_id,
                    chunk_id=record.chunk_id,
                    deterministic_score=score,
                )
            )
        return (
            tuple(records_by_id[chunk_id] for chunk_id in selected_chunk_ids),
            tuple(relations),
        )

    @staticmethod
    def _snapshot(
        snapshot: Sequence[KnowledgeChunkRecord],
    ) -> tuple[KnowledgeChunkRecord, ...]:
        records = tuple(
            KnowledgeChunkRecord.model_validate(record).model_copy(deep=True)
            for record in snapshot
        )
        chunk_ids = [record.chunk_id for record in records]
        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("知识计划执行快照包含重复 Chunk ID")
        return records

    @staticmethod
    def _normalize_scope(document_ids: Sequence[str]) -> tuple[str, ...]:
        normalized: list[str] = []
        seen: set[str] = set()
        for value in document_ids:
            if not isinstance(value, str) or not value.strip():
                raise ValueError("知识计划文档范围 ID 不能为空")
            cleaned = value.strip()
            if cleaned not in seen:
                normalized.append(cleaned)
                seen.add(cleaned)
        return tuple(normalized)

    @classmethod
    def _validate_reuse(
        cls,
        plan: KnowledgeReasoningPlan,
        *,
        prior_outcome: KnowledgePlanRoundOutcome | None,
        reusable_step_ids: Sequence[str],
    ) -> frozenset[str]:
        reusable = frozenset(reusable_step_ids)
        if len(reusable) != len(tuple(reusable_step_ids)):
            raise ValueError("知识计划复用步骤 ID 不能重复")
        if not reusable:
            return reusable
        if prior_outcome is None or plan.revision != 2:
            raise ValueError("知识计划步骤复用只允许发生在第二版")
        if not reusable.issubset(plan.kept_step_ids):
            raise ValueError("知识计划复用步骤必须属于 kept_step_ids")
        prior_by_id = {
            step.step_id: step for step in prior_outcome.plan.steps
        }
        current_by_id = {step.step_id: step for step in plan.steps}
        for step_id in reusable:
            if (
                step_id not in prior_by_id
                or current_by_id[step_id] != prior_by_id[step_id]
            ):
                raise ValueError("知识计划复用步骤与首版字段不一致")
        return reusable

    @staticmethod
    def _reuse_prior(
        reusable_ids: frozenset[str],
        *,
        prior_outcome: KnowledgePlanRoundOutcome | None,
    ) -> tuple[
        tuple[KnowledgePlanEvidenceRelation, ...],
        tuple[KnowledgeChunkRecord, ...],
        dict[str, KnowledgePlanReasonCode],
    ]:
        if not reusable_ids or prior_outcome is None:
            return (), (), {}
        relations = tuple(
            relation.model_copy(deep=True)
            for relation in prior_outcome.relations
            if relation.step_id in reusable_ids
        )
        referenced_chunk_ids = {relation.chunk_id for relation in relations}
        records = tuple(
            record.model_copy(deep=True)
            for record in prior_outcome.records
            if record.chunk_id in referenced_chunk_ids
        )
        reasons = {
            step_id: reason
            for step_id, reason in prior_outcome.empty_reason_by_step.items()
            if step_id in reusable_ids
        }
        return relations, records, reasons

    @staticmethod
    def _valid_records(
        retrieval: KnowledgeSearchResult,
        *,
        scope: tuple[str, ...],
        records_by_id: Mapping[str, KnowledgeChunkRecord],
    ) -> tuple[KnowledgeChunkRecord, ...]:
        scope_set = set(scope)
        selected: list[KnowledgeChunkRecord] = []
        seen_content: set[tuple[str, str]] = set()
        for hit in retrieval.hits:
            record = records_by_id.get(hit.chunk_id)
            if record is None or record.content_hash != hit.content_hash:
                continue
            if scope_set and record.document_id not in scope_set:
                continue
            content_key = (record.document_id, record.content_hash)
            if content_key in seen_content:
                continue
            selected.append(record.model_copy(deep=True))
            seen_content.add(content_key)
        return tuple(selected)

    @staticmethod
    def _empty_reason(
        retrieval: KnowledgeSearchResult,
        *,
        scope: tuple[str, ...],
        records_by_id: Mapping[str, KnowledgeChunkRecord],
    ) -> KnowledgePlanReasonCode:
        if not retrieval.hits:
            return "search_failed" if (
                retrieval.diagnostics.bm25_status == "skipped"
                and retrieval.diagnostics.vector_status == "skipped"
            ) else "no_hits"
        scope_set = set(scope)
        saw_scope_filtered = False
        saw_stale = False
        for hit in retrieval.hits:
            record = records_by_id.get(hit.chunk_id)
            if record is None or record.content_hash != hit.content_hash:
                saw_stale = True
                continue
            if scope_set and record.document_id not in scope_set:
                saw_scope_filtered = True
        if saw_scope_filtered:
            return "scope_filtered"
        if saw_stale:
            return "stale_candidates"
        return "no_hits"

    @staticmethod
    def _combined_records(
        relations: Sequence[KnowledgePlanEvidenceRelation],
        records_by_id: Mapping[str, KnowledgeChunkRecord],
        *,
        preferred_records: Sequence[KnowledgeChunkRecord],
    ) -> tuple[KnowledgeChunkRecord, ...]:
        referenced_ids = {relation.chunk_id for relation in relations}
        result: list[KnowledgeChunkRecord] = []
        seen: set[str] = set()
        for record in preferred_records:
            if record.chunk_id in referenced_ids and record.chunk_id not in seen:
                result.append(records_by_id[record.chunk_id].model_copy(deep=True))
                seen.add(record.chunk_id)
        for chunk_id in sorted(referenced_ids - seen):
            result.append(records_by_id[chunk_id].model_copy(deep=True))
        return tuple(result)

    @staticmethod
    def _validate_output_relations(
        plan: KnowledgeReasoningPlan,
        relations: Sequence[KnowledgePlanEvidenceRelation],
        records_by_id: Mapping[str, KnowledgeChunkRecord],
    ) -> None:
        step_ids = {step.step_id for step in plan.steps}
        pairs: set[tuple[str, str]] = set()
        for relation in relations:
            pair = (relation.step_id, relation.chunk_id)
            if pair in pairs:
                raise ValueError("知识计划执行结果包含重复关系")
            pairs.add(pair)
            if (
                relation.step_id not in step_ids
                or relation.chunk_id not in records_by_id
            ):
                raise ValueError("知识计划执行结果引用未知步骤或 Chunk")

    @staticmethod
    def _retrieval_mode(
        retrievals: Sequence[KnowledgeSearchResult],
    ) -> KnowledgeRetrievalMode:
        return "hybrid" if any(
            retrieval.mode == "hybrid" for retrieval in retrievals
        ) else "bm25"

    @staticmethod
    def _diagnostics(
        retrievals: Sequence[KnowledgeSearchResult],
    ) -> KnowledgeRetrievalDiagnostics:
        if not retrievals:
            return KnowledgeRetrievalDiagnostics()
        vector_statuses = {
            retrieval.diagnostics.vector_status for retrieval in retrievals
        }
        if "degraded" in vector_statuses:
            vector_status = "degraded"
        elif "executed" in vector_statuses:
            vector_status = "executed"
        else:
            vector_status = "skipped"
        bm25_status = "degraded" if any(
            retrieval.diagnostics.bm25_status == "degraded"
            for retrieval in retrievals
        ) else "executed"
        return KnowledgeRetrievalDiagnostics(
            bm25_status=bm25_status,
            vector_status=vector_status,
        )

    @classmethod
    def _cached_query(
        cls,
        query: str,
        *,
        scope: tuple[str, ...],
        cache: KnowledgePlanRequestCache,
    ) -> KnowledgePlanCachedQuery:
        return cache.query_results[
            cls._cache_key(cls._normalized_query(query), scope, cache)
        ]

    @staticmethod
    def _cache_key(
        normalized_query: str,
        scope: tuple[str, ...],
        cache: KnowledgePlanRequestCache,
    ) -> tuple[str, tuple[str, ...], str]:
        return normalized_query, scope, cache.snapshot_fingerprint

    @staticmethod
    def _normalized_query(value: str) -> str:
        return " ".join(value.split()).casefold()


def _snapshot_fingerprint(records: Sequence[KnowledgeChunkRecord]) -> str:
    digest = hashlib.sha256()
    for record in sorted(
        records,
        key=lambda item: (item.chunk_id, item.content_hash),
    ):
        digest.update(record.chunk_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(record.content_hash.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def merge_evidence(
    plan: KnowledgeReasoningPlan,
    coverage: KnowledgePlanCoverage,
    round_outcomes: Sequence[KnowledgePlanRoundOutcome],
) -> KnowledgePlanMergedEvidence:
    """只合并 covered 步骤的 direct 关系并执行最终证据预算。"""

    normalized_plan = KnowledgeReasoningPlan.model_validate(plan).model_copy(
        deep=True
    )
    normalized_coverage = KnowledgePlanCoverage.model_validate(
        coverage
    ).model_copy(deep=True)
    outcomes = tuple(round_outcomes)
    if not outcomes:
        return KnowledgePlanMergedEvidence(
            records=(),
            scores={},
            supporting_step_ids={},
        )
    plan_steps_by_id = {step.step_id: step for step in normalized_plan.steps}
    covered_step_ids = {
        result.step_id
        for result in normalized_coverage.step_results
        if result.status == "covered"
    }
    if not covered_step_ids.issubset(plan_steps_by_id):
        raise ValueError("最终覆盖结果引用计划外步骤")
    records_by_id: dict[str, KnowledgeChunkRecord] = {}
    relation_by_pair: dict[
        tuple[str, str],
        KnowledgePlanEvidenceRelation,
    ] = {}
    for raw_outcome in outcomes:
        outcome = raw_outcome
        for record in outcome.records:
            normalized_record = KnowledgeChunkRecord.model_validate(
                record
            ).model_copy(deep=True)
            existing = records_by_id.get(normalized_record.chunk_id)
            if existing is not None and existing != normalized_record:
                raise ValueError("多轮计划结果中的同 ID Chunk 内容不一致")
            records_by_id[normalized_record.chunk_id] = normalized_record
        for raw_relation in outcome.relations:
            relation = KnowledgePlanEvidenceRelation.model_validate(
                raw_relation
            ).model_copy(deep=True)
            if (
                relation.step_id not in covered_step_ids
                or relation.support_level != "direct"
            ):
                continue
            if relation.chunk_id not in records_by_id:
                raise ValueError("最终证据关系引用未知 Chunk")
            pair = (relation.step_id, relation.chunk_id)
            existing_relation = relation_by_pair.get(pair)
            if (
                existing_relation is None
                or relation.score > existing_relation.score
            ):
                relation_by_pair[pair] = relation
    support_by_chunk: dict[str, dict[str, float]] = {}
    for relation in relation_by_pair.values():
        support_by_chunk.setdefault(relation.chunk_id, {})[
            relation.step_id
        ] = relation.score
    ranked_chunk_ids = sorted(
        support_by_chunk,
        key=lambda chunk_id: (
            -sum(
                plan_steps_by_id[step_id].required
                for step_id in support_by_chunk[chunk_id]
            ),
            -len(support_by_chunk[chunk_id]),
            -max(support_by_chunk[chunk_id].values()),
            chunk_id,
        ),
    )
    selected_chunk_ids: list[str] = []
    selected_tokens = 0
    document_counts: dict[str, int] = {}
    document_order: list[str] = []
    for chunk_id in ranked_chunk_ids:
        if len(selected_chunk_ids) >= 6:
            break
        record = records_by_id[chunk_id]
        if document_counts.get(record.document_id, 0) >= 3:
            continue
        if selected_tokens + record.token_count > 3000:
            continue
        selected_chunk_ids.append(chunk_id)
        selected_tokens += record.token_count
        document_counts[record.document_id] = (
            document_counts.get(record.document_id, 0) + 1
        )
        if record.document_id not in document_order:
            document_order.append(record.document_id)
    document_rank = {
        document_id: index for index, document_id in enumerate(document_order)
    }
    selected_records = tuple(
        sorted(
            (records_by_id[chunk_id] for chunk_id in selected_chunk_ids),
            key=lambda record: (
                document_rank[record.document_id],
                record.position,
                record.chunk_id,
            ),
        )
    )
    selected_id_set = set(selected_chunk_ids)
    return KnowledgePlanMergedEvidence(
        records=selected_records,
        scores={
            chunk_id: max(support_by_chunk[chunk_id].values())
            for chunk_id in selected_chunk_ids
        },
        supporting_step_ids={
            chunk_id: tuple(sorted(support_by_chunk[chunk_id]))
            for chunk_id in selected_chunk_ids
            if chunk_id in selected_id_set
        },
    )


__all__ = [
    "KnowledgePlanCachedQuery",
    "KnowledgePlanExecutor",
    "KnowledgePlanMergedEvidence",
    "KnowledgePlanRequestCache",
    "KnowledgePlanRoundOutcome",
    "merge_evidence",
]
