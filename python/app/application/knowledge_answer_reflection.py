"""协调确定性风险检查与结构化知识答案反思。"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from typing import Protocol

from app.domain.services.knowledge_answer_reflection_policy import (
    KnowledgeAnswerReflectionPolicy,
)
from app.models.evidence_routing import EvidenceOption
from app.models.knowledge_qa import (
    KnowledgeChunkRecord,
    KnowledgeGeneratedAnswer,
    KnowledgeImageEvidence,
    KnowledgePlanCoverage,
    KnowledgeQuestionType,
)
from app.models.knowledge_reflection import (
    KnowledgeAnswerReflectionAnalysis,
    KnowledgeAnswerReflectionDecision,
    KnowledgeAnswerRiskSignals,
)


logger = logging.getLogger(__name__)


class KnowledgeAnswerReviewer(Protocol):
    """应用协调器依赖的结构化反思边界。"""

    async def review(
        self,
        *,
        question: str,
        standalone_query: str,
        answer: str,
        evidence: Sequence[KnowledgeChunkRecord],
        question_type: KnowledgeQuestionType,
        images: Sequence[KnowledgeImageEvidence] = (),
        coverage: KnowledgePlanCoverage | None = None,
    ) -> KnowledgeAnswerReflectionAnalysis:
        """检查当前答案草稿。"""

    async def aclose(self) -> None:
        """关闭反思资源。"""


class KnowledgeAnswerReflectionService:
    """每次请求先做确定性检查，必要时调用一次反思 Agent。"""

    def __init__(
        self,
        *,
        policy: KnowledgeAnswerReflectionPolicy,
        agent: KnowledgeAnswerReviewer | None,
    ) -> None:
        self._policy = policy
        self._agent = agent
        self._closed = False

    async def review(
        self,
        *,
        question: str,
        standalone_query: str,
        question_type: KnowledgeQuestionType,
        answer: KnowledgeGeneratedAnswer,
        evidence: Sequence[KnowledgeChunkRecord],
        images: Sequence[KnowledgeImageEvidence] = (),
        retrieval_degraded: bool = False,
        coverage: KnowledgePlanCoverage | None = None,
        retry_query: str | None = None,
        query_rewrite_attempted: bool = False,
        repair_attempted: bool = False,
        force_semantic_review: bool = False,
        trusted_options: Sequence[EvidenceOption] = (),
        allow_retrieval_retry: bool = True,
    ) -> KnowledgeAnswerReflectionDecision:
        """先执行硬边界检查，必要时只调用一次反思 Agent。"""

        protected_answer, records, image_records, signals = self._inputs(
            question_type=question_type,
            answer=answer,
            evidence=evidence,
            images=images,
            retrieval_degraded=retrieval_degraded,
            coverage=coverage,
            query_rewrite_attempted=query_rewrite_attempted,
            repair_attempted=repair_attempted,
            force_semantic_review=force_semantic_review,
        )
        prechecked = self._policy.precheck(signals)
        if prechecked is not None:
            return prechecked
        if self._agent is None:
            return self._policy.fallback(signals)
        try:
            analysis = await self._agent.review(
                question=question,
                standalone_query=standalone_query,
                answer=protected_answer.answer,
                evidence=records,
                question_type=question_type,
                images=image_records,
                coverage=coverage,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "知识答案反思模型不可用，按确定性结果继续",
                extra={"exception_type": type(exc).__name__},
            )
            return self._policy.fallback(signals)
        return self._policy.protect(
            analysis,
            signals=signals,
            retry_query=retry_query,
            trusted_options=trusted_options,
            allow_retrieval_retry=allow_retrieval_retry,
        )

    def validate_repaired(
        self,
        *,
        question_type: KnowledgeQuestionType,
        answer: KnowledgeGeneratedAnswer,
        evidence: Sequence[KnowledgeChunkRecord],
        images: Sequence[KnowledgeImageEvidence] = (),
        retrieval_degraded: bool = False,
        coverage: KnowledgePlanCoverage | None = None,
    ) -> KnowledgeAnswerReflectionDecision:
        """修复后只复检生成和引用硬边界，不再次调用模型。"""

        _, _, _, signals = self._inputs(
            question_type=question_type,
            answer=answer,
            evidence=evidence,
            images=images,
            retrieval_degraded=retrieval_degraded,
            coverage=coverage,
            query_rewrite_attempted=False,
            repair_attempted=True,
            force_semantic_review=False,
        )
        return self._policy.validate_repaired(signals)

    async def aclose(self) -> None:
        """关闭协调器拥有的反思 Agent。"""

        if self._closed:
            return
        self._closed = True
        if self._agent is not None:
            await self._agent.aclose()

    @staticmethod
    def _inputs(
        *,
        question_type: KnowledgeQuestionType,
        answer: KnowledgeGeneratedAnswer,
        evidence: Sequence[KnowledgeChunkRecord],
        images: Sequence[KnowledgeImageEvidence],
        retrieval_degraded: bool,
        coverage: KnowledgePlanCoverage | None,
        query_rewrite_attempted: bool,
        repair_attempted: bool,
        force_semantic_review: bool,
    ) -> tuple[
        KnowledgeGeneratedAnswer,
        tuple[KnowledgeChunkRecord, ...],
        tuple[KnowledgeImageEvidence, ...],
        KnowledgeAnswerRiskSignals,
    ]:
        protected_answer = KnowledgeGeneratedAnswer.model_validate(answer).model_copy(
            deep=True
        )
        records = tuple(
            KnowledgeChunkRecord.model_validate(record).model_copy(deep=True)
            for record in evidence
        )
        if not records:
            raise ValueError("知识答案反思至少需要一条批准 Evidence")
        image_records = tuple(
            KnowledgeImageEvidence.model_validate(image).model_copy(deep=True)
            for image in images
        )
        protected_coverage = (
            KnowledgePlanCoverage.model_validate(coverage).model_copy(deep=True)
            if coverage is not None
            else None
        )
        signals = KnowledgeAnswerRiskSignals(
            question_type=question_type,
            approved_chunk_ids=tuple(record.chunk_id for record in records),
            approved_image_ids=tuple(image.image_id for image in image_records),
            cited_chunk_ids=protected_answer.cited_chunk_ids,
            cited_image_ids=protected_answer.cited_image_ids,
            document_count=len(
                {record.document_id for record in records}
                | {image.document_id for image in image_records}
            ),
            answer_degraded=protected_answer.degraded,
            retrieval_degraded=retrieval_degraded,
            has_complex_coverage=protected_coverage is not None,
            query_rewrite_attempted=query_rewrite_attempted,
            repair_attempted=repair_attempted,
            force_semantic_review=force_semantic_review,
        )
        return protected_answer, records, image_records, signals


__all__ = [
    "KnowledgeAnswerReflectionService",
    "KnowledgeAnswerReviewer",
]
