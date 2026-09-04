"""用源文档事实执行无会话的相似文章推荐。"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import structlog

from app.agents.document_recall_agent import DocumentRecallAgent
from app.agents.document_rerank_agent import DocumentRerankAgent
from app.agents.user_profile_agent import UserProfileAgent
from app.application.conversation_service import ServiceUnavailableError
from app.domain.services.document_result_aggregator import (
    DocumentResultAggregator,
)
from app.infrastructure.database.json.feature_store import (
    UserNotFoundError,
)
from app.infrastructure.llm.client import llm_upgrade_scope
from app.models.article import (
    DocumentCandidate,
    DocumentRecommendation,
    RankedDocument,
)
from app.models.document import Document
from app.models.knowledge_qa import KnowledgeChunkRecord


logger = structlog.get_logger()


class SimilarDocumentUserStore(Protocol):
    """相似推荐前置用户校验的最小读取边界。"""

    async def get_user(self, user_id: str) -> dict[str, Any] | None:
        """读取用户基础事实，不存在时返回空。"""

        ...


class SimilarDocumentRepository(Protocol):
    """相似推荐生成查询需要的文档与 Chunk 读取边界。"""

    def get_document(self, document_id: str) -> Document | None:
        """读取源文档。"""

        ...

    def list_ready_chunks(
        self,
        document_ids: Sequence[str] = (),
    ) -> tuple[KnowledgeChunkRecord, ...]:
        """按稳定顺序读取源文档 Chunk。"""

        ...


@dataclass(frozen=True, slots=True)
class SimilarDocumentRecommendationResult:
    """应用服务交给 Controller 的最小结果。"""

    source_document_id: str
    recommendations: tuple[DocumentRecommendation, ...]
    agent_statuses: Mapping[str, str]


class DocumentNotFoundError(LookupError):
    """表示公开 `document_id` 不存在。"""


class SimilarDocumentRecommendationService:
    """复用对话推荐组件，执行独立相似文章用例。"""

    _RESULT_SIZE = 5
    _MAX_QUERY_LENGTH = 500

    def __init__(
        self,
        *,
        user_store: SimilarDocumentUserStore,
        repository: SimilarDocumentRepository,
        profile_agent: UserProfileAgent,
        recall_agent: DocumentRecallAgent,
        rerank_agent: DocumentRerankAgent,
        aggregator: DocumentResultAggregator,
    ) -> None:
        self._user_store = user_store
        self._repository = repository
        self._profile_agent = profile_agent
        self._recall_agent = recall_agent
        self._rerank_agent = rerank_agent
        self._aggregator = aggregator

    async def recommend(
        self,
        *,
        user_id: str,
        document_id: str,
    ) -> SimilarDocumentRecommendationResult:
        """返回最多五篇相关文档，不产生会话或行为副作用。"""

        normalized_user_id = self._required_text(user_id, "用户 ID")
        normalized_document_id = self._required_text(document_id, "文档 ID")
        user = await self._user_store.get_user(normalized_user_id)
        if user is None:
            raise UserNotFoundError(f"用户不存在：{normalized_user_id}")

        source_document = await asyncio.to_thread(
            self._repository.get_document,
            normalized_document_id,
        )
        if source_document is None:
            raise DocumentNotFoundError(normalized_document_id)
        source_chunks = await asyncio.to_thread(
            self._repository.list_ready_chunks,
            (normalized_document_id,),
        )
        query = self._project_query(source_document, source_chunks)

        profile_outcome, recall_outcome = await asyncio.gather(
            self._profile_agent.run(user_id=normalized_user_id),
            self._recall_agent.run(
                query=query,
                size=self._RESULT_SIZE,
                seen_document_ids=(normalized_document_id,),
            ),
            return_exceptions=True,
        )
        if isinstance(profile_outcome, asyncio.CancelledError):
            raise profile_outcome
        if isinstance(recall_outcome, asyncio.CancelledError):
            raise recall_outcome
        if isinstance(recall_outcome, BaseException) or not getattr(
            recall_outcome,
            "success",
            False,
        ):
            logger.error(
                "相似文章核心召回失败",
                exception_type=(
                    type(recall_outcome).__name__
                    if isinstance(recall_outcome, BaseException)
                    else type(getattr(recall_outcome, "error", None)).__name__
                ),
            )
            raise ServiceUnavailableError("文档召回暂时不可用")

        profile = None
        if not isinstance(profile_outcome, BaseException) and getattr(
            profile_outcome,
            "success",
            False,
        ):
            profile = getattr(profile_outcome, "profile", None)
        elif isinstance(profile_outcome, BaseException):
            logger.warning(
                "相似文章用户画像不可用，继续确定性重排",
                exception_type=type(profile_outcome).__name__,
            )
        profile_status = "success" if profile is not None else "failed"
        recall_status = self._result_status(recall_outcome)

        try:
            with llm_upgrade_scope(deadline=None):
                rerank_result = await self._rerank_agent.run(
                    query=query,
                    candidates=list(recall_outcome.candidates),
                    user_profile=profile,
                    current_topics=list(source_document.topics),
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "相似文章重排不可用，保留召回排序",
                exception_type=type(exc).__name__,
            )
            rerank_result = None

        rerank_status = self._result_status(rerank_result)
        rerank_documents = (
            list(getattr(rerank_result, "ranked_documents", ()))
            if rerank_result is not None
            else []
        )
        rerank_usable = bool(getattr(rerank_result, "success", False)) and (
            bool(rerank_documents) or not recall_outcome.candidates
        )
        ranked_documents = (
            rerank_documents
            if rerank_usable
            else self._fallback_ranked_documents(recall_outcome.candidates)
        )
        if rerank_status == "failed" or not rerank_usable:
            rerank_status = "degraded"
        aggregated = self._aggregator.aggregate(
            candidates=list(recall_outcome.candidates),
            ranked_documents=ranked_documents,
            seen_document_ids=(normalized_document_id,),
            size=self._RESULT_SIZE,
        )
        return SimilarDocumentRecommendationResult(
            source_document_id=normalized_document_id,
            recommendations=tuple(self._to_recommendation(item) for item in aggregated),
            agent_statuses={
                "user_profile": profile_status,
                "document_recall": recall_status,
                "document_rerank": rerank_status,
            },
        )

    @classmethod
    def _project_query(
        cls,
        document: Document,
        chunks: Sequence[KnowledgeChunkRecord],
    ) -> str:
        """将单篇文档事实确定性投影为一次召回查询。"""

        values: list[str] = [document.title, *document.topics]
        for chunk in sorted(chunks, key=lambda item: item.position):
            values.extend(chunk.heading_path)
            values.append(chunk.content)
        normalized: list[str] = []
        seen: set[str] = set()
        for value in values:
            cleaned = " ".join(value.split())
            key = cleaned.casefold()
            if cleaned and key not in seen:
                normalized.append(cleaned)
                seen.add(key)
        query = " ".join(normalized)[: cls._MAX_QUERY_LENGTH].strip()
        if not query:
            raise ServiceUnavailableError("源文档无法生成召回查询")
        return query

    @classmethod
    def _fallback_ranked_documents(
        cls,
        candidates: Sequence[DocumentCandidate],
    ) -> list[RankedDocument]:
        """在重排不可用时保留召回顺序和真实候选事实。"""

        return [
            RankedDocument(
                **candidate.model_dump(),
                llm_score=candidate.recall_score,
                relevance_score=candidate.recall_score,
                profile_score=0.0,
                length_level=cls._length_level(candidate.total_token_count),
                final_score=candidate.recall_score,
                rerank_reason=f"命中内容：{' '.join(candidate.excerpt.split())[:120]}",
            )
            for candidate in candidates
        ]

    @staticmethod
    def _to_recommendation(document: RankedDocument) -> DocumentRecommendation:
        return DocumentRecommendation(
            document_id=document.document_id,
            title=document.title,
            excerpt=document.excerpt,
            score=document.final_score,
            reason=document.rerank_reason,
            recall_score=document.recall_score,
            llm_score=document.llm_score,
            profile_score=document.profile_score,
        )

    @staticmethod
    def _result_status(result: Any) -> str:
        if result is None or not getattr(result, "success", False):
            return "failed"
        if getattr(result, "degraded_reason", None):
            return "degraded"
        data = getattr(result, "data", None) or {}
        if data.get("degraded_reason"):
            return "degraded"
        diagnostics = getattr(result, "retrieval_diagnostics", None)
        if getattr(diagnostics, "vector_status", None) in {"failed", "degraded"}:
            return "degraded"
        return "success"

    @staticmethod
    def _length_level(total_token_count: int) -> str:
        if total_token_count <= 800:
            return "short"
        if total_token_count <= 3000:
            return "medium"
        return "long"

    @staticmethod
    def _required_text(value: object, field_name: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field_name}不能为空")
        return " ".join(value.split())


__all__ = [
    "DocumentNotFoundError",
    "SimilarDocumentRecommendationResult",
    "SimilarDocumentRecommendationService",
]
