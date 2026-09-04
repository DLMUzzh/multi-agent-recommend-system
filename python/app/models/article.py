"""文档推荐候选、重排和展示结果契约。"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from app.models.common import AgentResult, _StrictModel
from app.models.document import DocumentContentType, DocumentDifficulty
from app.models.knowledge_qa import (
    KnowledgeRetrievalDiagnostics,
    KnowledgeRetrievalMode,
)


class DocumentCandidate(_StrictModel):
    """由 SQLite Chunk 召回恢复出的最小可信文档候选。"""

    document_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    topics: list[str] = Field(min_length=1, max_length=20)
    content_type: DocumentContentType
    difficulty: DocumentDifficulty
    author_id: str = Field(min_length=1)
    total_token_count: int = Field(ge=1)
    excerpt: str = Field(min_length=1, max_length=1200)
    matched_chunk_ids: list[str] = Field(min_length=1, max_length=4)
    recall_score: float = Field(ge=0.0, le=1.0)


class RankedDocument(DocumentCandidate):
    """文档候选经过可选 LLM 重排后的受保护结果。"""

    llm_score: float = Field(ge=0.0, le=1.0)
    relevance_score: float = Field(ge=0.0, le=1.0)
    profile_score: float = Field(ge=0.0, le=1.0)
    length_level: Literal["short", "medium", "long"]
    final_score: float = Field(ge=0.0, le=1.0)
    rerank_reason: str = Field(min_length=1, max_length=200)


class DocumentRecommendation(_StrictModel):
    """推荐链交给会话与 Controller 的最小真实文档结果。"""

    document_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    excerpt: str = Field(min_length=1, max_length=1200)
    score: float = Field(ge=0.0, le=1.0)
    reason: str = Field(min_length=1, max_length=200)
    recall_score: float | None = Field(default=None, ge=0.0, le=1.0)
    llm_score: float | None = Field(default=None, ge=0.0, le=1.0)
    profile_score: float | None = Field(default=None, ge=0.0, le=1.0)


class DocumentRecallResult(_StrictModel):
    """SQLite Chunk 推荐召回的候选和通道诊断。"""

    success: bool = True
    error: str | None = None
    candidates: list[DocumentCandidate] = Field(default_factory=list)
    retrieval_mode: KnowledgeRetrievalMode = "bm25"
    retrieval_diagnostics: KnowledgeRetrievalDiagnostics = Field(
        default_factory=KnowledgeRetrievalDiagnostics
    )


class DocumentRerankResult(AgentResult):
    """文档证据重排 Agent 的完整受保护结果。"""

    agent_name: str = "document_rerank"
    ranked_documents: list[RankedDocument] = Field(default_factory=list)
    degraded_reason: str | None = None


__all__ = [
    "DocumentCandidate",
    "DocumentRecallResult",
    "DocumentRecommendation",
    "DocumentRerankResult",
    "RankedDocument",
]
