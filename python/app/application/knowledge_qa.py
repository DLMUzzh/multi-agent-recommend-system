"""知识文档导入、检索、回查和回答应用服务。"""

from __future__ import annotations

import asyncio
import hashlib
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import logging
from pathlib import Path
from typing import Literal, Protocol

from app.agents.knowledge_chunk_rerank_agent import (
    KnowledgeChunkRerankOutcome,
)
from app.agents.knowledge_answer_agent import KnowledgeAnswerAgent
from app.agents.knowledge_query_analysis_agent import KnowledgeQueryAnalysisAgent
from app.application.knowledge_plan_execution import (
    KnowledgePlanExecutor,
    KnowledgePlanRequestCache,
    KnowledgePlanRoundOutcome,
    merge_evidence,
)
from app.domain.services.knowledge_document_chunker import (
    KnowledgeDocumentChunker,
)
from app.domain.services.knowledge_document_preprocessor import (
    KnowledgeDocumentPreprocessor,
)
from app.domain.services.knowledge_document_media import (
    KnowledgeDocumentMediaExtractor,
)
from app.domain.services.knowledge_evidence_selector import (
    KnowledgeEvidenceSelector,
)
from app.domain.services.knowledge_evidence_gate import KnowledgeEvidenceGate
from app.domain.services.knowledge_scope_resolver import KnowledgeScopeResolver
from app.domain.services.runtime_skill_matcher import RuntimeSkillMatcher
from app.infrastructure.database.sqlite.knowledge_repository import (
    KnowledgeRepository,
)
from app.infrastructure.observability.conversation_trace import (
    current_conversation_stream,
    emit_stream_event,
)
from app.infrastructure.llm.client import llm_upgrade_scope
from app.infrastructure.retrieval.knowledge_search import InMemoryKnowledgeSearch
from app.models.document import Document, ImageMimeType
from app.models.evidence_routing import (
    EvidenceOption,
    EvidenceSignals,
    KnowledgeEvidenceDecision,
)
from app.models.conversation import ConversationTurn
from app.models.knowledge_qa import (
    KnowledgeAnswerResult,
    KnowledgeChunkRecord,
    KnowledgeCitation,
    KnowledgeDocumentIngestResult,
    KnowledgeDocumentEvidence,
    KnowledgeExecutionChunk,
    KnowledgeExecutionDocument,
    KnowledgeExecutionInput,
    KnowledgeExecutionResult,
    KnowledgeExecutionTrace,
    KnowledgeGeneratedAnswer,
    KnowledgeImageReference,
    KnowledgeImageEvidence,
    KnowledgeImageCitation,
    KnowledgeImageUploadResult,
    KnowledgePlanCoverage,
    KnowledgePlanEvidenceRelation,
    KnowledgePlanReasonCode,
    KnowledgePlanStepResult,
    KnowledgePlanTraceStep,
    KnowledgeQueryAnalysis,
    KnowledgeQuestionType,
    KnowledgeReasoningPlan,
    KnowledgeReasoningStrategy,
    KnowledgeRetrievalDiagnostics,
    KnowledgeScopeResolution,
    KnowledgeSearchHit,
    KnowledgeSearchResult,
)
from app.models.interaction_memory import UserInteractionMemoryProjection
from app.models.knowledge_reflection import (
    KnowledgeAnswerReflectionDecision,
    KnowledgeAnswerRevisionPolicy,
)
from app.models.runtime_skill import (
    CompiledRuntimeSkill,
    RuntimeSkillMatchResult,
    RuntimeSkillResponsePolicy,
    RuntimeSkillSnapshot,
)


logger = logging.getLogger(__name__)


class KnowledgeSearch(Protocol):
    """应用服务依赖的可刷新知识检索边界。"""

    async def refresh(self, chunks: Sequence[KnowledgeChunkRecord]) -> None:
        """从 SQLite 快照刷新派生索引。"""

        ...

    async def search(
        self,
        question: str,
        *,
        limit: int = 5,
        document_ids: Sequence[str] = (),
    ) -> KnowledgeSearchResult:
        """检索当前问题的候选 Chunk。"""

        ...

    async def aclose(self) -> None:
        """关闭检索资源。"""

        ...


class KnowledgeAnswerGenerator(Protocol):
    """应用服务依赖的证据回答边界。"""

    async def generate(
        self,
        *,
        question: str,
        evidence: Sequence[KnowledgeChunkRecord],
        standalone_query: str | None = None,
        history: Sequence[ConversationTurn] = (),
        conversation_summary: str | None = None,
        interaction_memory: UserInteractionMemoryProjection | None = None,
        images: Sequence[KnowledgeImageEvidence] = (),
        response_policy: RuntimeSkillResponsePolicy | None = None,
        revision_policy: KnowledgeAnswerRevisionPolicy | None = None,
    ) -> KnowledgeGeneratedAnswer:
        """根据可信证据生成答案。"""

        ...

    async def aclose(self) -> None:
        """关闭回答资源。"""

        ...


class KnowledgeAnswerReflectionReviewer(Protocol):
    """应用服务依赖的知识答案反思边界。"""

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
        """返回受保护的五类草稿决策。"""

        ...

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
        """修复后只执行确定性复检。"""

        ...

    async def aclose(self) -> None:
        """关闭反思资源。"""

        ...


class KnowledgeQueryAnalyzer(Protocol):
    """应用服务依赖的一次性知识查询分析边界。"""

    async def analyze(
        self,
        question: str,
        *,
        history: Sequence[ConversationTurn] = (),
        conversation_summary: str | None = None,
    ) -> KnowledgeQueryAnalysis:
        """返回独立查询、问题类型和直接或分解检索计划。"""

        ...

    async def aclose(self) -> None:
        """关闭查询分析资源。"""

        ...


class KnowledgeReasoningPlanner(Protocol):
    """应用服务依赖的复杂问题首版规划与唯一一次修订边界。"""

    async def plan(
        self,
        *,
        standalone_query: str,
        question_type: Literal["comparative", "analytical", "exploratory"],
        sub_queries: Sequence[str] = (),
    ) -> KnowledgeReasoningPlan:
        """生成复杂问题首版计划。"""

        ...

    async def replan(
        self,
        *,
        standalone_query: str,
        question_type: Literal["comparative", "analytical", "exploratory"],
        previous_plan: KnowledgeReasoningPlan,
        step_results: Sequence[KnowledgePlanStepResult],
        remaining_step_limit: int,
    ) -> KnowledgeReasoningPlan:
        """根据安全覆盖结果修订一次计划。"""

        ...

    async def aclose(self) -> None:
        """关闭 Planner 资源。"""

        ...


class KnowledgePlanCoverageService(Protocol):
    """应用服务依赖的确定性计划覆盖检查边界。"""

    def evaluate(
        self,
        plan: KnowledgeReasoningPlan,
        *,
        relations: Sequence[KnowledgePlanEvidenceRelation],
        records: Sequence[KnowledgeChunkRecord],
        empty_reason_by_step: Mapping[str, KnowledgePlanReasonCode],
        replanned: bool,
        allow_replan: bool,
    ) -> KnowledgePlanCoverage:
        """返回严格步骤状态与下一动作。"""

        ...


class KnowledgeScopeResolutionService(Protocol):
    """应用服务依赖的当前轮知识范围解析边界。"""

    def resolve(
        self,
        question: str,
        *,
        history: Sequence[ConversationTurn],
        documents: Sequence[tuple[str, str]],
    ) -> KnowledgeScopeResolution:
        """解析临时文档范围或返回澄清。"""

        ...


class RuntimeSkillSnapshotProvider(Protocol):
    """应用服务依赖的请求级 Skill Snapshot 边界。"""

    def capture_snapshot(self) -> RuntimeSkillSnapshot:
        """返回当前不可变 Snapshot。"""

        ...


class RuntimeSkillMatchingService(Protocol):
    """应用服务依赖的确定性 Skill 匹配边界。"""

    def match(
        self,
        question: str,
        *,
        skills: Sequence[object],
        document_ids: Sequence[str] = (),
        document_topics_by_id: Mapping[str, Sequence[str]] | None = None,
    ) -> RuntimeSkillMatchResult:
        """返回唯一 Skill、可信并列或范围冲突。"""

        ...


class KnowledgeChunkReranker(Protocol):
    """应用服务依赖的 QA Passage 重排边界。"""

    async def rerank(
        self,
        *,
        query: str,
        candidates: Sequence[KnowledgeChunkRecord],
        deterministic_scores: dict[str, float],
    ) -> KnowledgeChunkRerankOutcome:
        """批量重排 SQLite 回查后的真实 Passage。"""

        ...

    async def aclose(self) -> None:
        """关闭重排资源。"""

        ...


class StoredKnowledgeImage(Protocol):
    """图片 Store 写入结果的最小结构契约。"""

    content_hash: str
    storage_key: str
    mime_type: ImageMimeType
    byte_size: int


class KnowledgeImageStore(Protocol):
    """可由本地文件或 OSS 实现替换的图片二进制边界。"""

    def put(self, content: bytes, mime_type: str) -> StoredKnowledgeImage:
        """校验并写入图片，返回安全存储元数据。"""

        ...

    def resolve(self, storage_key: str) -> Path:
        """把内部相对 Key 解析为当前实现可读取的位置。"""

        ...

    def delete_unreferenced(self, referenced_keys: Sequence[str]) -> int:
        """删除当前实现内未被事实表引用的合法对象。"""

        ...


class KnowledgeExecutionRecordWriter(Protocol):
    """运行期测试记录的可替换异步边界。"""

    async def append(self, trace: KnowledgeExecutionTrace) -> bool:
        """追加安全执行摘要；失败不得抛给问答调用方。"""

        ...


@dataclass(frozen=True, slots=True)
class KnowledgeImageFile:
    """Controller 可读取但不应直接公开路径的图片文件。"""

    path: Path
    mime_type: ImageMimeType
    content_hash: str


@dataclass(frozen=True, slots=True)
class _KnowledgeSimpleRetrievalStage:
    """简单知识链一次检索、回查、重排和证据选择结果。"""

    retrieval: KnowledgeSearchResult
    snapshot: tuple[KnowledgeChunkRecord, ...]
    plan_search_degraded: bool
    rerank: KnowledgeChunkRerankOutcome
    deterministic_scores: dict[str, float]
    evidence: tuple[KnowledgeChunkRecord, ...]


class KnowledgeQaService:
    """协调知识问答闭环，不拥有或持久化聊天会话状态。"""

    _PLAN_RRF_K = 60
    _RETRIEVAL_LIMIT = 20
    _ABSTAIN_ANSWER = "当前证据不足，无法可靠回答该问题。"

    def __init__(
        self,
        *,
        repository: KnowledgeRepository,
        search: KnowledgeSearch | None = None,
        answer_agent: KnowledgeAnswerGenerator | None = None,
        query_analysis_agent: KnowledgeQueryAnalyzer | None = None,
        scope_resolver: KnowledgeScopeResolutionService | None = None,
        evidence_selector: KnowledgeEvidenceSelector | None = None,
        chunker: KnowledgeDocumentChunker | None = None,
        preprocessor: KnowledgeDocumentPreprocessor | None = None,
        media_extractor: KnowledgeDocumentMediaExtractor | None = None,
        chunk_rerank_agent: KnowledgeChunkReranker | None = None,
        image_store: KnowledgeImageStore | None = None,
        execution_record_writer: KnowledgeExecutionRecordWriter | None = None,
        reasoning_planner_agent: KnowledgeReasoningPlanner | None = None,
        plan_executor: KnowledgePlanExecutor | None = None,
        plan_coverage_checker: KnowledgePlanCoverageService | None = None,
        runtime_skill_registry: RuntimeSkillSnapshotProvider | None = None,
        runtime_skill_matcher: RuntimeSkillMatchingService | None = None,
        evidence_gate: KnowledgeEvidenceGate | None = None,
        reflection_service: KnowledgeAnswerReflectionReviewer | None = None,
        request_timeout_seconds: float | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if request_timeout_seconds is not None and request_timeout_seconds <= 0:
            raise ValueError("知识问答请求超时必须大于零")
        self._repository = repository
        self._search = search or InMemoryKnowledgeSearch()
        self._answer_agent = answer_agent or KnowledgeAnswerAgent(llm=None)
        self._query_analysis_agent = (
            query_analysis_agent or KnowledgeQueryAnalysisAgent(llm=None)
        )
        self._scope_resolver = scope_resolver or KnowledgeScopeResolver()
        self._evidence_selector = evidence_selector or KnowledgeEvidenceSelector()
        self._chunker = chunker or KnowledgeDocumentChunker()
        self._preprocessor = preprocessor or KnowledgeDocumentPreprocessor()
        self._media_extractor = media_extractor or KnowledgeDocumentMediaExtractor(
            preprocessor=self._preprocessor,
            chunker=self._chunker,
        )
        self._chunk_rerank_agent = chunk_rerank_agent
        self._image_store = image_store
        self._execution_record_writer = execution_record_writer
        self._reasoning_planner_agent = reasoning_planner_agent
        self._plan_executor = plan_executor
        self._plan_coverage_checker = plan_coverage_checker
        self._runtime_skill_registry = runtime_skill_registry
        self._runtime_skill_matcher = runtime_skill_matcher or RuntimeSkillMatcher()
        self._evidence_gate = evidence_gate or KnowledgeEvidenceGate()
        self._reflection_service = reflection_service
        self._request_timeout_seconds = request_timeout_seconds
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._index_lock = asyncio.Lock()
        self._closed = False

    async def ingest_document(
        self,
        *,
        document_id: str,
        title: str,
        content_markdown: str,
        topics: Sequence[str],
        content_type: str,
        difficulty: str,
        author_id: str,
    ) -> KnowledgeDocumentIngestResult:
        """使用切分策略原子写入文档，并刷新当前服务实例的派生索引。"""
        normalized_id = self._required_text(document_id, "document_id")
        normalized_title = self._required_text(title, "文档标题")
        if not isinstance(content_markdown, str) or not content_markdown.strip():
            raise ValueError("文档正文不能为空")
        derivation = self._media_extractor.derive(
            document_id=normalized_id,
            content_markdown=content_markdown,
        )
        chunks = derivation.chunks
        if not chunks:
            raise ValueError("文档正文没有可切分内容")
        now = self._clock()
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("clock 必须返回带时区的时间")
        content_hash = hashlib.sha256(content_markdown.encode("utf-8")).hexdigest()

        async with self._index_lock:
            existing = await asyncio.to_thread(
                self._repository.get_document,
                normalized_id,
            )
            document = Document(
                document_id=normalized_id,
                title=normalized_title,
                content_markdown=content_markdown,
                topics=list(topics),
                content_type=content_type,
                difficulty=difficulty,
                author_id=self._required_text(author_id, "author_id"),
                content_hash=content_hash,
                created_at=existing.created_at if existing is not None else now,
                updated_at=now,
            )
            await asyncio.to_thread(
                self._repository.replace_document_bundle,
                document,
                chunks,
                derivation.images,
                derivation.links,
            )
            persisted_images = tuple(
                await asyncio.gather(
                    *(
                        asyncio.to_thread(
                            self._repository.get_image,
                            declaration.image_id,
                        )
                        for declaration in derivation.images
                    )
                )
            )
            snapshot = await asyncio.to_thread(self._repository.list_ready_chunks)
            await self._search.refresh(snapshot)

        return KnowledgeDocumentIngestResult(
            document_id=normalized_id,
            title=normalized_title,
            content_hash=content_hash,
            chunk_count=len(chunks),
            images=tuple(
                KnowledgeImageReference(
                    image_id=image.image_id,
                    image_key=image.image_key,
                    caption=image.caption,
                    status=image.status,
                )
                for image in persisted_images
                if image is not None
            ),
        )

    async def upload_image(
        self,
        *,
        image_id: str,
        content: bytes,
        mime_type: str,
    ) -> KnowledgeImageUploadResult:
        """保存文档已声明的图片，并在完整写入后更新 ready 状态。"""

        normalized_id = self._required_text(image_id, "image_id")
        if self._image_store is None:
            raise RuntimeError("知识图片存储未配置")
        declared = await asyncio.to_thread(
            self._repository.get_image,
            normalized_id,
        )
        if declared is None:
            raise ValueError("图片不存在")

        stored = await asyncio.to_thread(
            self._image_store.put,
            content,
            mime_type,
        )
        ready = await asyncio.to_thread(
            self._repository.mark_image_ready,
            image_id=normalized_id,
            content_hash=stored.content_hash,
            storage_key=stored.storage_key,
            mime_type=stored.mime_type,
            byte_size=stored.byte_size,
        )
        return KnowledgeImageUploadResult(
            image_id=ready.image_id,
            content_hash=stored.content_hash,
            mime_type=stored.mime_type,
            byte_size=stored.byte_size,
        )

    def get_image_file(self, image_id: str) -> KnowledgeImageFile | None:
        """回查 ready 图片及安全文件；pending 或文件缺失时返回空。"""

        normalized_id = self._required_text(image_id, "image_id")
        if self._image_store is None:
            return None
        image = self._repository.get_image(normalized_id)
        if (
            image is None
            or image.status != "ready"
            or image.storage_key is None
            or image.mime_type is None
            or image.content_hash is None
        ):
            return None
        try:
            path = self._image_store.resolve(image.storage_key)
        except ValueError:
            return None
        if not path.is_file():
            return None
        return KnowledgeImageFile(
            path=path,
            mime_type=image.mime_type,
            content_hash=image.content_hash,
        )

    async def ask(
        self,
        question: str,
        *,
        limit: int = 5,
        history: Sequence[ConversationTurn] = (),
        conversation_summary: str | None = None,
        document_ids: Sequence[str] = (),
        prepared_query: str | None = None,
        interaction_memory: UserInteractionMemoryProjection | None = None,
        request_route: Literal[
            "/api/v1/knowledge/ask",
            "/api/v1/chat",
        ] = "/api/v1/knowledge/ask",
    ) -> KnowledgeAnswerResult:
        """在统一请求预算内执行一次知识问答。"""

        timeout = self._request_timeout_seconds
        if timeout is None:
            with llm_upgrade_scope(deadline=None):
                return await self._ask_with_deadline(
                    question,
                    limit=limit,
                    history=history,
                    conversation_summary=conversation_summary,
                    document_ids=document_ids,
                    prepared_query=prepared_query,
                    interaction_memory=interaction_memory,
                    request_route=request_route,
                    deadline=None,
                )
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        with llm_upgrade_scope(deadline=deadline):
            return await asyncio.wait_for(
                self._ask_with_deadline(
                    question,
                    limit=limit,
                    history=history,
                    conversation_summary=conversation_summary,
                    document_ids=document_ids,
                    prepared_query=prepared_query,
                    interaction_memory=interaction_memory,
                    request_route=request_route,
                    deadline=deadline,
                ),
                timeout=timeout,
            )

    async def regenerate_from_evidence(
        self,
        question: str,
        *,
        chunk_ids: Sequence[str],
        interaction_memory: UserInteractionMemoryProjection | None = None,
        request_route: Literal["/api/v1/chat"] = "/api/v1/chat",
    ) -> KnowledgeAnswerResult:
        """重新回查可信 Chunk 并生成答案，不调用 Search 或查询分析 Agent。"""

        normalized_question = self._required_text(question, "知识问题")
        if request_route != "/api/v1/chat":
            raise ValueError("证据再回答只允许会话补救路径")
        normalized_ids = tuple(
            dict.fromkeys(
                self._required_text(chunk_id, "chunk_id") for chunk_id in chunk_ids
            )
        )
        if not normalized_ids or len(normalized_ids) > 20:
            raise ValueError("证据 Chunk 数量必须在 1 到 20 之间")
        records = await asyncio.to_thread(
            self._repository.get_chunks_by_ids,
            normalized_ids,
        )
        if tuple(record.chunk_id for record in records) != normalized_ids:
            return KnowledgeAnswerResult(
                status="insufficient_evidence",
                answer="原回答使用的部分知识证据已失效，请重新检索。",
                citations=(),
            )
        protected_interaction_memory = self._protect_interaction_memory(
            interaction_memory
        )
        linked_images = await asyncio.to_thread(
            self._repository.list_ready_images_by_chunk_ids,
            normalized_ids,
        )
        generated = await self._answer_agent.generate(
            question=normalized_question,
            standalone_query=normalized_question,
            evidence=records,
            interaction_memory=protected_interaction_memory,
            images=linked_images,
        )
        if generated.outcome == "abstain":
            return self._abstain_result(generated)
        reflection_decision = await self._review_generated_answer(
            question=normalized_question,
            standalone_query=normalized_question,
            question_type="factual",
            generated=generated,
            evidence=records,
            images=linked_images,
            repair_attempted=True,
            force_semantic_review=True,
            allow_retrieval_retry=False,
        )
        if reflection_decision.action == "rewrite":
            reflection_decision = KnowledgeAnswerReflectionDecision(
                action="refuse",
                confidence=1.0,
                reason_code="repair_exhausted",
            )
        if reflection_decision.action in {"ask", "select", "refuse"}:
            return self._reflection_result(reflection_decision)
        records_by_id = {record.chunk_id: record for record in records}
        cited_records = tuple(
            records_by_id[chunk_id]
            for chunk_id in generated.cited_chunk_ids
            if chunk_id in records_by_id
        )
        if not cited_records:
            return KnowledgeAnswerResult(
                status="insufficient_evidence",
                answer="原证据未能支持修正后的回答，请重新检索。",
                citations=(),
            )
        citation_numbers: dict[str, int] = {}
        citations: list[KnowledgeCitation] = []
        for record in cited_records:
            citation_id = citation_numbers.get(record.document_id)
            if citation_id is None:
                citation_id = len(citation_numbers) + 1
                citation_numbers[record.document_id] = citation_id
            citations.append(self._citation(record, citation_id=str(citation_id)))
        return KnowledgeAnswerResult(
            status="degraded" if generated.degraded else "success",
            answer=generated.answer,
            citations=tuple(citations),
            degraded_components=("answer",) if generated.degraded else (),
            resolved_document_ids=tuple(
                dict.fromkeys(record.document_id for record in cited_records)
            ),
            resolved_document_titles=tuple(
                dict.fromkeys(record.title for record in cited_records)
            ),
        )

    async def _ask_with_deadline(
        self,
        question: str,
        *,
        limit: int,
        history: Sequence[ConversationTurn],
        conversation_summary: str | None,
        document_ids: Sequence[str],
        prepared_query: str | None,
        interaction_memory: UserInteractionMemoryProjection | None,
        request_route: Literal[
            "/api/v1/knowledge/ask",
            "/api/v1/chat",
        ],
        deadline: float | None,
    ) -> KnowledgeAnswerResult:
        """读取当前文档集合，并可直接消费聊天路由预生成的受保护查询。"""

        started = time.perf_counter()
        trace_id = uuid.uuid4().hex
        normalized_question = self._required_text(question, "知识问题")
        normalized_prepared_query = (
            self._required_query(prepared_query, "预生成知识查询")
            if prepared_query is not None
            else None
        )
        protected_interaction_memory = self._protect_interaction_memory(
            interaction_memory
        )
        emit_stream_event(
            stage="知识问答",
            component="knowledge_qa_service",
            status="started",
            title="知识问答链开始",
            summary="准备范围解析、查询分析和 Chunk 检索",
            details={
                "question": normalized_question,
                "history_message_count": min(len(tuple(history)), 12),
                "has_conversation_summary": bool(conversation_summary),
                "prepared_query": normalized_prepared_query,
                "requested_document_ids": tuple(document_ids)[:20],
            },
        )
        if limit <= 0:
            raise ValueError("limit 必须大于零")
        if request_route not in {
            "/api/v1/knowledge/ask",
            "/api/v1/chat",
        }:
            raise ValueError("知识问答请求路由无效")
        initial_snapshot = await asyncio.to_thread(self._repository.list_ready_chunks)
        documents = tuple(
            dict.fromkeys(
                (record.document_id, record.title) for record in initial_snapshot
            )
        )
        titles_by_document_id = {document_id: title for document_id, title in documents}
        forced_document_ids = tuple(
            dict.fromkeys(
                self._required_text(document_id, "document_id")
                for document_id in document_ids
            )
        )
        if forced_document_ids:
            if any(
                document_id not in titles_by_document_id
                for document_id in forced_document_ids
            ):
                return await self._finalize_answer(
                    KnowledgeAnswerResult(
                        status="needs_clarification",
                        answer="指定的知识文档不存在或尚未就绪。",
                        citations=(),
                    ),
                    trace_id=trace_id,
                    started=started,
                    question=normalized_question,
                    request_route=request_route,
                    history=history,
                    conversation_summary=conversation_summary,
                    prepared_query=normalized_prepared_query,
                    requested_document_ids=forced_document_ids,
                )
        runtime_skill_match = self._match_runtime_skill(
            normalized_question,
            snapshot=initial_snapshot,
            document_ids=forced_document_ids,
        )
        runtime_skill = (
            runtime_skill_match.primary.skill
            if runtime_skill_match.primary is not None
            else None
        )
        skill_scope_ids = (
            runtime_skill_match.primary.resolved_document_ids
            if runtime_skill_match.primary is not None
            else ()
        )
        skill_decision = self._evidence_gate.precheck(
            skill_scope_conflict=runtime_skill_match.scope_conflict,
            skill_candidates=tuple(
                EvidenceOption(
                    option_id=candidate.skill.skill_id,
                    label=(f"{candidate.skill.skill_id} ({candidate.skill.version})"),
                )
                for candidate in runtime_skill_match.candidates
            ),
            scope_resolved=not runtime_skill_match.too_many_candidates,
            clarification_question=(
                "匹配到较多运行时 Skill，请补充更明确的领域或对象。"
                if runtime_skill_match.too_many_candidates
                else None
            ),
        )
        if skill_decision is not None:
            return await self._finalize_answer(
                self._gate_answer_result(
                    skill_decision,
                    resolved_document_ids=forced_document_ids,
                    resolved_document_titles=tuple(
                        titles_by_document_id[document_id]
                        for document_id in forced_document_ids
                    ),
                ),
                trace_id=trace_id,
                started=started,
                question=normalized_question,
                request_route=request_route,
                history=history,
                conversation_summary=conversation_summary,
                prepared_query=normalized_prepared_query,
                requested_document_ids=forced_document_ids,
            )
        if (
            runtime_skill is not None
            and (runtime_skill.document_ids or runtime_skill.document_topics)
            and not skill_scope_ids
        ):
            conflict = self._evidence_gate.precheck(skill_scope_conflict=True)
            if conflict is None:
                raise RuntimeError("Skill 范围冲突必须形成门控结果")
            return await self._finalize_answer(
                self._gate_answer_result(conflict),
                trace_id=trace_id,
                started=started,
                question=normalized_question,
                request_route=request_route,
                history=history,
                conversation_summary=conversation_summary,
                prepared_query=normalized_prepared_query,
                requested_document_ids=forced_document_ids,
            )
        if forced_document_ids:
            scope = KnowledgeScopeResolution(document_ids=forced_document_ids)
        else:
            scoped_documents = (
                tuple(item for item in documents if item[0] in skill_scope_ids)
                if skill_scope_ids
                else documents
            )
            scope = self._scope_resolver.resolve(
                normalized_question,
                history=history,
                documents=scoped_documents,
            )
            if (
                not scope.needs_clarification
                and not scope.document_ids
                and skill_scope_ids
            ):
                scope = KnowledgeScopeResolution(document_ids=skill_scope_ids)
        emit_stream_event(
            stage="范围解析",
            component="knowledge_scope_resolver",
            status=("degraded" if scope.needs_clarification else "success"),
            title="知识范围解析完成",
            summary=(
                "需要补充文档范围"
                if scope.needs_clarification
                else f"限定 {len(scope.document_ids)} 篇文档"
                if scope.document_ids
                else "检索全知识库"
            ),
            details={
                "resolved_document_ids": scope.document_ids,
                "needs_clarification": scope.needs_clarification,
                "clarification_question": scope.clarification_question,
                "candidate_document_ids": getattr(
                    scope,
                    "candidate_document_ids",
                    (),
                ),
            },
        )
        resolved_titles = tuple(
            titles_by_document_id[document_id]
            for document_id in scope.document_ids
            if document_id in titles_by_document_id
        )
        scope_decision = self._evidence_gate.precheck(
            scope_candidates=tuple(
                EvidenceOption(
                    option_id=document_id,
                    label=(
                        f"{titles_by_document_id.get(document_id, document_id)} "
                        f"({document_id})"
                    ),
                )
                for document_id in getattr(
                    scope,
                    "candidate_document_ids",
                    (),
                )
            ),
            scope_resolved=not scope.needs_clarification,
            clarification_question=scope.clarification_question,
        )
        if scope_decision is not None:
            return await self._finalize_answer(
                self._gate_answer_result(
                    scope_decision,
                    resolved_document_ids=scope.document_ids,
                    resolved_document_titles=resolved_titles,
                ),
                trace_id=trace_id,
                started=started,
                question=normalized_question,
                request_route=request_route,
                history=history,
                conversation_summary=conversation_summary,
                prepared_query=normalized_prepared_query,
                requested_document_ids=forced_document_ids,
            )
        query_analysis = await self._analyze_query(
            normalized_question,
            prepared_query=normalized_prepared_query,
            history=history,
            conversation_summary=conversation_summary,
        )
        emit_stream_event(
            stage="查询分析",
            component="knowledge_query_analysis_agent",
            status="degraded" if query_analysis.degraded else "success",
            title="Query Analysis 完成",
            summary=f"{query_analysis.question_type} / {query_analysis.strategy}",
            details={
                "standalone_query": query_analysis.standalone_query,
                "uses_history": query_analysis.uses_history,
                "question_type": query_analysis.question_type,
                "strategy": query_analysis.strategy,
                "sub_queries": query_analysis.sub_queries,
                "retry_query": query_analysis.retry_query,
                "missing_information": query_analysis.missing_information,
                "clarification_question": query_analysis.clarification_question,
                "confidence": query_analysis.confidence,
            },
        )
        missing_decision = self._evidence_gate.precheck(
            missing_information=query_analysis.missing_information,
            clarification_question=query_analysis.clarification_question,
        )
        if missing_decision is not None:
            return await self._finalize_answer(
                self._gate_answer_result(
                    missing_decision,
                    resolved_document_ids=scope.document_ids,
                    resolved_document_titles=resolved_titles,
                ),
                trace_id=trace_id,
                started=started,
                question=normalized_question,
                request_route=request_route,
                history=history,
                conversation_summary=conversation_summary,
                prepared_query=normalized_prepared_query,
                requested_document_ids=forced_document_ids,
                analysis=query_analysis,
            )
        reasoning_planner_degraded = False
        if (
            query_analysis.question_type in {"comparative", "analytical", "exploratory"}
            and self._reasoning_planner_agent is not None
            and self._plan_executor is not None
            and self._plan_coverage_checker is not None
        ):
            try:
                reasoning_plan = await self._reasoning_planner_agent.plan(
                    standalone_query=query_analysis.standalone_query,
                    question_type=query_analysis.question_type,
                    sub_queries=query_analysis.sub_queries,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                reasoning_planner_degraded = True
                logger.warning(
                    "知识推理 Planner 失败，退回现有检索链路",
                    extra={"exception_type": type(exc).__name__},
                )
            else:
                return await self._ask_with_reasoning_plan(
                    question=normalized_question,
                    query_analysis=query_analysis,
                    initial_plan=reasoning_plan,
                    scope=scope,
                    resolved_titles=resolved_titles,
                    history=history,
                    conversation_summary=conversation_summary,
                    interaction_memory=protected_interaction_memory,
                    trace_id=trace_id,
                    started=started,
                    request_route=request_route,
                    prepared_query=normalized_prepared_query,
                    requested_document_ids=forced_document_ids,
                    deadline=deadline,
                    runtime_skill=runtime_skill,
                )
        stage = await self._retrieve_simple_stage(
            query_analysis,
            scope=scope,
            limit=limit,
        )
        decision = self._decide_simple_evidence(
            stage,
            query_analysis=query_analysis,
            runtime_skill=runtime_skill,
            rewrite_attempted=False,
        )
        query_rewrite_attempted = False
        search_queries_override = (
            (query_analysis.standalone_query,)
            if query_analysis.strategy == "direct"
            else query_analysis.sub_queries
        )
        if decision.action == "rewrite":
            rewritten_query = decision.rewritten_query
            if rewritten_query is None:
                raise RuntimeError("改写门控结果缺少查询")
            stage = await self._retrieve_simple_stage(
                query_analysis,
                scope=scope,
                limit=limit,
                search_query_override=rewritten_query,
            )
            query_rewrite_attempted = True
            search_queries_override = (*search_queries_override, rewritten_query)
            decision = self._decide_simple_evidence(
                stage,
                query_analysis=query_analysis,
                runtime_skill=runtime_skill,
                rewrite_attempted=True,
            )
        retrieval = stage.retrieval
        latest_snapshot = stage.snapshot
        plan_search_degraded = stage.plan_search_degraded
        rerank = stage.rerank
        deterministic_scores = stage.deterministic_scores
        if decision.action != "answer":
            degraded_components = tuple(
                component
                for component, degraded in (
                    (
                        "query_analysis",
                        query_analysis.degraded or plan_search_degraded,
                    ),
                    ("planner", reasoning_planner_degraded),
                    (
                        "vector",
                        retrieval.diagnostics.vector_status == "degraded",
                    ),
                    ("rerank", rerank.degraded),
                )
                if degraded
            )
            refused = self._gate_answer_result(
                decision,
                resolved_document_ids=scope.document_ids,
                resolved_document_titles=resolved_titles,
            ).model_copy(
                update={
                    "retrieval_mode": retrieval.mode,
                    "diagnostics": retrieval.diagnostics,
                    "degraded_components": degraded_components,
                }
            )
            return await self._finalize_answer(
                refused,
                trace_id=trace_id,
                started=started,
                question=normalized_question,
                request_route=request_route,
                history=history,
                conversation_summary=conversation_summary,
                prepared_query=normalized_prepared_query,
                requested_document_ids=forced_document_ids,
                analysis=query_analysis,
                retrieval=retrieval,
                snapshot=latest_snapshot,
                selected_evidence=(),
                document_evidence=(),
                plan_search_degraded=plan_search_degraded,
                search_queries_override=search_queries_override,
            )
        approved_ids = set(decision.approved_evidence_ids)
        evidence_by_id = {record.chunk_id: record for record in stage.evidence}
        if not approved_ids.issubset(evidence_by_id):
            invalid = KnowledgeEvidenceDecision(
                action="refuse",
                confidence=1.0,
                reason_code="invalid_gate_input",
            )
            return await self._finalize_answer(
                self._gate_answer_result(invalid),
                trace_id=trace_id,
                started=started,
                question=normalized_question,
                request_route=request_route,
                history=history,
                conversation_summary=conversation_summary,
                prepared_query=normalized_prepared_query,
                requested_document_ids=forced_document_ids,
                analysis=query_analysis,
                retrieval=retrieval,
                snapshot=latest_snapshot,
                search_queries_override=search_queries_override,
            )
        evidence = tuple(
            record for record in stage.evidence if record.chunk_id in approved_ids
        )

        evidence_scores = {
            record.chunk_id: rerank.scores.get(
                record.chunk_id,
                deterministic_scores.get(record.chunk_id, 0.0),
            )
            for record in evidence
        }
        document_evidence = self._evidence_selector.group_by_document(
            evidence,
            scores=evidence_scores,
        )
        emit_stream_event(
            stage="文档证据",
            component="knowledge_evidence_selector",
            status="success",
            title="按文档组织最终证据",
            summary=f"形成 {len(document_evidence)} 个文档证据包",
            details={
                "documents": [
                    {
                        "document_id": document.document_id,
                        "title": document.title,
                        "score": document.score,
                        "selected_chunk_ids": tuple(
                            record.chunk_id for record in document.chunks
                        ),
                    }
                    for document in document_evidence
                ],
            },
        )
        evidence = tuple(
            record for document in document_evidence for record in document.chunks
        )

        linked_images = await asyncio.to_thread(
            self._repository.list_ready_images_by_chunk_ids,
            tuple(record.chunk_id for record in evidence),
        )
        candidate_images = self._evidence_selector.select_linked_images(
            evidence=evidence,
            images=linked_images,
            max_images=6,
        )
        image_availability = await asyncio.gather(
            *(
                asyncio.to_thread(self.get_image_file, image.image_id)
                for image in candidate_images
            )
        )
        candidate_images = tuple(
            image
            for image, image_file in zip(
                candidate_images,
                image_availability,
                strict=True,
            )
            if image_file is not None
        )
        emit_stream_event(
            stage="图片证据",
            component="knowledge_evidence_selector",
            status="success",
            title="图片证据检查完成",
            summary=f"可用图片 {len(candidate_images)} 张",
            details={
                "images": [
                    {
                        "image_id": image.image_id,
                        "document_id": image.document_id,
                        "linked_chunk_ids": image.linked_chunk_ids,
                        "caption": image.caption,
                    }
                    for image in candidate_images
                ],
            },
        )
        emit_stream_event(
            stage="答案生成",
            component="knowledge_answer_agent",
            status="started",
            title="开始生成答案",
            summary="仅发送白名单 Chunk 与图片证据",
            details={
                "evidence_chunk_ids": tuple(record.chunk_id for record in evidence),
                "image_ids": tuple(image.image_id for image in candidate_images),
            },
        )
        generated = await self._answer_agent.generate(
            question=normalized_question,
            standalone_query=query_analysis.standalone_query,
            history=history,
            conversation_summary=conversation_summary,
            evidence=evidence,
            interaction_memory=protected_interaction_memory,
            images=candidate_images,
            response_policy=(
                runtime_skill.response_policy if runtime_skill is not None else None
            ),
        )
        if generated.outcome == "abstain":
            return await self._finalize_answer(
                self._abstain_result(
                    generated,
                    retrieval_mode=retrieval.mode,
                    diagnostics=retrieval.diagnostics,
                    resolved_document_ids=scope.document_ids,
                    resolved_document_titles=resolved_titles,
                ),
                trace_id=trace_id,
                started=started,
                question=normalized_question,
                request_route=request_route,
                history=history,
                conversation_summary=conversation_summary,
                prepared_query=normalized_prepared_query,
                requested_document_ids=forced_document_ids,
                analysis=query_analysis,
                retrieval=retrieval,
                snapshot=latest_snapshot,
                selected_evidence=(),
                document_evidence=(),
                plan_search_degraded=plan_search_degraded,
                search_queries_override=search_queries_override,
            )
        reflection_decision = await self._review_generated_answer(
            question=normalized_question,
            standalone_query=query_analysis.standalone_query,
            question_type=query_analysis.question_type,
            generated=generated,
            evidence=evidence,
            images=candidate_images,
            retrieval_degraded=(
                plan_search_degraded
                or retrieval.diagnostics.vector_status == "degraded"
                or rerank.degraded
            ),
            retry_query=query_analysis.retry_query,
            query_rewrite_attempted=query_rewrite_attempted,
        )
        if (
            reflection_decision.action == "rewrite"
            and reflection_decision.rewrite_mode == "regenerate_answer"
            and reflection_decision.revision_policy is not None
        ):
            generated = await self._regenerate_answer_once(
                question=normalized_question,
                standalone_query=query_analysis.standalone_query,
                history=history,
                conversation_summary=conversation_summary,
                evidence=evidence,
                interaction_memory=protected_interaction_memory,
                images=candidate_images,
                response_policy=(
                    runtime_skill.response_policy if runtime_skill is not None else None
                ),
                revision_policy=reflection_decision.revision_policy,
            )
            if generated.outcome == "abstain":
                return await self._finalize_answer(
                    self._abstain_result(
                        generated,
                        retrieval_mode=retrieval.mode,
                        diagnostics=retrieval.diagnostics,
                        resolved_document_ids=scope.document_ids,
                        resolved_document_titles=resolved_titles,
                    ),
                    trace_id=trace_id,
                    started=started,
                    question=normalized_question,
                    request_route=request_route,
                    history=history,
                    conversation_summary=conversation_summary,
                    prepared_query=normalized_prepared_query,
                    requested_document_ids=forced_document_ids,
                    analysis=query_analysis,
                    retrieval=retrieval,
                    snapshot=latest_snapshot,
                    selected_evidence=(),
                    document_evidence=(),
                    plan_search_degraded=plan_search_degraded,
                    search_queries_override=search_queries_override,
                )
            if self._reflection_service is not None:
                reflection_decision = self._reflection_service.validate_repaired(
                    question_type=query_analysis.question_type,
                    answer=generated,
                    evidence=evidence,
                    images=candidate_images,
                    retrieval_degraded=(
                        plan_search_degraded
                        or retrieval.diagnostics.vector_status == "degraded"
                        or rerank.degraded
                    ),
                )
        if (
            reflection_decision.action == "rewrite"
            and reflection_decision.rewrite_mode == "retry_retrieval"
        ):
            rewritten_query = reflection_decision.rewritten_query
            revision_policy = reflection_decision.revision_policy
            if (
                query_rewrite_attempted
                or rewritten_query is None
                or revision_policy is None
            ):
                reflection_decision = KnowledgeAnswerReflectionDecision(
                    action="refuse",
                    confidence=1.0,
                    reason_code="retry_query_unavailable",
                )
            else:
                stage = await self._retrieve_simple_stage(
                    query_analysis,
                    scope=scope,
                    limit=limit,
                    search_query_override=rewritten_query,
                )
                query_rewrite_attempted = True
                search_queries_override = (
                    *search_queries_override,
                    rewritten_query,
                )
                retrieval = stage.retrieval
                latest_snapshot = stage.snapshot
                plan_search_degraded = stage.plan_search_degraded
                rerank = stage.rerank
                deterministic_scores = stage.deterministic_scores
                repaired_gate_decision = self._decide_simple_evidence(
                    stage,
                    query_analysis=query_analysis,
                    runtime_skill=runtime_skill,
                    rewrite_attempted=True,
                )
                if repaired_gate_decision.action != "answer":
                    refused = self._gate_answer_result(
                        repaired_gate_decision,
                        resolved_document_ids=scope.document_ids,
                        resolved_document_titles=resolved_titles,
                    ).model_copy(
                        update={
                            "retrieval_mode": retrieval.mode,
                            "diagnostics": retrieval.diagnostics,
                        }
                    )
                    return await self._finalize_answer(
                        refused,
                        trace_id=trace_id,
                        started=started,
                        question=normalized_question,
                        request_route=request_route,
                        history=history,
                        conversation_summary=conversation_summary,
                        prepared_query=normalized_prepared_query,
                        requested_document_ids=forced_document_ids,
                        analysis=query_analysis,
                        retrieval=retrieval,
                        snapshot=latest_snapshot,
                        selected_evidence=(),
                        document_evidence=(),
                        plan_search_degraded=plan_search_degraded,
                        search_queries_override=search_queries_override,
                    )
                approved_ids = set(repaired_gate_decision.approved_evidence_ids)
                evidence_by_id = {record.chunk_id: record for record in stage.evidence}
                if not approved_ids.issubset(evidence_by_id):
                    reflection_decision = KnowledgeAnswerReflectionDecision(
                        action="refuse",
                        confidence=1.0,
                        reason_code="invalid_reflection_input",
                    )
                    evidence = ()
                    document_evidence = ()
                    candidate_images = ()
                else:
                    repaired_evidence = tuple(
                        record
                        for record in stage.evidence
                        if record.chunk_id in approved_ids
                    )
                    evidence_scores = {
                        record.chunk_id: rerank.scores.get(
                            record.chunk_id,
                            deterministic_scores.get(record.chunk_id, 0.0),
                        )
                        for record in repaired_evidence
                    }
                    document_evidence = self._evidence_selector.group_by_document(
                        repaired_evidence,
                        scores=evidence_scores,
                    )
                    evidence = tuple(
                        record
                        for document in document_evidence
                        for record in document.chunks
                    )
                    candidate_images = await self._ready_images_for_evidence(evidence)
                    generated = await self._regenerate_answer_once(
                        question=normalized_question,
                        standalone_query=query_analysis.standalone_query,
                        history=history,
                        conversation_summary=conversation_summary,
                        evidence=evidence,
                        interaction_memory=protected_interaction_memory,
                        images=candidate_images,
                        response_policy=(
                            runtime_skill.response_policy
                            if runtime_skill is not None
                            else None
                        ),
                        revision_policy=revision_policy,
                    )
                    if generated.outcome == "abstain":
                        return await self._finalize_answer(
                            self._abstain_result(
                                generated,
                                retrieval_mode=retrieval.mode,
                                diagnostics=retrieval.diagnostics,
                                resolved_document_ids=scope.document_ids,
                                resolved_document_titles=resolved_titles,
                            ),
                            trace_id=trace_id,
                            started=started,
                            question=normalized_question,
                            request_route=request_route,
                            history=history,
                            conversation_summary=conversation_summary,
                            prepared_query=normalized_prepared_query,
                            requested_document_ids=forced_document_ids,
                            analysis=query_analysis,
                            retrieval=retrieval,
                            snapshot=latest_snapshot,
                            selected_evidence=(),
                            document_evidence=(),
                            plan_search_degraded=plan_search_degraded,
                            search_queries_override=search_queries_override,
                        )
                    if self._reflection_service is not None:
                        reflection_decision = (
                            self._reflection_service.validate_repaired(
                                question_type=query_analysis.question_type,
                                answer=generated,
                                evidence=evidence,
                                images=candidate_images,
                                retrieval_degraded=(
                                    plan_search_degraded
                                    or retrieval.diagnostics.vector_status == "degraded"
                                    or rerank.degraded
                                ),
                            )
                        )
        if reflection_decision.action in {"ask", "select", "refuse"}:
            return await self._finalize_answer(
                self._reflection_result(
                    reflection_decision,
                    resolved_document_ids=scope.document_ids,
                    resolved_document_titles=resolved_titles,
                ).model_copy(
                    update={
                        "retrieval_mode": retrieval.mode,
                        "diagnostics": retrieval.diagnostics,
                    }
                ),
                trace_id=trace_id,
                started=started,
                question=normalized_question,
                request_route=request_route,
                history=history,
                conversation_summary=conversation_summary,
                prepared_query=normalized_prepared_query,
                requested_document_ids=forced_document_ids,
                analysis=query_analysis,
                retrieval=retrieval,
                snapshot=latest_snapshot,
                selected_evidence=(),
                document_evidence=(),
                plan_search_degraded=plan_search_degraded,
                search_queries_override=search_queries_override,
            )
        emit_stream_event(
            stage="答案生成",
            component="knowledge_answer_agent",
            status="degraded" if generated.degraded else "success",
            title="答案生成完成",
            summary=(
                f"引用 {len(generated.cited_chunk_ids)} 个 Chunk，"
                f"{len(generated.cited_image_ids)} 张图片"
            ),
            details={
                "cited_chunk_ids": generated.cited_chunk_ids,
                "cited_image_ids": generated.cited_image_ids,
                "degraded": generated.degraded,
            },
        )
        evidence_by_id = {record.chunk_id: record for record in evidence}
        cited_records = tuple(
            evidence_by_id[chunk_id]
            for chunk_id in generated.cited_chunk_ids
            if chunk_id in evidence_by_id
        )
        images_by_id = {image.image_id: image for image in candidate_images}
        cited_images = tuple(
            images_by_id[image_id]
            for image_id in generated.cited_image_ids
            if image_id in images_by_id
        )
        degraded_components: list[str] = []
        if query_analysis.degraded or plan_search_degraded:
            degraded_components.append("query_analysis")
        if reasoning_planner_degraded:
            degraded_components.append("planner")
        if retrieval.diagnostics.vector_status == "degraded":
            degraded_components.append("vector")
        if rerank.degraded:
            degraded_components.append("rerank")
        if generated.degraded:
            degraded_components.append("answer")
        citation_numbers: dict[str, int] = {}
        citations: list[KnowledgeCitation] = []
        for record in cited_records:
            citation_id = citation_numbers.get(record.document_id)
            if citation_id is None:
                citation_id = len(citation_numbers) + 1
                citation_numbers[record.document_id] = citation_id
            citations.append(self._citation(record, citation_id=str(citation_id)))
        return await self._finalize_answer(
            KnowledgeAnswerResult(
                status="degraded" if degraded_components else "success",
                answer=generated.answer,
                citations=tuple(citations),
                images=tuple(
                    self._image_citation(image, citation_id=f"图{index}")
                    for index, image in enumerate(cited_images, start=1)
                ),
                retrieval_mode=retrieval.mode,
                diagnostics=retrieval.diagnostics,
                degraded_components=tuple(degraded_components),
                resolved_document_ids=scope.document_ids,
                resolved_document_titles=resolved_titles,
            ),
            trace_id=trace_id,
            started=started,
            question=normalized_question,
            request_route=request_route,
            history=history,
            conversation_summary=conversation_summary,
            prepared_query=normalized_prepared_query,
            requested_document_ids=forced_document_ids,
            analysis=query_analysis,
            retrieval=retrieval,
            snapshot=latest_snapshot,
            selected_evidence=evidence,
            document_evidence=document_evidence,
            plan_search_degraded=plan_search_degraded,
            search_queries_override=search_queries_override,
        )

    async def _review_generated_answer(
        self,
        *,
        question: str,
        standalone_query: str,
        question_type: KnowledgeQuestionType,
        generated: KnowledgeGeneratedAnswer,
        evidence: Sequence[KnowledgeChunkRecord],
        images: Sequence[KnowledgeImageEvidence] = (),
        retrieval_degraded: bool = False,
        coverage: KnowledgePlanCoverage | None = None,
        retry_query: str | None = None,
        query_rewrite_attempted: bool = False,
        repair_attempted: bool = False,
        force_semantic_review: bool = False,
        allow_retrieval_retry: bool = True,
    ) -> KnowledgeAnswerReflectionDecision:
        """执行一次草稿反思；未装配时按当前确定性结果通过。"""

        if generated.outcome != "answer":
            raise ValueError("主动拒答必须在进入答案反思前完成短路")
        if self._reflection_service is None:
            return KnowledgeAnswerReflectionDecision(
                action="answer",
                confidence=1.0,
                reason_code="deterministic_pass",
                approved=True,
            )
        emit_stream_event(
            stage="答案反思",
            component="knowledge_answer_reflection_service",
            status="started",
            title="开始检查知识答案草稿",
            summary="执行确定性检查并按风险决定是否调用小模型",
            details={
                "question_type": question_type,
                "evidence_count": len(tuple(evidence)),
                "image_count": len(tuple(images)),
                "retrieval_degraded": retrieval_degraded,
                "query_rewrite_attempted": query_rewrite_attempted,
                "repair_attempted": repair_attempted,
                "force_semantic_review": force_semantic_review,
            },
        )
        decision = await self._reflection_service.review(
            question=question,
            standalone_query=standalone_query,
            question_type=question_type,
            answer=generated,
            evidence=evidence,
            images=images,
            retrieval_degraded=retrieval_degraded,
            coverage=coverage,
            retry_query=retry_query,
            query_rewrite_attempted=query_rewrite_attempted,
            repair_attempted=repair_attempted,
            force_semantic_review=force_semantic_review,
            allow_retrieval_retry=allow_retrieval_retry,
        )
        emit_stream_event(
            stage="答案反思",
            component="knowledge_answer_reflection_service",
            status="degraded" if decision.reflection_degraded else "success",
            title="知识答案草稿检查完成",
            summary=f"{decision.action} / {decision.reason_code}",
            details={
                "action": decision.action,
                "reason_code": decision.reason_code,
                "rewrite_mode": decision.rewrite_mode,
                "repair_attempted": repair_attempted,
                "reflection_degraded": decision.reflection_degraded,
            },
        )
        return decision

    async def _regenerate_answer_once(
        self,
        *,
        question: str,
        standalone_query: str,
        history: Sequence[ConversationTurn],
        conversation_summary: str | None,
        evidence: Sequence[KnowledgeChunkRecord],
        interaction_memory: UserInteractionMemoryProjection | None,
        images: Sequence[KnowledgeImageEvidence],
        response_policy: RuntimeSkillResponsePolicy | None,
        revision_policy: KnowledgeAnswerRevisionPolicy,
    ) -> KnowledgeGeneratedAnswer:
        """使用枚举修订策略和同一 Evidence 再生成一次。"""

        emit_stream_event(
            stage="答案修复",
            component="knowledge_answer_agent",
            status="started",
            title="开始自动修订知识答案",
            summary="使用相同 Evidence 和枚举修订策略重新生成一次",
            details={
                "revision_focus": revision_policy.focus,
                "evidence_count": len(tuple(evidence)),
                "image_count": len(tuple(images)),
            },
        )
        generated = await self._answer_agent.generate(
            question=question,
            standalone_query=standalone_query,
            history=history,
            conversation_summary=conversation_summary,
            evidence=evidence,
            interaction_memory=interaction_memory,
            images=images,
            response_policy=response_policy,
            revision_policy=revision_policy,
        )
        emit_stream_event(
            stage="答案修复",
            component="knowledge_answer_agent",
            status="degraded" if generated.degraded else "success",
            title="知识答案自动修订完成",
            summary="修订草稿将进入确定性引用复检",
            details={
                "revision_focus": revision_policy.focus,
                "degraded": generated.degraded,
                "cited_chunk_count": len(generated.cited_chunk_ids),
                "cited_image_count": len(generated.cited_image_ids),
            },
        )
        return generated

    @staticmethod
    def _abstain_result(
        generated: KnowledgeGeneratedAnswer,
        *,
        retrieval_mode: Literal["bm25", "hybrid"] = "bm25",
        diagnostics: KnowledgeRetrievalDiagnostics | None = None,
        resolved_document_ids: Sequence[str] = (),
        resolved_document_titles: Sequence[str] = (),
    ) -> KnowledgeAnswerResult:
        """把受控主动拒答投影为既有公开状态且清空全部 Evidence。"""

        protected = KnowledgeGeneratedAnswer.model_validate(generated)
        if protected.outcome != "abstain":
            raise ValueError("只有主动拒答结果可以投影为证据不足")
        return KnowledgeAnswerResult(
            status="insufficient_evidence",
            answer=KnowledgeQaService._ABSTAIN_ANSWER,
            citations=(),
            images=(),
            retrieval_mode=retrieval_mode,
            diagnostics=diagnostics or KnowledgeRetrievalDiagnostics(),
            resolved_document_ids=tuple(resolved_document_ids),
            resolved_document_titles=tuple(resolved_document_titles),
        )

    @staticmethod
    def _reflection_result(
        decision: KnowledgeAnswerReflectionDecision,
        *,
        resolved_document_ids: Sequence[str] = (),
        resolved_document_titles: Sequence[str] = (),
    ) -> KnowledgeAnswerResult:
        """把反思问、选、拒安全投影为现有公开结果。"""

        protected = KnowledgeAnswerReflectionDecision.model_validate(decision)
        if protected.action == "ask":
            answer = protected.clarification_question or "请补充必要信息。"
            status = "needs_clarification"
        elif protected.action == "select":
            options = "\n".join(
                f"{index}. {option.label} [{option.option_id}]"
                for index, option in enumerate(protected.options, start=1)
            )
            answer = f"请从以下可信选项中选择：\n{options}"
            status = "needs_clarification"
        elif protected.action == "refuse":
            messages = {
                "generation_unavailable": "答案生成暂时不可用，请稍后重试。",
                "invalid_citation": "当前答案引用无效，无法安全回答。",
                "unsupported_claim": "当前答案缺少可信证据支持。",
                "unsafe_answer": "当前答案未通过安全检查。",
                "retry_query_unavailable": "当前没有可用的受控重试查询。",
                "repair_exhausted": "自动修复后仍无法形成可信答案。",
                "invalid_reflection_input": "当前答案检查输入无效。",
            }
            answer = messages.get(
                protected.reason_code,
                "当前知识库中没有找到足够依据。",
            )
            status = "insufficient_evidence"
        else:
            raise ValueError("只有 ask、select 或 refuse 可以直接投影结果")
        return KnowledgeAnswerResult(
            status=status,
            answer=answer,
            citations=(),
            images=(),
            resolved_document_ids=tuple(resolved_document_ids),
            resolved_document_titles=tuple(resolved_document_titles),
        )

    async def _retrieve_simple_stage(
        self,
        query_analysis: KnowledgeQueryAnalysis,
        *,
        scope: KnowledgeScopeResolution,
        limit: int,
        search_query_override: str | None = None,
    ) -> _KnowledgeSimpleRetrievalStage:
        """在固定范围内执行一次简单检索、回查、重排和证据选择。"""

        search_query = search_query_override or query_analysis.standalone_query
        search_queries = (
            (search_query,)
            if search_query_override is not None or query_analysis.strategy == "direct"
            else query_analysis.sub_queries
        )
        emit_stream_event(
            stage="检索",
            component="knowledge_search",
            status="started",
            title="开始 Chunk 检索",
            summary=f"发送 {len(search_queries)} 条查询",
            details={
                "search_queries": search_queries,
                "document_ids": scope.document_ids,
                "limit_per_query": self._RETRIEVAL_LIMIT,
            },
        )
        async with self._index_lock:
            if search_query_override is None:
                retrieval, plan_search_degraded = await self._search_with_plan(
                    query_analysis,
                    document_ids=scope.document_ids,
                )
            else:
                retrieval = await self._search.search(
                    search_query,
                    limit=self._RETRIEVAL_LIMIT,
                    document_ids=scope.document_ids,
                )
                plan_search_degraded = False
            latest_snapshot = await asyncio.to_thread(
                self._repository.list_ready_chunks
            )
        emit_stream_event(
            stage="检索",
            component="knowledge_search",
            status=(
                "degraded"
                if plan_search_degraded
                or retrieval.diagnostics.vector_status == "degraded"
                else "success"
            ),
            title="Chunk 检索完成",
            summary=f"召回 {len(retrieval.hits)} 个 Chunk",
            details={
                "retrieval_mode": retrieval.mode,
                "diagnostics": retrieval.diagnostics,
                "plan_search_degraded": plan_search_degraded,
            },
        )

        candidates = self._recheck_candidates(
            retrieval,
            latest_snapshot,
            scope_document_ids=scope.document_ids,
        )
        records_by_id = {record.chunk_id: record for record in latest_snapshot}
        for rank, hit in enumerate(retrieval.hits, start=1):
            record = records_by_id.get(hit.chunk_id)
            if record is None:
                continue
            excerpt = " ".join(record.content.split())
            if len(excerpt) > 220:
                excerpt = excerpt[:219].rstrip() + "…"
            emit_stream_event(
                stage="Chunk 召回",
                component="knowledge_search",
                status="success",
                title=f"召回 Chunk #{rank}",
                summary=f"{record.title} · {record.chunk_id}",
                details={
                    "rank": rank,
                    "chunk_id": record.chunk_id,
                    "document_id": record.document_id,
                    "title": record.title,
                    "heading_path": record.heading_path,
                    "score": hit.score,
                    "bm25_rank": hit.bm25_rank,
                    "vector_rank": hit.vector_rank,
                    "excerpt": excerpt,
                },
            )
        candidate_ids = {record.chunk_id for record in candidates}
        deterministic_scores = {
            hit.chunk_id: hit.score
            for hit in retrieval.hits
            if hit.chunk_id in candidate_ids
        }
        if self._chunk_rerank_agent is None:
            rerank = KnowledgeChunkRerankOutcome(
                records=candidates,
                scores=deterministic_scores,
            )
        else:
            rerank = await self._chunk_rerank_agent.rerank(
                query=search_query,
                candidates=candidates,
                deterministic_scores=deterministic_scores,
            )
        emit_stream_event(
            stage="Chunk 重排",
            component="knowledge_chunk_rerank_agent",
            status="degraded" if rerank.degraded else "success",
            title="Chunk 重排完成",
            summary=f"重排 {len(rerank.records)} 个可信 Chunk",
            details={
                "ranked_chunks": [
                    {
                        "chunk_id": record.chunk_id,
                        "document_id": record.document_id,
                        "score": rerank.scores.get(record.chunk_id, 0.0),
                    }
                    for record in rerank.records[:20]
                ],
            },
        )
        if (
            query_analysis.question_type == "summarization"
            and len(scope.document_ids) == 1
        ):
            evidence = self._evidence_selector.select_document_summary_context(
                document_id=scope.document_ids[0],
                snapshot=latest_snapshot,
            )
        else:
            evidence = self._evidence_selector.select_full_parent_context(
                ranked_records=rerank.records,
                scores=rerank.scores,
                snapshot=latest_snapshot,
                seed_limit=limit,
            )
        if query_analysis.question_type == "procedural":
            evidence = self._evidence_selector.select_direct_support(
                query_analysis.standalone_query,
                evidence,
            )
        elif query_analysis.question_type == "comparative":
            evidence = self._evidence_selector.select_comparative_support(
                object_queries=query_analysis.sub_queries,
                records=evidence,
            )
        emit_stream_event(
            stage="证据选择",
            component="knowledge_evidence_selector",
            status="success" if evidence else "degraded",
            title="最终 Chunk 证据选择完成",
            summary=f"保留 {len(evidence)} 个 Chunk",
            details={
                "selected_chunk_ids": tuple(record.chunk_id for record in evidence),
            },
        )
        return _KnowledgeSimpleRetrievalStage(
            retrieval=retrieval,
            snapshot=tuple(latest_snapshot),
            plan_search_degraded=plan_search_degraded,
            rerank=rerank,
            deterministic_scores=deterministic_scores,
            evidence=tuple(evidence),
        )

    def _decide_simple_evidence(
        self,
        stage: _KnowledgeSimpleRetrievalStage,
        *,
        query_analysis: KnowledgeQueryAnalysis,
        runtime_skill: CompiledRuntimeSkill | None,
        rewrite_attempted: bool,
    ) -> KnowledgeEvidenceDecision:
        """把简单链可信证据组装为统一信号并保护门控输出。"""

        evidence_scores = tuple(
            stage.rerank.scores.get(
                record.chunk_id,
                stage.deterministic_scores.get(record.chunk_id, 0.0),
            )
            for record in stage.evidence
        )
        signals = EvidenceSignals(
            relevance=max(evidence_scores, default=0.0),
            answerability=1.0 if stage.evidence else 0.0,
            ambiguity=0.0,
            gate_profile=(
                runtime_skill.gate_profile
                if runtime_skill is not None
                else "default_evidence"
            ),
            selected_evidence_ids=tuple(record.chunk_id for record in stage.evidence),
        )
        try:
            return KnowledgeEvidenceDecision.model_validate(
                self._evidence_gate.decide_after_retrieval(
                    signals,
                    retry_query=query_analysis.retry_query,
                    rewrite_attempted=rewrite_attempted,
                )
            )
        except (TypeError, ValueError):
            return KnowledgeEvidenceDecision(
                action="refuse",
                confidence=1.0,
                reason_code="invalid_gate_input",
            )

    def _decide_complex_evidence(
        self,
        records: Sequence[KnowledgeChunkRecord],
        *,
        scores: Mapping[str, float],
        coverage: KnowledgePlanCoverage,
        runtime_skill: CompiledRuntimeSkill | None,
    ) -> KnowledgeEvidenceDecision:
        """把复杂 Coverage 和合并证据投影到同一后置门控。"""

        evidence = tuple(records)
        signals = EvidenceSignals(
            relevance=max(
                (scores.get(record.chunk_id, 0.0) for record in evidence),
                default=0.0,
            ),
            answerability=(
                coverage.coverage_ratio if coverage.decision == "answer" else 0.0
            ),
            ambiguity=0.0,
            gate_profile=(
                runtime_skill.gate_profile
                if runtime_skill is not None
                else "default_evidence"
            ),
            selected_evidence_ids=tuple(record.chunk_id for record in evidence),
        )
        try:
            decision = KnowledgeEvidenceDecision.model_validate(
                self._evidence_gate.decide_after_retrieval(
                    signals,
                    retry_query=None,
                    rewrite_attempted=True,
                )
            )
        except (TypeError, ValueError):
            return KnowledgeEvidenceDecision(
                action="refuse",
                confidence=1.0,
                reason_code="invalid_gate_input",
            )
        if coverage.decision != "answer" and decision.action == "answer":
            return KnowledgeEvidenceDecision(
                action="refuse",
                confidence=1.0,
                reason_code="invalid_gate_input",
            )
        return decision

    async def _ask_with_reasoning_plan(
        self,
        *,
        question: str,
        query_analysis: KnowledgeQueryAnalysis,
        initial_plan: KnowledgeReasoningPlan,
        scope: KnowledgeScopeResolution,
        resolved_titles: Sequence[str],
        history: Sequence[ConversationTurn],
        conversation_summary: str | None,
        interaction_memory: UserInteractionMemoryProjection | None,
        trace_id: str,
        started: float,
        request_route: Literal[
            "/api/v1/knowledge/ask",
            "/api/v1/chat",
        ],
        prepared_query: str | None,
        requested_document_ids: Sequence[str],
        deadline: float | None,
        runtime_skill: CompiledRuntimeSkill | None,
    ) -> KnowledgeAnswerResult:
        """在固定快照内执行最多两版复杂计划并只回答一次。"""

        assert self._plan_executor is not None
        assert self._plan_coverage_checker is not None
        assert self._reasoning_planner_agent is not None
        emit_stream_event(
            stage="制定计划",
            component="knowledge_reasoning_planner_agent",
            status="success",
            title="复杂知识问题计划已生成",
            summary=f"{initial_plan.strategy} · {len(initial_plan.steps)} 个步骤",
            details={
                "revision": initial_plan.revision,
                "strategy": initial_plan.strategy,
                "step_ids": tuple(step.step_id for step in initial_plan.steps),
            },
        )
        planner_degraded = False
        outcomes: list[KnowledgePlanRoundOutcome] = []
        coverages: list[KnowledgePlanCoverage] = []
        final_plan = initial_plan
        try:
            async with self._index_lock:
                snapshot = await asyncio.to_thread(self._repository.list_ready_chunks)
                cache = KnowledgePlanRequestCache.from_snapshot(snapshot)
                first_outcome = await self._plan_executor.execute_round(
                    initial_plan,
                    document_ids=scope.document_ids,
                    snapshot=snapshot,
                    cache=cache,
                )
                outcomes.append(first_outcome)
                self._emit_plan_execution_event(first_outcome)
                first_coverage = self._plan_coverage_checker.evaluate(
                    initial_plan,
                    relations=first_outcome.relations,
                    records=first_outcome.records,
                    empty_reason_by_step=first_outcome.empty_reason_by_step,
                    replanned=False,
                    allow_replan=True,
                )
                coverages.append(first_coverage)
                self._emit_plan_coverage_event(
                    initial_plan,
                    first_coverage,
                )
                final_coverage = first_coverage
                if first_coverage.decision == "replan":
                    remaining_seconds = (
                        deadline - asyncio.get_running_loop().time()
                        if deadline is not None
                        else None
                    )
                    if remaining_seconds is not None and remaining_seconds < 15.0:
                        final_coverage = self._plan_coverage_checker.evaluate(
                            initial_plan,
                            relations=first_outcome.relations,
                            records=first_outcome.records,
                            empty_reason_by_step=(first_outcome.empty_reason_by_step),
                            replanned=False,
                            allow_replan=False,
                        )
                        coverages[-1] = final_coverage
                        emit_stream_event(
                            stage="调整计划",
                            component="knowledge_reasoning_planner_agent",
                            status="skipped",
                            title="剩余预算不足，跳过计划调整",
                            summary="按首轮证据执行最终覆盖判断",
                            details={
                                "revision": initial_plan.revision,
                                "decision": first_coverage.decision,
                            },
                        )
                        self._emit_plan_coverage_event(
                            initial_plan,
                            final_coverage,
                        )
                    else:
                        try:
                            revised_plan = await self._reasoning_planner_agent.replan(
                                standalone_query=query_analysis.standalone_query,
                                question_type=query_analysis.question_type,
                                previous_plan=initial_plan,
                                step_results=first_coverage.step_results,
                                remaining_step_limit=5,
                            )
                        except asyncio.CancelledError:
                            raise
                        except Exception as exc:
                            planner_degraded = True
                            logger.warning(
                                "知识推理重规划失败，按首轮证据执行最终覆盖判断",
                                extra={"exception_type": type(exc).__name__},
                            )
                            emit_stream_event(
                                stage="调整计划",
                                component="knowledge_reasoning_planner_agent",
                                status="degraded",
                                title="复杂知识计划调整失败",
                                summary="按首轮证据执行最终覆盖判断",
                                details={
                                    "revision": initial_plan.revision,
                                    "decision": first_coverage.decision,
                                },
                            )
                            final_coverage = self._plan_coverage_checker.evaluate(
                                initial_plan,
                                relations=first_outcome.relations,
                                records=first_outcome.records,
                                empty_reason_by_step=(
                                    first_outcome.empty_reason_by_step
                                ),
                                replanned=False,
                                allow_replan=False,
                            )
                            coverages[-1] = final_coverage
                            self._emit_plan_coverage_event(
                                initial_plan,
                                final_coverage,
                            )
                        else:
                            final_plan = revised_plan
                            emit_stream_event(
                                stage="调整计划",
                                component="knowledge_reasoning_planner_agent",
                                status="success",
                                title="复杂知识计划已调整",
                                summary=(
                                    f"第 {revised_plan.revision} 版 · "
                                    f"{len(revised_plan.steps)} 个步骤"
                                ),
                                details={
                                    "revision": revised_plan.revision,
                                    "strategy": revised_plan.strategy,
                                    "kept_step_ids": revised_plan.kept_step_ids,
                                    "step_ids": tuple(
                                        step.step_id for step in revised_plan.steps
                                    ),
                                },
                            )
                            reusable_step_ids = tuple(
                                result.step_id
                                for result in first_coverage.step_results
                                if result.status == "covered"
                                and result.step_id in revised_plan.kept_step_ids
                            )
                            second_outcome = await self._plan_executor.execute_round(
                                revised_plan,
                                document_ids=scope.document_ids,
                                snapshot=snapshot,
                                cache=cache,
                                prior_outcome=first_outcome,
                                reusable_step_ids=reusable_step_ids,
                            )
                            outcomes.append(second_outcome)
                            self._emit_plan_execution_event(second_outcome)
                            final_coverage = self._plan_coverage_checker.evaluate(
                                revised_plan,
                                relations=second_outcome.relations,
                                records=second_outcome.records,
                                empty_reason_by_step=(
                                    second_outcome.empty_reason_by_step
                                ),
                                replanned=True,
                                allow_replan=False,
                            )
                            coverages.append(final_coverage)
                            self._emit_plan_coverage_event(
                                revised_plan,
                                final_coverage,
                            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            component = (
                "coverage"
                if outcomes and len(coverages) < len(outcomes)
                else "plan_execution"
            )
            logger.warning(
                "知识计划执行或覆盖检查失败，保守返回证据不足",
                extra={
                    "component": component,
                    "exception_type": type(exc).__name__,
                },
            )
            latest_snapshot = tuple(snapshot) if "snapshot" in locals() else ()
            retrieval = self._plan_trace_retrieval(outcomes)
            return await self._finalize_answer(
                KnowledgeAnswerResult(
                    status="insufficient_evidence",
                    answer="当前知识库中没有找到足够依据。",
                    retrieval_mode=retrieval.mode,
                    diagnostics=retrieval.diagnostics,
                    degraded_components=(component,),
                    resolved_document_ids=scope.document_ids,
                    resolved_document_titles=tuple(resolved_titles),
                ),
                trace_id=trace_id,
                started=started,
                question=question,
                request_route=request_route,
                history=history,
                conversation_summary=conversation_summary,
                prepared_query=prepared_query,
                requested_document_ids=requested_document_ids,
                analysis=query_analysis,
                retrieval=retrieval,
                snapshot=latest_snapshot,
                selected_evidence=(),
                document_evidence=(),
                reasoning_strategy=initial_plan.strategy,
                plan_revision_count=len(outcomes),
                plan_steps=self._plan_trace_steps(outcomes, coverages),
                coverage=coverages[-1] if coverages else None,
                search_queries_override=self._plan_search_queries(outcomes),
            )
        retrieval = self._plan_trace_retrieval(outcomes)
        trace_steps = self._plan_trace_steps(outcomes, coverages)
        search_queries = self._plan_search_queries(outcomes)
        merged = merge_evidence(final_plan, final_coverage, outcomes)
        emit_stream_event(
            stage="合并证据",
            component="knowledge_plan_execution",
            status="success" if merged.records else "degraded",
            title="复杂知识证据合并完成",
            summary=f"合并 {len(merged.records)} 个可信 Chunk",
            details={
                "revision": final_plan.revision,
                "decision": final_coverage.decision,
                "chunk_count": len(merged.records),
                "chunk_ids": tuple(record.chunk_id for record in merged.records),
            },
        )
        decision = self._decide_complex_evidence(
            merged.records,
            scores=merged.scores,
            coverage=final_coverage,
            runtime_skill=runtime_skill,
        )
        if decision.action != "answer":
            degraded_components = tuple(
                component
                for component, degraded in (
                    ("query_analysis", query_analysis.degraded),
                    ("planner", planner_degraded),
                    (
                        "vector",
                        retrieval.diagnostics.vector_status == "degraded",
                    ),
                    (
                        "rerank",
                        any(outcome.rerank_degraded for outcome in outcomes),
                    ),
                )
                if degraded
            )
            refused = self._gate_answer_result(
                decision,
                resolved_document_ids=scope.document_ids,
                resolved_document_titles=resolved_titles,
            ).model_copy(
                update={
                    "retrieval_mode": retrieval.mode,
                    "diagnostics": retrieval.diagnostics,
                    "degraded_components": degraded_components,
                }
            )
            return await self._finalize_answer(
                refused,
                trace_id=trace_id,
                started=started,
                question=question,
                request_route=request_route,
                history=history,
                conversation_summary=conversation_summary,
                prepared_query=prepared_query,
                requested_document_ids=requested_document_ids,
                analysis=query_analysis,
                retrieval=retrieval,
                snapshot=snapshot,
                reasoning_strategy=final_plan.strategy,
                plan_revision_count=len(outcomes),
                plan_steps=trace_steps,
                coverage=final_coverage,
                search_queries_override=search_queries,
            )
        approved_ids = set(decision.approved_evidence_ids)
        merged_by_id = {record.chunk_id: record for record in merged.records}
        if not approved_ids.issubset(merged_by_id):
            invalid = KnowledgeEvidenceDecision(
                action="refuse",
                confidence=1.0,
                reason_code="invalid_gate_input",
            )
            return await self._finalize_answer(
                self._gate_answer_result(
                    invalid,
                    resolved_document_ids=scope.document_ids,
                    resolved_document_titles=resolved_titles,
                ),
                trace_id=trace_id,
                started=started,
                question=question,
                request_route=request_route,
                history=history,
                conversation_summary=conversation_summary,
                prepared_query=prepared_query,
                requested_document_ids=requested_document_ids,
                analysis=query_analysis,
                retrieval=retrieval,
                snapshot=snapshot,
                selected_evidence=(),
                document_evidence=(),
                reasoning_strategy=final_plan.strategy,
                plan_revision_count=len(outcomes),
                plan_steps=trace_steps,
                coverage=final_coverage,
                search_queries_override=search_queries,
            )
        approved_records = tuple(
            record for record in merged.records if record.chunk_id in approved_ids
        )
        document_evidence = self._evidence_selector.group_by_document(
            approved_records,
            scores={
                record.chunk_id: merged.scores.get(record.chunk_id, 0.0)
                for record in approved_records
            },
        )
        evidence = tuple(
            record for document in document_evidence for record in document.chunks
        )
        candidate_images = await self._ready_images_for_evidence(evidence)
        generated = await self._answer_agent.generate(
            question=question,
            standalone_query=query_analysis.standalone_query,
            history=history,
            conversation_summary=conversation_summary,
            evidence=evidence,
            interaction_memory=interaction_memory,
            images=candidate_images,
            response_policy=(
                runtime_skill.response_policy if runtime_skill is not None else None
            ),
        )
        if generated.outcome == "abstain":
            return await self._finalize_answer(
                self._abstain_result(
                    generated,
                    retrieval_mode=retrieval.mode,
                    diagnostics=retrieval.diagnostics,
                    resolved_document_ids=scope.document_ids,
                    resolved_document_titles=resolved_titles,
                ),
                trace_id=trace_id,
                started=started,
                question=question,
                request_route=request_route,
                history=history,
                conversation_summary=conversation_summary,
                prepared_query=prepared_query,
                requested_document_ids=requested_document_ids,
                analysis=query_analysis,
                retrieval=retrieval,
                snapshot=snapshot,
                selected_evidence=(),
                document_evidence=(),
                reasoning_strategy=final_plan.strategy,
                plan_revision_count=len(outcomes),
                plan_steps=trace_steps,
                coverage=final_coverage,
                search_queries_override=search_queries,
            )
        reflection_decision = await self._review_generated_answer(
            question=question,
            standalone_query=query_analysis.standalone_query,
            question_type=query_analysis.question_type,
            generated=generated,
            evidence=evidence,
            images=candidate_images,
            retrieval_degraded=(
                planner_degraded
                or retrieval.diagnostics.vector_status == "degraded"
                or any(outcome.rerank_degraded for outcome in outcomes)
            ),
            coverage=final_coverage,
            allow_retrieval_retry=False,
        )
        if (
            reflection_decision.action == "rewrite"
            and reflection_decision.rewrite_mode == "regenerate_answer"
            and reflection_decision.revision_policy is not None
        ):
            generated = await self._regenerate_answer_once(
                question=question,
                standalone_query=query_analysis.standalone_query,
                history=history,
                conversation_summary=conversation_summary,
                evidence=evidence,
                interaction_memory=interaction_memory,
                images=candidate_images,
                response_policy=(
                    runtime_skill.response_policy if runtime_skill is not None else None
                ),
                revision_policy=reflection_decision.revision_policy,
            )
            if generated.outcome == "abstain":
                return await self._finalize_answer(
                    self._abstain_result(
                        generated,
                        retrieval_mode=retrieval.mode,
                        diagnostics=retrieval.diagnostics,
                        resolved_document_ids=scope.document_ids,
                        resolved_document_titles=resolved_titles,
                    ),
                    trace_id=trace_id,
                    started=started,
                    question=question,
                    request_route=request_route,
                    history=history,
                    conversation_summary=conversation_summary,
                    prepared_query=prepared_query,
                    requested_document_ids=requested_document_ids,
                    analysis=query_analysis,
                    retrieval=retrieval,
                    snapshot=snapshot,
                    selected_evidence=(),
                    document_evidence=(),
                    reasoning_strategy=final_plan.strategy,
                    plan_revision_count=len(outcomes),
                    plan_steps=trace_steps,
                    coverage=final_coverage,
                    search_queries_override=search_queries,
                )
            if self._reflection_service is not None:
                reflection_decision = self._reflection_service.validate_repaired(
                    question_type=query_analysis.question_type,
                    answer=generated,
                    evidence=evidence,
                    images=candidate_images,
                    retrieval_degraded=(
                        planner_degraded
                        or retrieval.diagnostics.vector_status == "degraded"
                        or any(outcome.rerank_degraded for outcome in outcomes)
                    ),
                    coverage=final_coverage,
                )
        elif reflection_decision.action == "rewrite":
            reflection_decision = KnowledgeAnswerReflectionDecision(
                action="refuse",
                confidence=1.0,
                reason_code="retry_query_unavailable",
            )
        if reflection_decision.action in {"ask", "select", "refuse"}:
            return await self._finalize_answer(
                self._reflection_result(
                    reflection_decision,
                    resolved_document_ids=scope.document_ids,
                    resolved_document_titles=resolved_titles,
                ).model_copy(
                    update={
                        "retrieval_mode": retrieval.mode,
                        "diagnostics": retrieval.diagnostics,
                    }
                ),
                trace_id=trace_id,
                started=started,
                question=question,
                request_route=request_route,
                history=history,
                conversation_summary=conversation_summary,
                prepared_query=prepared_query,
                requested_document_ids=requested_document_ids,
                analysis=query_analysis,
                retrieval=retrieval,
                snapshot=snapshot,
                selected_evidence=(),
                document_evidence=(),
                reasoning_strategy=final_plan.strategy,
                plan_revision_count=len(outcomes),
                plan_steps=trace_steps,
                coverage=final_coverage,
                search_queries_override=search_queries,
            )
        answer_text = self._append_uncovered_optional_facets(
            generated.answer,
            final_plan,
            final_coverage,
        )
        evidence_by_id = {record.chunk_id: record for record in evidence}
        cited_records = tuple(
            evidence_by_id[chunk_id]
            for chunk_id in generated.cited_chunk_ids
            if chunk_id in evidence_by_id
        )
        images_by_id = {image.image_id: image for image in candidate_images}
        cited_images = tuple(
            images_by_id[image_id]
            for image_id in generated.cited_image_ids
            if image_id in images_by_id
        )
        degraded_components: list[str] = []
        if query_analysis.degraded:
            degraded_components.append("query_analysis")
        if planner_degraded:
            degraded_components.append("planner")
        if retrieval.diagnostics.vector_status == "degraded":
            degraded_components.append("vector")
        if any(outcome.rerank_degraded for outcome in outcomes):
            degraded_components.append("rerank")
        if generated.degraded:
            degraded_components.append("answer")
        citation_numbers: dict[str, int] = {}
        citations: list[KnowledgeCitation] = []
        for record in cited_records:
            citation_id = citation_numbers.get(record.document_id)
            if citation_id is None:
                citation_id = len(citation_numbers) + 1
                citation_numbers[record.document_id] = citation_id
            citations.append(self._citation(record, citation_id=str(citation_id)))
        return await self._finalize_answer(
            KnowledgeAnswerResult(
                status="degraded" if degraded_components else "success",
                answer=answer_text,
                citations=tuple(citations),
                images=tuple(
                    self._image_citation(image, citation_id=f"图{index}")
                    for index, image in enumerate(cited_images, start=1)
                ),
                retrieval_mode=retrieval.mode,
                diagnostics=retrieval.diagnostics,
                degraded_components=tuple(degraded_components),
                resolved_document_ids=scope.document_ids,
                resolved_document_titles=tuple(resolved_titles),
            ),
            trace_id=trace_id,
            started=started,
            question=question,
            request_route=request_route,
            history=history,
            conversation_summary=conversation_summary,
            prepared_query=prepared_query,
            requested_document_ids=requested_document_ids,
            analysis=query_analysis,
            retrieval=retrieval,
            snapshot=snapshot,
            selected_evidence=evidence,
            document_evidence=document_evidence,
            reasoning_strategy=final_plan.strategy,
            plan_revision_count=len(outcomes),
            plan_steps=trace_steps,
            coverage=final_coverage,
            search_queries_override=search_queries,
        )

    async def aclose(self) -> None:
        """关闭当前服务生命周期托管的模型和检索客户端。"""

        if self._closed:
            return
        self._closed = True
        await self._query_analysis_agent.aclose()
        if self._reasoning_planner_agent is not None:
            await self._reasoning_planner_agent.aclose()
        if self._chunk_rerank_agent is not None:
            await self._chunk_rerank_agent.aclose()
        if self._reflection_service is not None:
            await self._reflection_service.aclose()
        await self._answer_agent.aclose()
        await self._search.aclose()

    async def _finalize_answer(
        self,
        result: KnowledgeAnswerResult,
        *,
        trace_id: str,
        started: float,
        question: str,
        request_route: Literal[
            "/api/v1/knowledge/ask",
            "/api/v1/chat",
        ],
        history: Sequence[ConversationTurn],
        conversation_summary: str | None,
        prepared_query: str | None,
        requested_document_ids: Sequence[str],
        analysis: KnowledgeQueryAnalysis | None = None,
        retrieval: KnowledgeSearchResult | None = None,
        snapshot: Sequence[KnowledgeChunkRecord] = (),
        selected_evidence: Sequence[KnowledgeChunkRecord] = (),
        document_evidence: Sequence[KnowledgeDocumentEvidence] = (),
        plan_search_degraded: bool = False,
        reasoning_strategy: KnowledgeReasoningStrategy | None = None,
        plan_revision_count: int = 0,
        plan_steps: Sequence[KnowledgePlanTraceStep] = (),
        coverage: KnowledgePlanCoverage | None = None,
        search_queries_override: Sequence[str] | None = None,
    ) -> KnowledgeAnswerResult:
        """统一生成公开诊断，并尽力追加运行期测试记录。"""

        protected_analysis = analysis or KnowledgeQueryAnalysis(
            standalone_query=prepared_query or question,
            question_type="factual",
            strategy="direct",
            confidence=0.0,
            degraded=False,
        )
        protected_retrieval = retrieval or KnowledgeSearchResult()
        records_by_id = {
            record.chunk_id: KnowledgeChunkRecord.model_validate(record)
            for record in snapshot
        }
        selected_ids = {record.chunk_id for record in selected_evidence}
        retrieved_chunks = tuple(
            self._execution_chunk(
                rank=rank,
                hit=hit,
                record=records_by_id.get(hit.chunk_id),
                selected=hit.chunk_id in selected_ids,
            )
            for rank, hit in enumerate(protected_retrieval.hits, start=1)
            if records_by_id.get(hit.chunk_id) is not None
        )
        retrieved_ids_by_document: dict[str, list[str]] = {}
        for chunk in retrieved_chunks:
            retrieved_ids_by_document.setdefault(
                chunk.document_id,
                [],
            ).append(chunk.chunk_id)
        documents = tuple(
            KnowledgeExecutionDocument(
                document_id=document.document_id,
                title=document.title,
                score=document.score,
                retrieved_chunk_ids=tuple(
                    retrieved_ids_by_document.get(document.document_id, ())
                ),
                selected_chunk_ids=tuple(record.chunk_id for record in document.chunks),
            )
            for document in document_evidence
        )
        search_queries = (
            tuple(search_queries_override)
            if (search_queries_override is not None)
            else (
                (protected_analysis.standalone_query,)
                if protected_analysis.strategy == "direct"
                else tuple(protected_analysis.sub_queries)
                + (
                    (protected_analysis.standalone_query,)
                    if plan_search_degraded
                    else ()
                )
            )
        )
        trace = KnowledgeExecutionTrace(
            trace_id=trace_id,
            request_route=request_route,
            question=question,
            input=KnowledgeExecutionInput(
                history_message_count=min(len(tuple(history)), 12),
                has_conversation_summary=bool(conversation_summary),
                prepared_query=prepared_query is not None,
                requested_document_ids=tuple(requested_document_ids)[:20],
            ),
            standalone_query=protected_analysis.standalone_query,
            uses_history=protected_analysis.uses_history,
            question_type=protected_analysis.question_type,
            strategy=protected_analysis.strategy,
            sub_queries=protected_analysis.sub_queries,
            confidence=protected_analysis.confidence,
            search_queries=search_queries,
            reasoning_strategy=reasoning_strategy,
            plan_revision_count=plan_revision_count,
            plan_steps=tuple(plan_steps),
            coverage=coverage,
            retrieved_chunks=retrieved_chunks,
            documents=documents,
            retrieval_mode=protected_retrieval.mode,
            diagnostics=protected_retrieval.diagnostics,
            result=KnowledgeExecutionResult(
                status=result.status,
                citation_count=len(result.citations),
                image_count=len(result.images),
                elapsed_ms=(time.perf_counter() - started) * 1000,
                degraded_components=result.degraded_components,
            ),
        )
        completed = result.model_copy(update={"execution_trace": trace})
        emit_stream_event(
            stage="最终结果",
            component="knowledge_qa_service",
            status=(
                "degraded"
                if completed.status == "degraded"
                else "success"
                if completed.status == "success"
                else "degraded"
            ),
            title="知识问答结果完成",
            summary=(
                f"{completed.status} · {len(completed.citations)} 个引用 · "
                f"{len(completed.images)} 张图片"
            ),
            details={
                "status": completed.status,
                "citation_count": len(completed.citations),
                "image_count": len(completed.images),
                "degraded_components": completed.degraded_components,
                "elapsed_ms": trace.result.elapsed_ms,
            },
        )
        if (
            self._execution_record_writer is not None
            and current_conversation_stream() is None
        ):
            try:
                await self._execution_record_writer.append(trace)
                emit_stream_event(
                    stage="测试记录",
                    component="knowledge_test_record_writer",
                    status="success",
                    title="知识问答摘要已写入测试记录",
                    summary="已保存当前知识问答安全摘要",
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(
                    "知识问答测试记录失败，保留当前回答",
                    extra={"exception_type": type(exc).__name__},
                )
                emit_stream_event(
                    stage="测试记录",
                    component="knowledge_test_record_writer",
                    status="degraded",
                    title="知识问答摘要记录失败",
                    summary="不影响当前回答",
                    details={"error_type": type(exc).__name__},
                )
        return completed

    async def _ready_images_for_evidence(
        self,
        evidence: Sequence[KnowledgeChunkRecord],
    ) -> tuple[KnowledgeImageEvidence, ...]:
        """只恢复最终合并 Chunk 真实关联且二进制可读的 ready 图片。"""

        linked_images = await asyncio.to_thread(
            self._repository.list_ready_images_by_chunk_ids,
            tuple(record.chunk_id for record in evidence),
        )
        candidate_images = self._evidence_selector.select_linked_images(
            evidence=evidence,
            images=linked_images,
            max_images=6,
        )
        image_availability = await asyncio.gather(
            *(
                asyncio.to_thread(self.get_image_file, image.image_id)
                for image in candidate_images
            )
        )
        return tuple(
            image
            for image, image_file in zip(
                candidate_images,
                image_availability,
                strict=True,
            )
            if image_file is not None
        )

    @classmethod
    def _plan_trace_retrieval(
        cls,
        outcomes: Sequence[KnowledgePlanRoundOutcome],
    ) -> KnowledgeSearchResult:
        """把多轮白名单关系投影为兼容的有界检索诊断。"""

        records_by_id = {
            record.chunk_id: record
            for outcome in outcomes
            for record in outcome.records
        }
        scores: dict[str, float] = {}
        for outcome in outcomes:
            for relation in outcome.relations:
                scores[relation.chunk_id] = max(
                    scores.get(relation.chunk_id, 0.0),
                    relation.score,
                )
        ranked_ids = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))
        hits = tuple(
            KnowledgeSearchHit(
                chunk_id=chunk_id,
                content_hash=records_by_id[chunk_id].content_hash,
                score=scores[chunk_id],
                bm25_rank=rank,
            )
            for rank, chunk_id in enumerate(ranked_ids[:20], start=1)
        )
        return KnowledgeSearchResult(
            hits=hits,
            mode=(
                "hybrid"
                if any(outcome.retrieval_mode == "hybrid" for outcome in outcomes)
                else "bm25"
            ),
            diagnostics=KnowledgeRetrievalDiagnostics(
                bm25_status=cls._aggregate_channel_status(
                    tuple(outcome.diagnostics.bm25_status for outcome in outcomes)
                ),
                vector_status=cls._aggregate_channel_status(
                    tuple(outcome.diagnostics.vector_status for outcome in outcomes)
                ),
            ),
        )

    @staticmethod
    def _plan_search_queries(
        outcomes: Sequence[KnowledgePlanRoundOutcome],
    ) -> tuple[str, ...]:
        """按首次出现顺序合并两轮实际查询并保持 Trace 上限。"""

        result: list[str] = []
        seen: set[str] = set()
        for outcome in outcomes:
            for query in outcome.search_queries:
                key = " ".join(query.split()).casefold()
                if key not in seen:
                    result.append(query)
                    seen.add(key)
        return tuple(result[:10])

    @staticmethod
    def _emit_plan_execution_event(
        outcome: KnowledgePlanRoundOutcome,
    ) -> None:
        """发布单版计划执行完成的有界业务事实。"""

        emit_stream_event(
            stage="执行步骤",
            component="knowledge_plan_execution",
            status="degraded" if outcome.rerank_degraded else "success",
            title=f"第 {outcome.plan.revision} 版计划步骤执行完成",
            summary=(
                f"{len(outcome.plan.steps)} 个步骤 · {len(outcome.records)} 个 Chunk"
            ),
            details={
                "revision": outcome.plan.revision,
                "strategy": outcome.plan.strategy,
                "step_ids": tuple(step.step_id for step in outcome.plan.steps),
                "search_query_count": len(outcome.search_queries),
                "chunk_count": len(outcome.records),
                "chunk_ids": tuple(record.chunk_id for record in outcome.records),
                "relation_count": len(outcome.relations),
                "rerank_degraded": outcome.rerank_degraded,
            },
        )

    @staticmethod
    def _emit_plan_coverage_event(
        plan: KnowledgeReasoningPlan,
        coverage: KnowledgePlanCoverage,
    ) -> None:
        """发布确定性覆盖检查的有界状态与原因码。"""

        emit_stream_event(
            stage="检查覆盖",
            component="knowledge_plan_coverage",
            status=("success" if coverage.decision == "answer" else "degraded"),
            title=f"第 {plan.revision} 版计划覆盖检查完成",
            summary=(
                f"必需步骤 {coverage.covered_required_steps}/"
                f"{coverage.required_steps} · {coverage.decision}"
            ),
            details={
                "revision": plan.revision,
                "required_steps": coverage.required_steps,
                "covered_required_steps": coverage.covered_required_steps,
                "covered_steps": coverage.covered_steps,
                "coverage_ratio": coverage.coverage_ratio,
                "replanned": coverage.replanned,
                "decision": coverage.decision,
                "step_results": tuple(
                    {
                        "step_id": result.step_id,
                        "status": result.status,
                        "reason_code": result.reason_code,
                        "selected_chunk_ids": result.selected_chunk_ids,
                    }
                    for result in coverage.step_results
                ),
            },
        )

    @staticmethod
    def _plan_trace_steps(
        outcomes: Sequence[KnowledgePlanRoundOutcome],
        coverages: Sequence[KnowledgePlanCoverage],
    ) -> tuple[KnowledgePlanTraceStep, ...]:
        """将各版严格覆盖结果投影为最多十条公开步骤事实。"""

        trace_steps: list[KnowledgePlanTraceStep] = []
        for outcome, coverage in zip(outcomes, coverages, strict=False):
            results_by_id = {result.step_id: result for result in coverage.step_results}
            for step in outcome.plan.steps:
                result = results_by_id.get(step.step_id)
                if result is None:
                    continue
                trace_steps.append(
                    KnowledgePlanTraceStep(
                        revision=outcome.plan.revision,
                        step_id=step.step_id,
                        facet=step.facet,
                        query=step.query,
                        required=step.required,
                        status=result.status,
                        reason_code=result.reason_code,
                        selected_chunk_ids=result.selected_chunk_ids,
                    )
                )
        return tuple(trace_steps[:10])

    @staticmethod
    def _append_uncovered_optional_facets(
        answer: str,
        plan: KnowledgeReasoningPlan,
        coverage: KnowledgePlanCoverage,
    ) -> str:
        """分析型答案只追加枚举化的未覆盖可选维度，不拼接自由理由。"""

        if plan.question_type != "analytical":
            return answer
        status_by_id = {
            result.step_id: result.status for result in coverage.step_results
        }
        facet_labels = {
            "subject": "事实基础",
            "definition": "定义",
            "mechanism": "机制",
            "cause": "原因",
            "impact": "影响",
            "constraint": "限制",
            "tradeoff": "权衡",
            "scenario": "场景",
            "alternative": "替代方案",
            "comparison": "比较",
            "example": "示例",
        }
        uncovered = tuple(
            dict.fromkeys(
                facet_labels[step.facet]
                for step in plan.steps
                if not step.required and status_by_id[step.step_id] != "covered"
            )
        )
        if not uncovered:
            return answer
        return answer.rstrip() + "\n\n未覆盖维度：" + "、".join(uncovered) + "。"

    @classmethod
    def _execution_chunk(
        cls,
        *,
        rank: int,
        hit: KnowledgeSearchHit,
        record: KnowledgeChunkRecord | None,
        selected: bool,
    ) -> KnowledgeExecutionChunk:
        """把可信 SQLite Chunk 和公开分数投影为安全诊断。"""

        if record is None:
            raise ValueError("执行诊断不得使用未回查的 Chunk")
        excerpt = " ".join(record.content.split())
        if len(excerpt) > 220:
            excerpt = excerpt[:219].rstrip() + "…"
        return KnowledgeExecutionChunk(
            rank=rank,
            chunk_id=record.chunk_id,
            document_id=record.document_id,
            title=record.title,
            heading_path=record.heading_path,
            score=hit.score,
            bm25_rank=hit.bm25_rank,
            vector_rank=hit.vector_rank,
            selected=selected,
            excerpt=excerpt,
        )

    def _match_runtime_skill(
        self,
        question: str,
        *,
        snapshot: Sequence[KnowledgeChunkRecord],
        document_ids: Sequence[str],
    ) -> RuntimeSkillMatchResult:
        """捕获一次 Skill Snapshot，并在异常时按无 Skill 继续。"""

        if self._runtime_skill_registry is None:
            return RuntimeSkillMatchResult()
        try:
            skill_snapshot = self._runtime_skill_registry.capture_snapshot()
            topics_by_document_id: dict[str, tuple[str, ...]] = {}
            for record in snapshot:
                existing = topics_by_document_id.get(record.document_id, ())
                topics_by_document_id[record.document_id] = tuple(
                    dict.fromkeys((*existing, *record.topics))
                )
            return RuntimeSkillMatchResult.model_validate(
                self._runtime_skill_matcher.match(
                    question,
                    skills=tuple(skill_snapshot.skills.values()),
                    document_ids=document_ids,
                    document_topics_by_id=topics_by_document_id,
                )
            )
        except Exception as exc:
            logger.warning(
                "运行时 Skill 匹配失败，按无 Skill 继续",
                extra={"exception_type": type(exc).__name__},
            )
            return RuntimeSkillMatchResult()

    @staticmethod
    def _gate_answer_result(
        decision: KnowledgeEvidenceDecision,
        *,
        resolved_document_ids: Sequence[str] = (),
        resolved_document_titles: Sequence[str] = (),
    ) -> KnowledgeAnswerResult:
        """把内部五类决策投影为兼容公开结果并清空非回答证据。"""

        protected = KnowledgeEvidenceDecision.model_validate(decision)
        if protected.action == "ask":
            answer = protected.clarification_question or "请补充必要信息。"
            status = "needs_clarification"
        elif protected.action == "select":
            options = "\n".join(
                f"{index}. {option.label} [{option.option_id}]"
                for index, option in enumerate(protected.options, start=1)
            )
            answer = f"请从以下可信选项中选择：\n{options}"
            status = "needs_clarification"
        elif protected.action == "refuse":
            messages = {
                "unsafe_request": "该请求不允许进入知识回答流程。",
                "out_of_scope": "当前请求超出可用知识范围。",
                "skill_scope_conflict": "请求范围与匹配到的运行时 Skill 范围冲突。",
                "invalid_gate_input": "当前证据信号无效，无法安全回答。",
            }
            answer = messages.get(
                protected.reason_code,
                "当前知识库中没有找到足够依据。",
            )
            status = "insufficient_evidence"
        else:
            raise ValueError("只有 ask、select 或 refuse 可以直接投影结果")
        return KnowledgeAnswerResult(
            status=status,
            answer=answer,
            citations=(),
            images=(),
            resolved_document_ids=tuple(resolved_document_ids),
            resolved_document_titles=tuple(resolved_document_titles),
        )

    async def _analyze_query(
        self,
        question: str,
        *,
        prepared_query: str | None,
        history: Sequence[ConversationTurn],
        conversation_summary: str | None,
    ) -> KnowledgeQueryAnalysis:
        """保护统一分析器，并禁止再次改写主聊天预生成查询。"""

        analysis_input = prepared_query or question
        try:
            analysis = KnowledgeQueryAnalysis.model_validate(
                await self._query_analysis_agent.analyze(
                    analysis_input,
                    history=() if prepared_query is not None else history,
                    conversation_summary=(
                        None if prepared_query is not None else conversation_summary
                    ),
                )
            )
            if (
                prepared_query is not None
                and analysis.standalone_query != prepared_query
            ):
                raise ValueError("知识查询分析不得改写预生成查询")
            return analysis
        except asyncio.CancelledError:
            raise
        except Exception:
            return KnowledgeQueryAnalysis(
                standalone_query=analysis_input,
                question_type="factual",
                strategy="direct",
                confidence=0.0,
                degraded=True,
            )

    async def _search_with_plan(
        self,
        plan: KnowledgeQueryAnalysis,
        *,
        document_ids: Sequence[str],
    ) -> tuple[KnowledgeSearchResult, bool]:
        """执行直接或有界并行检索，分解失败时退回原查询。"""

        if plan.strategy == "direct":
            return (
                await self._search.search(
                    plan.standalone_query,
                    limit=self._RETRIEVAL_LIMIT,
                    document_ids=document_ids,
                ),
                False,
            )
        try:
            outcomes = await asyncio.gather(
                *(
                    self._search.search(
                        sub_query,
                        limit=self._RETRIEVAL_LIMIT,
                        document_ids=document_ids,
                    )
                    for sub_query in plan.sub_queries
                ),
                return_exceptions=True,
            )
            results: list[KnowledgeSearchResult] = []
            for outcome in outcomes:
                if isinstance(outcome, asyncio.CancelledError):
                    raise outcome
                if isinstance(outcome, Exception):
                    raise outcome
                results.append(KnowledgeSearchResult.model_validate(outcome))
            return self._fuse_plan_results(tuple(results)), False
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "知识分解检索失败，退回原查询直接检索",
                extra={"exception_type": type(exc).__name__},
            )
            return (
                await self._search.search(
                    plan.standalone_query,
                    limit=self._RETRIEVAL_LIMIT,
                    document_ids=document_ids,
                ),
                True,
            )

    @classmethod
    def _fuse_plan_results(
        cls,
        results: Sequence[KnowledgeSearchResult],
    ) -> KnowledgeSearchResult:
        """按请求级 RRF 去重多查询命中，并稳定归一到零至一。"""

        if not results:
            return KnowledgeSearchResult()
        scores: dict[str, float] = {}
        hashes: dict[str, str] = {}
        bm25_ranks: dict[str, int] = {}
        vector_ranks: dict[str, int] = {}
        for result in results:
            seen_in_result: set[str] = set()
            for rank, hit in enumerate(result.hits, start=1):
                if hit.chunk_id in seen_in_result:
                    continue
                seen_in_result.add(hit.chunk_id)
                current_hash = hashes.setdefault(hit.chunk_id, hit.content_hash)
                if current_hash != hit.content_hash:
                    raise ValueError("知识分解检索命中 Hash 冲突")
                scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (
                    cls._PLAN_RRF_K + rank
                )
                if hit.bm25_rank is not None:
                    bm25_ranks[hit.chunk_id] = min(
                        bm25_ranks.get(hit.chunk_id, hit.bm25_rank),
                        hit.bm25_rank,
                    )
                if hit.vector_rank is not None:
                    vector_ranks[hit.chunk_id] = min(
                        vector_ranks.get(hit.chunk_id, hit.vector_rank),
                        hit.vector_rank,
                    )
        theoretical_max = len(results) / (cls._PLAN_RRF_K + 1)
        fused_hits = tuple(
            sorted(
                (
                    KnowledgeSearchHit(
                        chunk_id=chunk_id,
                        content_hash=hashes[chunk_id],
                        score=min(score / theoretical_max, 1.0),
                        bm25_rank=bm25_ranks.get(chunk_id),
                        vector_rank=vector_ranks.get(chunk_id),
                    )
                    for chunk_id, score in scores.items()
                ),
                key=lambda hit: (-hit.score, hit.chunk_id),
            )[: cls._RETRIEVAL_LIMIT]
        )
        return KnowledgeSearchResult(
            hits=fused_hits,
            mode=(
                "hybrid"
                if any(result.mode == "hybrid" for result in results)
                else "bm25"
            ),
            diagnostics=KnowledgeRetrievalDiagnostics(
                bm25_status=cls._aggregate_channel_status(
                    result.diagnostics.bm25_status for result in results
                ),
                vector_status=cls._aggregate_channel_status(
                    result.diagnostics.vector_status for result in results
                ),
            ),
        )

    @staticmethod
    def _aggregate_channel_status(statuses: Sequence[str]) -> str:
        """按 degraded、executed、skipped 的保护优先级汇总通道状态。"""

        values = tuple(statuses)
        if "degraded" in values:
            return "degraded"
        if "executed" in values:
            return "executed"
        return "skipped"

    async def refresh_index(self) -> None:
        """幂等重切 ready 文档，并刷新当前进程派生索引。"""

        async with self._index_lock:
            snapshot = await asyncio.to_thread(self._repository.list_ready_chunks)
            await asyncio.to_thread(
                self._rechunk_ready_documents,
                snapshot,
            )
            snapshot = await asyncio.to_thread(self._repository.list_ready_chunks)
            await self._search.refresh(snapshot)
            await self._collect_unreferenced_images()

    async def _collect_unreferenced_images(self) -> None:
        """清理不可见孤儿图片；失败不影响文本索引可用性。"""

        if self._image_store is None:
            return
        try:
            referenced_keys = await asyncio.to_thread(
                self._repository.list_ready_image_storage_keys
            )
            await asyncio.to_thread(
                self._image_store.delete_unreferenced,
                referenced_keys,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "知识图片孤儿清理失败，保留文本索引",
                extra={"exception_type": type(exc).__name__},
            )

    def _rechunk_ready_documents(
        self,
        snapshot: Sequence[KnowledgeChunkRecord],
    ) -> None:
        records_by_document: dict[str, list[KnowledgeChunkRecord]] = {}
        for record in snapshot:
            records_by_document.setdefault(record.document_id, []).append(record)
        for document_id, current_records in records_by_document.items():
            try:
                document = self._repository.get_document(document_id)
                if document is None:
                    logger.warning(
                        "ready Chunk 对应文档不存在，保留当前派生数据",
                        extra={"document_id": document_id},
                    )
                    continue
                derivation = self._media_extractor.derive(
                    document_id=document_id,
                    content_markdown=document.content_markdown,
                )
                chunks = derivation.chunks
                if not chunks:
                    logger.warning(
                        "ready 文档预处理后为空，保留当前派生数据",
                        extra={"document_id": document_id},
                    )
                    continue
                current_signature = tuple(
                    (
                        record.chunk_id,
                        record.position,
                        record.heading_path,
                        record.content_hash,
                        record.token_count,
                    )
                    for record in current_records
                )
                next_signature = tuple(
                    (
                        chunk.chunk_id,
                        chunk.position,
                        chunk.heading_path,
                        chunk.content_hash,
                        chunk.token_count,
                    )
                    for chunk in chunks
                )
                if current_signature != next_signature:
                    self._repository.replace_document_bundle(
                        document,
                        chunks,
                        derivation.images,
                        derivation.links,
                    )
            except Exception as exc:
                logger.warning(
                    "ready 文档重切失败，保留当前派生数据",
                    extra={
                        "document_id": document_id,
                        "exception_type": type(exc).__name__,
                    },
                )

    @staticmethod
    def _protect_interaction_memory(
        value: UserInteractionMemoryProjection | None,
    ) -> UserInteractionMemoryProjection | None:
        if value is None:
            return None
        try:
            return UserInteractionMemoryProjection.model_validate(value).model_copy(
                deep=True
            )
        except ValueError:
            logger.warning("用户交互记忆投影无效，按默认回答方式继续")
            return None

    @staticmethod
    def _recheck_candidates(
        retrieval: KnowledgeSearchResult,
        records: Sequence[KnowledgeChunkRecord],
        *,
        scope_document_ids: Sequence[str],
    ) -> tuple[KnowledgeChunkRecord, ...]:
        if not retrieval.hits:
            return ()
        records_by_id = {
            record.chunk_id: KnowledgeChunkRecord.model_validate(record)
            for record in records
        }
        scope = frozenset(scope_document_ids)
        selected: list[KnowledgeChunkRecord] = []
        seen_ids: set[str] = set()
        seen_content: set[tuple[str, str]] = set()
        for hit in retrieval.hits:
            if hit.chunk_id in seen_ids:
                continue
            record = records_by_id.get(hit.chunk_id)
            if record is None or record.content_hash != hit.content_hash:
                continue
            if scope and record.document_id not in scope:
                continue
            content_key = (record.document_id, record.content_hash)
            if content_key in seen_content:
                continue
            selected.append(record.model_copy(deep=True))
            seen_ids.add(record.chunk_id)
            seen_content.add(content_key)
        return tuple(selected)

    @staticmethod
    def _citation(
        record: KnowledgeChunkRecord,
        *,
        citation_id: str,
    ) -> KnowledgeCitation:
        excerpt = " ".join(record.content.split())
        if len(excerpt) > 360:
            excerpt = excerpt[:359].rstrip() + "…"
        return KnowledgeCitation(
            citation_id=citation_id,
            document_id=record.document_id,
            title=record.title,
            chunk_id=record.chunk_id,
            heading_path=record.heading_path,
            excerpt=excerpt,
        )

    @staticmethod
    def _image_citation(
        image: KnowledgeImageEvidence,
        *,
        citation_id: str,
    ) -> KnowledgeImageCitation:
        """把实际使用的白名单图片渲染为公开引用。"""

        return KnowledgeImageCitation(
            citation_id=citation_id,
            image_id=image.image_id,
            document_id=image.document_id,
            title=image.title,
            heading_path=image.heading_path,
            caption=image.caption,
            url=(
                f"/api/v1/knowledge/images/{image.image_id}?v={image.content_hash[:12]}"
            ),
        )

    @staticmethod
    def _required_text(value: str, label: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{label}不能为空")
        return value.strip()

    @classmethod
    def _required_query(cls, value: str, label: str) -> str:
        """清理并限制直接进入共享检索器的预生成查询。"""

        cleaned = " ".join(cls._required_text(value, label).split())
        if len(cleaned) > 500:
            raise ValueError(f"{label}长度不能超过 500 个字符")
        return cleaned


__all__ = ["KnowledgeImageFile", "KnowledgeImageStore", "KnowledgeQaService"]
