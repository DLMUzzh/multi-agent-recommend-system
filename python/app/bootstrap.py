"""应用启动期依赖装配与外部资源关闭。"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

import structlog
from fastapi import FastAPI

from app.agents.conversation_feedback_agent import ConversationFeedbackAgent
from app.agents.document_recall_agent import DocumentRecallAgent
from app.agents.document_rerank_agent import DocumentRerankAgent
from app.agents.feedback_recovery_agent import FeedbackRecoveryAgent
from app.agents.intent_recognition_agent import IntentRecognitionAgent
from app.agents.knowledge_answer_agent import KnowledgeAnswerAgent
from app.agents.knowledge_answer_reflection_agent import (
    KnowledgeAnswerReflectionAgent,
)
from app.agents.knowledge_chunk_rerank_agent import KnowledgeChunkRerankAgent
from app.agents.knowledge_query_analysis_agent import KnowledgeQueryAnalysisAgent
from app.agents.knowledge_reasoning_planner_agent import (
    KnowledgeReasoningPlannerAgent,
)
from app.agents.user_profile_agent import UserProfileAgent
from app.application.conversation_service import ConversationService
from app.application.knowledge_qa import KnowledgeQaService
from app.application.knowledge_answer_reflection import (
    KnowledgeAnswerReflectionService,
)
from app.application.knowledge_plan_execution import KnowledgePlanExecutor
from app.application.runtime_skill_registry import RuntimeSkillRegistry
from app.application.personal_feedback_learning import (
    PersonalFeedbackLearningService,
)
from app.application.similar_document_recommendation import (
    SimilarDocumentRecommendationService,
)
from app.application.user_interaction_memory import UserInteractionMemoryWorker
from app.config import Settings, get_settings
from app.config.paths import KNOWLEDGE_IMAGE_ROOT, RUNTIME_SKILL_ROOT
from app.domain.services.conversation_arbitrator import ConversationArbitrator
from app.domain.services.document_result_aggregator import DocumentResultAggregator
from app.domain.services.feedback_recovery_policy import FeedbackRecoveryPolicy
from app.domain.services.knowledge_plan_coverage import (
    KnowledgePlanCoverageChecker,
)
from app.domain.services.knowledge_evidence_gate import KnowledgeEvidenceGate
from app.domain.services.knowledge_answer_reflection_policy import (
    KnowledgeAnswerReflectionPolicy,
)
from app.domain.services.runtime_skill_matcher import RuntimeSkillMatcher
from app.domain.services.user_interaction_memory import (
    UserInteractionMemoryService,
)
from app.infrastructure.database.json.feature_store import FeatureStore
from app.infrastructure.database.sqlite.conversation_store import (
    SQLiteConversationStore,
)
from app.infrastructure.database.sqlite.knowledge_repository import (
    SQLiteKnowledgeRepository,
)
from app.infrastructure.database.sqlite.user_profile_repository import (
    SQLiteUserProfileRepository,
)
from app.infrastructure.database.sqlite.user_intent_memory_repository import (
    SQLiteUserIntentMemoryRepository,
)
from app.infrastructure.database.sqlite.user_interaction_memory_repository import (
    SQLiteUserInteractionMemoryRepository,
)
from app.infrastructure.observability.conversation_trace import ConversationTraceWriter
from app.infrastructure.observability.knowledge_test_record import (
    KnowledgeTestRecordWriter,
)
from app.infrastructure.skills.file_skill_catalog import FileSkillCatalog
from app.infrastructure.retrieval.article_embedding import create_embedding_client
from app.infrastructure.retrieval.knowledge_search import InMemoryKnowledgeSearch
from app.infrastructure.storage.local_knowledge_image_store import (
    LocalKnowledgeImageStore,
)


logger = structlog.get_logger()


def _create_knowledge_answer_reflection(
    settings: Settings,
) -> KnowledgeAnswerReflectionService:
    """尽力创建小模型反思组件，失败时保留确定性检查。"""

    try:
        agent = KnowledgeAnswerReflectionAgent.from_settings(settings)
    except Exception as exc:
        logger.warning(
            "启动期知识答案反思 Agent 创建失败，按确定性检查继续",
            exception_type=type(exc).__name__,
        )
        agent = None
    return KnowledgeAnswerReflectionService(
        policy=KnowledgeAnswerReflectionPolicy(),
        agent=agent,
    )


@asynccontextmanager
async def lifespan(current_app: FastAPI) -> AsyncIterator[None]:
    """启动时一次组装共享数据、Agent、编排和会话服务。"""

    settings = get_settings()
    repository = SQLiteKnowledgeRepository()
    user_repository = SQLiteUserProfileRepository()
    intent_memory_repository = SQLiteUserIntentMemoryRepository(user_repository.path)
    interaction_memory_repository = SQLiteUserInteractionMemoryRepository(
        user_repository.path
    )
    interaction_memory_service = UserInteractionMemoryService()
    interaction_memory_worker = UserInteractionMemoryWorker(
        repository=interaction_memory_repository,
        memory_service=interaction_memory_service,
        feedback_agent=ConversationFeedbackAgent.from_settings(settings),
    )
    interaction_feedback_agent = ConversationFeedbackAgent.from_settings(settings)
    feedback_recovery_agent = FeedbackRecoveryAgent.from_settings(settings)
    user_store = FeatureStore(
        user_repository=user_repository,
        document_repository=repository,
    )
    profile_agent = UserProfileAgent(
        feature_store=user_store,
        settings=settings,
    )
    intent_agent = IntentRecognitionAgent(settings=settings)
    arbitrator = ConversationArbitrator()
    embedding_client = create_embedding_client(settings)
    document_search = InMemoryKnowledgeSearch(
        embedding_client=embedding_client,
        embedding_dimensions=(
            settings.embedding_dimensions if embedding_client is not None else None
        ),
        embedding_batch_size=settings.embedding_batch_size,
        rrf_k=settings.recall_rrf_k,
        bm25_k1=settings.recall_bm25_k1,
        bm25_b=settings.recall_bm25_b,
    )
    query_analysis_agent = KnowledgeQueryAnalysisAgent.from_settings(settings)
    chunk_rerank_agent = KnowledgeChunkRerankAgent.from_settings(settings)
    reasoning_planner_agent = KnowledgeReasoningPlannerAgent.from_settings(settings)
    plan_executor = KnowledgePlanExecutor(
        search=document_search,
        reranker=chunk_rerank_agent,
    )
    image_store = LocalKnowledgeImageStore(KNOWLEDGE_IMAGE_ROOT)
    knowledge_test_record_writer = KnowledgeTestRecordWriter()
    runtime_skill_registry = RuntimeSkillRegistry(
        catalog=FileSkillCatalog(root=RUNTIME_SKILL_ROOT)
    )
    runtime_skill_reload = runtime_skill_registry.reload()
    if not runtime_skill_reload.reloaded:
        logger.warning(
            "启动期运行时 Skill 加载失败，按空或旧 Snapshot 继续",
            error_code=runtime_skill_reload.error_code,
        )
    runtime_skill_matcher = RuntimeSkillMatcher()
    evidence_gate = KnowledgeEvidenceGate()
    reflection_service = _create_knowledge_answer_reflection(settings)
    knowledge_qa_service = KnowledgeQaService(
        repository=repository,
        search=document_search,
        answer_agent=KnowledgeAnswerAgent.from_settings(settings),
        chunk_rerank_agent=chunk_rerank_agent,
        query_analysis_agent=query_analysis_agent,
        reasoning_planner_agent=reasoning_planner_agent,
        plan_executor=plan_executor,
        plan_coverage_checker=KnowledgePlanCoverageChecker(),
        request_timeout_seconds=settings.chat_request_timeout_seconds,
        image_store=image_store,
        execution_record_writer=knowledge_test_record_writer,
        runtime_skill_registry=runtime_skill_registry,
        runtime_skill_matcher=runtime_skill_matcher,
        evidence_gate=evidence_gate,
        reflection_service=reflection_service,
    )
    await knowledge_qa_service.refresh_index()
    recall_agent = DocumentRecallAgent(
        repository=repository,
        search=document_search,
    )
    rerank_agent = DocumentRerankAgent(settings=settings)
    aggregator = DocumentResultAggregator()
    conversation_store = SQLiteConversationStore()
    personal_feedback_learning = PersonalFeedbackLearningService(
        feedback_store=conversation_store,
        feature_store=user_store,
    )
    now = datetime.now(timezone.utc)
    try:
        await conversation_store.purge_feedback_raw_before(
            now - timedelta(days=30),
            purged_at=now,
        )
    except Exception as exc:
        logger.warning(
            "启动期个人反馈原文清理失败，保留数据等待运行期重试",
            exception_type=type(exc).__name__,
        )
    conversation_trace_writer = ConversationTraceWriter()
    conversation_service = ConversationService(
        user_store=user_store,
        intent_agent=intent_agent,
        profile_agent=profile_agent,
        recall_agent=recall_agent,
        rerank_agent=rerank_agent,
        arbitrator=arbitrator,
        aggregator=aggregator,
        conversation_store=conversation_store,
        knowledge_qa_service=knowledge_qa_service,
        intent_memory_repository=intent_memory_repository,
        interaction_memory_repository=interaction_memory_repository,
        interaction_memory_service=interaction_memory_service,
        interaction_memory_worker=interaction_memory_worker,
        interaction_feedback_agent=interaction_feedback_agent,
        feedback_agent=feedback_recovery_agent,
        feedback_policy=FeedbackRecoveryPolicy(),
        personal_feedback_learning=personal_feedback_learning,
    )
    similar_document_recommendation_service = SimilarDocumentRecommendationService(
        user_store=user_store,
        repository=repository,
        profile_agent=profile_agent,
        recall_agent=recall_agent,
        rerank_agent=rerank_agent,
        aggregator=aggregator,
    )
    current_app.state.settings = settings
    current_app.state.user_store = user_store
    current_app.state.document_repository = repository
    current_app.state.document_search = document_search
    current_app.state.conversation_store = conversation_store
    current_app.state.conversation_trace_writer = conversation_trace_writer
    current_app.state.knowledge_test_record_writer = knowledge_test_record_writer
    current_app.state.runtime_skill_registry = runtime_skill_registry
    current_app.state.conversation_service = conversation_service
    current_app.state.similar_document_recommendation_service = (
        similar_document_recommendation_service
    )
    current_app.state.knowledge_qa_service = knowledge_qa_service
    current_app.state.user_interaction_memory_worker = interaction_memory_worker
    interaction_memory_worker.start()
    try:
        yield
    finally:
        await interaction_memory_worker.stop()
        await feedback_recovery_agent.aclose()
        await interaction_feedback_agent.aclose()
        await knowledge_qa_service.aclose()
        await rerank_agent.aclose()
        for agent in (
            intent_agent,
            profile_agent,
            conversation_service.summary_agent,
        ):
            close = getattr(agent.llm, "aclose", None)
            if close is not None:
                await close()


__all__ = ["lifespan"]
