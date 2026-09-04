"""从 FastAPI 应用状态读取已装配的应用服务。"""

from fastapi import Request

from app.application.conversation_service import (
    ConversationService,
    ServiceUnavailableError,
)
from app.application.similar_document_recommendation import (
    SimilarDocumentRecommendationService,
)
from app.infrastructure.observability.conversation_trace import (
    ConversationTraceWriter,
)


def get_conversation_service(request: Request) -> ConversationService:
    """返回聊天 Controller 使用的会话应用服务。"""

    service = getattr(request.app.state, "conversation_service", None)
    if service is None:
        raise ServiceUnavailableError("文章推荐服务暂时不可用")
    return service


def get_conversation_trace_writer(
    request: Request,
) -> ConversationTraceWriter | None:
    """返回可选请求追踪写入器，并保护错误装配。"""

    writer = getattr(request.app.state, "conversation_trace_writer", None)
    if writer is None:
        return None
    if not isinstance(writer, ConversationTraceWriter):
        raise ServiceUnavailableError("文章推荐服务暂时不可用")
    return writer


def get_similar_document_recommendation_service(
    request: Request,
) -> SimilarDocumentRecommendationService:
    """返回独立相似文章 Controller 使用的应用服务。"""

    service = getattr(
        request.app.state,
        "similar_document_recommendation_service",
        None,
    )
    if service is None:
        raise ServiceUnavailableError("文章推荐服务暂时不可用")
    return service


__all__ = [
    "get_conversation_service",
    "get_conversation_trace_writer",
    "get_similar_document_recommendation_service",
]
