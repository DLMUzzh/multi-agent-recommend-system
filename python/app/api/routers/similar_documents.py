"""独立相似文章推荐 HTTP 接口。"""

from __future__ import annotations

from fastapi import APIRouter, FastAPI, Query, Request, status
from fastapi.responses import JSONResponse

from app.api.dependencies import get_similar_document_recommendation_service
from app.api.errors import (
    degraded_components,
    error_response,
    required_text,
)
from app.application.similar_document_recommendation import DocumentNotFoundError
from app.models.schemas import (
    ErrorResponse,
    PublicArticleRecommendation,
    SimilarDocumentResponse,
)


router = APIRouter(prefix="/api/v1", tags=["相似文章推荐"])


@router.get(
    "/documents/{document_id}/similar",
    response_model=SimilarDocumentResponse,
    responses={
        404: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
async def recommend_similar_documents(
    request: Request,
    document_id: str,
    user_id: str = Query(min_length=1),
) -> SimilarDocumentResponse:
    """根据用户当前阅读文档返回最多五篇相关文档。"""

    normalized_document_id = required_text(document_id)
    normalized_user_id = required_text(user_id)
    result = await get_similar_document_recommendation_service(request).recommend(
        user_id=normalized_user_id,
        document_id=normalized_document_id,
    )
    components = degraded_components(result.agent_statuses)
    return SimilarDocumentResponse(
        source_document_id=result.source_document_id,
        recommendations=[
            PublicArticleRecommendation(
                document_id=item.document_id,
                title=item.title,
                excerpt=item.excerpt,
                score=item.score,
                reason=item.reason,
            )
            for item in result.recommendations
        ],
        degraded=bool(components),
        degraded_components=components,
    )


def register_similar_document_error_handlers(app: FastAPI) -> None:
    """注册源文档不存在的稳定安全错误。"""

    app.add_exception_handler(
        DocumentNotFoundError,
        _document_not_found_handler,
    )


async def _document_not_found_handler(
    _: Request,
    __: DocumentNotFoundError,
) -> JSONResponse:
    return error_response(
        status.HTTP_404_NOT_FOUND,
        "DOCUMENT_NOT_FOUND",
        "文档不存在",
    )


__all__ = [
    "register_similar_document_error_handlers",
    "router",
]
