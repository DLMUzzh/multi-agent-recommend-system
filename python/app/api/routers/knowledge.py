"""无会话知识文档导入与单轮问答 HTTP 路由。"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse

from app.application.knowledge_qa import KnowledgeQaService
from app.models.knowledge_qa import (
    KnowledgeAnswerResult,
    KnowledgeAskRequest,
    KnowledgeDocumentIngestRequest,
    KnowledgeDocumentIngestResult,
    KnowledgeImageUploadResult,
)


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/knowledge", tags=["knowledge"])


class InvalidKnowledgeRequestError(RuntimeError):
    """知识请求通过 HTTP Schema 后仍违反业务输入约束。"""


class KnowledgeServiceUnavailableError(RuntimeError):
    """无会话知识链无法安全完成当前请求。"""


class KnowledgeImageNotFoundError(RuntimeError):
    """图片不存在、尚未就绪或对应二进制已不可读。"""


def _service(request: Request) -> KnowledgeQaService:
    return request.app.state.knowledge_qa_service


@router.post(
    "/documents",
    response_model=KnowledgeDocumentIngestResult,
    status_code=201,
)
async def ingest_document(
    payload: KnowledgeDocumentIngestRequest,
    request: Request,
) -> KnowledgeDocumentIngestResult:
    """导入或替换一篇完整 Markdown 文档。"""

    try:
        return await _service(request).ingest_document(
            document_id=payload.document_id,
            title=payload.title,
            content_markdown=payload.content_markdown,
            topics=payload.topics,
            content_type=payload.content_type,
            difficulty=payload.difficulty,
            author_id=payload.author_id,
        )
    except ValueError as exc:
        raise InvalidKnowledgeRequestError(str(exc)) from None
    except Exception as exc:
        logger.warning(
            "知识文档导入失败",
            extra={"exception_type": type(exc).__name__},
        )
        raise KnowledgeServiceUnavailableError from None


@router.post("/ask", response_model=KnowledgeAnswerResult)
async def ask_knowledge(
    payload: KnowledgeAskRequest,
    request: Request,
) -> KnowledgeAnswerResult:
    """执行一次不持久化会话的知识问答。"""

    try:
        return await _service(request).ask(
            payload.question,
            limit=payload.limit,
        )
    except ValueError as exc:
        raise InvalidKnowledgeRequestError(str(exc)) from None
    except Exception as exc:
        logger.warning(
            "知识问答请求失败",
            extra={"exception_type": type(exc).__name__},
        )
        raise KnowledgeServiceUnavailableError from None


@router.put(
    "/images/{image_id}",
    response_model=KnowledgeImageUploadResult,
)
async def upload_knowledge_image(
    image_id: str,
    request: Request,
) -> KnowledgeImageUploadResult:
    """上传文档中已声明图片的原始字节。"""

    content = bytearray()
    async for chunk in request.stream():
        content.extend(chunk)
        if len(content) > 8 * 1024 * 1024:
            raise InvalidKnowledgeRequestError("图片大小不能超过 8 MiB")
    try:
        return await _service(request).upload_image(
            image_id=image_id,
            content=bytes(content),
            mime_type=request.headers.get("content-type", ""),
        )
    except ValueError as exc:
        raise InvalidKnowledgeRequestError(str(exc)) from None
    except Exception as exc:
        logger.warning(
            "知识图片上传失败",
            extra={"exception_type": type(exc).__name__},
        )
        raise KnowledgeServiceUnavailableError from None


@router.get("/images/{image_id}")
async def get_knowledge_image(
    image_id: str,
    request: Request,
) -> FileResponse:
    """按受保护图片 ID 返回已就绪二进制。"""

    try:
        image_file = _service(request).get_image_file(image_id)
    except ValueError:
        raise KnowledgeImageNotFoundError from None
    except Exception as exc:
        logger.warning(
            "知识图片读取失败",
            extra={"exception_type": type(exc).__name__},
        )
        raise KnowledgeServiceUnavailableError from None
    if image_file is None:
        raise KnowledgeImageNotFoundError
    return FileResponse(
        image_file.path,
        media_type=image_file.mime_type,
        headers={
            "X-Content-Type-Options": "nosniff",
            "ETag": f'"{image_file.content_hash}"',
            "Cache-Control": "private, max-age=3600",
        },
    )


def register_knowledge_error_handlers(app: FastAPI) -> None:
    """注册不暴露内部异常、路径或模型输入的稳定错误响应。"""

    @app.exception_handler(InvalidKnowledgeRequestError)
    async def invalid_request_handler(
        request: Request,
        exc: InvalidKnowledgeRequestError,
    ) -> JSONResponse:
        del request
        return JSONResponse(
            status_code=400,
            content=_error_body("INVALID_KNOWLEDGE_REQUEST", str(exc)),
        )

    @app.exception_handler(KnowledgeServiceUnavailableError)
    async def unavailable_handler(
        request: Request,
        exc: KnowledgeServiceUnavailableError,
    ) -> JSONResponse:
        del request, exc
        return JSONResponse(
            status_code=503,
            content=_error_body(
                "KNOWLEDGE_SERVICE_UNAVAILABLE",
                "知识问答服务暂时不可用",
            ),
        )

    @app.exception_handler(KnowledgeImageNotFoundError)
    async def image_not_found_handler(
        request: Request,
        exc: KnowledgeImageNotFoundError,
    ) -> JSONResponse:
        del request, exc
        return JSONResponse(
            status_code=404,
            content=_error_body(
                "KNOWLEDGE_IMAGE_NOT_FOUND",
                "知识图片不存在或尚未就绪",
            ),
        )


def _error_body(code: str, message: str) -> dict[str, dict[str, Any]]:
    return {"error": {"code": code, "message": message}}


__all__ = [
    "InvalidKnowledgeRequestError",
    "KnowledgeImageNotFoundError",
    "KnowledgeServiceUnavailableError",
    "register_knowledge_error_handlers",
    "router",
]
