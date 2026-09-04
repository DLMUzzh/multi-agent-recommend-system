"""HTTP Controller 共用的公开校验与安全错误体。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

from fastapi import Request, status
from fastapi.responses import JSONResponse

from app.models.schemas import DegradedComponent, ErrorDetail, ErrorResponse


ErrorCode = Literal[
    "USER_NOT_FOUND",
    "DOCUMENT_NOT_FOUND",
    "VALIDATION_ERROR",
    "SERVICE_UNAVAILABLE",
]


class PublicValidationError(ValueError):
    """表示未经过请求模型的公开文本参数校验失败。"""


def required_text(value: str) -> str:
    """返回去除首尾空白的非空文本。"""

    normalized = value.strip()
    if not normalized:
        raise PublicValidationError("公开请求字段不能为空")
    return normalized


async def validation_error_handler(
    _: Request,
    __: Exception,
) -> JSONResponse:
    """把公开参数和 Pydantic 校验失败映射为稳定 422。"""

    return error_response(
        status.HTTP_422_UNPROCESSABLE_ENTITY,
        "VALIDATION_ERROR",
        "请求参数无效",
    )


def error_response(
    status_code: int,
    code: ErrorCode,
    message: str,
) -> JSONResponse:
    """构造不包含内部异常详情的统一错误响应。"""

    payload = ErrorResponse(error=ErrorDetail(code=code, message=message))
    return JSONResponse(
        status_code=status_code,
        content=payload.model_dump(mode="json"),
    )


def degraded_components(
    agent_statuses: Mapping[str, str],
) -> list[DegradedComponent]:
    """把内部 Agent 状态映射为聊天接口公开的降级组件。"""

    return [
        public_name
        for public_name, internal_name in (
            ("user_profile", "user_profile"),
            ("document_recall", "document_recall"),
            ("document_rerank", "document_rerank"),
            ("knowledge_query_analysis", "knowledge_query_analysis"),
            ("knowledge_planner", "knowledge_planner"),
            ("knowledge_plan_execution", "knowledge_plan_execution"),
            ("knowledge_plan_coverage", "knowledge_plan_coverage"),
            ("knowledge_vector", "knowledge_vector"),
            ("knowledge_answer", "knowledge_answer"),
        )
        if agent_statuses.get(internal_name) in {"degraded", "failed"}
    ]


__all__ = [
    "ErrorCode",
    "PublicValidationError",
    "degraded_components",
    "error_response",
    "required_text",
    "validation_error_handler",
]
