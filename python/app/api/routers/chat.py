"""统一聊天、会话查询与会话重置 HTTP 接口。"""

from __future__ import annotations

import asyncio

import structlog
from fastapi import APIRouter, FastAPI, Query, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse

from app.models.schemas import (
    ChatRequest,
    ChatResponse,
    ChatStreamErrorEvent,
    ChatStreamProcessEvent,
    ChatStreamResultEvent,
    ConversationCompressionInfo,
    ConversationReply,
    ConversationSession,
    DegradedComponent,
    ErrorResponse,
    PublicArticleRecommendation,
    PublicRecommendationContext,
    RecognitionSource,
    RecommendationContext,
    SessionHistoryResponse,
)
from app.api.dependencies import (
    get_conversation_service,
    get_conversation_trace_writer,
)
from app.api.errors import (
    PublicValidationError,
    degraded_components,
    error_response,
    required_text,
    validation_error_handler,
)
from app.application.conversation_service import ServiceUnavailableError
from app.infrastructure.database.json.feature_store import UserNotFoundError
from app.infrastructure.observability.conversation_trace import (
    ConversationStreamRecorder,
    conversation_stream_context,
)


logger = structlog.get_logger()
router = APIRouter(prefix="/api/v1", tags=["文章推荐"])


@router.post(
    "/chat",
    response_model=ChatResponse,
    responses={
        404: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
async def chat(request: Request, payload: ChatRequest) -> ChatResponse:
    """继续指定会话，或生成新会话并完成一轮推荐或知识问答。"""

    service = get_conversation_service(request)
    trace_writer = get_conversation_trace_writer(request)
    if trace_writer is None:
        reply = await service.chat(
            payload.user_id,
            payload.message,
            session_id=payload.session_id,
        )
        return _to_chat_response(reply)

    async with trace_writer.trace_request(
        user_id=payload.user_id,
        message=payload.message,
        supplied_session_id=payload.session_id,
    ) as trace:
        reply = await service.chat(
            payload.user_id,
            payload.message,
            session_id=payload.session_id,
        )
        trace.set_resolved_session_id(reply.session_id)
        response = _to_chat_response(reply)
        trace.set_response(response)
        return response


@router.post(
    "/chat/stream",
    response_class=StreamingResponse,
    responses={
        200: {
            "content": {"application/x-ndjson": {}},
            "description": "逐行返回安全执行过程，最后一行为完整聊天结果。",
        },
        422: {"model": ErrorResponse},
    },
)
async def chat_stream(request: Request, payload: ChatRequest) -> StreamingResponse:
    """以 NDJSON 实时返回同一聊天用例的安全业务执行过程。"""

    service = get_conversation_service(request)
    trace_writer = get_conversation_trace_writer(request)
    test_record_writer = getattr(
        request.app.state,
        "knowledge_test_record_writer",
        None,
    )

    async def event_stream():
        recorder = ConversationStreamRecorder()
        recorder.emit(
            stage="请求",
            component="chat_controller",
            status="started",
            title="收到请求",
            summary="开始处理当前聊天请求",
            details={
                "request_route": "/api/v1/chat/stream",
                "message": payload.message,
                "supplied_session_id": payload.session_id,
            },
        )

        async def produce() -> None:
            try:
                with conversation_stream_context(recorder):
                    if trace_writer is None:
                        reply = await service.chat(
                            payload.user_id,
                            payload.message,
                            session_id=payload.session_id,
                        )
                        response = _to_chat_response(reply)
                    else:
                        async with trace_writer.trace_request(
                            user_id=payload.user_id,
                            message=payload.message,
                            supplied_session_id=payload.session_id,
                        ) as trace:
                            reply = await service.chat(
                                payload.user_id,
                                payload.message,
                                session_id=payload.session_id,
                            )
                            trace.set_resolved_session_id(reply.session_id)
                            response = _to_chat_response(reply)
                            trace.set_response(response)
                    if test_record_writer is not None:
                        append_stream = getattr(
                            test_record_writer,
                            "append_stream",
                            None,
                        )
                        if append_stream is not None:
                            try:
                                recorded = await append_stream(
                                    trace_id=recorder.trace_id,
                                    events=recorder.snapshot(),
                                    response=response,
                                )
                            except asyncio.CancelledError:
                                raise
                            except Exception as exc:
                                logger.warning(
                                    "聊天全链路测试记录失败，保留当前结果",
                                    exception_type=type(exc).__name__,
                                )
                                recorded = False
                            recorder.emit(
                                stage="测试记录",
                                component="knowledge_test_record_writer",
                                status="success" if recorded else "degraded",
                                title=(
                                    "全链路测试记录已写入"
                                    if recorded
                                    else "全链路测试记录写入失败"
                                ),
                                summary=(
                                    "已保存本轮安全业务执行轨迹"
                                    if recorded
                                    else "不影响当前聊天结果"
                                ),
                            )
                    recorder.emit(
                        stage="最终结果",
                        component="chat_controller",
                        status="success",
                        title="聊天结果准备完成",
                        summary=(
                            f"{response.action.value} · "
                            f"{len(response.recommendations)} 篇推荐 · "
                            f"{len(response.citations)} 个引用 · "
                            f"{len(response.images)} 张图片"
                        ),
                        details={
                            "session_id": response.session_id,
                            "action": response.action,
                            "intent_state": response.intent_state,
                            "degraded_components": response.degraded_components,
                        },
                    )
                    recorder.terminal(
                        "result",
                        {
                            "response": response.model_dump(mode="json"),
                        },
                    )
            except asyncio.CancelledError:
                raise
            except UserNotFoundError:
                recorder.terminal(
                    "error",
                    {
                        "error": {
                            "code": "USER_NOT_FOUND",
                            "message": "用户不存在",
                        }
                    },
                )
            except ServiceUnavailableError:
                recorder.terminal(
                    "error",
                    {
                        "error": {
                            "code": "SERVICE_UNAVAILABLE",
                            "message": "文章推荐服务暂时不可用",
                        }
                    },
                )
            except Exception:
                logger.exception("聊天流处理失败")
                recorder.terminal(
                    "error",
                    {
                        "error": {
                            "code": "SERVICE_UNAVAILABLE",
                            "message": "文章推荐服务暂时不可用",
                        }
                    },
                )

        producer = asyncio.create_task(produce(), name="chat-stream-producer")
        try:
            while True:
                event = await recorder.next_event()
                if event["event"] == "process":
                    protected = ChatStreamProcessEvent.model_validate(event)
                elif event["event"] == "result":
                    protected = ChatStreamResultEvent.model_validate(event)
                else:
                    protected = ChatStreamErrorEvent.model_validate(event)
                yield (
                    protected.model_dump_json(exclude_none=True)
                    + "\n"
                ).encode("utf-8")
                if event["event"] in {"result", "error"}:
                    break
        finally:
            if not producer.done():
                producer.cancel()
            await asyncio.gather(producer, return_exceptions=True)

    return StreamingResponse(
        event_stream(),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-store",
            "X-Accel-Buffering": "no",
        },
    )


@router.delete(
    "/sessions/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    responses={
        404: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
async def reset_session(
    request: Request,
    session_id: str,
    user_id: str = Query(min_length=1),
) -> Response:
    """幂等删除指定用户的持久会话。"""

    normalized_session_id = required_text(session_id)
    normalized_user_id = required_text(user_id)
    await get_conversation_service(request).reset_session(
        normalized_user_id,
        session_id=normalized_session_id,
    )
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get(
    "/sessions/{session_id}",
    response_model=SessionHistoryResponse,
    responses={
        404: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
async def get_session_history(
    request: Request,
    session_id: str,
    user_id: str = Query(min_length=1),
) -> SessionHistoryResponse:
    """读取指定用户和会话的近期历史、摘要与压缩状态。"""

    normalized_session_id = required_text(session_id)
    normalized_user_id = required_text(user_id)
    session = await get_conversation_service(request).read_session(
        normalized_user_id,
        normalized_session_id,
    )
    return _to_session_history_response(session)


def register_error_handlers(app: FastAPI) -> None:
    """注册稳定错误码，并隐藏内部异常文本和堆栈。"""

    app.add_exception_handler(RequestValidationError, validation_error_handler)
    app.add_exception_handler(PublicValidationError, validation_error_handler)
    app.add_exception_handler(UserNotFoundError, _user_not_found_handler)
    app.add_exception_handler(ServiceUnavailableError, _service_error_handler)
    app.add_exception_handler(Exception, _unexpected_error_handler)


def _to_chat_response(reply: ConversationReply) -> ChatResponse:
    degraded_components = _degraded_components(reply)
    active_context = _to_public_context(reply.active_context)
    recommendations = [
        PublicArticleRecommendation(
            document_id=item.document_id,
            title=item.title,
            excerpt=item.excerpt,
            score=item.score,
            reason=item.reason,
        )
        for item in reply.recommendations
    ]
    return ChatResponse(
        session_id=reply.session_id,
        session_type=reply.session_type,
        parent_session_id=reply.parent_session_id,
        active_child_session_id=reply.active_child_session_id,
        focus_document_id=reply.focus_document_id,
        focus_document_title=reply.focus_document_title,
        session_status=reply.session_status,
        message=reply.message,
        action=reply.action,
        intent_state=reply.intent_state,
        active_context=active_context,
        recommendations=recommendations,
        citations=[citation.model_copy(deep=True) for citation in reply.citations],
        images=[image.model_copy(deep=True) for image in reply.images],
        execution_trace=(
            reply.execution_trace.model_copy(deep=True)
            if reply.execution_trace is not None
            else None
        ),
        degraded=bool(degraded_components),
        degraded_components=degraded_components,
        compression=reply.compression.model_copy(deep=True),
    )


def _to_session_history_response(
    session: ConversationSession,
) -> SessionHistoryResponse:
    return SessionHistoryResponse(
        session_id=session.session_id,
        user_id=session.user_id,
        session_type=session.session_type,
        parent_session_id=session.parent_session_id,
        active_child_session_id=session.active_child_session_id,
        focus_document_id=session.focus_document_id,
        focus_document_title=session.focus_document_title,
        session_status=session.session_status,
        intent_state=session.intent_state,
        history=list(session.history),
        active_context=_to_public_context(session.active_context),
        turn_count=session.turn_count,
        compression=_session_compression_info(session),
    )


def _session_compression_info(
    session: ConversationSession,
) -> ConversationCompressionInfo:
    """根据近期历史和持久摘要生成公开压缩状态。"""

    recent_history = session.history[-12:]
    retained_turn_count = sum(turn.role == "user" for turn in recent_history)
    recent_start = max(0, len(session.history) - 12)
    if session.summary_watermark + 1 < recent_start:
        compression_status = "pending"
    elif session.summary is not None or session.summarized_turn_count > 0:
        compression_status = "compressed"
    else:
        compression_status = "not_needed"
    return ConversationCompressionInfo(
        status=compression_status,
        summary=session.summary,
        summarized_turn_count=session.summarized_turn_count,
        retained_turn_count=retained_turn_count,
        dropped_turn_count=session.dropped_turn_count,
    )


def _to_public_context(
    context: RecommendationContext | None,
) -> PublicRecommendationContext | None:
    if context is None:
        return None
    return PublicRecommendationContext(
        query=context.query,
        size=context.size,
    )


def _degraded_components(reply: ConversationReply) -> list[DegradedComponent]:
    components = degraded_components(reply.agent_statuses)
    if reply.intent_source is RecognitionSource.FALLBACK:
        components.insert(0, "intent_recognition")
    return components


async def _user_not_found_handler(
    _: Request,
    __: UserNotFoundError,
) -> JSONResponse:
    return error_response(
        status.HTTP_404_NOT_FOUND,
        "USER_NOT_FOUND",
        "用户不存在",
    )


async def _service_error_handler(
    _: Request,
    __: ServiceUnavailableError,
) -> JSONResponse:
    return error_response(
        status.HTTP_503_SERVICE_UNAVAILABLE,
        "SERVICE_UNAVAILABLE",
        "文章推荐服务暂时不可用",
    )


async def _unexpected_error_handler(
    _: Request,
    exc: Exception,
) -> JSONResponse:
    logger.error("HTTP 请求处理失败", error_type=type(exc).__name__)
    return error_response(
        status.HTTP_503_SERVICE_UNAVAILABLE,
        "SERVICE_UNAVAILABLE",
        "文章推荐服务暂时不可用",
    )


__all__ = ["register_error_handlers", "router"]
