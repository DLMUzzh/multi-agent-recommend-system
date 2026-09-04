"""不暴露敏感配置的服务健康检查。"""

from fastapi import APIRouter

from app import APP_VERSION
from app.config import get_settings
from app.infrastructure.llm.client import is_llm_configured
from app.infrastructure.retrieval.article_embedding import is_embedding_configured
from app.models.schemas import HealthResponse


router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """返回不含密钥、模型地址和内部配置的运行状态。"""

    settings = get_settings()
    return HealthResponse(
        status="healthy",
        version=APP_VERSION,
        llm_configured=is_llm_configured(settings),
        embedding_configured=is_embedding_configured(settings),
    )


__all__ = ["router"]
