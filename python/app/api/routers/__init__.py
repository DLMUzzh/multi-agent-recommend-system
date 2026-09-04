"""FastAPI 路由公共导出。"""

from .chat import register_error_handlers, router as chat_router
from .health import router as health_router
from .similar_documents import (
    register_similar_document_error_handlers,
    router as similar_documents_router,
)

__all__ = [
    "chat_router",
    "health_router",
    "register_error_handlers",
    "register_similar_document_error_handlers",
    "similar_documents_router",
]
