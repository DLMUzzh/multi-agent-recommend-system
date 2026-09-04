"""文章推荐项目的 FastAPI 入口。"""

from __future__ import annotations

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import APP_VERSION
from app.api.routers import (
    chat_router,
    health_router,
    register_error_handlers,
    register_similar_document_error_handlers,
    similar_documents_router,
)
from app.api.routers.knowledge import (
    register_knowledge_error_handlers,
    router as knowledge_router,
)
from app.bootstrap import lifespan


def create_app() -> FastAPI:
    """创建统一承载聊天、推荐、知识导入与无会话问答的应用。"""

    current_app = FastAPI(
        title="Multi-Agent Article Recommendation System",
        description="基于统一 SQLite Chunk 召回的文档推荐与知识问答服务",
        version=APP_VERSION,
        lifespan=lifespan,
    )
    current_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    current_app.include_router(chat_router)
    current_app.include_router(similar_documents_router)
    current_app.include_router(knowledge_router)
    current_app.include_router(health_router)
    register_error_handlers(current_app)
    register_similar_document_error_handlers(current_app)
    register_knowledge_error_handlers(current_app)
    return current_app


app = create_app()


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)


__all__ = ["app", "create_app"]
