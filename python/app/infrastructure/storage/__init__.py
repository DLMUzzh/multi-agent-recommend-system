"""知识资源存储基础设施。"""

from app.infrastructure.storage.local_knowledge_image_store import (
    LocalKnowledgeImageStore,
    StoredKnowledgeImage,
)

__all__ = [
    "LocalKnowledgeImageStore",
    "StoredKnowledgeImage",
]
