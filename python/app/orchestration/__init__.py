"""新项目会话编排的延迟导出。"""

from typing import Any


__all__ = ["ConversationGraph", "ConversationGraphResult"]


def __getattr__(name: str) -> Any:
    if name in {"ConversationGraph", "ConversationGraphResult"}:
        from .conversation_graph import ConversationGraph, ConversationGraphResult

        return {
            "ConversationGraph": ConversationGraph,
            "ConversationGraphResult": ConversationGraphResult,
        }[name]
    raise AttributeError(name)
