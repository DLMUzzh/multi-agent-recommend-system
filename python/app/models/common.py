"""跨组件模型共享的严格基础契约。"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class _StrictModel(BaseModel):
    """拒绝未知字段，避免跨组件契约静默漂移。"""

    model_config = ConfigDict(extra="forbid")


class AgentResult(_StrictModel):
    """所有 Agent 共用的最小运行结果。"""

    agent_name: str
    success: bool = True
    latency_ms: float = 0.0
    error: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)
    confidence: float = 1.0


__all__ = ["AgentResult"]
