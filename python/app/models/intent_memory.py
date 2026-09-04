"""跨会话用户意图记忆及其受保护投影契约。"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field, field_validator

from app.models.common import _StrictModel
from app.models.intent import IntentState


INTENT_MEMORY_VERSION = "v1"
RememberedIntent = Literal["recommend_articles", "knowledge_qa"]


def _aware_datetime(value: datetime) -> datetime:
    """拒绝缺少时区的记忆时间，避免衰减和排序产生歧义。"""

    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("意图记忆时间必须包含时区")
    return value


class RecommendationSizeEvidence(_StrictModel):
    """用户推荐数量习惯的结构化证据。"""

    size: int = Field(ge=1, le=10)
    evidence_count: int = Field(ge=1)
    explicit: bool = False
    last_observed_at: datetime

    _validate_time = field_validator("last_observed_at")(_aware_datetime)


class IntentCorrectionEvidence(_StrictModel):
    """用户主动纠正两个业务路由的累计证据。"""

    from_intent: IntentState
    to_intent: IntentState
    evidence_count: int = Field(ge=1)
    last_observed_at: datetime

    _validate_time = field_validator("last_observed_at")(_aware_datetime)


class IntentCorrectionProjection(_StrictModel):
    """允许进入意图 Prompt 的最小纠正事实。"""

    from_intent: IntentState
    to_intent: IntentState
    evidence_count: int = Field(ge=1)


class UserIntentMemoryProjection(_StrictModel):
    """提供给意图识别的白名单长期记忆，不包含用户 ID 或原始对话。"""

    default_recommendation_size: int | None = Field(default=None, ge=1, le=10)
    dominant_intent: RememberedIntent | None = None
    dominant_intent_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    corrections: list[IntentCorrectionProjection] = Field(
        default_factory=list,
        max_length=3,
    )


class UserIntentMemory(_StrictModel):
    """一个用户跨会话积累的稳定意图习惯和纠正证据。"""

    user_id: str = Field(min_length=1, max_length=128)
    recommendation_count: int = Field(default=0, ge=0)
    knowledge_qa_count: int = Field(default=0, ge=0)
    recommendation_sizes: list[RecommendationSizeEvidence] = Field(
        default_factory=list,
        max_length=10,
    )
    corrections: list[IntentCorrectionEvidence] = Field(
        default_factory=list,
        max_length=8,
    )
    updated_at: datetime
    memory_version: Literal["v1"] = INTENT_MEMORY_VERSION

    @field_validator("user_id", mode="before")
    @classmethod
    def validate_user_id(cls, value: object) -> str:
        """清理用户 ID，但不接受其他类型或空白值。"""

        if not isinstance(value, str) or not value.strip():
            raise ValueError("user_id 不能为空")
        return value.strip()

    _validate_time = field_validator("updated_at")(_aware_datetime)

    @classmethod
    def empty(cls, user_id: str, *, now: datetime) -> UserIntentMemory:
        """构造不携带任何推断的安全空记忆。"""

        return cls(user_id=user_id, updated_at=now)


__all__ = [
    "INTENT_MEMORY_VERSION",
    "IntentCorrectionEvidence",
    "IntentCorrectionProjection",
    "RecommendationSizeEvidence",
    "RememberedIntent",
    "UserIntentMemory",
    "UserIntentMemoryProjection",
]
