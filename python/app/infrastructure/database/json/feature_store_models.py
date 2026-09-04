"""Feature Store 的本地数据模型、事件常量和输入校验。"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


EventType = Literal[
    "click",
    "read",
    "like",
    "comment",
    "follow_author",
    "unfollow_author",
    "not_interested",
    "search",
    "share",
]

ALLOWED_EVENT_TYPES = {
    "click",
    "read",
    "like",
    "comment",
    "follow_author",
    "unfollow_author",
    "not_interested",
    "search",
    "share",
}

BASE_EVENT_WEIGHTS: dict[str, float] = {
    "click": 0.5,
    "read": 1.0,
    "like": 2.5,
    "comment": 1.5,
    "follow_author": 3.0,
    "unfollow_author": -3.0,
    "not_interested": -5.0,
    "search": 1.5,
    "share": 4.0,
}

STRONG_EVENT_TYPES = {"like", "comment", "share", "follow_author"}
EFFECTIVE_EVENT_TYPES = {
    "click",
    "read",
    "like",
    "comment",
    "follow_author",
    "search",
    "share",
}


def _require_string_id(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("业务 ID 必须是非空字符串")
    return value


def _require_timezone(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("时间必须包含时区")
    return value


class UserBaseProfile(BaseModel):
    """用户主动设置的文章推荐基础偏好。"""

    model_config = ConfigDict(extra="forbid")

    user_id: str
    topics: list[str] = Field(default_factory=list)
    blocked_topics: list[str] = Field(default_factory=list)
    preferred_content_types: list[str] = Field(default_factory=list)
    preferred_difficulty: str = ""
    preferred_reading_length: str = ""
    followed_author_ids: list[str] = Field(default_factory=list)
    blocked_author_ids: list[str] = Field(default_factory=list)
    created_at: datetime

    _user_id_is_string = field_validator("user_id", mode="before")(_require_string_id)
    _author_ids_are_strings = field_validator(
        "followed_author_ids", "blocked_author_ids", mode="before"
    )(lambda values: [_require_string_id(value) for value in (values or [])])
    _created_at_has_timezone = field_validator("created_at")(_require_timezone)


class BehaviorEvent(BaseModel):
    """经过校验且不可变的用户行为事实。"""

    model_config = ConfigDict(extra="forbid")

    event_id: str
    user_id: str
    event_type: EventType
    occurred_at: datetime
    document_id: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    _ids_are_strings = field_validator(
        "event_id", "user_id", "document_id", mode="before"
    )(_require_string_id)
    _occurred_at_has_timezone = field_validator("occurred_at")(_require_timezone)


class UserNotFoundError(LookupError):
    """请求未知用户的画像或数据时抛出。"""


__all__ = ["UserNotFoundError"]
