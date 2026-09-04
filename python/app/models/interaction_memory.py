"""用户交互反馈、回答偏好及其受保护投影契约。"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field, field_validator, model_validator

from app.models.common import _StrictModel


INTERACTION_MEMORY_VERSION = "v1"
FeedbackStatus = Literal["pending", "analyzed"]
FeedbackType = Literal[
    "preference_refinement",
    "detail_request",
    "format_preference",
    "factual_correction",
    "no_feedback",
]
PreferenceScope = Literal[
    "system_explanation",
    "knowledge_qa",
    "recommendation",
    "general",
]
ResponseFocus = Literal[
    "project_background",
    "architecture",
    "data_flow",
    "implementation_details",
    "tradeoffs",
    "examples",
]
DetailLevel = Literal["brief", "balanced", "detailed"]
AnswerStructure = Literal[
    "overview_first",
    "conclusion_first",
    "step_by_step",
    "example_first",
]
PreferencePersistence = Literal[
    "current_turn_only",
    "long_term_candidate",
    "explicit_long_term",
]


def _aware_datetime(value: datetime | None) -> datetime | None:
    if value is not None and (
        value.tzinfo is None or value.utcoffset() is None
    ):
        raise ValueError("交互记忆时间必须包含时区")
    return value


class ConversationFeedbackAnalysis(_StrictModel):
    """LLM 对相邻对话反馈的受控分析结果。"""

    is_preference_feedback: bool
    feedback_type: FeedbackType
    scope: PreferenceScope
    preferred_focus: list[ResponseFocus] = Field(default_factory=list, max_length=4)
    detail_level: DetailLevel | None = None
    answer_structure: AnswerStructure | None = None
    persistence: PreferencePersistence
    confidence: float = Field(ge=0.0, le=1.0)
    reason_code: str = Field(min_length=1, max_length=80)
    reason_summary: str = Field(min_length=1, max_length=300)

    @field_validator("preferred_focus")
    @classmethod
    def validate_focus(cls, value: list[ResponseFocus]) -> list[ResponseFocus]:
        if len(value) != len(set(value)):
            raise ValueError("回答关注点不能重复")
        return value

    @model_validator(mode="after")
    def validate_preference_boundary(self) -> ConversationFeedbackAnalysis:
        preference_fields = bool(
            self.preferred_focus
            or self.detail_level is not None
            or self.answer_structure is not None
        )
        if self.is_preference_feedback:
            if not preference_fields:
                raise ValueError("偏好反馈必须包含至少一个受控偏好")
            if self.feedback_type in {"factual_correction", "no_feedback"}:
                raise ValueError("事实纠错和无反馈不能形成偏好")
        else:
            if preference_fields:
                raise ValueError("非偏好反馈不能携带回答偏好")
            if self.persistence != "current_turn_only":
                raise ValueError("非偏好反馈只能作用于当前轮")
        return self


class ConversationFeedbackEvent(_StrictModel):
    """一个用户对上一轮回答的短期原始反馈窗口。"""

    event_id: str = Field(min_length=1, max_length=128)
    user_id: str = Field(min_length=1, max_length=128)
    session_id: str = Field(min_length=1, max_length=128)
    previous_user_message: str | None = Field(default=None, max_length=4000)
    previous_assistant_message: str | None = Field(default=None, max_length=8000)
    feedback_message: str | None = Field(default=None, max_length=4000)
    occurred_at: datetime
    status: FeedbackStatus = "pending"
    analysis: ConversationFeedbackAnalysis | None = None
    analysis_attempts: int = Field(default=0, ge=0)
    next_attempt_at: datetime | None = None
    last_error_type: str | None = Field(default=None, max_length=100)
    analyzed_at: datetime | None = None
    raw_purged_at: datetime | None = None

    _validate_occurred_at = field_validator("occurred_at")(_aware_datetime)
    _validate_optional_times = field_validator(
        "next_attempt_at",
        "analyzed_at",
        "raw_purged_at",
    )(_aware_datetime)

    @model_validator(mode="after")
    def validate_lifecycle(self) -> ConversationFeedbackEvent:
        raw_values = (
            self.previous_user_message,
            self.previous_assistant_message,
            self.feedback_message,
        )
        if self.raw_purged_at is None:
            if any(value is None or not value.strip() for value in raw_values):
                raise ValueError("未清理的反馈事件必须保留完整原始窗口")
        elif any(value is not None for value in raw_values):
            raise ValueError("已清理事件不能继续保留原始反馈文本")
        if self.status == "analyzed":
            if self.analysis is None or self.analyzed_at is None:
                raise ValueError("已分析事件必须保留结构化分析和时间")
        elif self.analysis is not None or self.analyzed_at is not None:
            raise ValueError("待分析事件不能提前写入分析结果")
        return self


class ResponsePreference(_StrictModel):
    """一个用户在特定回答场景下的有界交互偏好。"""

    scope: PreferenceScope
    preferred_focus: list[ResponseFocus] = Field(default_factory=list, max_length=4)
    detail_level: DetailLevel | None = None
    answer_structure: AnswerStructure | None = None
    evidence_count: int = Field(ge=1)
    confidence: float = Field(ge=0.0, le=1.0)
    source_event_ids: list[str] = Field(min_length=1, max_length=12)
    source_session_ids: list[str] = Field(min_length=1, max_length=12)
    first_observed_at: datetime
    last_observed_at: datetime

    _validate_times = field_validator(
        "first_observed_at",
        "last_observed_at",
    )(_aware_datetime)

    @field_validator(
        "preferred_focus",
        "source_event_ids",
        "source_session_ids",
    )
    @classmethod
    def validate_unique_lists(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("交互偏好证据不能重复")
        return value

    @model_validator(mode="after")
    def validate_payload(self) -> ResponsePreference:
        if not (
            self.preferred_focus
            or self.detail_level is not None
            or self.answer_structure is not None
        ):
            raise ValueError("交互偏好必须包含至少一个回答方式字段")
        if self.first_observed_at > self.last_observed_at:
            raise ValueError("首次观察时间不能晚于最近观察时间")
        return self


class ResponsePreferenceProjection(_StrictModel):
    """允许进入回答 Prompt 的最小交互偏好。"""

    scope: PreferenceScope
    preferred_focus: list[ResponseFocus] = Field(default_factory=list, max_length=4)
    detail_level: DetailLevel | None = None
    answer_structure: AnswerStructure | None = None
    evidence_count: int = Field(ge=1)
    confidence: float = Field(ge=0.0, le=1.0)


class UserInteractionMemoryProjection(_StrictModel):
    """不含用户身份、原文和原因说明的回答偏好白名单。"""

    preferences: list[ResponsePreferenceProjection] = Field(
        default_factory=list,
        max_length=3,
    )


class UserInteractionMemory(_StrictModel):
    """一个用户跨会话积累的有界回答方式偏好。"""

    user_id: str = Field(min_length=1, max_length=128)
    preferences: list[ResponsePreference] = Field(default_factory=list, max_length=8)
    updated_at: datetime
    memory_version: Literal["v1"] = INTERACTION_MEMORY_VERSION

    _validate_time = field_validator("updated_at")(_aware_datetime)

    @classmethod
    def empty(cls, user_id: str, *, now: datetime) -> UserInteractionMemory:
        return cls(user_id=user_id, updated_at=now)


__all__ = [
    "AnswerStructure",
    "ConversationFeedbackAnalysis",
    "ConversationFeedbackEvent",
    "DetailLevel",
    "FeedbackStatus",
    "FeedbackType",
    "INTERACTION_MEMORY_VERSION",
    "PreferencePersistence",
    "PreferenceScope",
    "ResponseFocus",
    "ResponsePreference",
    "ResponsePreferenceProjection",
    "UserInteractionMemory",
    "UserInteractionMemoryProjection",
]
