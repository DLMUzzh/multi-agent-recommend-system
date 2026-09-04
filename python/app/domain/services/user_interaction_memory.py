"""从细微会话反馈中确定性维护有界回答偏好。"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
import re

from app.models.interaction_memory import (
    ConversationFeedbackAnalysis,
    ConversationFeedbackEvent,
    ResponsePreference,
    ResponsePreferenceProjection,
    UserInteractionMemory,
    UserInteractionMemoryProjection,
)


class UserInteractionMemoryService:
    """LLM 只提出候选，本服务负责证据、幂等、边界和投影。"""

    _FEEDBACK_PATTERN = re.compile(
        r"(?:我更(?:关心|关注|想了解)|我想了解的是|重点(?:讲|说)|"
        r"详细(?:一点|讲)|展开(?:讲|说)|不要只讲|先讲|以后|每次|"
        r"我喜欢|回答时|项目背景|整体架构|实现细节|数据流|取舍)"
    )

    def __init__(
        self,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    def empty(self, user_id: str) -> UserInteractionMemory:
        """构造不带任何推断的空交互记忆。"""

        return UserInteractionMemory.empty(user_id, now=self._now())

    def is_feedback_candidate(
        self,
        message: str,
        *,
        has_previous_exchange: bool,
    ) -> bool:
        """只做宽松成本门控，是否形成偏好仍由结构化 LLM 决定。"""

        if not has_previous_exchange:
            return False
        normalized = " ".join(str(message).split())
        return bool(normalized and self._FEEDBACK_PATTERN.search(normalized))

    def apply_analysis(
        self,
        memory: UserInteractionMemory,
        *,
        event: ConversationFeedbackEvent,
        analysis: ConversationFeedbackAnalysis,
    ) -> UserInteractionMemory:
        """把一条已校验分析幂等折叠为低优先级交互偏好。"""

        current = UserInteractionMemory.model_validate(memory).model_copy(
            deep=True
        )
        validated_event = ConversationFeedbackEvent.model_validate(event)
        validated_analysis = ConversationFeedbackAnalysis.model_validate(analysis)
        if current.user_id != validated_event.user_id:
            raise ValueError("反馈事件与交互记忆用户不一致")
        if (
            not validated_analysis.is_preference_feedback
            or validated_analysis.persistence == "current_turn_only"
        ):
            return current

        existing = next(
            (
                item
                for item in current.preferences
                if item.scope == validated_analysis.scope
            ),
            None,
        )
        if (
            existing is not None
            and validated_event.event_id in existing.source_event_ids
        ):
            return current

        now = self._now()
        if existing is None:
            confidence = self._initial_confidence(validated_analysis)
            current.preferences.append(
                ResponsePreference(
                    scope=validated_analysis.scope,
                    preferred_focus=list(validated_analysis.preferred_focus),
                    detail_level=validated_analysis.detail_level,
                    answer_structure=validated_analysis.answer_structure,
                    evidence_count=1,
                    confidence=confidence,
                    source_event_ids=[validated_event.event_id],
                    source_session_ids=[validated_event.session_id],
                    first_observed_at=validated_event.occurred_at,
                    last_observed_at=validated_event.occurred_at,
                )
            )
        else:
            existing.preferred_focus = list(
                dict.fromkeys(
                    existing.preferred_focus
                    + list(validated_analysis.preferred_focus)
                )
            )[:4]
            if validated_analysis.detail_level is not None:
                existing.detail_level = validated_analysis.detail_level
            if validated_analysis.answer_structure is not None:
                existing.answer_structure = validated_analysis.answer_structure
            existing.evidence_count += 1
            existing.confidence = min(
                0.95,
                max(existing.confidence, self._initial_confidence(validated_analysis))
                + 0.08,
            )
            existing.source_event_ids = (
                existing.source_event_ids + [validated_event.event_id]
            )[-12:]
            if validated_event.session_id not in existing.source_session_ids:
                existing.source_session_ids = (
                    existing.source_session_ids + [validated_event.session_id]
                )[-12:]
            existing.last_observed_at = max(
                existing.last_observed_at,
                validated_event.occurred_at,
            )

        current.preferences = sorted(
            current.preferences,
            key=lambda item: (
                item.confidence,
                item.evidence_count,
                item.last_observed_at,
                item.scope,
            ),
            reverse=True,
        )[:8]
        current.updated_at = now
        return UserInteractionMemory.model_validate(current.model_dump())

    def project(
        self,
        memory: UserInteractionMemory,
    ) -> UserInteractionMemoryProjection:
        """生成最多三条、不含身份和原始反馈的回答偏好投影。"""

        validated = UserInteractionMemory.model_validate(memory)
        selected = sorted(
            validated.preferences,
            key=lambda item: (
                item.confidence,
                item.evidence_count,
                item.last_observed_at,
                item.scope,
            ),
            reverse=True,
        )[:3]
        return UserInteractionMemoryProjection(
            preferences=[
                ResponsePreferenceProjection(
                    scope=item.scope,
                    preferred_focus=list(item.preferred_focus),
                    detail_level=item.detail_level,
                    answer_structure=item.answer_structure,
                    evidence_count=item.evidence_count,
                    confidence=item.confidence,
                )
                for item in selected
            ]
        )

    def project_analysis(
        self,
        analysis: ConversationFeedbackAnalysis,
    ) -> UserInteractionMemoryProjection:
        """把单条已校验回答方式分析投影为无身份的当前轮提示。"""

        validated = ConversationFeedbackAnalysis.model_validate(analysis)
        if (
            not validated.is_preference_feedback
            or validated.persistence == "current_turn_only"
        ):
            return UserInteractionMemoryProjection()
        return UserInteractionMemoryProjection(
            preferences=[
                ResponsePreferenceProjection(
                    scope=validated.scope,
                    preferred_focus=list(validated.preferred_focus),
                    detail_level=validated.detail_level,
                    answer_structure=validated.answer_structure,
                    evidence_count=1,
                    confidence=self._initial_confidence(validated),
                )
            ]
        )

    @staticmethod
    def _initial_confidence(analysis: ConversationFeedbackAnalysis) -> float:
        if analysis.persistence == "explicit_long_term":
            return min(0.9, max(0.8, analysis.confidence))
        return min(0.75, analysis.confidence)

    def _now(self) -> datetime:
        now = self._clock()
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("交互记忆时钟必须包含时区")
        return now


__all__ = ["UserInteractionMemoryService"]
