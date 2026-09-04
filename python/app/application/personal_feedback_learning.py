"""把已保护的个人质量反馈路由到推荐画像事实。"""

from __future__ import annotations

from hashlib import sha256
from typing import Any

from app.models.personal_feedback import (
    PersonalFeedbackEvent,
    RecommendationMemorySignal,
)


class PersonalFeedbackLearningService:
    """只消费 Policy 已接纳的推荐负向信号，并保持个人隔离和幂等。"""

    _GENERIC_EVIDENCE_THRESHOLD = 3

    def __init__(self, *, feedback_store: Any, feature_store: Any) -> None:
        self.feedback_store = feedback_store
        self.feature_store = feature_store

    async def apply(self, *, event: PersonalFeedbackEvent) -> dict[str, str]:
        """应用一条已完成反馈的个人记忆路由，失败不抛出业务细节。"""

        protected = PersonalFeedbackEvent.model_validate(event).model_copy(
            deep=True
        )
        statuses = {
            route: protected.memory_statuses.get(route, "pending")
            for route in protected.memory_routes
        }
        for route, status in tuple(statuses.items()):
            if status in {"applied", "degraded", "skipped"}:
                continue
            if route != "recommendation_profile":
                statuses[route] = "skipped"
                continue
            statuses[route] = await self._apply_recommendation_profile(protected)
        return statuses

    async def _apply_recommendation_profile(
        self,
        event: PersonalFeedbackEvent,
    ) -> str:
        signals = tuple(event.recommendation_signals)
        if not signals:
            return "skipped"
        applied = False
        try:
            for signal in signals:
                protected = RecommendationMemorySignal.model_validate(signal)
                if not await self._ready_to_learn(event, protected):
                    continue
                document_id = self._source_document_id(protected)
                if document_id is None:
                    continue
                try:
                    await self.feature_store.record_behavior(
                        event.user_id,
                        "not_interested",
                        document_id,
                        metadata=self._behavior_metadata(event, protected),
                        occurred_at=event.updated_at,
                        event_id=self._behavior_event_id(event, protected),
                    )
                except ValueError as exc:
                    if "event_id 重复" not in str(exc):
                        raise
                applied = True
        except Exception:
            return "degraded"
        return "applied" if applied else "skipped"

    async def _ready_to_learn(
        self,
        event: PersonalFeedbackEvent,
        signal: RecommendationMemorySignal,
    ) -> bool:
        if signal.persistence == "current_recovery_only":
            return False
        if signal.specific or signal.persistence == "explicit_long_term":
            return True
        events = await self.feedback_store.list_feedback_events(
            event.user_id,
            limit=100,
        )
        key = self._signal_key(signal)
        evidence_ids = {
            candidate.feedback_id
            for candidate in events
            if candidate.status != "closed"
            and any(
                self._signal_key(item) == key
                for item in candidate.recommendation_signals
            )
        }
        return len(evidence_ids) >= self._GENERIC_EVIDENCE_THRESHOLD

    @staticmethod
    def _source_document_id(signal: RecommendationMemorySignal) -> str | None:
        if signal.source_document_ids:
            return signal.source_document_ids[0]
        if signal.target_type == "article":
            return signal.target_value
        return None

    @staticmethod
    def _behavior_metadata(
        event: PersonalFeedbackEvent,
        signal: RecommendationMemorySignal,
    ) -> dict[str, object]:
        metadata: dict[str, object] = {
            "target_type": signal.target_type,
            "source_feedback_id": event.feedback_id,
        }
        if signal.target_type == "difficulty":
            metadata["target_values"] = [signal.target_value]
            metadata["direction"] = "avoid"
        else:
            metadata["target_value"] = signal.target_value
        return metadata

    @classmethod
    def _behavior_event_id(
        cls,
        event: PersonalFeedbackEvent,
        signal: RecommendationMemorySignal,
    ) -> str:
        digest = sha256(
            f"{event.feedback_id}|{cls._signal_key(signal)}".encode("utf-8")
        ).hexdigest()[:24]
        return f"feedback-{digest}"

    @staticmethod
    def _signal_key(signal: RecommendationMemorySignal) -> str:
        protected = RecommendationMemorySignal.model_validate(signal)
        return "|".join(
            (
                protected.target_type,
                protected.target_value.casefold(),
                protected.direction,
            )
        )


__all__ = ["PersonalFeedbackLearningService"]
