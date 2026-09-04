"""确定性积累用户意图习惯，并生成受保护的有限记忆投影。"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
import re
from typing import Protocol

from app.models.intent import ArbitrationAction, IntentState
from app.models.intent_memory import (
    IntentCorrectionEvidence,
    IntentCorrectionProjection,
    RecommendationSizeEvidence,
    UserIntentMemory,
    UserIntentMemoryProjection,
)


class UserIntentMemoryRepository(Protocol):
    """会话服务依赖的同步长期意图记忆仓储边界。"""

    def get(self, user_id: str) -> UserIntentMemory | None: ...

    def save(self, memory: UserIntentMemory) -> None: ...


class UserIntentMemoryService:
    """只从成功业务结果积累统计事实，不保存自由文本用户记忆。"""

    _RECOMMENDATION_ACTIONS = frozenset(
        {
            ArbitrationAction.NEW,
            ArbitrationAction.REFINE,
            ArbitrationAction.REPEAT,
        }
    )
    _SIZE_PATTERN = re.compile(
        r"(?<!\d)(?P<size>10|[1-9]|[一二三四五六七八九十])(?!\d)\s*篇"
    )
    _EXPLICIT_DEFAULT_PATTERN = re.compile(r"(?:默认|以后|今后|每次)")
    _CORRECTION_PATTERN = re.compile(
        r"(?:不是|不对|错了|理解错|我的意思|我是想|我想说)"
    )
    _CHINESE_NUMBERS = {
        "一": 1,
        "二": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
        "十": 10,
    }

    def __init__(
        self,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    def empty(self, user_id: str) -> UserIntentMemory:
        """为尚无长期证据的用户构造安全空记忆。"""

        return UserIntentMemory.empty(user_id, now=self._now())

    def record_success(
        self,
        memory: UserIntentMemory,
        *,
        message: str,
        action: ArbitrationAction,
        previous_intent_state: IntentState,
        current_intent_state: IntentState,
    ) -> UserIntentMemory:
        """把一次已成功提交的业务结果折叠为有限统计证据。"""

        validated = UserIntentMemory.model_validate(memory)
        updated = validated.model_copy(deep=True)
        now = self._now()
        if action in self._RECOMMENDATION_ACTIONS:
            updated.recommendation_count += 1
            size = self._parse_size(message)
            if size is not None:
                self._record_size(
                    updated,
                    size=size,
                    explicit=self._EXPLICIT_DEFAULT_PATTERN.search(message)
                    is not None,
                    now=now,
                )
        elif action is ArbitrationAction.KNOWLEDGE_ANSWER:
            updated.knowledge_qa_count += 1
        else:
            return updated

        if (
            previous_intent_state is not current_intent_state
            and self._CORRECTION_PATTERN.search(message) is not None
        ):
            self._record_correction(
                updated,
                from_intent=previous_intent_state,
                to_intent=current_intent_state,
                now=now,
            )
        updated.updated_at = now
        return UserIntentMemory.model_validate(updated.model_dump())

    def project(
        self,
        memory: UserIntentMemory,
    ) -> UserIntentMemoryProjection:
        """生成不含用户 ID、原句和数据库细节的 Prompt 白名单投影。"""

        validated = UserIntentMemory.model_validate(memory)
        dominant_intent, dominant_confidence = self._dominant_intent(validated)
        corrections = sorted(
            validated.corrections,
            key=lambda item: (
                item.evidence_count,
                item.last_observed_at,
                item.from_intent.value,
                item.to_intent.value,
            ),
            reverse=True,
        )[:3]
        return UserIntentMemoryProjection(
            default_recommendation_size=self._default_size(validated),
            dominant_intent=dominant_intent,
            dominant_intent_confidence=dominant_confidence,
            corrections=[
                IntentCorrectionProjection(
                    from_intent=item.from_intent,
                    to_intent=item.to_intent,
                    evidence_count=item.evidence_count,
                )
                for item in corrections
            ],
        )

    def _now(self) -> datetime:
        now = self._clock()
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("意图记忆时钟必须包含时区")
        return now

    @classmethod
    def _parse_size(cls, message: str) -> int | None:
        matches = list(
            cls._SIZE_PATTERN.finditer(" ".join(str(message).split()))
        )
        if len(matches) != 1:
            return None
        match = matches[0]
        value = match.group("size")
        return cls._CHINESE_NUMBERS.get(
            value,
            int(value) if value.isdigit() else None,
        )

    @staticmethod
    def _record_size(
        memory: UserIntentMemory,
        *,
        size: int,
        explicit: bool,
        now: datetime,
    ) -> None:
        existing = next(
            (item for item in memory.recommendation_sizes if item.size == size),
            None,
        )
        if existing is None:
            memory.recommendation_sizes.append(
                RecommendationSizeEvidence(
                    size=size,
                    evidence_count=1,
                    explicit=explicit,
                    last_observed_at=now,
                )
            )
        else:
            existing.evidence_count += 1
            existing.explicit = existing.explicit or explicit
            existing.last_observed_at = now
        memory.recommendation_sizes = sorted(
            memory.recommendation_sizes,
            key=lambda item: (
                item.explicit,
                item.evidence_count,
                item.last_observed_at,
                -item.size,
            ),
            reverse=True,
        )[:10]

    @staticmethod
    def _record_correction(
        memory: UserIntentMemory,
        *,
        from_intent: IntentState,
        to_intent: IntentState,
        now: datetime,
    ) -> None:
        existing = next(
            (
                item
                for item in memory.corrections
                if item.from_intent is from_intent
                and item.to_intent is to_intent
            ),
            None,
        )
        if existing is None:
            memory.corrections.append(
                IntentCorrectionEvidence(
                    from_intent=from_intent,
                    to_intent=to_intent,
                    evidence_count=1,
                    last_observed_at=now,
                )
            )
        else:
            existing.evidence_count += 1
            existing.last_observed_at = now
        memory.corrections = sorted(
            memory.corrections,
            key=lambda item: (
                item.evidence_count,
                item.last_observed_at,
                item.from_intent.value,
                item.to_intent.value,
            ),
            reverse=True,
        )[:8]

    @staticmethod
    def _default_size(memory: UserIntentMemory) -> int | None:
        explicit = [item for item in memory.recommendation_sizes if item.explicit]
        candidates = explicit or [
            item
            for item in memory.recommendation_sizes
            if item.evidence_count >= 3
        ]
        if not candidates:
            return None
        selected = max(
            candidates,
            key=lambda item: (
                item.last_observed_at if explicit else item.evidence_count,
                item.evidence_count if explicit else item.last_observed_at,
                -item.size,
            ),
        )
        return selected.size

    @staticmethod
    def _dominant_intent(
        memory: UserIntentMemory,
    ) -> tuple[str | None, float]:
        total = memory.recommendation_count + memory.knowledge_qa_count
        if total < 3:
            return None, 0.0
        if memory.recommendation_count >= memory.knowledge_qa_count:
            dominant = "recommend_articles"
            count = memory.recommendation_count
        else:
            dominant = "knowledge_qa"
            count = memory.knowledge_qa_count
        share = count / total
        if share < 0.7:
            return None, 0.0
        return dominant, min(0.9, share)


__all__ = ["UserIntentMemoryRepository", "UserIntentMemoryService"]
