"""使用确定性专家决策树识别高置信聊天业务意图。"""

from __future__ import annotations

import re

from app.models.schemas import (
    IntentName,
    IntentRecognition,
    IntentState,
    RecognitionSource,
    RecommendationContext,
    RecommendationIntent,
    RelationHint,
)


class IntentDecisionTree:
    """只在规则可以唯一判断推荐、问答或无动作时返回结果。"""

    _NO_ACTION_PHRASES = frozenset(
        {"你好", "您好", "嗨", "谢谢", "感谢", "不错"}
    )
    _REPEAT_PHRASES = frozenset(
        {
            "换一批",
            "再来一批",
            "再来一些",
            "换一些",
            "重新推荐一批",
            "再换一批",
            "继续推荐",
        }
    )
    _RECOMMEND_PATTERN = re.compile(
        r"(?:^|请|想|可以|能否|麻烦)(?:给我|帮我)?"
        r"(?:推荐|找|来)(?:一些|几篇|一篇|\s*)"
        r"|(?:想看|有没有).*(?:文章|文档|教程|资料|内容)"
    )
    _QUESTION_PATTERN = re.compile(
        r"(?:[？?]$|是什么|为什么|怎么|如何|哪些|有什么|是否|能否|"
        r"区别|优缺点|原因|步骤|解释|说明一下|介绍一下)"
    )
    _HISTORY_DEPENDENT_PATTERN = re.compile(
        r"(?:它|这个|那个|这篇|那篇|该文|上面|刚才|之前|"
        r"第[一二三四五六七八九十\d]+篇)"
    )
    _SUBJECTLESS_QUESTION_PATTERN = re.compile(
        r"^(?:为什么|怎么|如何|哪些|有什么|是什么|是否|能否)"
    )
    _SIZE_PATTERN = re.compile(
        r"(?<!\d)(?P<size>10|[1-9]|[一二三四五六七八九十])(?!\d)\s*篇"
    )
    _ANY_ARABIC_SIZE_PATTERN = re.compile(r"(?P<size>\d+)\s*篇")
    _REPEAT_WITH_SIZE_PATTERN = re.compile(
        r"^(?:再来|再推荐|继续推荐|给我再来|给我再推荐)\s*"
        r"(?:10|[1-9]|[一二三四五六七八九十])\s*篇(?:文章)?$"
    )
    _TERMINAL_PUNCTUATION_PATTERN = re.compile(r"[。！？!?]{1,3}$")
    _TONE_SUFFIXES = ("一下", "吧")
    _NEGATION_PATTERN = re.compile(
        r"(?:不要|不用|无需|取消|排除|"
        r"别(?:再|给|要|用|推荐|换|来|加)?|"
        r"不(?:要|用|想|看|喜欢|需要|推荐|换|来|加|接受|考虑|"
        r"包括|包含|是|能|可|太|够))"
    )
    _COMBINATION_PATTERN = re.compile(
        r"(?:并(?:且|说明|解释|介绍)|同时|以及|另外|而且|再加|再解释|再说明|"
        r"或者|或|[、，,；;/])"
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

    def decide(
        self,
        message: str,
        *,
        active_context: RecommendationContext | None,
        intent_state: IntentState | str = IntentState.RECOMMENDATION,
        default_recommendation_size: int | None = None,
    ) -> IntentRecognition | None:
        """按短路、冲突、延续、问答和新推荐的顺序执行高置信规则。"""

        raw_message = " ".join(str(message).strip().split())
        cleaned = self._normalize_message(raw_message)
        if not cleaned:
            return self._recognition(
                intent=IntentName.NO_ACTION,
                relation=RelationHint.UNCLEAR,
            )
        if cleaned in self._NO_ACTION_PHRASES:
            return self._recognition(
                intent=IntentName.NO_ACTION,
                relation=RelationHint.UNCLEAR,
            )
        if self._has_unsafe_structure(raw_message):
            return None
        try:
            IntentState(intent_state)
        except ValueError:
            return None

        if cleaned in self._REPEAT_PHRASES:
            return self._repeat_recognition(active_context)
        if self._REPEAT_WITH_SIZE_PATTERN.fullmatch(cleaned):
            return self._repeat_recognition(
                active_context,
                size=self._quantity(cleaned),
            )
        if self._is_history_dependent_question(cleaned):
            return None
        if self._is_clear_question(cleaned):
            return self._recognition(
                intent=IntentName.KNOWLEDGE_QA,
                relation=RelationHint.NEW,
                rewritten_query=raw_message,
            )
        if self._is_clear_recommendation(cleaned):
            return self._recognition(
                intent=IntentName.RECOMMEND_ARTICLES,
                relation=RelationHint.NEW,
                rewritten_query=raw_message,
                size=(
                    self._quantity(cleaned)
                    or self._safe_default_size(default_recommendation_size)
                    or 5
                ),
            )
        return None

    @staticmethod
    def _safe_default_size(value: int | None) -> int | None:
        """长期记忆只允许补充当前消息没有声明的合法默认数量。"""

        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int):
            return None
        return value if 1 <= value <= 10 else None

    @classmethod
    def _normalize_message(cls, message: str) -> str:
        """只移除受控句末标点和一个明确语气后缀。"""

        cleaned = cls._TERMINAL_PUNCTUATION_PATTERN.sub("", message).strip()
        for suffix in cls._TONE_SUFFIXES:
            if cleaned.endswith(suffix):
                stem = cleaned[: -len(suffix)].rstrip()
                if stem:
                    return stem
        return cleaned

    @classmethod
    def _has_unsafe_structure(cls, message: str) -> bool:
        """否定、组合和多数量表达不能由单一规则安全决定。"""

        arabic_sizes = [
            int(match.group("size"))
            for match in cls._ANY_ARABIC_SIZE_PATTERN.finditer(message)
        ]
        if any(size < 1 or size > 10 for size in arabic_sizes):
            return True
        if cls._NEGATION_PATTERN.search(message) is not None:
            return True
        if cls._COMBINATION_PATTERN.search(message) is not None:
            return True
        return len(cls._SIZE_PATTERN.findall(message)) > 1

    @classmethod
    def _is_history_dependent_question(cls, message: str) -> bool:
        """识别必须依赖历史才能形成独立检索查询的问题。"""

        return bool(
            cls._HISTORY_DEPENDENT_PATTERN.search(message)
            or cls._SUBJECTLESS_QUESTION_PATTERN.search(message)
        )

    @classmethod
    def _is_clear_question(cls, message: str) -> bool:
        return cls._QUESTION_PATTERN.search(message) is not None

    @classmethod
    def _is_clear_recommendation(cls, message: str) -> bool:
        return cls._RECOMMEND_PATTERN.search(message) is not None

    @classmethod
    def _quantity(cls, message: str) -> int | None:
        match = cls._SIZE_PATTERN.search(message)
        if match is None:
            return None
        value = match.group("size")
        return cls._CHINESE_NUMBERS.get(value, int(value) if value.isdigit() else None)

    @classmethod
    def _repeat_recognition(
        cls,
        context: RecommendationContext | None,
        *,
        size: int | None = None,
    ) -> IntentRecognition:
        effective_size = size or (context.size if context is not None else 5)
        return cls._recognition(
            intent=IntentName.RECOMMEND_ARTICLES,
            relation=RelationHint.REPEAT,
            rewritten_query=(context.query if context is not None else None),
            size=effective_size,
        )

    @staticmethod
    def _recognition(
        *,
        intent: IntentName,
        relation: RelationHint,
        rewritten_query: str | None = None,
        size: int = 5,
    ) -> IntentRecognition:
        """构造通过规则树保护的完整路由结果。"""

        resolved_intent = (
            RecommendationIntent(size=size)
            if intent is IntentName.RECOMMEND_ARTICLES
            else None
        )
        return IntentRecognition(
            intent=intent,
            source=RecognitionSource.RULE,
            relation=relation,
            confidence=1.0,
            rewritten_query=rewritten_query,
            resolved_intent=resolved_intent,
        )


__all__ = ["IntentDecisionTree"]
