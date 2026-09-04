"""根据当前问题和唯一近期历史解析请求级知识文档范围。"""

from __future__ import annotations

import re
from collections.abc import Sequence

from app.models.conversation import ConversationTurn
from app.models.knowledge_qa import KnowledgeScopeResolution


_POSITION_PATTERN = re.compile(r"第\s*(\d+|[一二三四五六七八九十]+)\s*篇")
_REFERENCE_PATTERN = re.compile(
    r"^\[(\d+(?:\s*[,，]\s*\d+)*)\]\s+(.+)$"
)
_RECOMMENDATION_MARKER = "推荐结果："
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


class KnowledgeScopeResolver:
    """只生成当前轮临时范围；无法可靠解析时要求澄清。"""

    _MAX_HISTORY_MESSAGES = 12

    def resolve(
        self,
        question: str,
        *,
        history: Sequence[ConversationTurn],
        documents: Sequence[tuple[str, str]],
    ) -> KnowledgeScopeResolution:
        """解析明确标题、引用编号或“这篇”等近期指代。"""

        normalized_question = self._required_text(question)
        document_ids_by_title = self._document_ids_by_title(documents)
        explicit_titles = tuple(
            title
            for title in document_ids_by_title
            if title in normalized_question
        )
        if explicit_titles:
            explicit_ids = self._unique_ids_for_titles(
                explicit_titles,
                document_ids_by_title,
            )
            if explicit_ids is None:
                return self._clarification(
                    self._candidate_ids_for_titles(
                        explicit_titles,
                        document_ids_by_title,
                    )
                )
            return KnowledgeScopeResolution(
                document_ids=tuple(dict.fromkeys(explicit_ids))
            )

        position_match = _POSITION_PATTERN.search(normalized_question)
        if position_match is not None:
            referenced_titles = self._recent_position_titles(history)
            position = self._position(position_match.group(1))
            if 1 <= position <= len(referenced_titles):
                resolved_ids = self._unique_ids_for_titles(
                    (referenced_titles[position - 1],),
                    document_ids_by_title,
                )
                if resolved_ids is not None:
                    return KnowledgeScopeResolution(document_ids=resolved_ids)
            candidate_titles = (
                (referenced_titles[position - 1],)
                if 1 <= position <= len(referenced_titles)
                else ()
            )
            return self._clarification(
                self._candidate_ids_for_titles(
                    candidate_titles,
                    document_ids_by_title,
                )
            )

        if any(
            reference in normalized_question
            for reference in ("这篇", "本文", "上一篇", "上篇")
        ):
            referenced_titles = tuple(
                dict.fromkeys(self._recent_reference_titles(history))
            )
            resolved_ids = self._unique_ids_for_titles(
                referenced_titles,
                document_ids_by_title,
            )
            if resolved_ids is not None and len(resolved_ids) == 1:
                return KnowledgeScopeResolution(document_ids=resolved_ids)
            return self._clarification(
                self._candidate_ids_for_titles(
                    referenced_titles,
                    document_ids_by_title,
                )
            )

        if "它" in normalized_question:
            referenced_titles = tuple(
                dict.fromkeys(self._recent_reference_titles(history))
            )
            if not referenced_titles:
                return KnowledgeScopeResolution()
            resolved_ids = self._unique_ids_for_titles(
                referenced_titles,
                document_ids_by_title,
            )
            if resolved_ids is not None and len(resolved_ids) == 1:
                return KnowledgeScopeResolution(document_ids=resolved_ids)
            return self._clarification(
                self._candidate_ids_for_titles(
                    referenced_titles,
                    document_ids_by_title,
                )
            )

        return KnowledgeScopeResolution()

    @classmethod
    def _recent_reference_titles(
        cls,
        history: Sequence[ConversationTurn],
    ) -> tuple[str, ...]:
        for turn in reversed(history[-cls._MAX_HISTORY_MESSAGES :]):
            if turn.role != "assistant" or "参考资料：" not in turn.content:
                continue
            titles = cls._reference_titles(turn.content)
            if titles:
                return titles
        return ()

    @staticmethod
    def _reference_titles(content: str) -> tuple[str, ...]:
        titles: list[str] = []
        for line in content.splitlines():
            match = _REFERENCE_PATTERN.match(line.strip())
            if match is None:
                continue
            title = match.group(2).split("（", maxsplit=1)[0].strip()
            if title:
                titles.append(title)
        return tuple(titles)

    @staticmethod
    def _recommendation_titles(content: str) -> tuple[str, ...]:
        if _RECOMMENDATION_MARKER not in content:
            return ()
        recommendation_text = content.rsplit(
            _RECOMMENDATION_MARKER,
            maxsplit=1,
        )[1].splitlines()[0]
        return tuple(
            title.strip()
            for title in recommendation_text.split("；")
            if title.strip()
        )

    @classmethod
    def _recent_position_titles(
        cls,
        history: Sequence[ConversationTurn],
    ) -> tuple[str, ...]:
        for turn in reversed(history[-cls._MAX_HISTORY_MESSAGES :]):
            if turn.role != "assistant":
                continue
            recommendation_titles = cls._recommendation_titles(turn.content)
            if recommendation_titles:
                return recommendation_titles
            if "参考资料：" not in turn.content:
                continue
            reference_titles = cls._reference_titles(turn.content)
            if reference_titles:
                return reference_titles
        return ()

    @staticmethod
    def _document_ids_by_title(
        documents: Sequence[tuple[str, str]],
    ) -> dict[str, tuple[str, ...]]:
        result: dict[str, tuple[str, ...]] = {}
        for document_id, title in documents:
            normalized_id = " ".join(str(document_id).split())
            normalized_title = " ".join(str(title).split())
            if not normalized_id or not normalized_title:
                continue
            existing_ids = result.get(normalized_title, ())
            if normalized_id not in existing_ids:
                result[normalized_title] = (*existing_ids, normalized_id)
        return result

    @staticmethod
    def _unique_ids_for_titles(
        titles: Sequence[str],
        document_ids_by_title: dict[str, tuple[str, ...]],
    ) -> tuple[str, ...] | None:
        if not titles:
            return None
        resolved_ids: list[str] = []
        for title in titles:
            document_ids = document_ids_by_title.get(title, ())
            if len(document_ids) != 1:
                return None
            document_id = document_ids[0]
            if document_id not in resolved_ids:
                resolved_ids.append(document_id)
        return tuple(resolved_ids)

    @staticmethod
    def _candidate_ids_for_titles(
        titles: Sequence[str],
        document_ids_by_title: dict[str, tuple[str, ...]],
    ) -> tuple[str, ...]:
        """只在能够安全列举 2 到 5 个真实文档时返回候选。"""

        candidate_ids = tuple(
            dict.fromkeys(
                document_id
                for title in titles
                for document_id in document_ids_by_title.get(title, ())
            )
        )
        return candidate_ids if 2 <= len(candidate_ids) <= 5 else ()

    @staticmethod
    def _position(value: str) -> int:
        if value.isdigit():
            return int(value)
        return _CHINESE_NUMBERS.get(value, 0)

    @staticmethod
    def _required_text(value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("知识问题不能为空")
        return " ".join(value.split())

    @staticmethod
    def _clarification(
        candidate_document_ids: Sequence[str] = (),
    ) -> KnowledgeScopeResolution:
        return KnowledgeScopeResolution(
            needs_clarification=True,
            clarification_question=(
                "我还无法确定你指的是哪篇知识文档，请直接提供文档标题后再提问。"
            ),
            candidate_document_ids=tuple(candidate_document_ids),
        )


__all__ = ["KnowledgeScopeResolver"]
