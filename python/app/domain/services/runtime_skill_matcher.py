"""仅根据当前用户问题确定性匹配产品运行时 Skill。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from app.models.runtime_skill import (
    CompiledRuntimeSkill,
    RuntimeSkillMatchCandidate,
    RuntimeSkillMatchResult,
)


class RuntimeSkillMatcher:
    """按关键词、范围具体度和优先级选择零个或一个 Primary Skill。"""

    def match(
        self,
        question: str,
        *,
        skills: Sequence[CompiledRuntimeSkill],
        document_ids: Sequence[str] = (),
        document_topics_by_id: Mapping[str, Sequence[str]] | None = None,
    ) -> RuntimeSkillMatchResult:
        """使用原始问题和可信文档事实返回稳定匹配结果。"""

        normalized_question = self._required_text(question, "知识问题")
        requested_ids = self._unique_text(document_ids, "请求文档 ID")
        topics_by_id = self._topics_by_id(document_topics_by_id or {})
        candidates: list[RuntimeSkillMatchCandidate] = []
        scope_conflict = False
        for raw_skill in skills:
            skill = CompiledRuntimeSkill.model_validate(raw_skill).model_copy(
                deep=True
            )
            if not skill.enabled or "knowledge_qa" not in skill.applies_to:
                continue
            matched_keywords = tuple(
                keyword
                for keyword in skill.activation_keywords
                if keyword.casefold() in normalized_question.casefold()
            )
            if not matched_keywords:
                continue
            resolved_scope = self._resolved_scope(skill, topics_by_id)
            has_specific_scope = bool(skill.document_ids or skill.document_topics)
            if requested_ids and has_specific_scope:
                effective_scope = tuple(
                    document_id
                    for document_id in requested_ids
                    if document_id in resolved_scope
                )
                if not effective_scope:
                    scope_conflict = True
                    continue
            elif requested_ids:
                effective_scope = requested_ids
            else:
                effective_scope = resolved_scope
            candidates.append(
                RuntimeSkillMatchCandidate(
                    skill=skill,
                    matched_keywords=matched_keywords,
                    resolved_document_ids=effective_scope,
                    match_count=len(matched_keywords),
                    scope_specificity=(
                        2 if skill.document_ids else 1 if skill.document_topics else 0
                    ),
                )
            )
        if not candidates:
            return RuntimeSkillMatchResult(scope_conflict=scope_conflict)
        candidates.sort(
            key=lambda item: (
                -item.match_count,
                -item.scope_specificity,
                -item.skill.priority,
                item.skill.skill_id,
            )
        )
        best_score = self._score(candidates[0])
        best = tuple(item for item in candidates if self._score(item) == best_score)
        if len(best) == 1:
            return RuntimeSkillMatchResult(primary=best[0])
        if len(best) <= 5:
            return RuntimeSkillMatchResult(candidates=best)
        return RuntimeSkillMatchResult(too_many_candidates=True)

    @staticmethod
    def _resolved_scope(
        skill: CompiledRuntimeSkill,
        topics_by_id: Mapping[str, tuple[str, ...]],
    ) -> tuple[str, ...]:
        explicit = tuple(skill.document_ids)
        topic_keys = {topic.casefold() for topic in skill.document_topics}
        topic_matches = tuple(
            document_id
            for document_id, topics in topics_by_id.items()
            if topic_keys.intersection(topic.casefold() for topic in topics)
        )
        return tuple(dict.fromkeys((*explicit, *topic_matches)))

    @staticmethod
    def _score(candidate: RuntimeSkillMatchCandidate) -> tuple[int, int, int]:
        return (
            candidate.match_count,
            candidate.scope_specificity,
            candidate.skill.priority,
        )

    @classmethod
    def _topics_by_id(
        cls,
        values: Mapping[str, Sequence[str]],
    ) -> dict[str, tuple[str, ...]]:
        result: dict[str, tuple[str, ...]] = {}
        for document_id, topics in values.items():
            normalized_id = cls._required_text(document_id, "文档 ID")
            result[normalized_id] = cls._unique_text(topics, "文档主题")
        return result

    @classmethod
    def _unique_text(
        cls,
        values: Sequence[str],
        label: str,
    ) -> tuple[str, ...]:
        normalized = tuple(cls._required_text(value, label) for value in values)
        if len(normalized) != len(set(normalized)):
            raise ValueError(f"{label}不能重复")
        return normalized

    @staticmethod
    def _required_text(value: str, label: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{label}不能为空")
        normalized = " ".join(value.split())
        if "<" in normalized or ">" in normalized:
            raise ValueError(f"{label}包含非法字符")
        return normalized


__all__ = ["RuntimeSkillMatcher"]
