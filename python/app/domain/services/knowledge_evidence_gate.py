"""使用统一信号执行知识问答五类证据门控。"""

from __future__ import annotations

from collections.abc import Sequence

from app.models.evidence_routing import (
    EvidenceOption,
    EvidenceSignals,
    KnowledgeEvidenceDecision,
)


class KnowledgeEvidenceGate:
    """在检索前后执行无 I/O、可评估的固定决策规则。"""

    _STRICT_RELEVANCE = 0.5
    _STRICT_ANSWERABILITY = 0.75
    _MAX_AMBIGUITY = 0.5
    _DEFAULT_SCOPE_QUESTION = "请直接提供要询问的知识文档标题。"
    _TOO_MANY_OPTIONS_QUESTION = "候选范围较多，请补充更明确的对象或文档标题。"

    def precheck(
        self,
        *,
        safety_allowed: bool = True,
        skill_scope_conflict: bool = False,
        skill_candidates: Sequence[EvidenceOption] = (),
        scope_candidates: Sequence[EvidenceOption] = (),
        missing_information: Sequence[str] = (),
        clarification_question: str | None = None,
        scope_resolved: bool = True,
    ) -> KnowledgeEvidenceDecision | None:
        """在无需检索即可确定动作时短路，否则返回空。"""

        if not safety_allowed:
            return self._refuse("unsafe_request")
        if skill_scope_conflict:
            return self._refuse("skill_scope_conflict")
        skill_options = self._options(skill_candidates)
        if len(skill_options) > 5:
            return self._ask(self._TOO_MANY_OPTIONS_QUESTION, "unresolved_reference")
        if len(skill_options) >= 2:
            return self._select(skill_options, "multiple_skill_candidates")
        scope_options = self._options(scope_candidates)
        if len(scope_options) > 5:
            return self._ask(self._TOO_MANY_OPTIONS_QUESTION, "unresolved_reference")
        if len(scope_options) >= 2:
            return self._select(scope_options, "multiple_document_candidates")
        normalized_missing = self._missing_information(missing_information)
        if normalized_missing:
            question = clarification_question or (
                f"请补充完成回答所需的信息：{'、'.join(normalized_missing)}。"
            )
            return self._ask(question, "missing_information")
        if not scope_resolved:
            return self._ask(
                clarification_question or self._DEFAULT_SCOPE_QUESTION,
                "unresolved_reference",
            )
        return None

    def decide_after_retrieval(
        self,
        signals: EvidenceSignals | object,
        *,
        retry_query: str | None = None,
        rewrite_attempted: bool = False,
    ) -> KnowledgeEvidenceDecision:
        """根据已组装信号决定回答、一次改写或保守拒绝。"""

        try:
            protected = EvidenceSignals.model_validate(signals)
            normalized_retry = self._retry_query(retry_query)
        except (TypeError, ValueError):
            return self._refuse("invalid_gate_input")
        if not protected.safety_allowed:
            return self._refuse("unsafe_request")
        if not protected.scope_resolved:
            return self._refuse("out_of_scope")
        strict = protected.gate_profile == "strict_evidence"
        relevance_insufficient = not protected.selected_evidence_ids or (
            strict and protected.relevance < self._STRICT_RELEVANCE
        )
        if relevance_insufficient:
            if normalized_retry and not rewrite_attempted:
                return KnowledgeEvidenceDecision(
                    action="rewrite",
                    confidence=max(0.0, 1.0 - protected.relevance),
                    reason_code="low_relevance_retry_available",
                    rewritten_query=normalized_retry,
                )
            if normalized_retry and rewrite_attempted:
                return self._refuse("rewrite_exhausted")
            return self._refuse("no_relevant_evidence")
        minimum_answerability = self._STRICT_ANSWERABILITY if strict else 0.01
        if (
            protected.answerability < minimum_answerability
            or protected.ambiguity > self._MAX_AMBIGUITY
        ):
            return self._refuse("insufficient_answerability")
        return KnowledgeEvidenceDecision(
            action="answer",
            confidence=min(protected.relevance, protected.answerability),
            reason_code="enough_evidence",
            approved_evidence_ids=protected.selected_evidence_ids,
        )

    @staticmethod
    def _options(values: Sequence[EvidenceOption]) -> tuple[EvidenceOption, ...]:
        """复制可信候选并拒绝重复 ID。"""

        options = tuple(
            EvidenceOption.model_validate(value).model_copy(deep=True)
            for value in values
        )
        option_ids = tuple(option.option_id for option in options)
        if len(option_ids) != len(set(option_ids)):
            raise ValueError("证据门控候选 ID 不能重复")
        return options

    @staticmethod
    def _missing_information(values: Sequence[str]) -> tuple[str, ...]:
        """清理最多三个必要信息短语。"""

        normalized: list[str] = []
        for value in values:
            if not isinstance(value, str) or not value.strip():
                raise ValueError("必要信息不能为空")
            cleaned = " ".join(value.split())
            if len(cleaned) > 100 or "<" in cleaned or ">" in cleaned:
                raise ValueError("必要信息无效")
            if cleaned not in normalized:
                normalized.append(cleaned)
        if len(normalized) > 3:
            raise ValueError("必要信息不能超过三个")
        return tuple(normalized)

    @staticmethod
    def _retry_query(value: str | None) -> str | None:
        """校验查询分析阶段预先生成的唯一重试查询。"""

        if value is None:
            return None
        if not isinstance(value, str) or not value.strip():
            raise ValueError("重试查询不能为空")
        normalized = " ".join(value.split())
        if len(normalized) > 500 or "<" in normalized or ">" in normalized:
            raise ValueError("重试查询无效")
        return normalized

    @staticmethod
    def _ask(
        question: str,
        reason_code: str,
    ) -> KnowledgeEvidenceDecision:
        return KnowledgeEvidenceDecision(
            action="ask",
            confidence=1.0,
            reason_code=reason_code,
            clarification_question=question,
        )

    @staticmethod
    def _select(
        options: tuple[EvidenceOption, ...],
        reason_code: str,
    ) -> KnowledgeEvidenceDecision:
        return KnowledgeEvidenceDecision(
            action="select",
            confidence=1.0,
            reason_code=reason_code,
            options=options,
        )

    @staticmethod
    def _refuse(reason_code: str) -> KnowledgeEvidenceDecision:
        return KnowledgeEvidenceDecision(
            action="refuse",
            confidence=1.0,
            reason_code=reason_code,
        )


__all__ = ["KnowledgeEvidenceGate"]
