"""知识回答草稿的确定性风险检查与反思候选保护。"""

from __future__ import annotations

from collections.abc import Sequence

from app.models.evidence_routing import EvidenceOption
from app.models.knowledge_reflection import (
    KnowledgeAnswerReflectionAnalysis,
    KnowledgeAnswerReflectionDecision,
    KnowledgeAnswerRevisionPolicy,
    KnowledgeAnswerRiskSignals,
)


class KnowledgeAnswerReflectionPolicy:
    """保护自动反思的硬边界、动作预算和可信候选。"""

    _SEMANTIC_QUESTION_TYPES = {
        "comparative",
        "analytical",
        "exploratory",
        "verification",
    }
    _AMBIGUITY_QUESTION = "当前存在多个可信对象，请补充更明确的对象或文档标题。"

    def precheck(
        self,
        signals: KnowledgeAnswerRiskSignals | object,
    ) -> KnowledgeAnswerReflectionDecision | None:
        """硬错误直接短路，低风险直接通过，语义风险返回空。"""

        try:
            protected = KnowledgeAnswerRiskSignals.model_validate(signals)
        except (TypeError, ValueError):
            return self._refuse("invalid_reflection_input")
        hard_failure = self._hard_failure(protected)
        if hard_failure is not None:
            return hard_failure
        if (
            protected.force_semantic_review
            or protected.document_count > 1
            or protected.has_complex_coverage
            or protected.question_type in self._SEMANTIC_QUESTION_TYPES
        ):
            return None
        return self._answer("deterministic_pass")

    def protect(
        self,
        analysis: KnowledgeAnswerReflectionAnalysis | object,
        *,
        signals: KnowledgeAnswerRiskSignals,
        retry_query: str | None = None,
        trusted_options: Sequence[EvidenceOption] = (),
        allow_retrieval_retry: bool = True,
    ) -> KnowledgeAnswerReflectionDecision:
        """保护模型候选并返回可执行的五类动作。"""

        try:
            protected_signals = KnowledgeAnswerRiskSignals.model_validate(
                signals
            ).model_copy(deep=True)
            protected_analysis = KnowledgeAnswerReflectionAnalysis.model_validate(
                analysis
            ).model_copy(deep=True)
            options = self._options(trusted_options)
        except (TypeError, ValueError):
            return self._refuse("invalid_reflection_input")
        hard_failure = self._hard_failure(protected_signals)
        if hard_failure is not None:
            return hard_failure
        if protected_analysis.action == "answer":
            return self._answer(
                "semantic_pass",
                confidence=protected_analysis.confidence,
            )
        if protected_signals.repair_attempted and (
            protected_analysis.action == "rewrite"
        ):
            return self._refuse("repair_exhausted")
        if protected_analysis.action == "rewrite":
            assert protected_analysis.rewrite_mode is not None
            revision_policy = self._revision_policy(protected_analysis)
            if protected_analysis.rewrite_mode == "retry_retrieval":
                normalized_query = self._retry_query(retry_query)
                if (
                    not allow_retrieval_retry
                    or protected_signals.query_rewrite_attempted
                    or normalized_query is None
                ):
                    return self._refuse("retry_query_unavailable")
                return KnowledgeAnswerReflectionDecision(
                    action="rewrite",
                    confidence=protected_analysis.confidence,
                    reason_code=protected_analysis.issue,
                    rewrite_mode="retry_retrieval",
                    rewritten_query=normalized_query,
                    revision_policy=revision_policy,
                )
            return KnowledgeAnswerReflectionDecision(
                action="rewrite",
                confidence=protected_analysis.confidence,
                reason_code=protected_analysis.issue,
                rewrite_mode="regenerate_answer",
                revision_policy=revision_policy,
            )
        if protected_analysis.action == "ask":
            return KnowledgeAnswerReflectionDecision(
                action="ask",
                confidence=protected_analysis.confidence,
                reason_code="missing_information",
                clarification_question=(
                    "请补充完成回答所需的信息："
                    + "、".join(protected_analysis.missing_information)
                    + "。"
                ),
            )
        if protected_analysis.action == "select":
            if 2 <= len(options) <= 5:
                return KnowledgeAnswerReflectionDecision(
                    action="select",
                    confidence=protected_analysis.confidence,
                    reason_code="ambiguous_target",
                    options=options,
                )
            return KnowledgeAnswerReflectionDecision(
                action="ask",
                confidence=protected_analysis.confidence,
                reason_code="ambiguous_target",
                clarification_question=self._AMBIGUITY_QUESTION,
            )
        return self._refuse(protected_analysis.issue)

    def fallback(
        self,
        signals: KnowledgeAnswerRiskSignals,
    ) -> KnowledgeAnswerReflectionDecision:
        """模型不可用时保持已有确定性安全边界。"""

        prechecked = self.precheck(signals)
        if prechecked is not None:
            return prechecked
        return self._answer(
            "reflection_unavailable_fallback",
            reflection_degraded=True,
        )

    def validate_repaired(
        self,
        signals: KnowledgeAnswerRiskSignals | object,
    ) -> KnowledgeAnswerReflectionDecision:
        """修复后只复检生成和引用硬边界，不再请求语义判断。"""

        try:
            protected = KnowledgeAnswerRiskSignals.model_validate(signals)
        except (TypeError, ValueError):
            return self._refuse("invalid_reflection_input")
        if self._hard_failure(protected) is not None:
            return self._refuse("repair_exhausted")
        return self._answer("repair_validation_pass")

    @staticmethod
    def _hard_failure(
        signals: KnowledgeAnswerRiskSignals,
    ) -> KnowledgeAnswerReflectionDecision | None:
        if signals.answer_degraded:
            return KnowledgeAnswerReflectionPolicy._refuse("generation_unavailable")
        if not signals.cited_chunk_ids and not signals.cited_image_ids:
            return KnowledgeAnswerReflectionPolicy._refuse("invalid_citation")
        if not set(signals.cited_chunk_ids).issubset(
            signals.approved_chunk_ids
        ) or not set(signals.cited_image_ids).issubset(signals.approved_image_ids):
            return KnowledgeAnswerReflectionPolicy._refuse("invalid_citation")
        return None

    @staticmethod
    def _revision_policy(
        analysis: KnowledgeAnswerReflectionAnalysis,
    ) -> KnowledgeAnswerRevisionPolicy:
        return KnowledgeAnswerRevisionPolicy(focus=analysis.revision_focus)

    @staticmethod
    def _retry_query(value: str | None) -> str | None:
        if value is None or not isinstance(value, str) or not value.strip():
            return None
        normalized = " ".join(value.split())
        if len(normalized) > 500 or "<" in normalized or ">" in normalized:
            return None
        return normalized

    @staticmethod
    def _options(
        values: Sequence[EvidenceOption],
    ) -> tuple[EvidenceOption, ...]:
        options = tuple(
            EvidenceOption.model_validate(value).model_copy(deep=True)
            for value in values
        )
        option_ids = tuple(option.option_id for option in options)
        if len(option_ids) != len(set(option_ids)):
            raise ValueError("反思可信选项不能重复")
        return options

    @staticmethod
    def _answer(
        reason_code: str,
        *,
        confidence: float = 1.0,
        reflection_degraded: bool = False,
    ) -> KnowledgeAnswerReflectionDecision:
        return KnowledgeAnswerReflectionDecision(
            action="answer",
            confidence=confidence,
            reason_code=reason_code,
            approved=True,
            reflection_degraded=reflection_degraded,
        )

    @staticmethod
    def _refuse(reason_code: str) -> KnowledgeAnswerReflectionDecision:
        return KnowledgeAnswerReflectionDecision(
            action="refuse",
            confidence=1.0,
            reason_code=reason_code,
        )


__all__ = ["KnowledgeAnswerReflectionPolicy"]
