"""保护个人自然语言反馈候选并构造有限补救动作。"""

from __future__ import annotations

import re

from app.models.personal_feedback import (
    ConversationResultSnapshot,
    FeedbackAnalysis,
    FeedbackDecision,
    PersonalFeedbackEvent,
    RecommendationMemorySignal,
)


class FeedbackRecoveryPolicy:
    """确定性解析目标、限制动作次数并隔离个人记忆副作用。"""

    _CANDIDATE_PATTERN = re.compile(
        r"(?:不相关|不满意|不好|不对|错了|答错|没找到|未检索|漏了|"
        r"不完整|太基础|太简单|太难|太啰嗦|不是推荐|不是问答|"
        r"以后|每次|默认不要|不要再)"
    )
    _GENERIC_NEGATIVE_PATTERN = re.compile(
        r"^(?:这些|这个)?(?:回答|推荐)?(?:都)?(?:不太)?"
        r"(?:好|不好|不对|不满意)[。！!]?$"
    )
    _ORDINALS = {
        "一": 1,
        "二": 2,
        "两": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
        "十": 10,
    }

    def is_candidate(
        self,
        message: str,
        *,
        snapshot: ConversationResultSnapshot | None,
        pending_event: PersonalFeedbackEvent | None,
    ) -> bool:
        """只在存在合法反馈目标时执行高召回、低成本门控。"""

        normalized = " ".join(str(message).split())
        if not normalized or (snapshot is None and pending_event is None):
            return False
        if pending_event is not None and pending_event.status in {
            "classifying",
            "awaiting_detail",
        }:
            return True
        return self._CANDIDATE_PATTERN.search(normalized) is not None

    def fallback_decision(
        self,
        message: str,
        *,
        snapshot: ConversationResultSnapshot | None,
        pending_event: PersonalFeedbackEvent | None,
    ) -> FeedbackDecision:
        """无 LLM 时只对高精度泛化负面反馈追问，不猜测补救动作。"""

        normalized = " ".join(str(message).split())
        if snapshot is None and pending_event is None:
            return self._normal_decision("feedback_target_missing")
        if self._GENERIC_NEGATIVE_PATTERN.fullmatch(normalized):
            return FeedbackDecision(
                is_feedback=True,
                feedback_type="recommendation_irrelevant"
                if snapshot is not None and snapshot.result_type == "recommendation"
                else "answer_incorrect",
                completeness="incomplete",
                next_action="clarify",
                clarification_question=(
                    "你不满意的是内容不相关、已经看过，还是难度不合适？"
                    if snapshot is not None
                    and snapshot.result_type == "recommendation"
                    else "你不满意的是事实有误、内容不完整，还是回答方式不合适？"
                ),
                reason_code="generic_negative_needs_detail",
            )
        return self._normal_decision("fallback_no_high_precision_feedback")

    def protect(
        self,
        message: str,
        *,
        snapshot: ConversationResultSnapshot | None,
        pending_event: PersonalFeedbackEvent | None,
        analysis: FeedbackAnalysis,
    ) -> FeedbackDecision:
        """验证 LLM 候选的目标、查询、动作和记忆路由后返回新对象。"""

        validated = FeedbackAnalysis.model_validate(analysis).model_copy(deep=True)
        if not validated.is_feedback or validated.feedback_type == "no_feedback":
            return self._normal_decision(validated.reason_code)
        if snapshot is None:
            raise ValueError("反馈补救缺少结果快照")
        if pending_event is not None:
            if pending_event.source_result_id != snapshot.result_id:
                raise ValueError("待补充反馈与结果快照不一致")
            if pending_event.clarification_count >= 1 and (
                validated.completeness == "incomplete"
            ):
                return self._normal_decision("clarification_limit_reached")
            if pending_event.recovery_count >= 1:
                return self._normal_decision("recovery_limit_reached")

        target_document_ids = self._resolve_target_document_ids(
            message,
            snapshot=snapshot,
            proposed=validated.target_document_ids,
        )
        protected_query = validated.corrected_query or snapshot.query
        next_action = validated.suggested_action
        if validated.completeness == "incomplete":
            return FeedbackDecision(
                is_feedback=True,
                feedback_type=validated.feedback_type,
                completeness="incomplete",
                next_action="clarify",
                target_document_ids=target_document_ids,
                clarification_question=self._clarification_question(validated),
                reason_code=validated.reason_code,
            )

        if next_action == "retry_answer_from_evidence":
            if snapshot.result_type != "knowledge_answer" or not (
                snapshot.citation_chunk_ids
            ):
                raise ValueError("原结果没有可复用的可信回答证据")
        if next_action == "retry_recommendation" and (
            snapshot.result_type != "recommendation"
        ):
            raise ValueError("知识回答反馈不能执行推荐补救")
        if next_action in {"retry_recommendation", "retry_retrieval"} and not (
            protected_query
        ):
            raise ValueError("补救动作缺少可执行查询")

        signals = self._protect_recommendation_signals(
            validated,
            snapshot=snapshot,
            target_document_ids=target_document_ids,
        )
        memory_routes: list[str] = []
        if validated.feedback_type == "answer_style":
            memory_routes.append("interaction_memory")
        if validated.feedback_type == "route_correction":
            memory_routes.append("intent_memory")
        if signals:
            memory_routes.append("recommendation_profile")
        excluded_document_ids = (
            target_document_ids if next_action == "retry_recommendation" else ()
        )
        return FeedbackDecision.model_validate(
            {
                "is_feedback": True,
                "feedback_type": validated.feedback_type,
                "completeness": "complete",
                "next_action": next_action,
                "protected_query": protected_query,
                "target_document_ids": target_document_ids,
                "excluded_document_ids": excluded_document_ids,
                "recommendation_signals": signals,
                "route_target": validated.route_target,
                "memory_routes": tuple(memory_routes),
                "reason_code": validated.reason_code,
            }
        )

    def _resolve_target_document_ids(
        self,
        message: str,
        *,
        snapshot: ConversationResultSnapshot,
        proposed: tuple[str, ...],
    ) -> tuple[str, ...]:
        """目标只能来自快照；自然语言顺序由程序解析而非信任 LLM。"""

        allowed = (
            snapshot.recommendation_document_ids
            if snapshot.result_type == "recommendation"
            else snapshot.resolved_document_ids or snapshot.citation_document_ids
        )
        if any(document_id not in allowed for document_id in proposed):
            raise ValueError("反馈目标文档不属于上一轮结果")
        position = self._extract_position(message)
        if position is not None:
            if snapshot.result_type != "recommendation" or position > len(allowed):
                raise ValueError("反馈目标文档顺序无法从上一轮结果解析")
            resolved = (allowed[position - 1],)
            if proposed and proposed != resolved:
                raise ValueError("反馈目标文档与程序解析顺序不一致")
            return resolved
        return proposed

    def _protect_recommendation_signals(
        self,
        analysis: FeedbackAnalysis,
        *,
        snapshot: ConversationResultSnapshot,
        target_document_ids: tuple[str, ...],
    ) -> tuple[RecommendationMemorySignal, ...]:
        """事实纠错、检索失败和一次性信号不能污染推荐画像。"""

        if analysis.feedback_type in {
            "answer_incorrect",
            "answer_incomplete",
            "article_not_found",
            "answer_style",
            "route_correction",
        }:
            return ()
        allowed = set(snapshot.recommendation_document_ids)
        accepted: list[RecommendationMemorySignal] = []
        for signal in analysis.recommendation_signals:
            candidate = RecommendationMemorySignal.model_validate(signal)
            if any(document_id not in allowed for document_id in candidate.source_document_ids):
                raise ValueError("推荐记忆信号引用了快照外文档")
            if candidate.source_document_ids and target_document_ids and not set(
                candidate.source_document_ids
            ).issubset(set(target_document_ids)):
                raise ValueError("推荐记忆信号与当前反馈目标不一致")
            if candidate.persistence == "current_recovery_only":
                continue
            accepted.append(candidate)
        return tuple(accepted)

    @classmethod
    def _extract_position(cls, message: str) -> int | None:
        """解析“第二篇”等有界推荐顺序。"""

        match = re.search(r"第\s*(\d{1,2}|[一二两三四五六七八九十])\s*篇", message)
        if match is None:
            return None
        raw = match.group(1)
        return int(raw) if raw.isdigit() else cls._ORDINALS[raw]

    @staticmethod
    def _clarification_question(analysis: FeedbackAnalysis) -> str:
        """根据缺失字段生成单一有界追问。"""

        missing = set(analysis.missing_information)
        if "article_identity" in missing:
            return "你指的是哪一篇文章？可以提供标题、作者或它在上一轮中的顺序。"
        if "correct_fact" in missing:
            return "具体哪一点有误？请补充你期望的正确事实或范围。"
        if "topic" in missing or "scope" in missing:
            return "你希望我改为哪个主题或范围？"
        return "你不满意的是内容不相关、事实有误、内容不完整，还是回答方式不合适？"

    @staticmethod
    def _normal_decision(reason_code: str) -> FeedbackDecision:
        """构造不携带任何副作用的普通流程决策。"""

        return FeedbackDecision(
            is_feedback=False,
            feedback_type="no_feedback",
            completeness="complete",
            next_action="normal",
            reason_code=reason_code,
        )


__all__ = ["FeedbackRecoveryPolicy"]
