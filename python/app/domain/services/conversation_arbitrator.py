"""把完整推荐路由结果转换为可执行查询上下文。"""

from __future__ import annotations

from app.models.schemas import (
    ArbitrationAction,
    ArbitrationDecision,
    IntentName,
    IntentRecognition,
    RecommendationContext,
    RelationHint,
)


class ConversationArbitrator:
    """执行资源、置信度、会话关系和查询存在性保护。"""

    _MIN_CONFIDENCE = 0.6
    _RELATIVE_RELATIONS = {
        RelationHint.REFINE,
        RelationHint.REPEAT,
    }

    def decide(
        self,
        recognition: IntentRecognition,
        current: RecommendationContext | None,
    ) -> ArbitrationDecision:
        """根据推荐路由结果生成本轮确定性上下文和动作。"""

        if recognition.intent is not IntentName.RECOMMEND_ARTICLES:
            return ArbitrationDecision(
                action=ArbitrationAction.UNSUPPORTED,
                context=current,
                reason="当前轮不是文章推荐请求。",
            )
        if recognition.confidence < self._MIN_CONFIDENCE:
            return self._clarify(current, "推荐意图置信度不足。")
        if current is None and recognition.relation in self._RELATIVE_RELATIONS:
            return self._clarify(None, "请求引用了不存在的上一轮推荐。")
        if recognition.relation is RelationHint.UNCLEAR:
            return self._clarify(current, "无法确定本轮与上一推荐的关系。")

        intent = recognition.resolved_intent
        if intent is None:
            return self._clarify(current, "推荐请求缺少数量与资源参数。")
        if intent.resource_type != "article":
            return ArbitrationDecision(
                action=ArbitrationAction.UNSUPPORTED,
                context=current,
                reason="当前只支持文章推荐，暂不支持书籍或其他资源。",
                clarification_question="我目前只能推荐文章，请说明想查找的文章内容。",
            )

        if recognition.relation is RelationHint.REPEAT:
            assert current is not None
            context = current.model_copy(deep=True)
            context.size = intent.size
            context.avoid_seen = True
            action = ArbitrationAction.REPEAT
        else:
            query = recognition.rewritten_query
            if query is None:
                return self._clarify(current, "推荐请求缺少可执行检索查询。")
            context = RecommendationContext(query=query, size=intent.size)
            if current is None or recognition.relation is RelationHint.NEW:
                action = ArbitrationAction.NEW
            else:
                context.seen_article_ids = list(current.seen_article_ids)
                action = ArbitrationAction.REFINE

        return ArbitrationDecision(
            action=action,
            context=context,
            reason="已采用经过保护的自然语言查询和数量。",
        )

    @staticmethod
    def _clarify(
        current: RecommendationContext | None,
        reason: str,
    ) -> ArbitrationDecision:
        return ArbitrationDecision(
            action=ArbitrationAction.CLARIFY,
            context=current,
            reason=reason,
            clarification_question=(
                "我还不能确定这次要推荐什么内容，请直接说明想看的文章主题或问题。"
            ),
        )


__all__ = ["ConversationArbitrator"]
