"""推荐意图、执行上下文和确定性仲裁契约。"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from app.models.common import _StrictModel


def _normalize_query(value: object, *, allow_none: bool) -> str | None:
    """清理不可信查询，并拒绝空值、非字符串和超长文本。"""

    if value is None:
        if allow_none:
            return None
        raise ValueError("检索查询不能为空")
    if not isinstance(value, str):
        raise ValueError("检索查询必须是字符串")
    cleaned = " ".join(value.split())
    if not cleaned:
        raise ValueError("检索查询不能为空")
    if len(cleaned) > 500:
        raise ValueError("检索查询长度不能超过 500 个字符")
    return cleaned


class IntentName(str, Enum):
    """意图识别允许返回的业务意图。"""

    RECOMMEND_ARTICLES = "recommend_articles"
    KNOWLEDGE_QA = "knowledge_qa"
    NO_ACTION = "no_action"
    UNKNOWN = "unknown"


class IntentState(str, Enum):
    """唯一会话当前所在的业务路由。"""

    RECOMMENDATION = "recommendation"
    KNOWLEDGE_QA = "knowledge_qa"


class RecognitionSource(str, Enum):
    """意图识别结果的来源。"""

    LLM = "llm"
    RULE = "rule"
    FALLBACK = "fallback"


class RelationHint(str, Enum):
    """当前消息与既有会话意图的关系。"""

    NEW = "new"
    REFINE = "refine"
    REPEAT = "repeat"
    UNCLEAR = "unclear"


class RecommendationIntent(_StrictModel):
    """一次推荐决策中需要仲裁的最小业务参数。"""

    resource_type: Literal["article", "book", "other"] = "article"
    size: int = Field(default=5, ge=1, le=10)


class ArbitrationAction(str, Enum):
    """确定性仲裁器允许执行的会话动作。"""

    NEW = "new"
    REFINE = "refine"
    REPEAT = "repeat"
    CLARIFY = "clarify"
    UNSUPPORTED = "unsupported"
    KNOWLEDGE_ANSWER = "knowledge_answer"
    RETURN_TO_PARENT = "return_to_parent"


class IntentRecognition(_StrictModel):
    """意图 Agent 输出给仲裁层的完整识别结果。"""

    intent: IntentName
    source: RecognitionSource
    relation: RelationHint
    confidence: float = Field(ge=0.0, le=1.0)
    rewritten_query: str | None = None
    resolved_intent: RecommendationIntent | None = None

    @field_validator("rewritten_query", mode="before")
    @classmethod
    def validate_rewritten_query(cls, value: object) -> str | None:
        """统一保护规则结果和 LLM 结果中的推荐或问答查询。"""

        return _normalize_query(value, allow_none=True)

    @model_validator(mode="after")
    def validate_business_payload(self) -> IntentRecognition:
        """保证推荐、问答和短路结果不会混写负载。"""

        if self.intent is IntentName.RECOMMEND_ARTICLES:
            if self.resolved_intent is None:
                raise ValueError("推荐意图必须携带推荐参数")
            if (
                self.relation is not RelationHint.REPEAT
                and self.rewritten_query is None
            ):
                raise ValueError("新推荐或修改推荐必须携带检索查询")
            return self
        if self.intent is IntentName.KNOWLEDGE_QA:
            if self.resolved_intent is not None:
                raise ValueError("知识问答意图不能携带推荐参数")
            if self.rewritten_query is None:
                raise ValueError("知识问答意图必须携带检索查询")
            return self
        if self.resolved_intent is not None or self.rewritten_query is not None:
            raise ValueError("短路意图不能携带业务执行负载")
        return self


class RecommendationContext(_StrictModel):
    """推荐链共享的查询、数量和已展示文档上下文。"""

    query: str = Field(min_length=1, max_length=500)
    size: int = Field(default=5, ge=1, le=10)
    seen_article_ids: list[str] = Field(default_factory=list)
    avoid_seen: bool = False

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_payload(cls, value: Any) -> Any:
        """读取旧 SQLite JSON，但不把已退役字段写回新上下文。"""

        if not isinstance(value, dict):
            return value
        payload = dict(value)
        if not payload.get("query"):
            payload["query"] = payload.get("retrieval_query") or payload.get(
                "semantic_query"
            )
        legacy_fields = {
            "primary_topics",
            "expanded_topics",
            "topic_weights",
            "semantic_query",
            "retrieval_query",
            "excluded_topics",
            "excluded_author_ids",
            "content_types",
            "difficulty",
            "requested_resource",
            "language",
            "publish_time_after",
            "publish_time_before",
        }
        for field_name in legacy_fields:
            payload.pop(field_name, None)
        return payload

    @field_validator("query", mode="before")
    @classmethod
    def validate_query(cls, value: object) -> str:
        """保护实际送入共享 Chunk Search 的查询。"""

        normalized = _normalize_query(value, allow_none=False)
        assert normalized is not None
        return normalized


class ArbitrationDecision(_StrictModel):
    """仲裁器对当前轮次给出的确定性决策。"""

    action: ArbitrationAction
    context: RecommendationContext | None = None
    reason: str
    clarification_question: str | None = None


__all__ = [
    "ArbitrationAction",
    "ArbitrationDecision",
    "IntentName",
    "IntentRecognition",
    "IntentState",
    "RecognitionSource",
    "RecommendationContext",
    "RecommendationIntent",
    "RelationHint",
]
