"""知识回答自动反思的严格内部契约。"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator, model_validator

from app.models.common import _StrictModel
from app.models.evidence_routing import EvidenceAction, EvidenceOption
from app.models.knowledge_qa import KnowledgeQuestionType


KnowledgeReflectionIssue = Literal[
    "none",
    "generation_unavailable",
    "invalid_citation",
    "unsupported_claim",
    "incomplete_answer",
    "off_topic",
    "missing_information",
    "ambiguous_target",
    "unsafe_answer",
]
KnowledgeReflectionRewriteMode = Literal[
    "regenerate_answer",
    "retry_retrieval",
]
KnowledgeRevisionFocus = Literal[
    "grounding",
    "coverage",
    "relevance",
    "organization",
]
KnowledgeReflectionReasonCode = Literal[
    "deterministic_pass",
    "semantic_pass",
    "repair_validation_pass",
    "reflection_unavailable_fallback",
    "generation_unavailable",
    "invalid_citation",
    "unsupported_claim",
    "incomplete_answer",
    "off_topic",
    "missing_information",
    "ambiguous_target",
    "unsafe_answer",
    "retry_query_unavailable",
    "repair_exhausted",
    "invalid_reflection_input",
]


def _normalize_text(
    value: object,
    *,
    field_name: str,
    max_length: int,
) -> str:
    """清理受控短文本并拒绝可解释标记。"""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name}不能为空")
    normalized = " ".join(value.split())
    if len(normalized) > max_length:
        raise ValueError(f"{field_name}长度不能超过 {max_length} 个字符")
    if "<" in normalized or ">" in normalized:
        raise ValueError(f"{field_name}包含非法字符")
    return normalized


def _normalize_unique_texts(
    value: object,
    *,
    field_name: str,
    max_items: int,
    item_max_length: int,
    min_items: int = 0,
) -> tuple[str, ...]:
    """复制有界文本序列并拒绝重复项。"""

    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field_name}必须是列表或元组")
    normalized = tuple(
        _normalize_text(
            item,
            field_name=field_name,
            max_length=item_max_length,
        )
        for item in value
    )
    if not min_items <= len(normalized) <= max_items:
        raise ValueError(f"{field_name}数量必须在 {min_items} 到 {max_items} 之间")
    if len({item.casefold() for item in normalized}) != len(normalized):
        raise ValueError(f"{field_name}不能重复")
    return normalized


class KnowledgeAnswerRevisionPolicy(_StrictModel):
    """回答 Agent 可接收的枚举化修订策略。"""

    focus: tuple[KnowledgeRevisionFocus, ...] = Field(
        min_length=1,
        max_length=4,
    )

    @field_validator("focus", mode="before")
    @classmethod
    def normalize_focus(cls, value: object) -> tuple[str, ...]:
        """修订关注点必须唯一且来自固定枚举。"""

        return _normalize_unique_texts(
            value,
            field_name="答案修订关注点",
            min_items=1,
            max_items=4,
            item_max_length=40,
        )


class KnowledgeAnswerRiskSignals(_StrictModel):
    """确定性风险检查消费的请求期信号。"""

    question_type: KnowledgeQuestionType
    approved_chunk_ids: tuple[str, ...] = Field(min_length=1, max_length=20)
    approved_image_ids: tuple[str, ...] = Field(default=(), max_length=6)
    cited_chunk_ids: tuple[str, ...] = Field(default=(), max_length=20)
    cited_image_ids: tuple[str, ...] = Field(default=(), max_length=6)
    document_count: int = Field(ge=1, le=20)
    answer_degraded: bool = False
    retrieval_degraded: bool = False
    has_complex_coverage: bool = False
    query_rewrite_attempted: bool = False
    repair_attempted: bool = False
    force_semantic_review: bool = False

    @field_validator(
        "approved_chunk_ids",
        "approved_image_ids",
        "cited_chunk_ids",
        "cited_image_ids",
        mode="before",
    )
    @classmethod
    def normalize_ids(cls, value: object, info: object) -> tuple[str, ...]:
        """身份序列必须有界、唯一并保持顺序。"""

        field_name = getattr(info, "field_name", "反思证据 ID")
        limit = 6 if "image" in field_name else 20
        minimum = 1 if field_name == "approved_chunk_ids" else 0
        return _normalize_unique_texts(
            value,
            field_name=field_name,
            min_items=minimum,
            max_items=limit,
            item_max_length=200,
        )


class KnowledgeAnswerReflectionAnalysis(_StrictModel):
    """反思 Agent 只能提出的结构化候选。"""

    action: EvidenceAction
    issue: KnowledgeReflectionIssue
    confidence: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    rewrite_mode: KnowledgeReflectionRewriteMode | None = None
    revision_focus: tuple[KnowledgeRevisionFocus, ...] = Field(
        default=(),
        max_length=4,
    )
    missing_information: tuple[str, ...] = Field(default=(), max_length=3)

    @field_validator("revision_focus", mode="before")
    @classmethod
    def normalize_revision_focus(cls, value: object) -> tuple[str, ...]:
        return _normalize_unique_texts(
            value,
            field_name="反思修订关注点",
            max_items=4,
            item_max_length=40,
        )

    @field_validator("missing_information", mode="before")
    @classmethod
    def normalize_missing_information(cls, value: object) -> tuple[str, ...]:
        return _normalize_unique_texts(
            value,
            field_name="反思缺失信息",
            max_items=3,
            item_max_length=100,
        )

    @model_validator(mode="after")
    def validate_action_issue(self) -> KnowledgeAnswerReflectionAnalysis:
        """约束模型候选动作、问题和修复负载的一致性。"""

        if self.action == "answer":
            if self.issue != "none" or (
                self.rewrite_mode or self.revision_focus or self.missing_information
            ):
                raise ValueError("通过候选不能携带答案问题或修复负载")
            return self
        if self.issue == "none":
            raise ValueError("非回答候选必须说明问题类型")
        if self.action == "rewrite":
            if self.issue not in {
                "unsupported_claim",
                "incomplete_answer",
                "off_topic",
            }:
                raise ValueError("改写候选的问题类型无效")
            if self.rewrite_mode is None or not self.revision_focus:
                raise ValueError("改写候选必须携带模式和修订关注点")
            if self.missing_information:
                raise ValueError("改写候选不能携带缺失信息")
            return self
        if self.rewrite_mode or self.revision_focus:
            raise ValueError("非改写候选不能携带改写负载")
        if self.action == "ask":
            if self.issue != "missing_information" or not self.missing_information:
                raise ValueError("询问候选必须说明缺失信息")
            return self
        if self.missing_information:
            raise ValueError("非询问候选不能携带缺失信息")
        if self.action == "select" and self.issue != "ambiguous_target":
            raise ValueError("选择候选必须对应目标歧义")
        if self.action == "refuse" and self.issue in {
            "missing_information",
            "ambiguous_target",
        }:
            raise ValueError("可澄清问题不能直接建议拒绝")
        return self


class KnowledgeAnswerReflectionDecision(_StrictModel):
    """确定性保护后供知识问答执行的五类动作。"""

    action: EvidenceAction
    confidence: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    reason_code: KnowledgeReflectionReasonCode
    approved: bool = False
    rewrite_mode: KnowledgeReflectionRewriteMode | None = None
    rewritten_query: str | None = Field(default=None, max_length=500)
    revision_policy: KnowledgeAnswerRevisionPolicy | None = None
    clarification_question: str | None = Field(default=None, max_length=500)
    options: tuple[EvidenceOption, ...] = Field(default=(), max_length=5)
    reflection_degraded: bool = False

    @field_validator("rewritten_query", mode="before")
    @classmethod
    def normalize_rewritten_query(cls, value: object) -> str | None:
        if value is None:
            return None
        return _normalize_text(
            value,
            field_name="反思重试查询",
            max_length=500,
        )

    @field_validator("clarification_question", mode="before")
    @classmethod
    def normalize_clarification_question(cls, value: object) -> str | None:
        if value is None:
            return None
        return _normalize_text(
            value,
            field_name="反思澄清问题",
            max_length=500,
        )

    @field_validator("options", mode="before")
    @classmethod
    def normalize_options(cls, value: object) -> tuple[EvidenceOption, ...]:
        if value is None:
            return ()
        if not isinstance(value, (list, tuple)):
            raise ValueError("反思选择项必须是列表或元组")
        options = tuple(EvidenceOption.model_validate(item) for item in value)
        option_ids = tuple(option.option_id for option in options)
        if len(option_ids) != len(set(option_ids)):
            raise ValueError("反思选择项 ID 不能重复")
        return options

    @model_validator(mode="after")
    def validate_action_payload(self) -> KnowledgeAnswerReflectionDecision:
        """保证五类动作只携带各自允许的字段组合。"""

        if self.action == "answer":
            if not self.approved:
                raise ValueError("回答动作必须批准当前草稿")
            if (
                self.rewrite_mode
                or self.rewritten_query
                or self.revision_policy
                or self.clarification_question
                or self.options
            ):
                raise ValueError("回答动作不能携带修复、问题或选项")
            if self.reason_code not in {
                "deterministic_pass",
                "semantic_pass",
                "repair_validation_pass",
                "reflection_unavailable_fallback",
            }:
                raise ValueError("回答动作原因码无效")
            if self.reflection_degraded != (
                self.reason_code == "reflection_unavailable_fallback"
            ):
                raise ValueError("反思降级标记与原因码不一致")
            return self
        if self.approved:
            raise ValueError("非回答动作不能批准当前草稿")
        if self.reflection_degraded:
            raise ValueError("非回答动作不能标记反思降级通过")
        if self.action == "rewrite":
            if self.reason_code not in {
                "unsupported_claim",
                "incomplete_answer",
                "off_topic",
            }:
                raise ValueError("改写动作原因码无效")
            if self.rewrite_mode is None or self.revision_policy is None:
                raise ValueError("改写动作必须携带模式和修订策略")
            if self.rewrite_mode == "retry_retrieval":
                if self.rewritten_query is None:
                    raise ValueError("重新检索必须携带受保护查询")
            elif self.rewritten_query is not None:
                raise ValueError("同证据再生成不能携带检索查询")
            if self.clarification_question or self.options:
                raise ValueError("改写动作不能携带问题或选项")
            return self
        if self.rewrite_mode or self.rewritten_query or self.revision_policy:
            raise ValueError("非改写动作不能携带修复负载")
        if self.action == "ask":
            if (
                self.reason_code not in {"missing_information", "ambiguous_target"}
                or not self.clarification_question
                or self.options
            ):
                raise ValueError("询问动作负载无效")
            return self
        if self.action == "select":
            if (
                self.reason_code != "ambiguous_target"
                or not 2 <= len(self.options) <= 5
                or self.clarification_question
            ):
                raise ValueError("选择动作负载无效")
            return self
        if self.clarification_question or self.options:
            raise ValueError("拒绝动作不能携带问题或选项")
        if self.reason_code in {
            "deterministic_pass",
            "semantic_pass",
            "repair_validation_pass",
            "reflection_unavailable_fallback",
            "missing_information",
            "ambiguous_target",
        }:
            raise ValueError("拒绝动作原因码无效")
        return self


__all__ = [
    "KnowledgeAnswerReflectionAnalysis",
    "KnowledgeAnswerReflectionDecision",
    "KnowledgeAnswerRevisionPolicy",
    "KnowledgeAnswerRiskSignals",
    "KnowledgeReflectionIssue",
    "KnowledgeReflectionReasonCode",
    "KnowledgeReflectionRewriteMode",
    "KnowledgeRevisionFocus",
]
