"""知识问答五类证据路由的严格内部契约。"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator, model_validator

from app.models.common import _StrictModel


EvidenceAction = Literal["answer", "rewrite", "ask", "select", "refuse"]
EvidenceReasonCode = Literal[
    "enough_evidence",
    "low_relevance_retry_available",
    "missing_information",
    "unresolved_reference",
    "multiple_document_candidates",
    "multiple_skill_candidates",
    "unsafe_request",
    "out_of_scope",
    "skill_scope_conflict",
    "no_relevant_evidence",
    "insufficient_answerability",
    "invalid_gate_input",
    "rewrite_exhausted",
]
EvidenceGateProfile = Literal["default_evidence", "strict_evidence"]


def _normalize_text(value: object, *, field_name: str, max_length: int) -> str:
    """清理受控短文本并拒绝可被解释为标记的尖括号。"""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name}不能为空")
    normalized = " ".join(value.split())
    if len(normalized) > max_length:
        raise ValueError(f"{field_name}长度不能超过 {max_length} 个字符")
    if "<" in normalized or ">" in normalized:
        raise ValueError(f"{field_name}包含非法字符")
    return normalized


def _normalize_unique_ids(
    value: object,
    *,
    field_name: str,
    max_length: int,
) -> tuple[str, ...]:
    """复制并校验有界、顺序稳定的内部 ID。"""

    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field_name}必须是列表或元组")
    normalized = tuple(
        _normalize_text(item, field_name=field_name, max_length=200)
        for item in value
    )
    if len(normalized) > max_length:
        raise ValueError(f"{field_name}数量不能超过 {max_length}")
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{field_name}不能重复")
    return normalized


class EvidenceOption(_StrictModel):
    """由可信 Snapshot 回填的单个选择项。"""

    option_id: str = Field(min_length=1, max_length=200)
    label: str = Field(min_length=1, max_length=500)

    @field_validator("option_id", mode="before")
    @classmethod
    def normalize_option_id(cls, value: object) -> str:
        """选择项 ID 只允许受控短文本。"""

        return _normalize_text(value, field_name="选择项 ID", max_length=200)

    @field_validator("label", mode="before")
    @classmethod
    def normalize_label(cls, value: object) -> str:
        """展示标题不能携带标记文本。"""

        return _normalize_text(value, field_name="选择项标题", max_length=500)


class EvidenceSignals(_StrictModel):
    """EvidenceSelector 或 Coverage 后供门控消费的统一信号。"""

    relevance: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    answerability: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    ambiguity: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    safety_allowed: bool = True
    scope_resolved: bool = True
    gate_profile: EvidenceGateProfile = "default_evidence"
    selected_evidence_ids: tuple[str, ...] = Field(default=(), max_length=20)

    @field_validator("selected_evidence_ids", mode="before")
    @classmethod
    def normalize_selected_evidence_ids(cls, value: object) -> tuple[str, ...]:
        """证据 ID 必须来自同一已校验结果且保持唯一。"""

        return _normalize_unique_ids(
            value,
            field_name="选中证据 ID",
            max_length=20,
        )


class KnowledgeEvidenceDecision(_StrictModel):
    """五类门控的互斥决策输出。"""

    action: EvidenceAction
    confidence: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    reason_code: EvidenceReasonCode
    rewritten_query: str | None = Field(default=None, max_length=500)
    clarification_question: str | None = Field(default=None, max_length=500)
    options: tuple[EvidenceOption, ...] = Field(default=(), max_length=5)
    approved_evidence_ids: tuple[str, ...] = Field(default=(), max_length=20)

    @field_validator("rewritten_query", mode="before")
    @classmethod
    def normalize_rewritten_query(cls, value: object) -> str | None:
        """改写查询必须是安全的有界文本。"""

        if value is None:
            return None
        return _normalize_text(value, field_name="改写查询", max_length=500)

    @field_validator("clarification_question", mode="before")
    @classmethod
    def normalize_clarification_question(cls, value: object) -> str | None:
        """澄清问题必须是安全的有界文本。"""

        if value is None:
            return None
        return _normalize_text(value, field_name="澄清问题", max_length=500)

    @field_validator("options", mode="before")
    @classmethod
    def normalize_options(cls, value: object) -> tuple[EvidenceOption, ...]:
        """复制选择项并拒绝重复可信 ID。"""

        if value is None:
            return ()
        if not isinstance(value, (list, tuple)):
            raise ValueError("选择项必须是列表或元组")
        options = tuple(EvidenceOption.model_validate(item) for item in value)
        option_ids = tuple(option.option_id for option in options)
        if len(option_ids) != len(set(option_ids)):
            raise ValueError("选择项 ID 不能重复")
        return options

    @field_validator("approved_evidence_ids", mode="before")
    @classmethod
    def normalize_approved_evidence_ids(cls, value: object) -> tuple[str, ...]:
        """批准证据必须是顺序稳定的唯一 ID。"""

        return _normalize_unique_ids(
            value,
            field_name="批准证据 ID",
            max_length=20,
        )

    @model_validator(mode="after")
    def validate_action_payload(self) -> KnowledgeEvidenceDecision:
        """保证五类动作只携带各自允许的字段组合。"""

        reason_codes = {
            "answer": {"enough_evidence"},
            "rewrite": {"low_relevance_retry_available"},
            "ask": {"missing_information", "unresolved_reference"},
            "select": {
                "multiple_document_candidates",
                "multiple_skill_candidates",
            },
            "refuse": {
                "unsafe_request",
                "out_of_scope",
                "skill_scope_conflict",
                "no_relevant_evidence",
                "insufficient_answerability",
                "invalid_gate_input",
                "rewrite_exhausted",
            },
        }
        if self.reason_code not in reason_codes[self.action]:
            raise ValueError("证据动作与原因码不一致")
        if self.action == "answer":
            if not self.approved_evidence_ids:
                raise ValueError("回答动作必须批准至少一条证据")
            if self.rewritten_query or self.clarification_question or self.options:
                raise ValueError("回答动作不能携带改写、问题或选择项")
            return self
        if self.approved_evidence_ids:
            raise ValueError("非回答动作不能携带批准证据")
        if self.action == "rewrite":
            if not self.rewritten_query:
                raise ValueError("改写动作必须携带查询")
            if self.clarification_question or self.options:
                raise ValueError("改写动作不能携带问题或选择项")
            return self
        if self.action == "ask":
            if not self.clarification_question:
                raise ValueError("询问动作必须携带一个澄清问题")
            if self.rewritten_query or self.options:
                raise ValueError("询问动作不能携带查询或选择项")
            return self
        if self.action == "select":
            if not 2 <= len(self.options) <= 5:
                raise ValueError("选择动作必须携带 2 到 5 个可信选项")
            if self.rewritten_query or self.clarification_question:
                raise ValueError("选择动作不能携带查询或澄清问题")
            return self
        if self.rewritten_query or self.clarification_question or self.options:
            raise ValueError("拒绝动作不能携带查询、问题或选择项")
        return self


__all__ = [
    "EvidenceAction",
    "EvidenceGateProfile",
    "EvidenceOption",
    "EvidenceReasonCode",
    "EvidenceSignals",
    "KnowledgeEvidenceDecision",
]
