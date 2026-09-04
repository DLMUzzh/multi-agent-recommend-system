"""产品请求期动态 Skill 的严格 Manifest 与匹配契约。"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType
from typing import Literal

from pydantic import Field, field_validator, model_validator

from app.models.common import _StrictModel


RuntimeSkillGateProfile = Literal["default_evidence", "strict_evidence"]
RuntimeSkillFocus = Literal[
    "background",
    "architecture",
    "mechanism",
    "steps",
    "constraints",
    "tradeoffs",
    "examples",
    "risks",
]
RuntimeSkillOrganization = Literal[
    "default",
    "conclusion_then_details",
    "steps",
    "comparison_table",
]


def _normalize_text(value: object, *, field_name: str, max_length: int) -> str:
    """清理 Manifest 文本并拒绝换行和可解释标记。"""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name}不能为空")
    if "\n" in value or "\r" in value or "<" in value or ">" in value:
        raise ValueError(f"{field_name}包含非法字符")
    normalized = " ".join(value.split())
    if len(normalized) > max_length:
        raise ValueError(f"{field_name}长度不能超过 {max_length} 个字符")
    return normalized


def _normalize_text_tuple(
    value: object,
    *,
    field_name: str,
    max_items: int,
    item_max_length: int,
    min_items: int = 0,
) -> tuple[str, ...]:
    """复制并按大小写不敏感规则拒绝重复文本。"""

    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field_name}必须是列表或元组")
    normalized: list[str] = []
    seen: set[str] = set()
    for item in value:
        cleaned = _normalize_text(
            item,
            field_name=field_name,
            max_length=item_max_length,
        )
        key = cleaned.casefold()
        if key in seen:
            raise ValueError(f"{field_name}不能重复")
        normalized.append(cleaned)
        seen.add(key)
    if not min_items <= len(normalized) <= max_items:
        raise ValueError(
            f"{field_name}数量必须在 {min_items} 到 {max_items} 之间"
        )
    return tuple(normalized)


class RuntimeSkillActivation(_StrictModel):
    """Skill 的确定性激活关键词。"""

    keywords: tuple[str, ...] = Field(min_length=1, max_length=32)

    @field_validator("keywords", mode="before")
    @classmethod
    def normalize_keywords(cls, value: object) -> tuple[str, ...]:
        return _normalize_text_tuple(
            value,
            field_name="Skill 激活关键词",
            min_items=1,
            max_items=32,
            item_max_length=80,
        )


class RuntimeSkillDocumentScope(_StrictModel):
    """Skill 能够收窄到的文档主题或稳定 ID。"""

    topics: tuple[str, ...] = Field(default=(), max_length=20)
    document_ids: tuple[str, ...] = Field(default=(), max_length=20)

    @field_validator("topics", mode="before")
    @classmethod
    def normalize_topics(cls, value: object) -> tuple[str, ...]:
        return _normalize_text_tuple(
            value,
            field_name="Skill 文档主题",
            max_items=20,
            item_max_length=100,
        )

    @field_validator("document_ids", mode="before")
    @classmethod
    def normalize_document_ids(cls, value: object) -> tuple[str, ...]:
        return _normalize_text_tuple(
            value,
            field_name="Skill 文档 ID",
            max_items=20,
            item_max_length=100,
        )


class RuntimeSkillResponsePolicy(_StrictModel):
    """只影响表达方式的枚举化低优先级回答策略。"""

    focus: tuple[RuntimeSkillFocus, ...] = Field(default=(), max_length=8)
    organization: RuntimeSkillOrganization = "default"

    @field_validator("focus", mode="before")
    @classmethod
    def normalize_focus(cls, value: object) -> tuple[str, ...]:
        return _normalize_text_tuple(
            value,
            field_name="Skill 回答关注点",
            max_items=8,
            item_max_length=40,
        )


class RuntimeSkillManifest(_StrictModel):
    """磁盘 `manifest.json` 的完整白名单 Schema。"""

    skill_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{1,63}$")
    version: str = Field(
        pattern=r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$"
    )
    enabled: bool
    applies_to: tuple[Literal["knowledge_qa"], ...] = Field(
        min_length=1,
        max_length=1,
    )
    activation: RuntimeSkillActivation
    document_scope: RuntimeSkillDocumentScope
    query_terms: tuple[str, ...] = Field(default=(), max_length=20)
    gate_profile: RuntimeSkillGateProfile
    response_policy: RuntimeSkillResponsePolicy
    allowed_tools: tuple[str, ...] = Field(default=(), max_length=0)
    priority: int = Field(ge=0, le=1000)
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("query_terms", mode="before")
    @classmethod
    def normalize_query_terms(cls, value: object) -> tuple[str, ...]:
        return _normalize_text_tuple(
            value,
            field_name="Skill 查询术语",
            max_items=20,
            item_max_length=100,
        )

    @field_validator("allowed_tools", mode="before")
    @classmethod
    def require_no_tools(cls, value: object) -> tuple[str, ...]:
        """首期运行时 Skill 不得声明任何工具。"""

        if value not in ((), []):
            raise ValueError("首期运行时 Skill 不允许声明工具")
        return ()


class CompiledRuntimeSkill(_StrictModel):
    """Manifest 通过全量校验后的请求期只读投影。"""

    skill_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{1,63}$")
    version: str = Field(
        pattern=r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$"
    )
    enabled: bool
    applies_to: tuple[Literal["knowledge_qa"], ...]
    activation_keywords: tuple[str, ...] = Field(min_length=1, max_length=32)
    document_topics: tuple[str, ...] = Field(default=(), max_length=20)
    document_ids: tuple[str, ...] = Field(default=(), max_length=20)
    query_terms: tuple[str, ...] = Field(default=(), max_length=20)
    gate_profile: RuntimeSkillGateProfile
    response_policy: RuntimeSkillResponsePolicy
    priority: int = Field(ge=0, le=1000)
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def from_manifest(
        cls,
        manifest: RuntimeSkillManifest,
    ) -> CompiledRuntimeSkill:
        """只投影请求期需要的受控字段。"""

        protected = RuntimeSkillManifest.model_validate(manifest)
        return cls(
            skill_id=protected.skill_id,
            version=protected.version,
            enabled=protected.enabled,
            applies_to=protected.applies_to,
            activation_keywords=protected.activation.keywords,
            document_topics=protected.document_scope.topics,
            document_ids=protected.document_scope.document_ids,
            query_terms=protected.query_terms,
            gate_profile=protected.gate_profile,
            response_policy=protected.response_policy.model_copy(deep=True),
            priority=protected.priority,
            content_hash=protected.content_hash,
        )


class RuntimeSkillMatchCandidate(_StrictModel):
    """单个已匹配 Skill 及其可信范围。"""

    skill: CompiledRuntimeSkill
    matched_keywords: tuple[str, ...] = Field(min_length=1, max_length=32)
    resolved_document_ids: tuple[str, ...] = Field(default=(), max_length=20)
    match_count: int = Field(ge=1, le=32)
    scope_specificity: int = Field(ge=0, le=2)


class RuntimeSkillMatchResult(_StrictModel):
    """确定性匹配的唯一结果、可信并列或范围冲突。"""

    primary: RuntimeSkillMatchCandidate | None = None
    candidates: tuple[RuntimeSkillMatchCandidate, ...] = Field(
        default=(),
        max_length=5,
    )
    scope_conflict: bool = False
    too_many_candidates: bool = False

    @model_validator(mode="after")
    def validate_result(self) -> RuntimeSkillMatchResult:
        """唯一 Primary、并列候选和过多候选三者互斥。"""

        active_states = sum(
            (
                self.primary is not None,
                bool(self.candidates),
                self.too_many_candidates,
                self.scope_conflict,
            )
        )
        if active_states > 1:
            raise ValueError("Skill 匹配结果状态互斥")
        if self.candidates and not 2 <= len(self.candidates) <= 5:
            raise ValueError("Skill 并列候选必须包含 2 到 5 项")
        return self


@dataclass(frozen=True, slots=True)
class RuntimeSkillSnapshot:
    """一次请求可安全长期持有的不可变 Skill Catalog 代。"""

    generation: int
    loaded_at: datetime
    catalog_hash: str
    skills: Mapping[str, CompiledRuntimeSkill]

    @classmethod
    def build(
        cls,
        *,
        generation: int,
        loaded_at: datetime,
        skills: Sequence[CompiledRuntimeSkill],
    ) -> RuntimeSkillSnapshot:
        """复制编译结果并生成稳定 Catalog Hash。"""

        if isinstance(generation, bool) or generation < 0:
            raise ValueError("Skill Snapshot generation 必须是非负整数")
        if loaded_at.tzinfo is None or loaded_at.utcoffset() is None:
            raise ValueError("Skill Snapshot 时间必须带时区")
        protected = tuple(
            sorted(
                (
                    CompiledRuntimeSkill.model_validate(skill).model_copy(
                        deep=True
                    )
                    for skill in skills
                ),
                key=lambda skill: skill.skill_id,
            )
        )
        skill_ids = tuple(skill.skill_id for skill in protected)
        if len(skill_ids) != len(set(skill_ids)):
            raise ValueError("Skill Snapshot 不能包含重复 ID")
        payload = [skill.model_dump(mode="json") for skill in protected]
        canonical = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return cls(
            generation=generation,
            loaded_at=loaded_at,
            catalog_hash=hashlib.sha256(canonical).hexdigest(),
            skills=MappingProxyType(
                {skill.skill_id: skill for skill in protected}
            ),
        )


RuntimeSkillReloadError = Literal["catalog_invalid", "catalog_io"]


@dataclass(frozen=True, slots=True)
class RuntimeSkillReloadResult:
    """reload 的安全结果，不包含 Manifest 正文或内部路径。"""

    reloaded: bool
    snapshot: RuntimeSkillSnapshot
    error_code: RuntimeSkillReloadError | None = None

    def __post_init__(self) -> None:
        if self.reloaded == (self.error_code is not None):
            raise ValueError("Skill reload 状态与错误码不一致")


__all__ = [
    "CompiledRuntimeSkill",
    "RuntimeSkillActivation",
    "RuntimeSkillDocumentScope",
    "RuntimeSkillGateProfile",
    "RuntimeSkillManifest",
    "RuntimeSkillMatchCandidate",
    "RuntimeSkillMatchResult",
    "RuntimeSkillOrganization",
    "RuntimeSkillResponsePolicy",
    "RuntimeSkillReloadError",
    "RuntimeSkillReloadResult",
    "RuntimeSkillSnapshot",
]
