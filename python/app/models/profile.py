"""用户基础画像、行为画像和语义画像契约。"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field

from app.models.common import AgentResult, _StrictModel


PROFILE_VERSION = "v4"


class BaseProfileSnapshot(_StrictModel):
    """用户 SQLite 提供的显式基础画像。"""

    topics: list[str] = Field(default_factory=list)
    blocked_topics: list[str] = Field(default_factory=list)
    preferred_content_types: list[str] = Field(default_factory=list)
    preferred_difficulty: str = ""
    preferred_reading_length: str = ""
    followed_author_ids: list[str] = Field(default_factory=list)
    blocked_author_ids: list[str] = Field(default_factory=list)
    created_at: datetime


class TopicInterest(_StrictModel):
    """画像中的单个主题兴趣及其证据强度。"""

    topic: str
    weight: float = Field(ge=-1.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_count: int = Field(ge=0)
    last_interaction_at: datetime | None = None
    source: Literal["explicit", "behavior", "explicit_and_behavior"]


class ValuePreference(_StrictModel):
    """内容类型或难度等离散值偏好。"""

    value: str
    weight: float = Field(ge=-1.0, le=1.0)
    evidence_count: int = Field(ge=0)


class AuthorAffinity(_StrictModel):
    """用户与文章作者之间的确定性亲和关系。"""

    author_id: str
    affinity_score: float = Field(ge=-1.0, le=1.0)
    followed: bool
    blocked: bool
    evidence_count: int = Field(ge=0)


ActivityLevel = Literal[
    "new_user",
    "casual_reader",
    "active_reader",
    "deep_reader",
    "explorer",
    "churn_risk",
]


class ActivityProfile(_StrictModel):
    """由 REF 与阅读质量组成的用户活跃画像。"""

    recency_score: float = Field(ge=0.0, le=1.0)
    frequency_score: float = Field(ge=0.0, le=1.0)
    engagement_score: float = Field(ge=0.0, le=1.0)
    level: ActivityLevel
    active_days_30d: int = Field(default=0, ge=0)
    effective_read_count_30d: int = Field(default=0, ge=0)
    strong_interaction_count_30d: int = Field(default=0, ge=0)
    average_read_quality: float = Field(default=0.0, ge=0.0, le=1.0)
    distinct_topic_count_30d: int = Field(default=0, ge=0)


EvidenceSource = Literal[
    "user_description",
    "offline_features",
    "realtime_features",
    "rfe_activity",
]


class SemanticInterest(_StrictModel):
    """用于稳定画像输出的结构化主题兴趣。"""

    topic: str
    strength: float = Field(ge=0.0, le=1.0)
    source_topics: list[str] = Field(default_factory=list)
    evidence_sources: list[EvidenceSource] = Field(default_factory=list)
    reason: str = ""


class ExpansionInterest(_StrictModel):
    """画像允许探索的相关扩展兴趣。"""

    topic: str
    based_on: list[str] = Field(default_factory=list)
    relation: str = ""
    exploration_confidence: float = Field(ge=0.0, le=1.0)


class InterestAnalysis(_StrictModel):
    """按核心、新兴、衰退和负向分类的兴趣集合。"""

    core_interests: list[SemanticInterest] = Field(default_factory=list)
    emerging_interests: list[SemanticInterest] = Field(default_factory=list)
    fading_interests: list[SemanticInterest] = Field(default_factory=list)
    negative_interests: list[SemanticInterest] = Field(default_factory=list)
    expansion_interests: list[ExpansionInterest] = Field(default_factory=list)


class ReaderProfileAnalysis(_StrictModel):
    """用户阅读者类型、活跃等级和分析置信度。"""

    reader_type: str
    activity_level: ActivityLevel
    analysis_confidence: float = Field(ge=0.0, le=1.0)


DifficultyRecommendation = Literal[
    "beginner",
    "beginner_to_intermediate",
    "intermediate",
    "intermediate_to_advanced",
    "advanced",
    "mixed",
]


class ReadingPreferencesAnalysis(_StrictModel):
    """用户对难度、深度、篇幅和内容类型的阅读偏好。"""

    recommended_difficulty: DifficultyRecommendation
    content_depth: Literal["light", "medium", "deep", "mixed"]
    preferred_reading_length: Literal["short", "medium", "long", "mixed"]
    preferred_content_types: list[str] = Field(default_factory=list)
    technical_density: Literal["low", "medium", "high", "mixed"]
    reason: str = ""


class ExplorationStrategy(_StrictModel):
    """画像建议的聚焦与探索比例。"""

    mode: Literal["conservative", "balanced", "aggressive"]
    focus_ratio: float = Field(ge=0.0, le=1.0)
    exploration_ratio: float = Field(ge=0.0, le=1.0)
    diversity_level: Literal["low", "medium", "high"]
    reason: str = ""


class RecommendationStrategy(_StrictModel):
    """画像生成的主题、作者和排序策略摘要。"""

    primary_topics: list[str] = Field(default_factory=list)
    secondary_topics: list[str] = Field(default_factory=list)
    exploration_topics: list[str] = Field(default_factory=list)
    excluded_topics: list[str] = Field(default_factory=list)
    author_strategy: str = ""
    ranking_notes: list[str] = Field(default_factory=list)


class PreferenceConflict(_StrictModel):
    """画像证据之间检测到的偏好冲突。"""

    type: str
    description: str
    resolution: str


class SemanticProfile(_StrictModel):
    """兼容保留的结构化语义画像。"""

    reader_profile: ReaderProfileAnalysis
    interest_analysis: InterestAnalysis
    reading_preferences: ReadingPreferencesAnalysis
    exploration_strategy: ExplorationStrategy
    recommendation_strategy: RecommendationStrategy
    preference_conflicts: list[PreferenceConflict] = Field(default_factory=list)


class BehaviorProfile(_StrictModel):
    """由行为事件计算得到的兴趣与活跃特征。"""

    short_term_interests: list[TopicInterest] = Field(default_factory=list)
    long_term_interests: list[TopicInterest] = Field(default_factory=list)
    negative_interests: list[TopicInterest] = Field(default_factory=list)
    negative_document_ids: list[str] = Field(default_factory=list, max_length=50)
    content_type_preferences: list[ValuePreference] = Field(default_factory=list)
    difficulty_preferences: list[ValuePreference] = Field(default_factory=list)
    negative_difficulty_preferences: list[ValuePreference] = Field(
        default_factory=list
    )
    reading_length_preferences: list[ValuePreference] = Field(
        default_factory=list
    )
    author_affinities: list[AuthorAffinity] = Field(default_factory=list)
    search_intents: list[str] = Field(default_factory=list)
    activity: ActivityProfile


class ProfileEvidence(_StrictModel):
    """用户画像生成时使用的数据量和质量证据。"""

    valid_event_count: int = Field(ge=0)
    invalid_event_count: int = Field(ge=0)
    strong_signal_count: int = Field(ge=0)
    latest_event_at: datetime | None = None
    offline_profile_at: datetime | None = None
    realtime_event_count: int = Field(ge=0)


SemanticEnrichmentStatus = Literal["applied", "disabled", "failed", "not_needed"]


class UserProfile(_StrictModel):
    """文章推荐链路使用的稳定用户画像。"""

    user_id: str
    profile_status: Literal["ready", "cold_start", "degraded"]
    base_profile: BaseProfileSnapshot
    behavior_profile: BehaviorProfile
    semantic_profile: SemanticProfile
    profile_summary: str
    profile_confidence: float = Field(ge=0.0, le=1.0)
    profile_cold_start: bool
    semantic_enrichment_status: SemanticEnrichmentStatus = "not_needed"
    semantic_enrichment_reason: str | None = None
    evidence: ProfileEvidence
    generated_at: datetime
    expires_at: datetime
    profile_version: str = PROFILE_VERSION


class UserProfileResult(AgentResult):
    """用户画像 Agent 的稳定公开结果。"""

    agent_name: str = "user_profile"
    profile: UserProfile | None = None


__all__ = [
    "ActivityLevel",
    "ActivityProfile",
    "AuthorAffinity",
    "BaseProfileSnapshot",
    "BehaviorProfile",
    "DifficultyRecommendation",
    "EvidenceSource",
    "ExpansionInterest",
    "ExplorationStrategy",
    "InterestAnalysis",
    "PROFILE_VERSION",
    "PreferenceConflict",
    "ProfileEvidence",
    "ReaderProfileAnalysis",
    "ReadingPreferencesAnalysis",
    "RecommendationStrategy",
    "SemanticEnrichmentStatus",
    "SemanticInterest",
    "SemanticProfile",
    "TopicInterest",
    "UserProfile",
    "UserProfileResult",
    "ValuePreference",
]
