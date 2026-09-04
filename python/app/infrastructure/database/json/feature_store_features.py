"""Feature Store 的画像证据、质量与公开特征快照组装。"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from app.infrastructure.database.json.feature_store_models import (
    BehaviorEvent,
    UserBaseProfile,
    UserNotFoundError,
    _require_string_id,
    _require_timezone,
)


class _FeatureStoreFeaturesMixin:
    """依赖 FeatureStore 门面状态的确定性特征快照实现。"""

    async def get_user_features(
        self,
        user_id: str,
        *,
        as_of: datetime | str | None = None,
        max_events: int = 5000,
    ) -> dict[str, Any]:
        """合并用户描述、每日离线标签、七天在线标签与访问时 REF。"""

        del max_events
        user_id = _require_string_id(user_id)
        as_of_dt = self._coerce_datetime(as_of) if as_of else self._now()
        user_data = await self.get_user(user_id)
        if not user_data:
            raise UserNotFoundError(f"用户不存在：{user_id}")
        user = UserBaseProfile.model_validate(user_data)
        state = self._local_redis_state_for(user_id=user_id, as_of=as_of_dt)
        online = state.get("online_tags", {})
        offline = state.get("offline_tags", {})
        activity = self._calculate_ref_activity_from_state(state, as_of_dt)
        evidence = self._feature_evidence(
            user=user,
            online=online,
            offline=offline,
            state=state,
        )
        quality = self._feature_quality(
            user_id=user_id,
            as_of=as_of_dt,
            state=state,
            activity=activity,
            evidence=evidence,
        )
        return self._feature_snapshot(
            user=user,
            as_of=as_of_dt,
            online=online,
            offline=offline,
            activity=activity,
            evidence=evidence,
            quality=quality,
            previous_profile=await self.get_latest_historical_profile(user_id),
        )

    def _feature_evidence(
        self,
        *,
        user: UserBaseProfile,
        online: dict[str, Any],
        offline: dict[str, Any],
        state: dict[str, Any],
    ) -> dict[str, Any]:
        """把离线、在线标签转换为统一的画像证据行。"""

        short_topics = self._tag_score_rows(
            online.get("topic_scores"),
            "topic",
            updated_at=online.get("updated_at"),
        )
        long_topics = self._tag_score_rows(
            offline.get("topic_scores"),
            "topic",
            updated_at=offline.get("updated_at"),
        )
        negative_topics = self._tag_score_rows(
            self._merge_score_maps(
                offline.get("negative_topic_scores"),
                online.get("negative_topic_scores"),
            ),
            "topic",
            updated_at=online.get("updated_at") or offline.get("updated_at"),
            ascending=True,
        )
        negative_difficulties = self._tag_score_rows(
            self._merge_score_maps(
                offline.get("negative_difficulty_scores"),
                online.get("negative_difficulty_scores"),
            ),
            "value",
            updated_at=online.get("updated_at") or offline.get("updated_at"),
            ascending=True,
        )
        negative_document_ids = sorted(
            {
                str(item)
                for item in (
                    list(offline.get("negative_document_ids", []))
                    + list(online.get("negative_document_ids", []))
                )
                if isinstance(item, str) and item.strip()
            }
        )[:50]
        content_types = self._tag_score_rows(
            self._merge_score_maps(
                offline.get("content_type_scores"),
                online.get("content_type_scores"),
            ),
            "value",
            updated_at=online.get("updated_at") or offline.get("updated_at"),
        )
        difficulties = self._tag_score_rows(
            self._merge_score_maps(
                offline.get("difficulty_scores"),
                online.get("difficulty_scores"),
            ),
            "value",
            updated_at=online.get("updated_at") or offline.get("updated_at"),
        )
        reading_lengths = self._tag_score_rows(
            self._merge_score_maps(
                offline.get("reading_length_scores"),
                online.get("reading_length_scores"),
            ),
            "value",
            updated_at=online.get("updated_at") or offline.get("updated_at"),
        )
        authors = self._tag_author_rows(
            user=user,
            scores=self._merge_score_maps(
                offline.get("author_scores"),
                online.get("author_scores"),
            ),
        )
        recent_behaviors = state.get("recent_behaviors", {})
        return {
            "short_topics": short_topics,
            "long_topics": long_topics,
            "negative_topics": negative_topics,
            "negative_difficulties": negative_difficulties,
            "negative_document_ids": negative_document_ids,
            "content_types": content_types,
            "difficulties": difficulties,
            "reading_lengths": reading_lengths,
            "authors": authors,
            "recent_event_count": sum(
                len(items)
                for items in recent_behaviors.values()
                if isinstance(items, list)
            ),
            "event_type_counts_7d": {
                event_type: len(items)
                for event_type, items in sorted(recent_behaviors.items())
                if isinstance(items, list)
            },
            "searches": list(
                dict.fromkeys(
                    [
                        *online.get("search_queries", []),
                        *offline.get("search_queries", []),
                    ]
                )
            )[:10],
        }

    def _feature_quality(
        self,
        *,
        user_id: str,
        as_of: datetime,
        state: dict[str, Any],
        activity: dict[str, Any],
        evidence: dict[str, Any],
    ) -> dict[str, Any]:
        """根据原始行为与 REF 快照计算数据质量和画像置信度。"""

        raw_user_events = [
            event
            for event in self._events_by_user.get(user_id, [])
            if event.occurred_at <= as_of
        ]
        latest_event = max(
            raw_user_events,
            key=lambda item: (item.occurred_at, item.event_id),
            default=None,
        )
        valid_event_count = len(raw_user_events)
        strong_signal_count = sum(
            int(bucket.get("strong_behavior_count", 0))
            for bucket in state.get("ref_activity", {}).get("daily", {}).values()
        )
        metadata_completeness = (
            (
                activity["average_read_quality"]
                if activity["effective_read_count_30d"]
                else 1.0
            )
            if valid_event_count
            else 0.0
        )
        topic_metadata_ratio = float(
            bool(evidence["long_topics"] or evidence["short_topics"])
        )
        return {
            "valid_event_count": valid_event_count,
            "invalid_event_count": sum(
                1 for item in self._load_errors if item.get("user_id") == user_id
            ),
            "strong_signal_count": strong_signal_count,
            "metadata_completeness": metadata_completeness,
            "topic_metadata_ratio": topic_metadata_ratio,
            "confidence_inputs": {
                "valid_event_count": valid_event_count,
                "metadata_completeness": metadata_completeness,
                "strong_signal_count": strong_signal_count,
                "recency_score": activity["recency_score"],
                "has_consumption_signal": (
                    activity["effective_read_count_30d"] > 0
                ),
                "topic_metadata_ratio": topic_metadata_ratio,
            },
            "latest_event": latest_event,
        }

    def _feature_snapshot(
        self,
        *,
        user: UserBaseProfile,
        as_of: datetime,
        online: dict[str, Any],
        offline: dict[str, Any],
        activity: dict[str, Any],
        evidence: dict[str, Any],
        quality: dict[str, Any],
        previous_profile: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """组装保持公开字段稳定的用户特征快照。"""

        explicit_preferences = self._explicit_preferences(user)
        latest_event = quality["latest_event"]
        return {
            "user_id": user.user_id,
            "explicit_preferences": explicit_preferences,
            "user_description": explicit_preferences,
            "offline_features": self._offline_feature_payload(
                offline=offline,
                evidence=evidence,
                previous_profile=previous_profile,
            ),
            "realtime_features": self._realtime_feature_payload(
                online=online,
                evidence=evidence,
                latest_event=latest_event,
            ),
            "rfe_activity": activity,
            "short_term_topic_evidence": evidence["short_topics"],
            "long_term_topic_evidence": evidence["long_topics"],
            "negative_topic_evidence": evidence["negative_topics"],
            "negative_difficulty_evidence": evidence["negative_difficulties"],
            "negative_document_ids": evidence["negative_document_ids"],
            "content_type_evidence": evidence["content_types"],
            "difficulty_evidence": evidence["difficulties"],
            "reading_length_evidence": evidence["reading_lengths"],
            "author_evidence": evidence["authors"],
            "search_queries": evidence["searches"],
            "activity": activity,
            "data_quality": self._data_quality_payload(quality),
            "confidence_inputs": quality["confidence_inputs"],
            "latest_event_at": (
                latest_event.occurred_at.isoformat() if latest_event else None
            ),
            "offline_profile_at": offline.get("updated_at"),
            "realtime_event_count": evidence["recent_event_count"],
            "as_of": as_of.isoformat(),
            "_daily_buckets": [],
            "_search_events": [],
            "_latest_processed_event_id": (
                latest_event.event_id if latest_event else None
            ),
        }

    @staticmethod
    def _explicit_preferences(user: UserBaseProfile) -> dict[str, Any]:
        return {
            "topics": user.topics,
            "blocked_topics": user.blocked_topics,
            "preferred_content_types": user.preferred_content_types,
            "preferred_difficulty": user.preferred_difficulty,
            "preferred_reading_length": user.preferred_reading_length,
            "followed_author_ids": user.followed_author_ids,
            "blocked_author_ids": user.blocked_author_ids,
            "created_at": user.created_at.isoformat(),
        }

    def _offline_feature_payload(
        self,
        *,
        offline: dict[str, Any],
        evidence: dict[str, Any],
        previous_profile: dict[str, Any] | None,
    ) -> dict[str, Any]:
        return {
            "updated_at": offline.get("updated_at"),
            "topic_scores": offline.get("topic_scores", {}),
            "content_type_scores": offline.get("content_type_scores", {}),
            "difficulty_scores": offline.get("difficulty_scores", {}),
            "reading_length_scores": offline.get(
                "reading_length_scores",
                {},
            ),
            "author_scores": offline.get("author_scores", {}),
            "negative_topic_scores": offline.get("negative_topic_scores", {}),
            "negative_difficulty_scores": offline.get(
                "negative_difficulty_scores",
                {},
            ),
            "negative_document_ids": offline.get("negative_document_ids", []),
            "long_term_topic_evidence": evidence["long_topics"],
            "content_type_evidence": evidence["content_types"],
            "difficulty_evidence": evidence["difficulties"],
            "reading_length_evidence": evidence["reading_lengths"],
            "author_evidence": evidence["authors"],
            "previous_profile": self._offline_feature_snapshot(previous_profile),
        }

    def _realtime_feature_payload(
        self,
        *,
        online: dict[str, Any],
        evidence: dict[str, Any],
        latest_event: BehaviorEvent | None,
    ) -> dict[str, Any]:
        updated_at = online.get("updated_at")
        return {
            "window_days": self.ONLINE_WINDOW_DAYS,
            "updated_at": updated_at,
            "topic_scores": online.get("topic_scores", {}),
            "short_term_topic_evidence": evidence["short_topics"],
            "content_type_evidence": self._tag_score_rows(
                online.get("content_type_scores"),
                "value",
                updated_at=updated_at,
            ),
            "difficulty_evidence": self._tag_score_rows(
                online.get("difficulty_scores"),
                "value",
                updated_at=updated_at,
            ),
            "reading_length_evidence": self._tag_score_rows(
                online.get("reading_length_scores"),
                "value",
                updated_at=updated_at,
            ),
            "author_evidence": self._tag_score_rows(
                online.get("author_scores"),
                "author_id",
                updated_at=updated_at,
            ),
            "negative_topic_evidence": self._tag_score_rows(
                online.get("negative_topic_scores"),
                "topic",
                updated_at=updated_at,
                ascending=True,
            ),
            "negative_difficulty_evidence": self._tag_score_rows(
                online.get("negative_difficulty_scores"),
                "value",
                updated_at=updated_at,
                ascending=True,
            ),
            "negative_document_ids": online.get("negative_document_ids", []),
            "search_queries": online.get("search_queries", []),
            "event_type_counts_7d": evidence["event_type_counts_7d"],
            "latest_event_at": (
                latest_event.occurred_at.isoformat() if latest_event else None
            ),
            "realtime_event_count": evidence["recent_event_count"],
        }

    @staticmethod
    def _data_quality_payload(quality: dict[str, Any]) -> dict[str, Any]:
        return {
            "valid_event_count": quality["valid_event_count"],
            "invalid_event_count": quality["invalid_event_count"],
            "invalid_reasons": {},
            "capped_event_count": 0,
            "metadata_completeness": round(
                quality["metadata_completeness"],
                4,
            ),
            "strong_signal_count": quality["strong_signal_count"],
            "topic_metadata_ratio": quality["topic_metadata_ratio"],
            "relationship_warnings": [],
        }

    async def compact_user_features(
        self,
        user_id: str,
        *,
        as_of: datetime | str | None = None,
    ) -> dict[str, Any]:
        """兼容入口：重算一个用户的在线、离线标签和 REF 统计。"""

        user_id = _require_string_id(user_id)
        as_of_dt = self._coerce_datetime(as_of) if as_of else self._now()
        self._local_redis_users[user_id] = self._build_local_redis_user_state(
            user_id=user_id,
            as_of=as_of_dt,
        )
        await self.invalidate_cached_profile(user_id)
        return dict(self._local_redis_users[user_id]["offline_tags"])

    @staticmethod
    def _offline_feature_snapshot(
        profile: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """从上一份画像提取可用于 LLM 上下文的确定性字段。

        扩展主题和自由文本模型结论不会被回收为离线事实，避免一次推测结果在下一次
        请求中成为自我强化证据。
        """
        if not profile:
            return {"available": False}

        behavior = profile.get("behavior_profile")
        if not isinstance(behavior, dict):
            behavior = {}
        return {
            "available": True,
            "generated_at": profile.get("generated_at"),
            "profile_version": profile.get("profile_version"),
            "profile_confidence": profile.get("profile_confidence"),
            "long_term_interests": behavior.get("long_term_interests", []),
            "content_type_preferences": behavior.get("content_type_preferences", []),
            "difficulty_preferences": behavior.get("difficulty_preferences", []),
            "negative_difficulty_preferences": behavior.get(
                "negative_difficulty_preferences",
                [],
            ),
            "negative_document_ids": behavior.get("negative_document_ids", []),
            "reading_length_preferences": behavior.get(
                "reading_length_preferences",
                [],
            ),
            "author_affinities": behavior.get("author_affinities", []),
            "activity": behavior.get("activity", {}),
        }

    def _now(self) -> datetime:
        return _require_timezone(self._clock())

    @staticmethod
    def _coerce_datetime(value: datetime | str) -> datetime:
        if isinstance(value, datetime):
            return _require_timezone(value)
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return _require_timezone(parsed)


__all__ = []
