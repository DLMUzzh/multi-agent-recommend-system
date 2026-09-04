"""Feature Store 的行为有效性、标签和 REF 确定性计算。"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from datetime import date, datetime, timedelta
from typing import Any

from app.infrastructure.database.json.feature_store_models import (
    BASE_EVENT_WEIGHTS,
    EFFECTIVE_EVENT_TYPES,
    STRONG_EVENT_TYPES,
    BehaviorEvent,
    UserBaseProfile,
    UserNotFoundError,
)
from app.models.document import DocumentFact


class _FeatureStoreTagsMixin:
    """依赖 FeatureStore 门面状态的标签与活跃度计算实现。"""

    def _validated_user_events(
        self,
        *,
        user: UserBaseProfile,
        as_of: datetime,
    ) -> list[BehaviorEvent]:
        """返回可参与标签和 REF 的有效事件。"""

        seen_event_ids: set[str] = set()
        last_duplicate_key_at: dict[tuple[str, str, str], datetime] = {}
        valid_events: list[BehaviorEvent] = []
        for event in sorted(
            self._events_by_user.get(user.user_id, []),
            key=lambda item: (item.occurred_at, item.event_id),
        ):
            document_profile = self._document_profiles.get(event.document_id)
            invalid_reason = self._event_invalid_reason(
                event=event,
                document_profile=document_profile,
                user=user,
                as_of=as_of,
                seen_event_ids=seen_event_ids,
                last_duplicate_key_at=last_duplicate_key_at,
            )
            if invalid_reason:
                continue
            seen_event_ids.add(event.event_id)
            last_duplicate_key_at[
                (event.user_id, event.document_id, event.event_type)
            ] = event.occurred_at
            valid_events.append(event)
        return valid_events

    def _recent_behavior_payload(self, event: BehaviorEvent) -> dict[str, Any]:
        """去除父级已经表达的用户和行为类型，保留行为事实。"""

        document_profile = self._document_profiles.get(event.document_id)
        return {
            "event_id": event.event_id,
            "document_id": event.document_id,
            "author_id": (
                document_profile.author_id if document_profile else None
            ),
            "occurred_at": event.occurred_at.isoformat(),
            "metadata": event.metadata,
        }

    def _build_ref_activity_state(
        self,
        *,
        events: list[BehaviorEvent],
        as_of: datetime,
    ) -> dict[str, Any]:
        """构建计算最近三十天 REF 所需的每日充分统计。"""

        cutoff_date = (as_of - timedelta(days=self.REF_WINDOW_DAYS)).date()
        daily: dict[str, dict[str, Any]] = {}
        last_active_at = max(
            (
                event.occurred_at
                for event in events
                if event.event_type in EFFECTIVE_EVENT_TYPES
            ),
            default=None,
        )
        for event in events:
            if event.occurred_at.date() < cutoff_date:
                continue
            day = event.occurred_at.date().isoformat()
            bucket = daily.setdefault(
                day,
                {
                    "behavior_count": 0,
                    "effective_behavior_count": 0,
                    "read_count": 0,
                    "read_quality_sum": 0.0,
                    "strong_behavior_count": 0,
                    "last_active_at": None,
                    "topics": set(),
                },
            )
            bucket["behavior_count"] += 1
            if event.event_type in EFFECTIVE_EVENT_TYPES:
                bucket["effective_behavior_count"] += 1
                current_last = self._parse_optional_datetime(bucket["last_active_at"])
                if current_last is None or event.occurred_at > current_last:
                    bucket["last_active_at"] = event.occurred_at.isoformat()
                document_profile = self._document_profiles.get(event.document_id)
                if document_profile:
                    bucket["topics"].update(document_profile.topics)
            if event.event_type == "read":
                bucket["read_count"] += 1
                bucket["read_quality_sum"] += self._read_quality(event)
            if event.event_type in STRONG_EVENT_TYPES:
                bucket["strong_behavior_count"] += 1

        normalized_daily: dict[str, dict[str, Any]] = {}
        for day in sorted(daily):
            bucket = daily[day]
            normalized_daily[day] = {
                "behavior_count": int(bucket["behavior_count"]),
                "effective_behavior_count": int(
                    bucket["effective_behavior_count"]
                ),
                "read_count": int(bucket["read_count"]),
                "read_quality_sum": round(float(bucket["read_quality_sum"]), 4),
                "strong_behavior_count": int(bucket["strong_behavior_count"]),
                "last_active_at": bucket["last_active_at"],
                "topics": sorted(bucket["topics"]),
            }
        return {
            "last_active_at": last_active_at.isoformat() if last_active_at else None,
            "daily": normalized_daily,
        }

    @staticmethod
    def _rounded_scores(scores: dict[str, float]) -> dict[str, float]:
        return {
            key: round(value, 4)
            for key, value in sorted(
                scores.items(),
                key=lambda item: (-abs(item[1]), item[0]),
            )
            if abs(value) >= 0.0001
        }

    def _calculate_tag_state(
        self,
        *,
        user: UserBaseProfile,
        events: list[BehaviorEvent],
        as_of: datetime,
        apply_day_weights: bool,
    ) -> dict[str, Any]:
        """把一组行为聚合为可与用户描述对应的标签分数。"""

        topic_scores: dict[str, float] = defaultdict(float)
        content_type_scores: dict[str, float] = defaultdict(float)
        difficulty_scores: dict[str, float] = defaultdict(float)
        reading_length_scores: dict[str, float] = defaultdict(float)
        author_scores: dict[str, float] = defaultdict(float)
        negative_topic_scores: dict[str, float] = defaultdict(float)
        negative_difficulty_scores: dict[str, float] = defaultdict(float)
        negative_document_ids: set[str] = set()
        search_queries: list[tuple[datetime, str]] = []

        for event in events:
            document_profile = self._document_profiles.get(event.document_id)
            age_days = max((as_of.date() - event.occurred_at.date()).days, 0)
            day_weight = (
                self.ONLINE_DAY_WEIGHTS[age_days]
                if apply_day_weights and age_days < self.ONLINE_WINDOW_DAYS
                else 1.0
            )
            contribution = (
                BASE_EVENT_WEIGHTS[event.event_type]
                * self._read_quality(event)
                * day_weight
            )
            interest_factor = 0.5 if event.event_type == "comment" else 1.0

            if event.event_type in {"follow_author", "unfollow_author"}:
                if document_profile:
                    author_scores[document_profile.author_id] += contribution
                continue

            if event.event_type == "not_interested":
                target_type = str(event.metadata.get("target_type", "article"))
                if target_type == "author":
                    if document_profile:
                        author_scores[document_profile.author_id] += contribution
                    continue
                if target_type == "article":
                    negative_document_ids.add(event.document_id)
                    continue
                if target_type == "difficulty":
                    raw_values = event.metadata.get("target_values", [])
                    values = (
                        raw_values
                        if isinstance(raw_values, list)
                        else []
                    )
                    for value in values:
                        if isinstance(value, str) and value.strip():
                            negative_difficulty_scores[value.strip()] += contribution
                    continue
                if not document_profile:
                    continue
                target_value = event.metadata.get("target_value")
                if target_type == "topic" and isinstance(target_value, str):
                    topics = [target_value.strip()] if target_value.strip() else []
                else:
                    topics = document_profile.topics
                for topic in topics:
                    negative_topic_scores[topic] += contribution / max(len(topics), 1)
                continue

            if not document_profile:
                continue
            evidence_profile = document_profile
            if event.event_type == "search":
                query = str(event.metadata.get("query", "")).strip()
                if query:
                    search_queries.append((event.occurred_at, query))
                selected_id = event.metadata.get("selected_result_document_id")
                if (
                    isinstance(selected_id, str)
                    and selected_id in self._document_profiles
                ):
                    evidence_profile = self._document_profiles[selected_id]

            adjusted = contribution * interest_factor
            for topic in evidence_profile.topics:
                topic_scores[topic] += adjusted / max(
                    len(evidence_profile.topics), 1
                )
            content_type_scores[evidence_profile.content_type] += adjusted
            difficulty_scores[evidence_profile.difficulty] += adjusted
            reading_length_scores[
                self._reading_length_level(evidence_profile.total_token_count)
            ] += adjusted
            author_scores[evidence_profile.author_id] += 0.5 * adjusted

        return {
            "topic_scores": self._rounded_scores(topic_scores),
            "content_type_scores": self._rounded_scores(content_type_scores),
            "difficulty_scores": self._rounded_scores(difficulty_scores),
            "reading_length_scores": self._rounded_scores(
                reading_length_scores
            ),
            "author_scores": self._rounded_scores(author_scores),
            "negative_topic_scores": self._rounded_scores(negative_topic_scores),
            "negative_difficulty_scores": self._rounded_scores(
                negative_difficulty_scores
            ),
            "negative_document_ids": sorted(negative_document_ids)[:50],
            "search_queries": self._latest_unique_queries(search_queries),
            "updated_at": as_of.isoformat(),
        }

    def _build_local_redis_user_state(
        self,
        *,
        user_id: str,
        as_of: datetime,
    ) -> dict[str, Any]:
        """从永久行为数据库重建一个用户的模拟 Redis 状态。"""

        user = self._users.get(user_id)
        if user is None:
            raise UserNotFoundError(f"用户不存在：{user_id}")
        events = self._validated_user_events(user=user, as_of=as_of)
        recent_events: list[BehaviorEvent] = []
        offline_events: list[BehaviorEvent] = []
        recent_behaviors: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for event in events:
            age_days = (as_of.date() - event.occurred_at.date()).days
            if 0 <= age_days < self.ONLINE_WINDOW_DAYS:
                recent_events.append(event)
                recent_behaviors[event.event_type].append(
                    self._recent_behavior_payload(event)
                )
            elif age_days >= self.ONLINE_WINDOW_DAYS:
                offline_events.append(event)

        normalized_recent = {
            event_type: sorted(
                items,
                key=lambda item: (item["occurred_at"], item["event_id"]),
            )
            for event_type, items in sorted(recent_behaviors.items())
        }
        return {
            "recent_behaviors": normalized_recent,
            "ref_activity": self._build_ref_activity_state(
                events=events,
                as_of=as_of,
            ),
            "online_tags": self._calculate_tag_state(
                user=user,
                events=recent_events,
                as_of=as_of,
                apply_day_weights=True,
            ),
            "offline_tags": self._calculate_tag_state(
                user=user,
                events=offline_events,
                as_of=as_of,
                apply_day_weights=False,
            ),
        }

    async def refresh_daily_tags(
        self,
        *,
        as_of: datetime | str | None = None,
    ) -> dict[str, dict[str, Any]]:
        """每日重算全部用户的七天在线标签、离线标签和 REF 统计。"""

        as_of_dt = self._coerce_datetime(as_of) if as_of else self._now()
        self._local_redis_users = {
            user_id: self._build_local_redis_user_state(
                user_id=user_id,
                as_of=as_of_dt,
            )
            for user_id in sorted(self._users)
        }
        for user_id in self._local_redis_users:
            await self.invalidate_cached_profile(user_id)
        return json.loads(json.dumps(self._local_redis_users, ensure_ascii=False))

    # ---------- 确定性特征计算 ----------

    def _local_redis_state_for(
        self,
        *,
        user_id: str,
        as_of: datetime,
    ) -> dict[str, Any]:
        """读取同日模拟 Redis 状态，缺失或跨日时临时从事实库重建。"""

        state = self._local_redis_users.get(user_id)
        updated_at = None
        if state:
            updated_at = self._parse_optional_datetime(
                state.get("online_tags", {}).get("updated_at")
            )
        if updated_at and updated_at.date() == as_of.date():
            return state
        return self._build_local_redis_user_state(user_id=user_id, as_of=as_of)

    def _calculate_ref_activity_from_state(
        self,
        state: dict[str, Any],
        as_of: datetime,
    ) -> dict[str, Any]:
        """访问时从 Redis 每日统计计算 Recency、Frequency 和 Engagement。"""

        ref_state = state.get("ref_activity", {})
        cutoff_date = (as_of - timedelta(days=self.REF_WINDOW_DAYS)).date()
        daily = [
            bucket
            for day, bucket in ref_state.get("daily", {}).items()
            if cutoff_date <= date.fromisoformat(day) <= as_of.date()
        ]
        last_active_at = self._parse_optional_datetime(ref_state.get("last_active_at"))
        if last_active_at and last_active_at <= as_of:
            age_days = max(
                (as_of - last_active_at).total_seconds() / 86400.0,
                0.0,
            )
            recency = self._time_decay(age_days, 14.0)
        else:
            recency = 0.0

        active_days = sum(
            int(bucket.get("effective_behavior_count", 0)) > 0 for bucket in daily
        )
        read_count = sum(int(bucket.get("read_count", 0)) for bucket in daily)
        frequency = 0.5 * min(active_days / 10.0, 1.0) + 0.5 * min(
            read_count / 20.0,
            1.0,
        )
        read_quality_sum = sum(
            float(bucket.get("read_quality_sum", 0.0)) for bucket in daily
        )
        average_read_quality = (
            read_quality_sum / read_count if read_count else 0.0
        )
        strong_count = sum(
            int(bucket.get("strong_behavior_count", 0)) for bucket in daily
        )
        engagement = 0.6 * average_read_quality + 0.4 * min(
            strong_count / 5.0,
            1.0,
        )
        valid_event_count = sum(
            int(bucket.get("behavior_count", 0)) for bucket in daily
        )
        distinct_topics = {
            str(topic)
            for bucket in daily
            for topic in bucket.get("topics", [])
            if str(topic).strip()
        }
        if valid_event_count < 5:
            level = "new_user"
        elif recency < 0.25 and valid_event_count >= 10:
            level = "churn_risk"
        elif engagement >= 0.75 and read_count >= 3 and strong_count >= 3:
            level = "deep_reader"
        elif len(distinct_topics) >= 8 and frequency >= 0.2:
            level = "explorer"
        elif recency >= 0.5 and frequency >= 0.25:
            level = "active_reader"
        else:
            level = "casual_reader"
        return {
            "recency_score": round(recency, 4),
            "frequency_score": round(frequency, 4),
            "engagement_score": round(engagement, 4),
            "level": level,
            "active_days_30d": active_days,
            "effective_read_count_30d": read_count,
            "strong_interaction_count_30d": strong_count,
            "average_read_quality": round(average_read_quality, 4),
            "distinct_topic_count_30d": len(distinct_topics),
        }

    @staticmethod
    def _merge_score_maps(*maps: Any) -> dict[str, float]:
        merged: dict[str, float] = defaultdict(float)
        for score_map in maps:
            if not isinstance(score_map, dict):
                continue
            for key, value in score_map.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    merged[str(key)] += float(value)
        return _FeatureStoreTagsMixin._rounded_scores(merged)

    @staticmethod
    def _tag_score_rows(
        scores: Any,
        key_name: str,
        *,
        updated_at: str | None,
        ascending: bool = False,
    ) -> list[dict[str, Any]]:
        if not isinstance(scores, dict):
            return []
        rows = [
            {
                key_name: str(key),
                "raw_score": round(float(value), 4),
                "event_count": 1,
                "last_interaction_at": updated_at,
            }
            for key, value in scores.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        ]
        rows.sort(key=lambda item: item["raw_score"], reverse=not ascending)
        return rows[:50]

    def _tag_author_rows(
        self,
        *,
        user: UserBaseProfile,
        scores: dict[str, float],
    ) -> list[dict[str, Any]]:
        author_ids = (
            set(scores)
            | set(user.followed_author_ids)
            | set(user.blocked_author_ids)
        )
        rows = [
            {
                "author_id": author_id,
                "raw_score": round(float(scores.get(author_id, 0.0)), 4),
                "event_count": 1 if author_id in scores else 0,
                "last_interaction_at": None,
                "followed": author_id in user.followed_author_ids,
                "blocked": author_id in user.blocked_author_ids,
            }
            for author_id in author_ids
        ]
        rows.sort(key=lambda item: (-item["raw_score"], item["author_id"]))
        return rows[:50]

    @staticmethod
    def _parse_optional_datetime(value: datetime | str | None) -> datetime | None:
        if value is None or isinstance(value, datetime):
            return value
        return datetime.fromisoformat(value.replace("Z", "+00:00"))

    def _event_invalid_reason(
        self,
        *,
        event: BehaviorEvent,
        document_profile: DocumentFact | None,
        user: UserBaseProfile,
        as_of: datetime,
        seen_event_ids: set[str],
        last_duplicate_key_at: dict[tuple[str, str, str], datetime],
    ) -> str | None:
        if event.event_id in seen_event_ids:
            return "duplicate_event_id"
        if event.user_id != user.user_id:
            return "user_mismatch"
        if event.occurred_at > as_of:
            return "future_event"
        if document_profile is None:
            return "document_not_found"

        duplicate_key = (event.user_id, event.document_id, event.event_type)
        previous_at = last_duplicate_key_at.get(duplicate_key)
        if (
            previous_at
            and (event.occurred_at - previous_at).total_seconds()
            <= self.NEAR_DUPLICATE_SECONDS
        ):
            return "near_duplicate"

        if event.event_type == "read":
            dwell_seconds = event.metadata.get("dwell_seconds")
            if not isinstance(dwell_seconds, (int, float)) or dwell_seconds <= 0:
                return "invalid_read_dwell_seconds"
            completion_rate = event.metadata.get("completion_rate")
            if completion_rate is not None and (
                not isinstance(completion_rate, (int, float))
                or not 0.0 <= float(completion_rate) <= 1.0
            ):
                return "invalid_completion_rate"
        if event.event_type == "search":
            query = event.metadata.get("query")
            if not isinstance(query, str) or not query.strip():
                return "empty_search_query"
        return None

    @staticmethod
    def _reading_length_level(total_token_count: int) -> str:
        """按批准边界把文档总 token 数映射为阅读长度。"""

        if total_token_count <= 800:
            return "short"
        if total_token_count <= 3000:
            return "medium"
        return "long"

    @staticmethod
    def _read_quality(event: BehaviorEvent) -> float:
        if event.event_type != "read":
            return 1.0
        dwell_seconds = max(float(event.metadata.get("dwell_seconds", 0.0)), 0.0)
        completion_rate = max(
            0.0, min(float(event.metadata.get("completion_rate", 0.0)), 1.0)
        )
        dwell_factor = min(dwell_seconds / 300.0, 1.0)
        return 0.5 + 0.3 * dwell_factor + 0.2 * completion_rate


    @staticmethod
    def _latest_unique_queries(items: list[tuple[datetime, str]]) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for _, query in sorted(items, key=lambda item: item[0], reverse=True):
            normalized = query.strip()
            if normalized and normalized not in seen:
                result.append(normalized)
                seen.add(normalized)
            if len(result) >= 10:
                break
        return result

    @staticmethod
    def _time_decay(age_days: float, half_life_days: float) -> float:
        return math.pow(0.5, age_days / half_life_days)

    @staticmethod
    def normalize_score(raw_score: float) -> float:
        """在不改变顺序的前提下，把累计分限制到 -1..1。"""
        return raw_score / (abs(raw_score) + 5.0)

__all__ = []
