"""Feature Store 的 SQLite、Redis 和画像缓存仓储原语。"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import Any

import structlog

from app.infrastructure.database.json.feature_store_models import (
    BehaviorEvent,
    _require_string_id,
    _require_timezone,
)


logger = structlog.get_logger()


class _FeatureStoreRepositoryMixin:
    """依赖 FeatureStore 门面状态的本地与可选 Redis 仓储实现。"""

    def reload_mock_data(self) -> None:
        """兼容入口：从两个 SQLite 事实库重新加载进程内快照。"""

        self._users.clear()
        self._document_profiles.clear()
        self._events_by_user.clear()
        self._local_redis_users.clear()
        self._profile_cache.clear()
        self._profile_history.clear()
        self._load_errors.clear()

        users = self._user_repository.list_users()
        events = self._user_repository.list_all_events()
        self._users.update({user.user_id: user for user in users})
        document_ids: list[str] = []
        for event in events:
            self._events_by_user[event.user_id].append(event)
            document_ids.append(event.document_id)
            selected_id = event.metadata.get("selected_result_document_id")
            if isinstance(selected_id, str) and selected_id.strip():
                document_ids.append(selected_id.strip())
        self._document_profiles.update(
            self._document_repository.get_document_facts(document_ids)
        )

        logger.info(
            "Feature Store SQLite 事实加载完成",
            user_count=len(self._users),
            document_profile_count=len(self._document_profiles),
            event_count=sum(len(items) for items in self._events_by_user.values()),
            invalid_record_count=0,
        )

    async def get_user(self, user_id: str) -> dict[str, Any] | None:
        """返回一份可 JSON 序列化的用户基础资料。"""
        user_id = _require_string_id(user_id)
        if self.redis:
            raw = await self.redis.get(f"article-profile:user:{user_id}")
            if raw:
                return self._decode_json(raw)
        user = await asyncio.to_thread(self._user_repository.get_user, user_id)
        if user is not None:
            self._users[user_id] = user
        return user.model_dump(mode="json") if user else None

    async def get_recent_events(
        self,
        user_id: str,
        since: datetime,
        limit: int = 50,
        until: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """按时间正序返回 ``since`` 到 ``until`` 范围内的事件。"""
        user_id = _require_string_id(user_id)
        since = _require_timezone(since)
        until = _require_timezone(until or self._now())

        events: list[BehaviorEvent]
        if self.redis:
            key = f"article-profile:events:{user_id}"
            raw_events = await self.redis.zrangebyscore(
                key, since.timestamp(), until.timestamp()
            )
            events = []
            for raw in raw_events:
                try:
                    events.append(BehaviorEvent.model_validate(self._decode_json(raw)))
                except Exception as exc:
                    logger.warning(
                        "Feature Store 拒绝非法 Redis 行为事件", reason=str(exc)
                    )
        else:
            events = list(
                await asyncio.to_thread(
                    self._user_repository.list_events,
                    user_id,
                    since=since,
                    until=until,
                )
            )
            self._events_by_user[user_id] = list(events)

        events.sort(key=lambda item: (item.occurred_at, item.event_id))
        if limit > 0:
            events = events[-limit:]
        return [event.model_dump(mode="json") for event in events]

    async def get_cached_profile(self, user_id: str) -> dict[str, Any] | None:
        user_id = _require_string_id(user_id)
        if self.redis:
            raw = await self.redis.get(f"article-profile:profile-cache:{user_id}:v2")
            if raw:
                return self._decode_json(raw)
        return self._profile_cache.get(user_id)

    async def set_cached_profile(self, user_id: str, profile: Any) -> None:
        user_id = _require_string_id(user_id)
        if hasattr(profile, "model_dump"):
            payload = profile.model_dump(mode="json")
        elif isinstance(profile, dict):
            payload = profile
        else:
            raise TypeError("profile 必须是 Pydantic 模型或字典")
        self._profile_cache[user_id] = payload
        if self.redis:
            await self.redis.set(
                f"article-profile:profile-cache:{user_id}:v2",
                json.dumps(payload, ensure_ascii=False),
                ex=self.ttl,
            )

    async def invalidate_cached_profile(self, user_id: str) -> None:
        user_id = _require_string_id(user_id)
        self._profile_cache.pop(user_id, None)
        if self.redis:
            await self.redis.delete(f"article-profile:profile-cache:{user_id}:v2")

    async def get_profile_history(self, user_id: str) -> list[dict[str, Any]]:
        """按生成时间正序返回有限条历史画像快照。"""

        user_id = _require_string_id(user_id)
        if self.redis:
            raw = await self.redis.get(f"article-profile:profile-history:{user_id}:v2")
            if raw:
                payload = self._decode_json(raw)
                profiles = payload.get("profiles", [])
                if isinstance(profiles, list):
                    return [dict(item) for item in profiles if isinstance(item, dict)]
        return [dict(item) for item in self._profile_history.get(user_id, [])]

    async def get_latest_historical_profile(
        self, user_id: str
    ) -> dict[str, Any] | None:
        """返回最近一份历史画像，供受限离线上下文使用。"""

        history = await self.get_profile_history(user_id)
        return history[-1] if history else None

    async def archive_profile(self, user_id: str, profile: Any) -> None:
        """保存过期画像，并限制单用户历史快照数量。"""

        user_id = _require_string_id(user_id)
        if hasattr(profile, "model_dump"):
            payload = profile.model_dump(mode="json")
        elif isinstance(profile, dict):
            payload = dict(profile)
        else:
            raise TypeError("profile 必须是 Pydantic 模型或字典")
        history = await self.get_profile_history(user_id)
        generated_at = payload.get("generated_at")
        history = [
            item for item in history if item.get("generated_at") != generated_at
        ]
        history.append(payload)
        history = history[-self.PROFILE_HISTORY_LIMIT :]
        self._profile_history[user_id] = history
        if self.redis:
            await self.redis.set(
                f"article-profile:profile-history:{user_id}:v2",
                json.dumps({"profiles": history}, ensure_ascii=False),
            )

    @staticmethod
    def _decode_json(raw: Any) -> dict[str, Any]:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        if isinstance(raw, str):
            decoded = json.loads(raw)
        else:
            decoded = raw
        if not isinstance(decoded, dict):
            raise ValueError("存储的 JSON 值必须是对象")
        return decoded



__all__ = []
