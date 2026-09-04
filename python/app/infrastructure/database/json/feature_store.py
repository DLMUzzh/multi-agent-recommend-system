"""文章画像使用的 SQLite 事实、可选 Redis 与确定性特征计算门面。

公开服务保留状态与调用顺序；事实读取、标签/REF 和特征快照分别由职责模块承接。
"""

from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.config.paths import DATA_ROOT
from app.infrastructure.database.json.feature_store_features import (
    _FeatureStoreFeaturesMixin,
)
from app.infrastructure.database.json.feature_store_models import (
    ALLOWED_EVENT_TYPES,
    BehaviorEvent,
    UserBaseProfile,
    UserNotFoundError,
    _require_string_id,
)
from app.infrastructure.database.json.feature_store_repository import (
    _FeatureStoreRepositoryMixin,
)
from app.infrastructure.database.json.feature_store_tags import (
    _FeatureStoreTagsMixin,
)
from app.infrastructure.database.sqlite.document_repository import (
    SQLiteDocumentRepository,
)
from app.infrastructure.database.sqlite.user_profile_repository import (
    SQLiteUserProfileRepository,
)
from app.models.document import DocumentFact


class FeatureStore(
    _FeatureStoreRepositoryMixin,
    _FeatureStoreTagsMixin,
    _FeatureStoreFeaturesMixin,
):
    """管理 SQLite 事实、可选 Redis、确定性标签和画像缓存的数据服务。

    参数：
        redis_client：可选异步 Redis 客户端；未注入时使用进程内缓存。
        ttl：远程画像缓存的过期秒数。
        data_dir：未注入仓储时，两个 SQLite 数据库所在目录。
        auto_load_mock：兼容参数；为真时在构造阶段从 SQLite 加载事实。
        clock：可注入的带时区时钟，用于确定性测试。
    """

    ONLINE_WINDOW_DAYS = 7
    REF_WINDOW_DAYS = 30
    ONLINE_DAY_WEIGHTS = (1.0, 0.85, 0.70, 0.55, 0.40, 0.25, 0.10)
    NEAR_DUPLICATE_SECONDS = 2.0
    PROFILE_HISTORY_LIMIT = 10

    def __init__(
        self,
        redis_client: Any = None,
        ttl: int = 86400,
        data_dir: str | Path | None = None,
        auto_load_mock: bool = True,
        clock: Callable[[], datetime] | None = None,
        user_repository: SQLiteUserProfileRepository | None = None,
        document_repository: SQLiteDocumentRepository | None = None,
    ):
        self.redis = redis_client
        self.ttl = ttl
        self.data_dir = Path(data_dir) if data_dir else DATA_ROOT
        self._user_repository = user_repository or SQLiteUserProfileRepository(
            self.data_dir / "user_profiles.sqlite3"
        )
        self._document_repository = (
            document_repository
            or SQLiteDocumentRepository(self.data_dir / "documents.sqlite3")
        )
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._users: dict[str, UserBaseProfile] = {}
        self._document_profiles: dict[str, DocumentFact] = {}
        self._events_by_user: dict[str, list[BehaviorEvent]] = defaultdict(list)
        self._local_redis_users: dict[str, dict[str, Any]] = {}
        self._profile_cache: dict[str, dict[str, Any]] = {}
        self._profile_history: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._load_errors: list[dict[str, str]] = []
        if auto_load_mock:
            self.reload_mock_data()

    async def record_behavior(
        self,
        user_id: str,
        behavior_type: str,
        item_id: str,
        metadata: dict[str, Any] | None = None,
        *,
        occurred_at: datetime | None = None,
        event_id: str | None = None,
    ) -> None:
        """持久化行为事实，并即时更新该用户的确定性缓存状态。"""
        if behavior_type not in ALLOWED_EVENT_TYPES:
            raise ValueError(f"不支持的 event_type：{behavior_type}")
        user_id = _require_string_id(user_id)
        if user_id not in self._users:
            raise UserNotFoundError(f"用户不存在：{user_id}")
        document_profile = self._document_profiles.get(item_id)
        if document_profile is None:
            facts = await asyncio.to_thread(
                self._document_repository.get_document_facts,
                (item_id,),
            )
            document_profile = facts.get(item_id)
            if document_profile is None:
                raise ValueError("行为必须引用 ready 文档")
            self._document_profiles[item_id] = document_profile
        event = BehaviorEvent(
            event_id=event_id or f"evt-{user_id}-{uuid4().hex}",
            user_id=user_id,
            event_type=behavior_type,
            occurred_at=occurred_at or self._now(),
            document_id=item_id,
            metadata=metadata or {},
        )
        if any(
            existing.event_id == event.event_id
            for events in self._events_by_user.values()
            for existing in events
        ):
            raise ValueError(f"event_id 重复：{event.event_id}")
        await asyncio.to_thread(self._user_repository.append_event, event)
        self._events_by_user[user_id].append(event)
        self._events_by_user[user_id].sort(
            key=lambda item: (item.occurred_at, item.event_id)
        )
        previous_offline_tags = self._local_redis_users.get(user_id, {}).get(
            "offline_tags"
        )
        refreshed_state = self._build_local_redis_user_state(
            user_id=user_id,
            as_of=self._now(),
        )
        if isinstance(previous_offline_tags, dict):
            refreshed_state["offline_tags"] = previous_offline_tags
        self._local_redis_users[user_id] = refreshed_state
        if self.redis:
            key = f"article-profile:events:{user_id}"
            payload = json.dumps(event.model_dump(mode="json"), ensure_ascii=False)
            await self.redis.zadd(key, {payload: event.occurred_at.timestamp()})
            await self.redis.expire(key, self.ttl)
        await self.invalidate_cached_profile(user_id)
