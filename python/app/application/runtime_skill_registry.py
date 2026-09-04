"""管理运行时 Skill 的请求级 Snapshot 和原子 reload。"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from threading import Lock
from typing import Protocol

from app.models.runtime_skill import (
    CompiledRuntimeSkill,
    RuntimeSkillReloadResult,
    RuntimeSkillSnapshot,
)


logger = logging.getLogger(__name__)


class RuntimeSkillCatalog(Protocol):
    """Registry 依赖的全量 Skill Catalog 边界。"""

    def load(self) -> Sequence[CompiledRuntimeSkill]:
        """返回一批已完整校验的编译 Skill。"""

        ...


class RuntimeSkillRegistry:
    """让每次请求固定使用同一代，并在失败 reload 时保留旧代。"""

    def __init__(
        self,
        *,
        catalog: RuntimeSkillCatalog,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._catalog = catalog
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._reload_lock = Lock()
        self._snapshot_lock = Lock()
        self._snapshot = RuntimeSkillSnapshot.build(
            generation=0,
            loaded_at=self._clock(),
            skills=(),
        )

    def capture_snapshot(self) -> RuntimeSkillSnapshot:
        """原子读取当前不可变 Snapshot 引用。"""

        return self._snapshot

    def reload(self) -> RuntimeSkillReloadResult:
        """全量构建候选 Snapshot，成功后在锁内单次替换。"""

        with self._reload_lock:
            previous = self._snapshot
            try:
                skills = tuple(self._catalog.load())
                candidate = RuntimeSkillSnapshot.build(
                    generation=previous.generation + 1,
                    loaded_at=self._clock(),
                    skills=skills,
                )
            except OSError as exc:
                logger.warning(
                    "运行时 Skill reload 发生 I/O 失败，保留旧 Snapshot",
                    extra={"exception_type": type(exc).__name__},
                )
                return RuntimeSkillReloadResult(
                    reloaded=False,
                    snapshot=previous,
                    error_code="catalog_io",
                )
            except ValueError as exc:
                logger.warning(
                    "运行时 Skill reload 校验失败，保留旧 Snapshot",
                    extra={"exception_type": type(exc).__name__},
                )
                return RuntimeSkillReloadResult(
                    reloaded=False,
                    snapshot=previous,
                    error_code="catalog_invalid",
                )
            with self._snapshot_lock:
                self._snapshot = candidate
            return RuntimeSkillReloadResult(
                reloaded=True,
                snapshot=candidate,
            )


__all__ = ["RuntimeSkillCatalog", "RuntimeSkillRegistry"]
