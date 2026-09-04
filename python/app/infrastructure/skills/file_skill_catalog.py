"""从固定根目录安全加载运行时 Skill Manifest。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic import ValidationError

from app.models.runtime_skill import (
    CompiledRuntimeSkill,
    RuntimeSkillManifest,
)


class FileSkillCatalog:
    """全量校验受控目录，成功后返回编译 Skill 快照素材。"""

    _MAX_SKILLS = 100
    _MAX_MANIFEST_BYTES = 64 * 1024
    _MANIFEST_NAME = "manifest.json"

    def __init__(self, *, root: Path) -> None:
        self._root = Path(root)

    def load(self) -> tuple[CompiledRuntimeSkill, ...]:
        """读取并编译整个 Catalog；任一非法项使整批失败。"""

        if not self._root.exists():
            return ()
        if self._root.is_symlink() or not self._root.is_dir():
            raise ValueError("运行时 Skill 根目录无效")
        root = self._root.resolve()
        entries = tuple(sorted(self._root.iterdir(), key=lambda path: path.name))
        if len(entries) > self._MAX_SKILLS:
            raise ValueError("运行时 Skill 数量超过上限")
        compiled: list[CompiledRuntimeSkill] = []
        seen_ids: set[str] = set()
        for entry in entries:
            skill = self._load_skill(entry, root=root)
            if skill.skill_id in seen_ids:
                raise ValueError("运行时 Skill ID 重复")
            compiled.append(skill)
            seen_ids.add(skill.skill_id)
        return tuple(compiled)

    def _load_skill(
        self,
        skill_dir: Path,
        *,
        root: Path,
    ) -> CompiledRuntimeSkill:
        """校验一个目录只能包含一个普通 Manifest 文件。"""

        if skill_dir.is_symlink() or not skill_dir.is_dir():
            raise ValueError("运行时 Skill 目录无效")
        resolved_dir = skill_dir.resolve()
        if resolved_dir.parent != root:
            raise ValueError("运行时 Skill 目录越出根目录")
        children = tuple(skill_dir.iterdir())
        if len(children) != 1 or children[0].name != self._MANIFEST_NAME:
            raise ValueError("运行时 Skill 目录只能包含 manifest.json")
        manifest_path = children[0]
        if manifest_path.is_symlink() or not manifest_path.is_file():
            raise ValueError("运行时 Skill Manifest 必须是普通文件")
        if manifest_path.resolve().parent != resolved_dir:
            raise ValueError("运行时 Skill Manifest 越出目录")
        try:
            if manifest_path.stat().st_size > self._MAX_MANIFEST_BYTES:
                raise ValueError("运行时 Skill Manifest 超过大小上限")
            raw_bytes = manifest_path.read_bytes()
            raw = json.loads(raw_bytes.decode("utf-8"))
        except ValueError:
            raise
        except (OSError, UnicodeError, json.JSONDecodeError):
            raise ValueError("运行时 Skill Manifest 无法安全读取") from None
        if not isinstance(raw, dict):
            raise ValueError("运行时 Skill Manifest 必须是 JSON 对象")
        try:
            manifest = RuntimeSkillManifest.model_validate(raw)
        except ValidationError:
            raise ValueError("运行时 Skill Manifest 未通过 Schema 校验") from None
        if manifest.skill_id != skill_dir.name:
            raise ValueError("运行时 Skill 目录名与 skill_id 不一致")
        expected_hash = self._content_hash(raw)
        if manifest.content_hash != expected_hash:
            raise ValueError("运行时 Skill Manifest Hash 不一致")
        return CompiledRuntimeSkill.from_manifest(manifest)

    @staticmethod
    def _content_hash(raw: dict[str, object]) -> str:
        """对移除 Hash 字段后的规范 JSON 计算 SHA-256。"""

        payload = {key: value for key, value in raw.items() if key != "content_hash"}
        canonical = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()


__all__ = ["FileSkillCatalog"]
