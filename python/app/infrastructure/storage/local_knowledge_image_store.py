"""本地知识图片的内容寻址存储。"""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
from collections.abc import Collection
from pathlib import Path

from pydantic import Field

from app.models.common import _StrictModel
from app.models.document import ImageMimeType


_STORAGE_KEY_PATTERN = re.compile(
    r"^(?P<prefix>[0-9a-f]{2})/"
    r"(?P<content_hash>[0-9a-f]{64})"
    r"(?P<extension>\.(?:png|jpg|webp|gif))$"
)
_MIME_EXTENSIONS: dict[str, str] = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
    "image/gif": ".gif",
}


class StoredKnowledgeImage(_StrictModel):
    """已通过校验并写入存储的图片元数据。"""

    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    storage_key: str = Field(
        pattern=r"^[0-9a-f]{2}/[0-9a-f]{64}\.(?:png|jpg|webp|gif)$"
    )
    mime_type: ImageMimeType
    byte_size: int = Field(ge=1, le=8 * 1024 * 1024)


class LocalKnowledgeImageStore:
    """把已验证图片原子写入固定根目录，供后续 OSS 适配替换。"""

    MAX_BYTES = 8 * 1024 * 1024

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.root.chmod(0o700)

    def put(self, content: bytes, mime_type: str) -> StoredKnowledgeImage:
        """校验图片内容，并按 SHA-256 使用原子替换写入。"""

        if not content:
            raise ValueError("图片内容不能为空")
        if len(content) > self.MAX_BYTES:
            raise ValueError("图片大小不能超过 8 MiB")

        extension = _MIME_EXTENSIONS.get(mime_type)
        if extension is None:
            raise ValueError("不支持的图片类型")
        if not self._matches_magic(content, mime_type):
            raise ValueError("图片类型与内容不一致")

        content_hash = hashlib.sha256(content).hexdigest()
        storage_key = f"{content_hash[:2]}/{content_hash}{extension}"
        target = self.resolve(storage_key)
        target.parent.mkdir(mode=0o700, parents=False, exist_ok=True)
        target.parent.chmod(0o700)

        if not target.exists():
            self._atomic_write(target, content)
        elif not target.is_file():
            raise ValueError("图片存储 Key 无效")

        return StoredKnowledgeImage(
            content_hash=content_hash,
            storage_key=storage_key,
            mime_type=mime_type,
            byte_size=len(content),
        )

    def resolve(self, storage_key: str) -> Path:
        """解析合法相对 Key，并保证最终路径仍位于存储根目录内。"""

        match = _STORAGE_KEY_PATTERN.fullmatch(storage_key)
        if match is None:
            raise ValueError("图片存储 Key 无效")
        if match.group("prefix") != match.group("content_hash")[:2]:
            raise ValueError("图片存储 Key 无效")

        lexical_candidate = self.root / storage_key
        if lexical_candidate.is_symlink() or lexical_candidate.parent.is_symlink():
            raise ValueError("图片存储 Key 无效")
        candidate = lexical_candidate.resolve()
        if not candidate.is_relative_to(self.root):
            raise ValueError("图片存储 Key 无效")
        return candidate

    def delete_unreferenced(self, referenced_keys: Collection[str]) -> int:
        """删除根目录内未被数据库引用的合法内容寻址图片。"""

        referenced = {self._validated_key(key) for key in referenced_keys}
        deleted = 0
        for shard in self.root.iterdir():
            if not shard.is_dir() or not re.fullmatch(r"[0-9a-f]{2}", shard.name):
                continue
            if not shard.resolve().is_relative_to(self.root):
                continue
            for candidate in shard.iterdir():
                if candidate.is_symlink():
                    continue
                storage_key = f"{shard.name}/{candidate.name}"
                try:
                    resolved = self.resolve(storage_key)
                except ValueError:
                    continue
                if storage_key in referenced or not resolved.is_file():
                    continue
                resolved.unlink()
                deleted += 1
            try:
                shard.rmdir()
            except OSError:
                pass
        return deleted

    def _validated_key(self, storage_key: str) -> str:
        self.resolve(storage_key)
        return storage_key

    @staticmethod
    def _matches_magic(content: bytes, mime_type: str) -> bool:
        if mime_type == "image/png":
            return content.startswith(b"\x89PNG\r\n\x1a\n")
        if mime_type == "image/jpeg":
            return content.startswith(b"\xff\xd8\xff")
        if mime_type == "image/gif":
            return content.startswith((b"GIF87a", b"GIF89a"))
        if mime_type == "image/webp":
            return (
                len(content) >= 12
                and content.startswith(b"RIFF")
                and content[8:12] == b"WEBP"
            )
        return False

    @staticmethod
    def _atomic_write(target: Path, content: bytes) -> None:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".knowledge-image-",
            dir=target.parent,
        )
        temporary_path = Path(temporary_name)
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, target)
            target.chmod(0o600)
        except BaseException:
            try:
                os.close(descriptor)
            except OSError:
                pass
            temporary_path.unlink(missing_ok=True)
            raise
