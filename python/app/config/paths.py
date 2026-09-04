"""集中维护不依赖当前工作目录的项目路径。"""

from pathlib import Path


PYTHON_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = PYTHON_ROOT.parent
DATA_ROOT = PROJECT_ROOT / "data"
DOCUMENT_DATABASE_PATH = DATA_ROOT / "documents.sqlite3"
KNOWLEDGE_IMAGE_ROOT = DATA_ROOT / "knowledge_images"
KNOWLEDGE_TEST_RECORD_PATH = DATA_ROOT / "test_records" / "knowledge_qa.md"
USER_PROFILE_DATABASE_PATH = DATA_ROOT / "user_profiles.sqlite3"
RUNTIME_ROOT = PROJECT_ROOT / "runtime"
RUNTIME_SKILL_ROOT = RUNTIME_ROOT / "skills"
CONVERSATION_DATABASE_PATH = RUNTIME_ROOT / "conversations.sqlite3"
LOG_ROOT = PROJECT_ROOT / "log"
RECALL_INDEX_ROOT = DATA_ROOT / "recall_index"
ENV_FILE = PYTHON_ROOT / ".env"

__all__ = [
    "DATA_ROOT",
    "CONVERSATION_DATABASE_PATH",
    "DOCUMENT_DATABASE_PATH",
    "ENV_FILE",
    "LOG_ROOT",
    "KNOWLEDGE_IMAGE_ROOT",
    "KNOWLEDGE_TEST_RECORD_PATH",
    "PROJECT_ROOT",
    "PYTHON_ROOT",
    "RECALL_INDEX_ROOT",
    "RUNTIME_ROOT",
    "RUNTIME_SKILL_ROOT",
    "USER_PROFILE_DATABASE_PATH",
]
