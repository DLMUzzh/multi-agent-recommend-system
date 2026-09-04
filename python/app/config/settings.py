"""文章推荐服务的集中环境配置与进程内缓存入口。"""

from functools import lru_cache
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings

from .paths import ENV_FILE


class Settings(BaseSettings):
    """文章推荐服务当前实际使用的运行配置。"""

    llm_provider: str = "deepseek"
    llm_api_key: SecretStr = SecretStr("")
    llm_base_url: str = "https://api.deepseek.com/v1"
    llm_model: str = "deepseek-chat"
    llm_small_model: str = ""
    llm_large_model: str = ""
    llm_temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    llm_max_tokens: int = Field(default=2048, ge=1)
    llm_request_timeout_seconds: float = Field(default=30.0, gt=0.0)
    llm_max_retries: int = Field(default=2, ge=0)

    llm_intent_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    llm_intent_max_tokens: int = Field(default=1400, ge=1)
    llm_profile_temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    llm_profile_max_tokens: int = Field(default=3072, ge=1)
    llm_rerank_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    llm_rerank_max_tokens: int = Field(default=3072, ge=1)
    recall_bm25_k1: float = Field(default=1.2, gt=0.0)
    recall_bm25_b: float = Field(default=0.75, ge=0.0, le=1.0)

    embedding_api_key: SecretStr = SecretStr("")
    embedding_base_url: str = (
        "https://llm-oo27tfovq98x83un.cn-beijing.maas.aliyuncs.com/"
        "compatible-mode/v1"
    )
    embedding_model: str = Field(default="qwen3.7-text-embedding", min_length=1)
    embedding_dimensions: int = Field(default=1024, gt=0)
    embedding_batch_size: int = Field(default=10, gt=0)
    embedding_request_timeout_seconds: float = Field(default=10.0, gt=0.0)
    embedding_max_retries: int = Field(default=2, ge=0)
    recall_rrf_k: int = Field(default=60, gt=0)

    agent_timeout_user_profile: float = Field(default=5.0, gt=0.0)
    chat_request_timeout_seconds: float = Field(default=45.0, gt=0.0)

    model_config = {"env_file": ENV_FILE, "env_prefix": "ARTICLE_REC_"}


@lru_cache()
def get_settings() -> Settings:
    """返回进程内缓存的服务配置。"""

    return Settings()
