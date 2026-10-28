from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", frozen=True)

    app_api_key: SecretStr = SecretStr("")
    openai_api_key: SecretStr = SecretStr("")
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = Field(default=256, ge=1, le=3072)
    answer_model: str = "gpt-4o-mini"
    top_k: int = Field(default=4, ge=1, le=10)
    min_similarity: float = Field(default=0.25, ge=-1, le=1)
