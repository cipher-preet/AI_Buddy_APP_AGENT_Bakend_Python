from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    APP_NAME: str = "Speech To Vector API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True

    REDIS_URL: str
    SARVAM_API_KEY: str
    OPENAI_API_KEY: str
    QDRANT_API_KEY: str
    QDRANT_URL: str
    EMBEDDING_MODEL: str
    VECTOR_SIZE: int
    QDRANT_COLLECTION: str

    class Config:
        env_file = ".env"


settings = Settings()
