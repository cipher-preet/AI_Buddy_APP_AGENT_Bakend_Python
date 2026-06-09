from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    APP_NAME: str = "AI Agent API"
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
    MONGO_URL: str
    MONGO_DB_NAME:str
    OPENAI_CHAT_MODEL: str = "gpt-4o-mini"

    class Config:
        env_file = ".env"


settings = Settings()
