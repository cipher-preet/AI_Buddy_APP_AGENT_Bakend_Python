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
    USER_TIMEZONE: str = "Asia/Kolkata"
    TRANSCRIPT_ANALYSIS_DEBOUNCE_SECONDS: int = 7
    TRANSCRIPT_ANALYSIS_MAX_BATCH_CHUNKS: int = 80
    TRANSCRIPT_ANALYSIS_RECENT_CHUNK_LIMIT: int = 30
    TRANSCRIPT_ANALYSIS_SEMANTIC_LIMIT: int = 8
    TRANSCRIPT_ANALYSIS_TASK_LIMIT: int = 20
    TRANSCRIPT_ANALYSIS_NOTE_LIMIT: int = 20
    TRANSCRIPT_ANALYSIS_LOCK_TTL_SECONDS: int = 120
    TRANSCRIPT_ANALYSIS_MAX_ATTEMPTS: int = 3
    TRANSCRIPT_SESSION_IDLE_SECONDS: int = 600
    TRANSCRIPT_SESSION_IDLE_CHECK_SECONDS: int = 30
    TASK_DEDUPE_SIMILARITY_THRESHOLD: float = 0.88

    class Config:
        env_file = ".env"


settings = Settings()
