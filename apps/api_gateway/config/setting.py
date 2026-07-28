from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    APP_NAME: str = "Buddy Conversation Processing API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True

    REDIS_URL: str
    MONGODB_URL: str = "mongodb://localhost:27017"
    MONGODB_DATABASE: str = "buddy"

    SARVAM_API_KEY: str
    SARVAM_BASE_URL: str = "https://api.sarvam.ai/v1"
    SARVAM_SPEECH_BASE_URL: str = "https://api.sarvam.ai"
    SARVAM_DEFAULT_MODEL: str = "sarvam-105b"
    SARVAM_FAST_MODEL: str = "sarvam-30b"
    SARVAM_TIMEOUT_SECONDS: float = 60
    SARVAM_MAX_RETRIES: int = 3
    SARVAM_MAX_CONCURRENCY: int = 8

    OPENAI_API_KEY: str
    OPENAI_BASE_URL: str = "https://api.openai.com/v1"

    QDRANT_API_KEY: str
    QDRANT_URL: str
    EMBEDDING_MODEL: str
    VECTOR_SIZE: int
    QDRANT_COLLECTION: str

    REDIS_AUDIO_STREAM: str = "buddy:audio:ingestion"
    REDIS_STT_STREAM: str = "buddy:stt:jobs"
    REDIS_TRANSCRIPT_READY_STREAM: str = "buddy:transcript:ready"
    REDIS_FINALIZATION_STREAM: str = "buddy:conversation:finalization"
    REDIS_PROCESSING_STREAM: str = "buddy:conversation:processing"
    REDIS_RETRY_STREAM: str = "buddy:conversation:retry"
    REDIS_DEAD_LETTER_STREAM: str = "buddy:dead-letter"

    REDIS_AUDIO_GROUP: str = "audio-workers"
    REDIS_STT_GROUP: str = "stt-workers"
    REDIS_TRANSCRIPT_GROUP: str = "transcript-workers"
    REDIS_FINALIZATION_GROUP: str = "finalization-workers"
    REDIS_PROCESSING_GROUP: str = "conversation-processing-workers"

    REDIS_CLAIM_IDLE_MS: int = 60000
    REDIS_BLOCK_MS: int = 5000
    REDIS_BATCH_SIZE: int = 10
    WORKER_CONCURRENCY: int = 4
    WORKER_MAX_RETRIES: int = 5
    WORKER_RETRY_BASE_SECONDS: float = 1
    WORKER_RETRY_MAX_SECONDS: float = 300

    CONVERSATION_INACTIVITY_TIMEOUT_SECONDS: int = 900
    RAW_TRANSCRIPT_RETENTION_DAYS: int = 30
    TRANSCRIPT_SEGMENT_TARGET_TOKENS: int = 2500
    TRANSCRIPT_SEGMENT_OVERLAP_RATIO: float = 0.12
    MAX_TRANSCRIPT_TOKENS: int = 120000
    MAX_TRANSCRIPT_SEGMENTS: int = 80
    MAX_REPAIR_ROUNDS: int = 2

    LLM_DEFAULT_PROVIDER: str = "sarvam"
    LLM_SECONDARY_PROVIDER: str = "openai"
    LLM_DEFAULT_MODEL: str = "sarvam-105b"
    LLM_FAST_MODEL: str = "sarvam-30b"
    LLM_VALIDATION_MODEL: str = "sarvam-105b"
    LLM_SUMMARY_MODEL: str = "sarvam-30b"
    LLM_TIMEOUT_SECONDS: float = 60
    LLM_MAX_CONCURRENCY: int = 8
    LLM_TEMPERATURE: float = 0.1

    MAX_QUEUED_CONVERSATIONS_PER_USER: int = 50
    MAX_ACTIVE_PROCESSING_JOBS_PER_USER: int = 2
    MAX_ACTIVE_LLM_CALLS_PER_CONVERSATION: int = 4
    ENABLE_TRANSCRIPT_DEBUG_LOGS: bool = False

    class Config:
        env_file = ".env"


settings = Settings()
