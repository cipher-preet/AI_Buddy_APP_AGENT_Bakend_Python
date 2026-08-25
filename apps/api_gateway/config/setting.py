from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    APP_NAME: str = "Buddy Conversation Processing API"
    APP_VERSION: str = "1.0.0"
    APP_ENV: str = "local"
    SERVICE_ROLE: str = "api"
    DEBUG: bool = True

    REDIS_URL: str = "redis://localhost:6379"
    REDIS_MAX_RETRIES: int | None = None
    REDIS_EVENT_RETENTION: int = Field(default=86400, ge=60)
    MONGODB_URI: str = ""
    MONGODB_URL: str = "mongodb://localhost:27017"
    MONGODB_DATABASE: str = "buddy"

    SARVAM_API_KEY: SecretStr | str = ""
    SARVAM_BASE_URL: str = "https://api.sarvam.ai/v1"
    SARVAM_SPEECH_BASE_URL: str = "https://api.sarvam.ai"
    SARVAM_DEFAULT_MODEL: str = "sarvam-105b"
    SARVAM_FAST_MODEL: str = "sarvam-105b"
    SARVAM_TIMEOUT_SECONDS: float = 60
    SARVAM_MAX_RETRIES: int = 3
    SARVAM_MAX_CONCURRENCY: int = 8
    SARVAM_MAX_TOKENS: int = 4096
    SARVAM_STT_MAX_DURATION_MS: int = Field(default=30000, ge=1000, le=600000)

    STT_PROVIDER_ORDER: str = "deepgram,sarvam"
    STT_ALLOW_SARVAM_FALLBACK: bool = True
    STT_TIMEOUT_SECONDS: float = 60
    STT_MAX_RETRIES: int = 2

    DEEPGRAM_API_KEY: SecretStr | str = ""
    DEEPGRAM_MODEL: str = "nova-3"
    DEEPGRAM_LANGUAGE: str = "multi"
    DEEPGRAM_SMART_FORMAT: bool = True
    DEEPGRAM_DETECT_LANGUAGE: bool = False

    OPENAI_API_KEY: SecretStr | str = ""
    OPENAI_BASE_URL: str = "https://api.openai.com/v1"

    GEMINI_API_KEY: SecretStr | str = ""
    GEMINI_BASE_URL: str = "https://generativelanguage.googleapis.com/v1beta/openai"
    GEMINI_FREE_MODEL: str = "gemini-3.5-flash-lite"
    GEMINI_MAX_RPM: int = 12
    GEMINI_MAX_RPD: int = 900

    # Groq Cloud (https://api.groq.com). This is not xAI Grok.
    GROQ_API_KEY: SecretStr | str = ""
    GROQ_BASE_URL: str = "https://api.groq.com/openai/v1"
    GROQ_FREE_MODEL: str = "openai/gpt-oss-20b"
    GROQ_MAX_RPM: int = 24
    GROQ_MAX_RPD: int = 900
    GROQ_MAX_TPM: int = 7000
    GROQ_MAX_TPD: int = 180000

    MISTRAL_API_KEY: SecretStr | str = ""
    MISTRAL_BASE_URL: str = "https://api.mistral.ai/v1"
    MISTRAL_CHEAP_MODEL: str = "ministral-3b-2512"

    QDRANT_API_KEY: SecretStr | str = ""
    QDRANT_URL: str = "http://localhost:6333"
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    VECTOR_SIZE: int = Field(default=1536, gt=0)
    QDRANT_COLLECTION: str = "speech_chunks"

    REDIS_AUDIO_STREAM: str = "buddy:audio:ingestion"
    REDIS_STT_STREAM: str = "buddy:stt:jobs"
    REDIS_TRANSCRIPT_READY_STREAM: str = "buddy:transcript:ready"
    REDIS_WINDOW_EXTRACTION_STREAM: str = "buddy:conversation:window-extraction"
    REDIS_FINALIZATION_STREAM: str = "buddy:conversation:finalization"
    REDIS_PROCESSING_STREAM: str = "buddy:conversation:processing"
    REDIS_RETRY_STREAM: str = "buddy:conversation:retry"
    REDIS_DEAD_LETTER_STREAM: str = "buddy:dead-letter"

    REDIS_AUDIO_GROUP: str = "audio-workers"
    REDIS_STT_GROUP: str = "stt-workers"
    REDIS_TRANSCRIPT_GROUP: str = "transcript-workers"
    REDIS_WINDOW_EXTRACTION_GROUP: str = "window-extraction-workers"
    REDIS_FINALIZATION_GROUP: str = "finalization-workers"
    REDIS_PROCESSING_GROUP: str = "conversation-processing-workers"

    REDIS_CLAIM_IDLE_MS: int = 60000
    REDIS_BLOCK_MS: int = 5000
    REDIS_BATCH_SIZE: int = 10
    WORKER_CONCURRENCY: int = Field(default=4, ge=1, le=64)
    AUDIO_WORKER_CONCURRENCY: int | None = Field(default=None, ge=1, le=64)
    STT_WORKER_CONCURRENCY: int | None = Field(default=None, ge=1, le=64)
    FINALIZATION_WORKER_CONCURRENCY: int | None = Field(default=None, ge=1, le=64)
    TRANSCRIPT_WINDOW_WORKER_CONCURRENCY: int | None = Field(default=None, ge=1, le=64)
    WINDOW_EXTRACTION_WORKER_CONCURRENCY: int | None = Field(default=None, ge=1, le=64)
    PROCESSING_WORKER_CONCURRENCY: int = Field(default=1, ge=1, le=8)
    WORKER_MAX_RETRIES: int = Field(default=5, ge=0, le=100)
    WORKER_GRACEFUL_SHUTDOWN_SECONDS: int = Field(default=30, ge=1, le=600)
    WORKER_TEMP_AUDIO_ROOT: str = "/tmp/buddy"
    WORKER_RETRY_BASE_SECONDS: float = Field(default=1, ge=0.1, le=300)
    WORKER_RETRY_MAX_SECONDS: float = Field(default=300, ge=1, le=3600)

    QUEUE_PROVIDER: str = "redis"
    QUEUE_API_BASE_URL: str = ""
    QUEUE_API_SERVICE_TOKEN: SecretStr | str = ""
    QUEUE_API_HMAC_SECRET: SecretStr | str = ""
    QUEUE_API_REQUEST_TIMEOUT_SECONDS: float = Field(default=5, gt=0, le=60)
    QUEUE_API_MAX_BODY_BYTES: int = Field(default=65536, ge=1024, le=1048576)
    QUEUE_API_SIGNATURE_TOLERANCE_SECONDS: int = Field(default=300, ge=30, le=3600)

    STORAGE_PROVIDER: str = "local"
    AWS_ACCESS_KEY_ID: SecretStr | str = ""
    AWS_SECRET_ACCESS_KEY: SecretStr | str = ""
    ACCESS_KEY_ID: SecretStr | str = ""
    SECREATE_KEY_ACCESS: SecretStr | str = ""
    AWS_REGION: str = "ap-south-1"
    S3_BUCKET: str = ""
    S3_REGION: str = ""
    S3_AUDIO_BUCKET: str = ""
    S3_AUDIO_PREFIX: str = "buddy/audio"
    S3_PRESIGNED_URL_TTL_SECONDS: int = Field(default=300, ge=30, le=3600)
    S3_MAX_AUDIO_SIZE_BYTES: int = Field(default=25 * 1024 * 1024, ge=1024, le=512 * 1024 * 1024)
    S3_ALLOWED_CONTENT_TYPES: str = "audio/aac,audio/aiff,audio/amr,audio/flac,audio/m4a,audio/mp4,audio/mpeg,audio/mp3,audio/ogg,audio/opus,audio/wav,audio/wave,audio/webm,audio/x-m4a,audio/x-wav,video/mp4,video/webm"
    S3_DELETE_AFTER_PROCESSING: bool = False
    S3_UPLOAD_TIMEOUT_SECONDS: float = 60
    S3_DOWNLOAD_TIMEOUT_SECONDS: float = 60
    S3_MAX_RETRIES: int = 3
    CLOUDFRONT_URL: str = ""
    CLOUDFRONT_KEY_PAIR_ID: str = ""
    CLOUDFRONT_PRIVATE_KEY: SecretStr | str = ""
    CLOUDFRONT_SIGNED_URL_EXPIRES_SECONDS: int = 3600

    CONVERSATION_INACTIVITY_TIMEOUT_SECONDS: int = 900
    RAW_TRANSCRIPT_RETENTION_DAYS: int = 30
    TRANSCRIPT_SEGMENT_TARGET_TOKENS: int = 2500
    TRANSCRIPT_SEGMENT_OVERLAP_RATIO: float = 0.12
    MAX_TRANSCRIPT_TOKENS: int = 120000
    MAX_TRANSCRIPT_SEGMENTS: int = 80
    MAX_REPAIR_ROUNDS: int = 2
    ENABLE_INCREMENTAL_MEETING_PROCESSING: bool = True
    SEMANTIC_WINDOW_MAX_USEFUL_MINUTES: int = Field(default=60, ge=1, le=480)
    SEMANTIC_WINDOW_MAX_TRANSCRIPT_TOKENS: int = Field(default=22000, ge=200, le=100000)
    SEMANTIC_WINDOW_SAFE_CONTEXT_RATIO: float = Field(default=0.72, ge=0.3, le=0.95)
    INCREMENTAL_WINDOW_TARGET_TOKENS: int = Field(default=22000, ge=200, le=100000)
    INCREMENTAL_WINDOW_MAX_TOKENS: int = Field(default=28000, ge=200, le=120000)
    INCREMENTAL_WINDOW_OVERLAP_TOKENS: int = Field(default=200, ge=0, le=8000)
    INCREMENTAL_WINDOW_OVERLAP_RATIO: float = Field(default=0.02, ge=0, le=0.5)
    INCREMENTAL_WINDOW_MAX_DURATION_MS: int = Field(default=60 * 60 * 1000, ge=1000, le=8 * 60 * 60 * 1000)
    SPARSE_WINDOW_MAX_WALL_CLOCK_MS: int = Field(default=0, ge=0, le=8 * 60 * 60 * 1000)
    SPARSE_WINDOW_MIN_USEFUL_TOKENS: int = Field(default=4, ge=1, le=500)
    LLM_PROVIDER_CONTEXT_TOKENS: str = "groq:8192,gemini:1048576,mistral:32768,sarvam:32768,openai:128000,anthropic:200000"
    WINDOW_PROCESSING_STALE_TIMEOUT_SECONDS: int = Field(default=180, ge=15, le=3600)
    STT_PROCESSING_STALE_TIMEOUT_SECONDS: int = Field(default=300, ge=30, le=3600)
    FINALIZATION_MAX_RETRIES: int = Field(default=8, ge=1, le=50)
    FINALIZATION_MISSING_SEQUENCE_TIMEOUT_SECONDS: int = Field(default=900, ge=15, le=3600)
    COVERAGE_SPARSE_WINDOW_ENABLED: bool = True
    MEETING_MEMORY_RETRIEVAL_LIMIT: int = Field(default=12, ge=1, le=50)
    MEETING_MEMORY_GLOBAL_ITEM_LIMIT: int = Field(default=30, ge=5, le=100)
    ARTIFACT_TITLE_JACCARD_DUPLICATE: float = Field(default=0.9, ge=0.5, le=1)
    ARTIFACT_TITLE_JACCARD_UPDATE: float = Field(default=0.75, ge=0.4, le=1)
    COVERAGE_COMPRESSION_RATIO_THRESHOLD: float = Field(default=0.75, ge=0.1, le=1)
    COVERAGE_MIN_PROVISIONAL_FOR_GUARD: int = Field(default=8, ge=1, le=100)
    COVERAGE_WEAK_WINDOW_MIN_TOKENS: int = Field(default=200, ge=20, le=5000)
    SELECTIVE_RECOVERY_MAX_WINDOWS: int = Field(default=3, ge=0, le=20)
    FINAL_MODEL_INPUT_TOKEN_LIMIT: int = Field(default=48000, ge=1000, le=200000)
    FINAL_COMPRESSION_GROUP_TOKENS: int = Field(default=8000, ge=1000, le=50000)
    MAX_QUALITY_REPAIR_ROUNDS: int = Field(default=1, ge=0, le=2)
    INTELLIGENCE_CONFIDENCE_PUBLISH_THRESHOLD: float = Field(default=0.55, ge=0, le=1)
    INTELLIGENCE_CONFIDENCE_SUGGESTION_THRESHOLD: float = Field(default=0.45, ge=0, le=1)
    INTELLIGENCE_CONFIDENCE_EVIDENCE_WEIGHT: float = Field(default=0.28, ge=0, le=1)
    INTELLIGENCE_CONFIDENCE_EXPLICITNESS_WEIGHT: float = Field(default=0.20, ge=0, le=1)
    INTELLIGENCE_CONFIDENCE_COMPLETENESS_WEIGHT: float = Field(default=0.16, ge=0, le=1)
    INTELLIGENCE_CONFIDENCE_CONTEXT_WEIGHT: float = Field(default=0.12, ge=0, le=1)
    INTELLIGENCE_CONFIDENCE_VALIDATION_WEIGHT: float = Field(default=0.16, ge=0, le=1)
    INTELLIGENCE_CONFIDENCE_CONFLICT_PENALTY: float = Field(default=0.22, ge=0, le=1)
    INTELLIGENCE_CONFIDENCE_SHALLOW_PENALTY: float = Field(default=0.12, ge=0, le=1)
    INTELLIGENCE_CONFIDENCE_SPECULATION_PENALTY: float = Field(default=0.22, ge=0, le=1)

    LLM_DEFAULT_PROVIDER: str = "sarvam"
    LLM_SECONDARY_PROVIDER: str = "openai"
    LLM_DEFAULT_MODEL: str = "sarvam-105b"
    LLM_FAST_MODEL: str = "sarvam-105b"
    LLM_VALIDATION_MODEL: str = "sarvam-105b"
    LLM_SUMMARY_MODEL: str = "sarvam-105b"
    LLM_ENABLE_COST_OPTIMIZED_ROUTING: bool = True
    LLM_TIMEOUT_SECONDS: float = 60
    LLM_MAX_CONCURRENCY: int = 8
    LLM_TEMPERATURE: float = 0.1
    LLM_STRUCTURED_MAX_TOKENS: int = 4096
    LLM_EXTRACTION_OUTPUT_MAX_TOKENS: int = Field(default=4096, ge=512, le=16000)

    MAX_QUEUED_CONVERSATIONS_PER_USER: int = 50
    MAX_ACTIVE_PROCESSING_JOBS_PER_USER: int = 2
    MAX_ACTIVE_LLM_CALLS_PER_CONVERSATION: int = 4
    CONVERSATION_PROCESSING_TIMEOUT_SECONDS: float = 1800
    ENABLE_REQUEST_LOGS: bool = False
    ENABLE_TRANSCRIPT_DEBUG_LOGS: bool = False

    @field_validator("APP_ENV", "SERVICE_ROLE", "QUEUE_PROVIDER", "STORAGE_PROVIDER")
    @classmethod
    def normalize_enumish(cls, value: str) -> str:
        return str(value or "").strip().lower()

    @field_validator("SARVAM_TIMEOUT_SECONDS", "STT_TIMEOUT_SECONDS", "LLM_TIMEOUT_SECONDS")
    @classmethod
    def positive_timeout(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("timeout settings must be positive")
        return value

    @model_validator(mode="after")
    def validate_selected_role(self):
        if self.S3_BUCKET and not self.S3_AUDIO_BUCKET:
            self.S3_AUDIO_BUCKET = self.S3_BUCKET
        if self.S3_REGION and not self.AWS_REGION:
            self.AWS_REGION = self.S3_REGION
        if self.MONGODB_URI and self.MONGODB_URL == "mongodb://localhost:27017":
            self.MONGODB_URL = self.MONGODB_URI
        if self.REDIS_MAX_RETRIES is not None:
            self.WORKER_MAX_RETRIES = self.REDIS_MAX_RETRIES

        allowed_roles = {"api", "worker", "queue_api", "all", "local"}
        if self.SERVICE_ROLE not in allowed_roles:
            raise ValueError(f"SERVICE_ROLE must be one of {sorted(allowed_roles)}")

        allowed_queue_providers = {"redis", "queue_api"}
        if self.QUEUE_PROVIDER not in allowed_queue_providers:
            raise ValueError(f"QUEUE_PROVIDER must be one of {sorted(allowed_queue_providers)}")

        if self.STORAGE_PROVIDER not in {"local", "s3"}:
            raise ValueError("STORAGE_PROVIDER must be one of ['local', 's3']")

        if self.SERVICE_ROLE in {"api", "all"} and self.QUEUE_PROVIDER == "queue_api":
            if not self.QUEUE_API_BASE_URL.strip():
                raise ValueError("QUEUE_API_BASE_URL is required when QUEUE_PROVIDER=queue_api")
            if not self._secret_value(self.QUEUE_API_SERVICE_TOKEN) and not self._secret_value(self.QUEUE_API_HMAC_SECRET):
                raise ValueError("QUEUE_API_SERVICE_TOKEN or QUEUE_API_HMAC_SECRET is required when QUEUE_PROVIDER=queue_api")

        if self.SERVICE_ROLE in {"worker", "queue_api", "all"} and not self.REDIS_URL.strip():
            raise ValueError("REDIS_URL is required for worker and queue_api roles")

        if self.SERVICE_ROLE in {"worker", "all"}:
            if not self.MONGODB_URL.strip() or self.MONGODB_URL == "mongodb://localhost:27017":
                raise ValueError("MONGODB_URL is required for worker roles and must not point to localhost")

        if self.STORAGE_PROVIDER == "s3" and self.SERVICE_ROLE in {"api", "worker", "all"}:
            if not self.S3_AUDIO_BUCKET.strip():
                raise ValueError("S3_AUDIO_BUCKET or S3_BUCKET is required when STORAGE_PROVIDER=s3")
        return self

    @property
    def resolved_aws_access_key_id(self) -> str:
        return self._secret_value(self.AWS_ACCESS_KEY_ID) or self._secret_value(self.ACCESS_KEY_ID)

    @property
    def resolved_aws_secret_access_key(self) -> str:
        return self._secret_value(self.AWS_SECRET_ACCESS_KEY) or self._secret_value(self.SECREATE_KEY_ACCESS)

    @property
    def allowed_audio_content_types(self) -> set[str]:
        return {
            item.strip().lower()
            for item in self.S3_ALLOWED_CONTENT_TYPES.split(",")
            if item.strip()
        }

    @property
    def semantic_window_useful_duration_ms(self) -> int:
        return int(self.SEMANTIC_WINDOW_MAX_USEFUL_MINUTES * 60 * 1000)

    @property
    def provider_context_token_limits(self) -> dict[str, int]:
        limits: dict[str, int] = {}
        for item in self.LLM_PROVIDER_CONTEXT_TOKENS.split(","):
            if ":" not in item:
                continue
            name, raw = item.split(":", 1)
            try:
                limits[name.strip().lower()] = int(raw.strip())
            except ValueError:
                continue
        return limits

    @property
    def stt_provider_order_list(self) -> list[str]:
        providers = [
            item.strip().lower()
            for item in self.STT_PROVIDER_ORDER.split(",")
            if item.strip()
        ]
        if not self.STT_ALLOW_SARVAM_FALLBACK:
            providers = [item for item in providers if item != "sarvam"]
        return providers or ["deepgram", "sarvam"]

    @staticmethod
    def _secret_value(value: SecretStr | str | None) -> str:
        if value is None:
            return ""
        if isinstance(value, SecretStr):
            return value.get_secret_value()
        return str(value)

    def secret_value(self, value: SecretStr | str | None) -> str:
        return self._secret_value(value)


settings = Settings()
