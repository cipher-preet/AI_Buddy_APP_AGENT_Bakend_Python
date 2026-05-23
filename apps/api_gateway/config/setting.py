from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    APP_NAME: str = "AI Agent API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True

    SARVAM_API_KEY: str

    class Config:
        env_file = ".env"


settings = Settings()