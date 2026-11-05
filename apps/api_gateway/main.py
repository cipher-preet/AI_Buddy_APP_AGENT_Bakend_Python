import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from apps.api_gateway.routes.health import router as health_router
from apps.api_gateway.routes.chat_routes import router as chat_router
from apps.api_gateway.middleware.logging import log_requests
from apps.api_gateway.routes.conversation_routes import router as conversation_router
from apps.api_gateway.routes.daily_briefing_routes import router as daily_briefing_router
from apps.api_gateway.routes.mindmap_routes import router as mindmap_router
from apps.api_gateway.routes.document_routes import router as document_router
from apps.api_gateway.routes.reminder_voice_routes import router as reminder_voice_router
from apps.api_gateway.routes.reminder_device_routes import router as reminder_device_router
from apps.api_gateway.routes.speech_routes import router as speech_router
from apps.api_gateway.config.setting import settings
from services.db.mongo import close_mongo_client, ensure_mongo_indexes
from services.llm.router import close_llm_runtime

app = FastAPI(title=settings.APP_NAME, version=settings.APP_VERSION)

_SENSITIVE_FIELDS = {"authtoken", "authorization", "access_token", "token", "password", "secret"}


def _redact_validation_error(error: dict[str, Any]) -> dict[str, Any]:
    location = {str(part).lower() for part in error.get("loc", ())}
    if not location & _SENSITIVE_FIELDS:
        return error
    cleaned = dict(error)
    if "input" in cleaned:
        cleaned["input"] = "[redacted]"
    cleaned.pop("ctx", None)
    return cleaned


@app.exception_handler(RequestValidationError)
async def redact_validation_errors(_request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={"detail": [_redact_validation_error(dict(error)) for error in exc.errors()]},
    )


app.middleware("http")(log_requests)

app.include_router(health_router, prefix="/api/health", tags=["Health"])


app.include_router(speech_router, prefix="/api/v1/speech", tags=["Speech"])
app.include_router(chat_router, prefix="/api/v1/chat", tags=["Chat"])
app.include_router(
    reminder_voice_router,
    prefix="/api/v1/reminders",
    tags=["Reminder Voice"],
)
app.include_router(
    reminder_device_router,
    prefix="/api/v1/reminders",
    tags=["Reminder Devices"],
)
app.include_router(
    conversation_router,
    prefix="/api/v1/conversations",
    tags=["Conversations"],
)
app.include_router(
    daily_briefing_router,
    prefix="/api/v1/daily-briefing",
    tags=["Daily Briefing"],
)
app.include_router(
    mindmap_router,
    prefix="/api/v1/mindmap",
    tags=["Mindmap"],
)
app.include_router(
    document_router,
    prefix="/api/v1/documents",
    tags=["Documents"],
)


@app.on_event("startup")
async def startup():
    if not settings.resolved_auth_token_secret:
        logging.getLogger("auth").warning(
            "auth.token_secret_missing: set AUTH_TOKEN_SECRET or SESSION_SECRET to the same value the Node API uses"
        )
    await ensure_mongo_indexes()


@app.on_event("shutdown")
async def shutdown():
    await close_llm_runtime()
    await close_mongo_client()


@app.get("/")
async def root():
    return {"message": "AI Agent System Running"}
