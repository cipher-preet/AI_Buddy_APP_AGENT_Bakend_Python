from fastapi import FastAPI

from apps.api_gateway.routes.health import router as health_router
from apps.api_gateway.routes.chat_routes import router as chat_router
from apps.api_gateway.middleware.logging import log_requests
from apps.api_gateway.routes.conversation_routes import router as conversation_router
from apps.api_gateway.routes.reminder_voice_routes import router as reminder_voice_router
from apps.api_gateway.routes.reminder_device_routes import router as reminder_device_router
from apps.api_gateway.routes.speech_routes import router as speech_router
from apps.api_gateway.config.setting import settings
from services.db.mongo import close_mongo_client, ensure_mongo_indexes
from services.llm.router import close_llm_runtime

app = FastAPI(title=settings.APP_NAME, version=settings.APP_VERSION)

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


@app.on_event("startup")
async def startup():
    await ensure_mongo_indexes()


@app.on_event("shutdown")
async def shutdown():
    await close_llm_runtime()
    await close_mongo_client()


@app.get("/")
async def root():
    return {"message": "AI Agent System Running"}
