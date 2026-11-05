from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from apps.api_gateway.auth import require_matching_user, require_user
from services.auth.identity import AuthenticatedUser
from services.db.mongo import get_database
from services.reminders.mongo_store import MongoReminderStore

router = APIRouter(dependencies=[Depends(require_user)])


class RegisterDeviceTokenRequest(BaseModel):
    userId: str = Field(min_length=1)
    token: str = Field(min_length=1)
    platform: str | None = None


@router.post("/device-tokens")
async def register_device_token(
    request: RegisterDeviceTokenRequest,
    actor: AuthenticatedUser = Depends(require_user),
):
    user_id = require_matching_user(actor, request.userId)
    token = request.token.strip()
    if not user_id or not token:
        raise HTTPException(status_code=400, detail="userId and token are required")
    store = MongoReminderStore(get_database())
    await store.upsert_token(user_id, token, request.platform)
    return {"ok": True, "userId": user_id}
