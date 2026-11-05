from fastapi import APIRouter, Depends, File, Form, Query, UploadFile

from apps.api_gateway.auth import require_matching_user, require_user
from apps.api_gateway.controllers.reminder_voice_controller import (
    get_reminder_prompt_controller,
    run_reminder_voice_turn_controller,
)
from services.auth.identity import AuthenticatedUser

router = APIRouter()


@router.get("/voice/prompt")
async def get_reminder_voice_prompt(
    kind: str = Query(default="greeting"),
    language: str | None = Query(default=None),
):
    return await get_reminder_prompt_controller(kind, language)


@router.post("/voice/turn")
async def reminder_voice_turn(
    actor: AuthenticatedUser = Depends(require_user),
    file: UploadFile = File(...),
    userId: str = Form(...),
    timezone: str | None = Form(default=None),
    collected: str | None = Form(default=None),
):
    user_id = require_matching_user(actor, userId)
    return await run_reminder_voice_turn_controller(
        file=file,
        user_id=user_id,
        timezone_name=timezone,
        collected_raw=collected,
    )
