from fastapi import APIRouter, File, Form, Query, UploadFile

from apps.api_gateway.controllers.reminder_voice_controller import (
    get_reminder_prompt_controller,
    run_reminder_voice_turn_controller,
)

router = APIRouter()


@router.get("/voice/prompt")
async def get_reminder_voice_prompt(
    kind: str = Query(default="greeting"),
    language: str | None = Query(default=None),
):
    return await get_reminder_prompt_controller(kind, language)


@router.post("/voice/turn")
async def reminder_voice_turn(
    file: UploadFile = File(...),
    userId: str = Form(...),
    timezone: str | None = Form(default=None),
    collected: str | None = Form(default=None),
):
    return await run_reminder_voice_turn_controller(
        file=file,
        user_id=userId,
        timezone_name=timezone,
        collected_raw=collected,
    )
