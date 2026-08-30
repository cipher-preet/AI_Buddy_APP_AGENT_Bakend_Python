from fastapi import UploadFile

from services.reminders.service import (
    get_reminder_prompt_service,
    run_reminder_voice_turn_service,
)


async def get_reminder_prompt_controller(kind: str, language: str | None = None):
    return await get_reminder_prompt_service(kind, language)


async def run_reminder_voice_turn_controller(
    file: UploadFile,
    user_id: str,
    timezone_name: str | None,
    collected_raw: str | None,
):
    return await run_reminder_voice_turn_service(
        file=file,
        user_id=user_id,
        timezone_name=timezone_name,
        collected_raw=collected_raw,
    )
