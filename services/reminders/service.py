from __future__ import annotations

from fastapi import HTTPException, UploadFile

from services.reminders.prompts import STATIC_PROMPTS, prompt_text
from services.reminders.tts import synthesize_speech


async def get_reminder_prompt_service(kind: str, language: str | None = None) -> dict:
    kind_key = (kind or "").strip().lower()
    if kind_key not in STATIC_PROMPTS:
        raise HTTPException(status_code=400, detail="Unknown reminder voice prompt.")
    lang = "hi" if str(language or "").lower().startswith("hi") else "en"
    text = prompt_text(kind_key, lang)

    spoken = await synthesize_speech(text)
    return {
        "success": True,
        "data": {
            "kind": kind,
            "text": text,
            "audioBase64": None if spoken is None else spoken["audioBase64"],
            "contentType": None if spoken is None else spoken["contentType"],
        },
    }


async def run_reminder_voice_turn_service(
    *,
    file: UploadFile,
    user_id: str,
    timezone_name: str | None,
    collected_raw: str | None,
) -> dict:
    del file, timezone_name, collected_raw
    if not (user_id or "").strip():
        raise HTTPException(status_code=400, detail="userId is required.")
    raise HTTPException(status_code=410, detail="Voice reminders are turned off.")
