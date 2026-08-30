from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from fastapi import HTTPException, UploadFile

from services.reminders.pipeline import run_reminder_turn, with_spoken_reply
from services.reminders.prompts import STATIC_PROMPTS, prompt_text
from services.reminders.schemas import ReminderCollected
from services.reminders.tts import synthesize_speech

_MAX_AUDIO_BYTES = 8 * 1024 * 1024


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
    if not (user_id or "").strip():
        raise HTTPException(status_code=400, detail="userId is required.")

    collected = _parse_collected(collected_raw)
    suffix = Path(file.filename or "reminder.m4a").suffix or ".m4a"
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Audio file is empty.")
    if len(content) > _MAX_AUDIO_BYTES:
        raise HTTPException(status_code=400, detail="Audio file is too large.")

    tmp_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        result = await run_reminder_turn(
            file_path=tmp_path,
            filename=file.filename or Path(tmp_path).name,
            content_type=file.content_type or "audio/mp4",
            collected=collected,
            timezone_name=timezone_name,
        )
        payload = await with_spoken_reply(result)
        return {
            "success": True,
            "data": payload,
        }
    except HTTPException:
        raise
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        print("Reminder voice turn failed:", {"error": str(error)[:400], "user_id": user_id})
        raise HTTPException(
            status_code=503,
            detail="Unable to process the reminder just now. Please try again.",
        ) from error
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def _parse_collected(raw: str | None) -> ReminderCollected:
    if not raw or not raw.strip():
        return ReminderCollected()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise HTTPException(status_code=400, detail="collected must be valid JSON.") from error
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="collected must be an object.")
    try:
        return ReminderCollected.model_validate(payload)
    except Exception as error:
        raise HTTPException(status_code=400, detail="Invalid collected reminder state.") from error
