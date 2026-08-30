"""Fast reminder-voice turn: Deepgram STT, hardcoded questions, Krutrim fallback."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Awaitable, Callable

from services.llm.models import LLMMessage, StructuredLLMRequest
from services.llm.router import LLMCapability, get_llm_router
from services.prompts.loader import load_prompt
from services.reminders.dates import now_in_timezone
from services.reminders.extract import (
    apply_llm_extract,
    clean_reminder_transcript,
    infer_intent,
    looks_like_reminder,
    merge_collected,
    rule_extract,
)
from services.reminders.language import detect_language
from services.reminders.prompts import (
    PROMPT_KIND_EMPTY,
    PROMPT_KIND_OUT_OF_CONTEXT,
    PROMPT_KIND_SAVED,
    confirmation_text,
    missing_field_prompt,
    prompt_text,
)
from services.reminders.schemas import (
    ReminderCollected,
    ReminderExtractorResponse,
    ReminderPayload,
    ReminderVoiceTurnResult,
)
from services.reminders.tts import synthesize_speech
from services.speech.transcription_router import transcribe_from_path_with_fallback

LlmExtractor = Callable[[str, ReminderCollected, datetime], Awaitable[ReminderExtractorResponse | None]]


async def run_reminder_turn(
    *,
    file_path: str,
    filename: str,
    content_type: str,
    collected: ReminderCollected | None = None,
    timezone_name: str | None = None,
    llm_extractor: LlmExtractor | None = None,
) -> ReminderVoiceTurnResult:
    previous = collected or ReminderCollected()
    now = now_in_timezone(timezone_name)

    stt = await transcribe_from_path_with_fallback(
        file_path,
        filename,
        content_type,
        keyterms=[
            "reminder",
            "tomorrow",
            "today",
            "wake",
            "alarm",
            "AM",
            "PM",
            "kal",
            "aaj",
            "subah",
            "shaam",
            "baje",
            "yaad",
            "कल",
            "आज",
            "सुबह",
            "शाम",
            "बजे",
            "रिमाइंडर",
        ],
        context={"feature": "reminder_voice"},
    )
    transcript = clean_reminder_transcript(str(stt.get("transcript") or ""))
    provider = str(stt.get("provider") or "deepgram")

    if not transcript:
        language = previous.language or "en"
        return ReminderVoiceTurnResult(
            status="unclear",
            transcript="",
            replyText=prompt_text(PROMPT_KIND_EMPTY, language),
            replyKind=PROMPT_KIND_EMPTY,
            collected=previous,
            sttProvider=provider,
            language=language,
        )

    return await decide_reminder_turn(
        transcript=transcript,
        collected=previous,
        now=now,
        stt_provider=provider,
        llm_extractor=llm_extractor,
    )


async def decide_reminder_turn(
    *,
    transcript: str,
    collected: ReminderCollected,
    now: datetime,
    stt_provider: str | None = None,
    llm_extractor: LlmExtractor | None = None,
) -> ReminderVoiceTurnResult:
    transcript = clean_reminder_transcript(transcript)
    used_llm = False
    incoming = rule_extract(transcript, now)
    merged = merge_collected(collected, incoming)
    language = detect_language(transcript, merged.language or collected.language)
    merged.language = language
    intent = infer_intent(transcript, merged)

    needs_llm = _should_use_llm(transcript, merged, intent)
    if needs_llm:
        extractor = llm_extractor or extract_with_krutrim
        llm_result = await extractor(transcript, merged, now)
        used_llm = llm_result is not None
        if llm_result is not None:
            merged = apply_llm_extract(merged, llm_result, now)
            intent = infer_intent(transcript, merged, llm_result.intent)

    if intent == "out_of_context" and not merged.is_complete():
        kind, text = (
            (PROMPT_KIND_OUT_OF_CONTEXT, prompt_text(PROMPT_KIND_OUT_OF_CONTEXT, language))
            if not collected.missing_field()
            else missing_field_prompt(collected.missing_field(), language)
        )
        if collected.missing_field():
            text = f"{prompt_text(PROMPT_KIND_OUT_OF_CONTEXT, language)} {text}"
        return ReminderVoiceTurnResult(
            status="out_of_context",
            transcript=transcript,
            replyText=text,
            replyKind=kind,
            collected=collected,
            sttProvider=stt_provider,
            usedLlm=used_llm,
            language=language,
        )

    missing = merged.missing_field()
    if intent == "unclear" and missing == "title" and not merged.dateKey and not merged.timeLabel:
        return ReminderVoiceTurnResult(
            status="unclear",
            transcript=transcript,
            replyText=prompt_text("unclear", language),
            replyKind="unclear",
            collected=merged,
            sttProvider=stt_provider,
            usedLlm=used_llm,
            language=language,
        )

    if missing:
        kind, text = missing_field_prompt(missing, language)
        return ReminderVoiceTurnResult(
            status="need_more",
            transcript=transcript,
            replyText=text,
            replyKind=kind,
            collected=merged,
            sttProvider=stt_provider,
            usedLlm=used_llm,
            language=language,
        )

    reminder = ReminderPayload(
        title=_pretty_title(merged.title or "Reminder"),
        description=(merged.description or merged.title or "Reminder")[:500],
        dateKey=merged.dateKey or "",
        dateLabel=merged.dateLabel or "",
        timeLabel=merged.timeLabel or "",
        repeat=merged.repeat or "once",
    )
    return ReminderVoiceTurnResult(
        status="ready",
        transcript=transcript,
        replyText=confirmation_text(
            reminder.title,
            reminder.dateLabel,
            reminder.timeLabel,
            language,
        ),
        replyKind=PROMPT_KIND_SAVED,
        collected=merged,
        reminder=reminder,
        sttProvider=stt_provider,
        usedLlm=used_llm,
        language=language,
    )


async def extract_with_krutrim(
    transcript: str,
    collected: ReminderCollected,
    now: datetime,
) -> ReminderExtractorResponse | None:
    router = get_llm_router()
    provider, model = router.route(LLMCapability.SEMANTIC_EXTRACTION)
    payload = {
        "now": now.isoformat(),
        "timezone": str(now.tzinfo or "Asia/Kolkata"),
        "transcript": transcript,
        "alreadyCollected": collected.model_dump(),
    }
    request = StructuredLLMRequest(
        model=model,
        temperature=0,
        max_tokens=256,
        schema_name=ReminderExtractorResponse.__name__,
        messages=[
            LLMMessage(role="system", content=load_prompt("reminder-extractor-v1")),
            LLMMessage(role="user", content=json.dumps(payload, ensure_ascii=False)),
        ],
    )
    try:
        parsed = await provider.generate_structured(request, ReminderExtractorResponse)
    except Exception as error:
        print("Reminder Krutrim extract failed:", {"error": str(error)[:400]})
        return None
    return parsed if isinstance(parsed, ReminderExtractorResponse) else None


async def with_spoken_reply(result: ReminderVoiceTurnResult) -> dict:
    spoken = await synthesize_speech(result.replyText)
    payload = result.model_dump()
    if spoken:
        payload["replyAudioBase64"] = spoken["audioBase64"]
        payload["replyAudioContentType"] = spoken["contentType"]
    else:
        payload["replyAudioBase64"] = None
        payload["replyAudioContentType"] = None
    return payload


def _should_use_llm(transcript: str, collected: ReminderCollected, intent: str) -> bool:
    if intent == "out_of_context" or collected.is_complete():
        return False
    if parseable_follow_up(transcript) or looks_like_reminder(transcript, collected):
        return False
    return intent == "unclear"


def parseable_follow_up(transcript: str) -> bool:
    from services.reminders.dates import parse_date, parse_time

    dummy_now = now_in_timezone("Asia/Kolkata")
    return bool(parse_date(transcript, dummy_now) or parse_time(transcript))


def _pretty_title(value: str) -> str:
    cleaned = " ".join((value or "").strip().split())
    if not cleaned:
        return "Reminder"
    return cleaned[:1].upper() + cleaned[1:]
