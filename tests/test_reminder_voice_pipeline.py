from datetime import datetime, timedelta, timezone
import asyncio

from services.reminders.dates import format_time_label, now_in_timezone, parse_date, parse_time
from services.reminders.extract import (
    clean_reminder_transcript,
    extract_title,
    merge_collected,
    rule_extract,
)
from services.reminders.pipeline import decide_reminder_turn
from services.reminders.prompts import ASK_DATE, ASK_TIME, ASK_TITLE, GREETING, OUT_OF_CONTEXT, prompt_text
from services.reminders.schemas import ReminderCollected


IST = timezone(timedelta(hours=5, minutes=30))
NOW = datetime(2026, 8, 30, 10, 0, tzinfo=IST)


def test_greeting_copy_is_hardcoded():
    assert GREETING == "Please tell me what reminder I should set."


def test_strips_deepgram_speaker_labels():
    cleaned = clean_reminder_transcript(
        "[Speaker 0] set reminder for tomorrow at 6am"
    )
    assert cleaned == "set reminder for tomorrow at 6am"
    assert "speaker" not in cleaned.casefold()


def test_speaker_label_is_not_saved_as_title():
    result = asyncio.run(
        decide_reminder_turn(
            transcript="[Speaker 0] set reminder for tomorrow at 6am",
            collected=ReminderCollected(),
            now=NOW,
            llm_extractor=_fail_if_called,
        )
    )

    assert result.status == "need_more"
    assert result.replyText == ASK_TITLE
    assert result.collected.title is None
    assert result.reminder is None
    assert "speaker" not in (result.transcript or "").casefold()


def test_speaker_label_does_not_block_real_title():
    result = asyncio.run(
        decide_reminder_turn(
            transcript="[Speaker 0] remind me to call mom tomorrow at 6am",
            collected=ReminderCollected(),
            now=NOW,
            llm_extractor=_fail_if_called,
        )
    )

    assert result.status == "ready"
    assert result.reminder is not None
    assert result.reminder.title.lower() == "call mom"
    assert "speaker" not in result.reminder.title.casefold()
    assert "speaker" not in result.reminder.description.casefold()


def test_parses_tomorrow_at_6am():
    extracted = rule_extract("set reminder for tomorrow at 6am", NOW)

    assert extracted.dateKey == "2026-08-31"
    assert extracted.timeLabel == "6:00 AM"
    assert extracted.title is None


def test_parses_call_mom_tomorrow_evening_time():
    extracted = rule_extract("remind me to call mom tomorrow at 6:30 pm", NOW)

    assert extracted.title == "call mom"
    assert extracted.dateKey == "2026-08-31"
    assert extracted.timeLabel == "6:30 PM"


def test_title_strips_date_and_time():
    assert extract_title("Remind me to take medicine tomorrow at 7 am") == "take medicine"


def test_does_not_invent_time_for_morning():
    extracted = rule_extract("remind me to walk the dog tomorrow morning", NOW)

    assert extracted.title == "walk the dog"
    assert extracted.dateKey == "2026-08-31"
    assert extracted.timeLabel is None


def test_time_label_matches_backend_pattern():
    assert format_time_label(6, 0) == "6:00 AM"
    assert format_time_label(18, 5) == "6:05 PM"
    assert format_time_label(0, 0) == "12:00 AM"
    assert format_time_label(12, 0) == "12:00 PM"


def test_numeric_date_without_year_rolls_forward():
    parsed = parse_date("on 3 January", NOW)

    assert parsed is not None
    assert parsed[0] == "2027-01-03"


def test_merge_keeps_previous_fields():
    previous = ReminderCollected(title="Call mom", dateKey="2026-08-31", dateLabel="31 Aug 2026")
    incoming = ReminderCollected(timeLabel="6:00 AM")

    merged = merge_collected(previous, incoming)

    assert merged.title == "Call mom"
    assert merged.dateKey == "2026-08-31"
    assert merged.timeLabel == "6:00 AM"


def test_complete_utterance_is_ready_without_llm():
    result = asyncio.run(
        decide_reminder_turn(
            transcript="remind me to call mom tomorrow at 6am",
            collected=ReminderCollected(),
            now=NOW,
            llm_extractor=_fail_if_called,
        )
    )

    assert result.status == "ready"
    assert result.usedLlm is False
    assert result.reminder is not None
    assert result.reminder.title.lower() == "call mom"
    assert result.reminder.dateKey == "2026-08-31"
    assert result.reminder.timeLabel == "6:00 AM"
    assert result.reminder.source == "ai"
    assert result.reminder.notification is True
    assert result.reminder.aiCalling is False
    assert "call mom" in result.replyText.lower()


def test_missing_time_asks_hardcoded_follow_up():
    result = asyncio.run(
        decide_reminder_turn(
            transcript="remind me to drink water tomorrow",
            collected=ReminderCollected(),
            now=NOW,
            llm_extractor=_fail_if_called,
        )
    )

    assert result.status == "need_more"
    assert result.replyText == ASK_TIME
    assert result.collected.title == "drink water"
    assert result.collected.dateKey == "2026-08-31"
    assert result.reminder is None


def test_missing_date_asks_hardcoded_follow_up():
    result = asyncio.run(
        decide_reminder_turn(
            transcript="remind me to pay rent at 9am",
            collected=ReminderCollected(),
            now=NOW,
            llm_extractor=_fail_if_called,
        )
    )

    assert result.status == "need_more"
    assert result.replyText == ASK_DATE
    assert result.collected.timeLabel == "9:00 AM"
    assert result.reminder is None


def test_missing_title_asks_what_to_remind():
    result = asyncio.run(
        decide_reminder_turn(
            transcript="set a reminder for tomorrow at 6am",
            collected=ReminderCollected(),
            now=NOW,
            llm_extractor=_fail_if_called,
        )
    )

    assert result.status == "need_more"
    assert result.replyText == ASK_TITLE
    assert result.collected.dateKey == "2026-08-31"
    assert result.collected.timeLabel == "6:00 AM"
    assert result.reminder is None


def test_setup_phrase_without_action_asks_for_title():
    extracted = rule_extract("set the reminder for tomorrow at 6am", NOW)

    assert extracted.title is None
    assert extracted.dateKey == "2026-08-31"
    assert extracted.timeLabel == "6:00 AM"

    result = asyncio.run(
        decide_reminder_turn(
            transcript="set the reminder for tomorrow at 6am",
            collected=ReminderCollected(),
            now=NOW,
            llm_extractor=_fail_if_called,
        )
    )

    assert result.status == "need_more"
    assert result.replyText == ASK_TITLE
    assert result.reminder is None


def test_title_follow_up_completes_after_date_and_time():
    previous = ReminderCollected(
        dateKey="2026-08-31",
        dateLabel="31 Aug 2026",
        timeLabel="6:00 AM",
    )
    result = asyncio.run(
        decide_reminder_turn(
            transcript="call mom",
            collected=previous,
            now=NOW,
            llm_extractor=_fail_if_called,
        )
    )

    assert result.status == "ready"
    assert result.reminder is not None
    assert result.reminder.title.lower() == "call mom"


def test_follow_up_time_completes_without_writing_early():
    previous = ReminderCollected(
        title="Call mom",
        description="Call mom",
        dateKey="2026-08-31",
        dateLabel="31 Aug 2026",
    )

    first = asyncio.run(
        decide_reminder_turn(
            transcript="remind me to call mom tomorrow",
            collected=ReminderCollected(),
            now=NOW,
            llm_extractor=_fail_if_called,
        )
    )
    assert first.status == "need_more"
    assert first.reminder is None

    second = asyncio.run(
        decide_reminder_turn(
            transcript="6 in the morning",
            collected=previous,
            now=NOW,
            llm_extractor=_fail_if_called,
        )
    )

    assert second.status == "ready"
    assert second.reminder is not None
    assert second.reminder.timeLabel == "6:00 AM"


def test_out_of_context_keeps_collected_and_replies_gently():
    previous = ReminderCollected(title="Call mom")
    result = asyncio.run(
        decide_reminder_turn(
            transcript="what's the weather today",
            collected=previous,
            now=NOW,
            llm_extractor=_fail_if_called,
        )
    )

    assert result.status == "out_of_context"
    assert OUT_OF_CONTEXT in result.replyText
    assert result.collected.title == "Call mom"
    assert result.reminder is None


def test_unclear_speech_does_not_save():
    result = asyncio.run(
        decide_reminder_turn(
            transcript="hmm okay maybe later",
            collected=ReminderCollected(),
            now=NOW,
            llm_extractor=_noop_llm,
        )
    )

    assert result.status == "unclear"
    assert result.reminder is None


async def _fail_if_called(*_args, **_kwargs):
    raise AssertionError("LLM should not be called for this reminder turn")


async def _noop_llm(*_args, **_kwargs):
    return None


def test_parse_time_ignores_year_digits():
    assert parse_time("tomorrow 2026") is None
    assert parse_time("at 6am") == "6:00 AM"
    assert parse_time("18:45") == "6:45 PM"


def test_asia_kolkata_does_not_crash_without_iana_tzdata(monkeypatch):
    import services.reminders.dates as dates

    def boom(_name):
        raise KeyError("No time zone found with key Asia/Kolkata")

    monkeypatch.setattr(dates, "ZoneInfo", boom)

    now = now_in_timezone("Asia/Kolkata")

    assert now.utcoffset() == timedelta(hours=5, minutes=30)


def test_unknown_timezone_falls_back_to_ist(monkeypatch):
    import services.reminders.dates as dates

    def boom(_name):
        raise KeyError("No time zone found")

    monkeypatch.setattr(dates, "ZoneInfo", boom)

    now = now_in_timezone("Not/ARealZone")

    assert now.tzinfo is not None
    assert now.utcoffset() == timedelta(hours=5, minutes=30)


def test_hinglish_full_utterance_is_ready_without_llm():
    result = asyncio.run(
        decide_reminder_turn(
            transcript="kal subah 6 baje mummy ko call karna",
            collected=ReminderCollected(),
            now=NOW,
            llm_extractor=_fail_if_called,
        )
    )

    assert result.status == "ready"
    assert result.reminder is not None
    assert result.reminder.title.lower() == "mummy ko call karna"
    assert result.reminder.dateKey == "2026-08-31"
    assert result.reminder.timeLabel == "6:00 AM"
    assert result.language == "hi"
    assert "रिमाइंडर" in result.replyText


def test_devanagari_full_utterance_is_ready_without_llm():
    result = asyncio.run(
        decide_reminder_turn(
            transcript="कल सुबह 6 बजे मम्मी को फोन लगाना",
            collected=ReminderCollected(),
            now=NOW,
            llm_extractor=_fail_if_called,
        )
    )

    assert result.status == "ready"
    assert result.reminder is not None
    assert "मम्मी" in result.reminder.title
    assert result.reminder.dateKey == "2026-08-31"
    assert result.reminder.timeLabel == "6:00 AM"
    assert result.language == "hi"


def test_hindi_missing_title_asks_in_hindi():
    result = asyncio.run(
        decide_reminder_turn(
            transcript="कल शाम 7 बजे",
            collected=ReminderCollected(),
            now=NOW,
            llm_extractor=_fail_if_called,
        )
    )

    assert result.status == "need_more"
    assert result.replyText == prompt_text("ask_title", "hi")
    assert result.collected.title is None
    assert result.collected.dateKey == "2026-08-31"
    assert result.collected.timeLabel == "7:00 PM"
    assert result.reminder is None


def test_hinglish_setup_phrase_is_not_a_title():
    result = asyncio.run(
        decide_reminder_turn(
            transcript="mujhe kal 6 baje yaad dila dena",
            collected=ReminderCollected(),
            now=NOW,
            llm_extractor=_fail_if_called,
        )
    )

    assert result.status == "need_more"
    assert result.replyText == prompt_text("ask_title", "hi")
    assert result.collected.title is None
    assert result.collected.dateKey == "2026-08-31"
    assert result.collected.timeLabel == "6:00 AM"


def test_hindi_title_follow_up_completes_reminder():
    previous = ReminderCollected(
        dateKey="2026-08-31",
        dateLabel="31 Aug 2026",
        timeLabel="6:00 AM",
        language="hi",
    )
    result = asyncio.run(
        decide_reminder_turn(
            transcript="mummy ko call karna",
            collected=previous,
            now=NOW,
            llm_extractor=_fail_if_called,
        )
    )

    assert result.status == "ready"
    assert result.reminder is not None
    assert result.reminder.title.lower() == "mummy ko call karna"
    assert result.language == "hi"


def test_hindi_speaker_label_is_stripped():
    result = asyncio.run(
        decide_reminder_turn(
            transcript="[Speaker 0] kal subah 6 baje mummy ko call karna",
            collected=ReminderCollected(),
            now=NOW,
            llm_extractor=_fail_if_called,
        )
    )

    assert result.status == "ready"
    assert result.reminder is not None
    assert "speaker" not in result.reminder.title.casefold()


def test_parse_hindi_clock_and_relative_days():
    assert parse_time("subah 6 baje") == "6:00 AM"
    assert parse_time("shaam 7 baje") == "7:00 PM"
    assert parse_time("6 baje") == "6:00 AM"
    assert parse_date("kal milna", NOW)[0] == "2026-08-31"
    assert parse_date("आज शाम", NOW)[0] == "2026-08-30"
    assert extract_title("kal subah 6 baje mummy ko call karna") == "mummy ko call karna"
