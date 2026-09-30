"""Deterministic checks that turn raw LLM candidates into safe, deduplicated schedule items."""

from __future__ import annotations

import hashlib
import re
from datetime import date, datetime, timedelta, timezone

from services.reminders.dates import format_time_label, parse_date, parse_time
from services.reminders.occurrence import parse_time_label, zoned_local_to_utc
from services.schedule_extraction.schemas import (
    EventCandidate,
    PipelineOutcome,
    ReminderCandidate,
    ScheduledEvent,
    ScheduledReminder,
)

MAX_DAYS_AHEAD = 366
DEFAULT_EVENT_TIME = "10:00 AM"
DEFAULT_REMINDER_TIME = "9:00 AM"
TIMED_EVENT_MINUTES = 60
DEADLINE_EVENT_MINUTES = 30
PAST_GRACE = timedelta(minutes=5)
TITLE_SIMILARITY = 0.6
GROUNDING_COVERAGE = 0.75

_SEQ_PREFIX = re.compile(r"\[\d+\]\s*")
_NON_WORD = re.compile(r"[^\w\s]", re.UNICODE)
_STOPWORDS = frozenset(
    "a an the to of for with and or on at in by is are be my me our we i you your this that "
    "ko ka ki ke se hai hain par pe aur meeting call reminder remind".split()
)


def _fold(text: str) -> str:
    text = _SEQ_PREFIX.sub(" ", text or "")
    return " ".join(_NON_WORD.sub(" ", text.casefold()).split())


def title_tokens(title: str) -> frozenset[str]:
    return frozenset(token for token in _fold(title).split() if token not in _STOPWORDS and len(token) > 1)


def titles_similar(left: str, right: str) -> bool:
    a, b = title_tokens(left), title_tokens(right)
    if not a or not b:
        return _fold(left) == _fold(right)
    if a <= b or b <= a:
        return True
    return len(a & b) / len(a | b) >= TITLE_SIMILARITY


def is_grounded(evidence: str, folded_transcript: str, transcript_tokens: frozenset[str]) -> bool:
    folded = _fold(evidence)
    if not folded:
        return False
    if folded in folded_transcript:
        return True
    tokens = [token for token in folded.split() if len(token) > 1]
    if len(tokens) < 3:
        return False
    return sum(1 for token in tokens if token in transcript_tokens) / len(tokens) >= GROUNDING_COVERAGE


def normalize_time(text: str) -> str | None:
    raw = (text or "").strip()
    if not raw:
        return None
    if parse_time_label(raw):
        hour, minute = parse_time_label(raw)
        return format_time_label(hour, minute)
    return parse_time(raw)


def resolve_date(date_key: str, date_text: str, recorded_at: datetime) -> str | None:
    base = recorded_at.date()
    try:
        value = date.fromisoformat(date_key) if len(date_key or "") == 10 else None
    except ValueError:
        value = None
    if value is None and date_text:
        parsed = parse_date(date_text, recorded_at)
        value = date.fromisoformat(parsed[0]) if parsed else None
    if value is None or value < base or value > base + timedelta(days=MAX_DAYS_AHEAD):
        return None
    return value.isoformat()


def local_to_utc(date_key: str, time_label: str, timezone_name: str) -> datetime | None:
    return zoned_local_to_utc(date_key, time_label, timezone_name)


def add_minutes(time_label: str, minutes: int) -> str:
    parsed = parse_time_label(time_label)
    if not parsed:
        return time_label
    total = (parsed[0] * 60 + parsed[1] + minutes) % (24 * 60)
    return format_time_label(total // 60, total % 60)


def shift_local(date_key: str, time_label: str, minutes: int) -> tuple[str, str]:
    parsed = parse_time_label(time_label)
    if not parsed:
        return date_key, time_label
    moment = datetime.fromisoformat(date_key) + timedelta(hours=parsed[0], minutes=parsed[1] + minutes)
    return moment.date().isoformat(), format_time_label(moment.hour, moment.minute)


def fingerprint(conversation_id: str, kind: str, title: str, date_key: str) -> str:
    key = "|".join([conversation_id, kind, " ".join(sorted(title_tokens(title))) or _fold(title), date_key])
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


class CandidateValidator:
    def __init__(
        self,
        conversation_id: str,
        transcript: str,
        recorded_at_local: datetime,
        timezone_name: str,
        min_confidence: float,
        max_items: int,
        now: datetime | None = None,
    ):
        self.conversation_id = conversation_id
        self.folded_transcript = _fold(transcript)
        self.transcript_tokens = frozenset(self.folded_transcript.split())
        self.recorded_at = recorded_at_local
        self.timezone_name = timezone_name
        self.min_confidence = min_confidence
        self.max_items = max_items
        self.now = now or datetime.now(timezone.utc)

    def _common(self, candidate, outcome: PipelineOutcome) -> str | None:
        if not candidate.title:
            outcome.drop("no_title")
            return None
        if candidate.confidence < self.min_confidence:
            outcome.drop("low_confidence")
            return None
        if not is_grounded(candidate.evidence, self.folded_transcript, self.transcript_tokens):
            outcome.drop("ungrounded")
            return None
        date_key = resolve_date(candidate.dateKey, candidate.dateText, self.recorded_at)
        if date_key is None:
            outcome.drop("invalid_date")
        return date_key

    def _in_past(self, date_key: str, time_label: str) -> bool:
        moment = local_to_utc(date_key, time_label, self.timezone_name)
        return moment is None or moment < self.now - PAST_GRACE

    def events(self, candidates: list[EventCandidate], outcome: PipelineOutcome) -> list[ScheduledEvent]:
        kept: list[ScheduledEvent] = []
        for candidate in candidates:
            date_key = self._common(candidate, outcome)
            if date_key is None:
                continue
            start = normalize_time(candidate.startTime) or normalize_time(candidate.timeText)
            inferred = start is None
            start = start or DEFAULT_EVENT_TIME
            if self._in_past(date_key, start):
                outcome.drop("in_past")
                continue
            span = DEADLINE_EVENT_MINUTES if candidate.kind == "deadline" else TIMED_EVENT_MINUTES
            end = normalize_time(candidate.endTime)
            if not end or not _after(start, end):
                end = add_minutes(start, span)
            item = ScheduledEvent(
                kind=candidate.kind,
                title=candidate.title[:80],
                description=candidate.description[:500],
                location=candidate.location[:120],
                dateKey=date_key,
                startTimeLabel=start,
                endTimeLabel=end,
                timeInferred=inferred,
                evidence=candidate.evidence[:500],
                confidence=candidate.confidence,
            )
            item.fingerprint = fingerprint(self.conversation_id, "event", item.title, date_key)
            _merge(kept, item, outcome)
        return _cap(kept, self.max_items, outcome)

    def reminders(self, candidates: list[ReminderCandidate], outcome: PipelineOutcome) -> list[ScheduledReminder]:
        kept: list[ScheduledReminder] = []
        for candidate in candidates:
            date_key = self._common(candidate, outcome)
            if date_key is None:
                continue
            time_label = normalize_time(candidate.time) or normalize_time(candidate.timeText)
            inferred = time_label is None
            time_label = time_label or DEFAULT_REMINDER_TIME
            if candidate.repeat == "once" and self._in_past(date_key, time_label):
                outcome.drop("in_past")
                continue
            item = ScheduledReminder(
                title=candidate.title[:80],
                description=candidate.description[:500],
                dateKey=date_key,
                timeLabel=time_label,
                repeat=candidate.repeat,
                timeInferred=inferred,
                evidence=candidate.evidence[:500],
                confidence=candidate.confidence,
            )
            item.fingerprint = fingerprint(self.conversation_id, "reminder", item.title, date_key)
            _merge(kept, item, outcome)
        return _cap(kept, self.max_items, outcome)


def drop_reminders_covered_by_events(
    reminders: list[ScheduledReminder],
    events: list[ScheduledEvent],
    outcome: PipelineOutcome,
) -> list[ScheduledReminder]:
    kept = []
    for reminder in reminders:
        if any(event.dateKey == reminder.dateKey and titles_similar(event.title, reminder.title) for event in events):
            outcome.drop("covered_by_event")
            continue
        kept.append(reminder)
    return kept


def _after(start: str, end: str) -> bool:
    a, b = parse_time_label(start), parse_time_label(end)
    return bool(a and b) and (b[0] * 60 + b[1]) > (a[0] * 60 + a[1])


def _merge(kept: list, item, outcome: PipelineOutcome) -> None:
    for index, existing in enumerate(kept):
        if existing.dateKey == item.dateKey and titles_similar(existing.title, item.title):
            outcome.drop("duplicate")
            if _better(item, existing):
                kept[index] = item
            return
    kept.append(item)


def _better(candidate, existing) -> bool:
    if existing.timeInferred and not candidate.timeInferred:
        return True
    if candidate.timeInferred and not existing.timeInferred:
        return False
    return candidate.confidence > existing.confidence


def _cap(items: list, limit: int, outcome: PipelineOutcome) -> list:
    if len(items) <= limit:
        return items
    ranked = sorted(items, key=lambda item: item.confidence, reverse=True)
    for _ in ranked[limit:]:
        outcome.drop("over_limit")
    return ranked[:limit]
