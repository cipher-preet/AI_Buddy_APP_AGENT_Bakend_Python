"""Idempotent persistence of extracted events/reminders with AI calling on by default."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any

from bson import ObjectId
from pymongo import ReturnDocument
from pymongo.errors import DuplicateKeyError

from services.conversation.repository import mongo_id_candidates, to_mongo_id
from services.reminders.dates import format_date_label
from services.schedule_extraction.reminder_schedule import ReminderScheduler, build_schedule_fields
from services.schedule_extraction.schemas import PipelineOutcome, ScheduledEvent, ScheduledReminder
from services.schedule_extraction.validation import local_to_utc, shift_local, titles_similar

REMINDER_TONES = ("rose", "lavender", "ochre", "teal")
EVENT_TONES = ("indigo", "violet", "cyan", "teal")
EXTRACTED_BY = "meeting_schedule_extraction_v1"


def _tone(fingerprint: str, tones: tuple[str, ...]) -> str:
    return tones[int(hashlib.sha1(fingerprint.encode()).hexdigest(), 16) % len(tones)]


def _date_label(date_key: str) -> str:
    return format_date_label(datetime.fromisoformat(date_key))


class ScheduleWriter:
    def __init__(
        self,
        database,
        scheduler: ReminderScheduler,
        *,
        ai_calling: bool,
        meeting_remind_before: int,
        deadline_remind_before: int,
        now: datetime | None = None,
    ):
        self.db = database
        self.scheduler = scheduler
        self.ai_calling = ai_calling
        self.meeting_remind_before = meeting_remind_before
        self.deadline_remind_before = deadline_remind_before
        self.now = now or datetime.now(timezone.utc)

    async def _existing_titles(self, collection, user_id: str, date_keys: set[str]) -> list[dict[str, Any]]:
        if not date_keys:
            return []
        cursor = collection.find(
            {"userId": {"$in": mongo_id_candidates(user_id)}, "dateKey": {"$in": sorted(date_keys)}},
            {"title": 1, "dateKey": 1, "sourceFingerprint": 1},
        )
        return await cursor.to_list(length=500)

    @staticmethod
    def _is_user_duplicate(existing: list[dict[str, Any]], title: str, date_key: str, fingerprint: str) -> bool:
        return any(
            doc.get("dateKey") == date_key
            and doc.get("sourceFingerprint") != fingerprint
            and titles_similar(str(doc.get("title") or ""), title)
            for doc in existing
        )

    async def _upsert(self, collection, doc: dict[str, Any]) -> dict[str, Any]:
        try:
            return await collection.find_one_and_update(
                {"sourceFingerprint": doc["sourceFingerprint"]},
                {"$setOnInsert": doc},
                upsert=True,
                return_document=ReturnDocument.AFTER,
            )
        except DuplicateKeyError:
            return await collection.find_one({"sourceFingerprint": doc["sourceFingerprint"]})

    async def _save_reminder(
        self,
        *,
        user_id: str,
        conversation_id: str,
        timezone_name: str,
        fingerprint: str,
        title: str,
        description: str,
        date_key: str,
        time_label: str,
        repeat: str,
        extraction: dict[str, Any],
    ) -> dict[str, Any] | None:
        reminder_id = ObjectId()
        fields = build_schedule_fields(
            date_key=date_key,
            time_label=time_label,
            repeat=repeat,
            timezone_name=timezone_name,
            ai_calling=self.ai_calling,
            beeping=not self.ai_calling,
            now=self.now,
        )
        if fields["deliveryStatus"] != "SCHEDULED":
            return None
        fields["scheduledOccurrenceId"] = self.scheduler.occurrence_for(str(reminder_id), fields)
        doc = {
            "_id": reminder_id,
            "userId": to_mongo_id(user_id),
            "title": title[:80],
            "description": description[:500],
            "dateKey": date_key,
            "dateLabel": _date_label(date_key),
            "timeLabel": time_label,
            "source": "ai",
            "tone": _tone(fingerprint, REMINDER_TONES),
            "repeat": repeat,
            "aiCalling": self.ai_calling,
            "notification": False,
            "beeping": not self.ai_calling,
            "lastTriggerAtUtc": None,
            "lastDeliveredOccurrenceKey": None,
            **fields,
            "sourceConversationId": to_mongo_id(conversation_id),
            "sourceFingerprint": fingerprint,
            "extraction": {"by": EXTRACTED_BY, **extraction},
            "createdAt": self.now,
            "updatedAt": self.now,
        }
        saved = await self._upsert(self.db.reminders, doc)
        if saved is not None:
            await self.scheduler.ensure_scheduled(saved)
        return saved

    async def write_reminders(
        self,
        user_id: str,
        conversation_id: str,
        timezone_name: str,
        reminders: list[ScheduledReminder],
        outcome: PipelineOutcome,
    ) -> list[str]:
        existing = await self._existing_titles(self.db.reminders, user_id, {item.dateKey for item in reminders})
        ids: list[str] = []
        for item in reminders:
            if self._is_user_duplicate(existing, item.title, item.dateKey, item.fingerprint):
                outcome.drop("already_exists")
                continue
            saved = await self._save_reminder(
                user_id=user_id,
                conversation_id=conversation_id,
                timezone_name=timezone_name,
                fingerprint=item.fingerprint,
                title=item.title,
                description=item.description,
                date_key=item.dateKey,
                time_label=item.timeLabel,
                repeat=item.repeat,
                extraction={
                    "pipeline": "reminders",
                    "confidence": item.confidence,
                    "evidence": item.evidence,
                    "timeInferred": item.timeInferred,
                },
            )
            if saved is None:
                outcome.drop("not_schedulable")
                continue
            ids.append(str(saved["_id"]))
        return ids

    def _linked_trigger(self, event: ScheduledEvent, timezone_name: str) -> tuple[str, str]:
        before = self.deadline_remind_before if event.kind == "deadline" else self.meeting_remind_before
        date_key, time_label = shift_local(event.dateKey, event.startTimeLabel, -before)
        trigger = local_to_utc(date_key, time_label, timezone_name)
        if trigger is None or trigger < self.now:
            return event.dateKey, event.startTimeLabel
        return date_key, time_label

    async def write_events(
        self,
        user_id: str,
        conversation_id: str,
        timezone_name: str,
        events: list[ScheduledEvent],
        outcome: PipelineOutcome,
    ) -> list[str]:
        existing = await self._existing_titles(self.db.calendar_events, user_id, {item.dateKey for item in events})
        ids: list[str] = []
        for event in events:
            if self._is_user_duplicate(existing, event.title, event.dateKey, event.fingerprint):
                outcome.drop("already_exists")
                continue
            before = self.deadline_remind_before if event.kind == "deadline" else self.meeting_remind_before
            trigger_date, trigger_time = self._linked_trigger(event, timezone_name)
            prefix = "Due" if event.kind == "deadline" else "Starting soon"
            reminder = await self._save_reminder(
                user_id=user_id,
                conversation_id=conversation_id,
                timezone_name=timezone_name,
                fingerprint=f"{event.fingerprint}:reminder",
                title=event.title,
                description=event.description or f"{prefix}: {event.title} at {event.startTimeLabel}",
                date_key=trigger_date,
                time_label=trigger_time,
                repeat="once",
                extraction={"pipeline": "calendar_events", "linkedEvent": True, "confidence": event.confidence},
            )
            doc = {
                "_id": ObjectId(),
                "userId": to_mongo_id(user_id),
                "title": event.title,
                "description": event.description,
                "location": event.location,
                "dateKey": event.dateKey,
                "dateLabel": _date_label(event.dateKey),
                "startTimeLabel": event.startTimeLabel,
                "endTimeLabel": event.endTimeLabel,
                "tone": _tone(event.fingerprint, EVENT_TONES),
                "aiReminder": reminder is not None,
                "aiCalling": self.ai_calling and reminder is not None,
                "notification": False,
                "beeping": not self.ai_calling and reminder is not None,
                "remindBeforeMinutes": before if reminder is not None else 0,
                "reminderId": reminder["_id"] if reminder is not None else None,
                "source": "ai",
                "kind": event.kind,
                "sourceConversationId": to_mongo_id(conversation_id),
                "sourceFingerprint": event.fingerprint,
                "extraction": {
                    "by": EXTRACTED_BY,
                    "pipeline": "calendar_events",
                    "confidence": event.confidence,
                    "evidence": event.evidence,
                    "timeInferred": event.timeInferred,
                },
                "createdAt": self.now,
                "updatedAt": self.now,
            }
            saved = await self._upsert(self.db.calendar_events, doc)
            if saved is not None:
                ids.append(str(saved["_id"]))
        return ids
