"""Durable attendance and recording imports for KukuNotes meeting bots."""
from __future__ import annotations

import asyncio
import logging
import os
import re
from contextlib import suppress
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import urlsplit
from uuid import uuid4

import httpx
from pymongo import ReturnDocument
from pymongo.errors import DuplicateKeyError

from apps.api_gateway.workers.meeting_bot_media import (
    InvalidRecording, audio_chunk_duration, extract_chunks, store_audio_chunk, store_video,
)
from services.db.mongo import get_database


log = logging.getLogger(__name__)
LEASE_SECONDS = 90
RECOVERY_DELAY = 180
ATTEMPT_LIMIT = 8
MAX_PENDING_AUDIO_CHUNKS = 8
LIVE_STATUSES = {"in_call_not_recording", "in_call_recording", "call_ended", "done"}
ATTENDANCE_POLL_SECONDS = 15


def now() -> datetime:
    return datetime.now(timezone.utc)


def as_utc(value: str | datetime) -> datetime:
    when = datetime.fromisoformat(value.replace("Z", "+00:00")) if isinstance(value, str) else value
    return when.replace(tzinfo=timezone.utc) if when.tzinfo is None else when.astimezone(timezone.utc)


def recording_media(bot: dict) -> tuple[str, str | None] | None:
    records = bot.get("recordings") or []
    assets = []
    for record in records:
        video = (record.get("media_shortcuts") or {}).get("video_mixed") or {}
        url = (video.get("data") or {}).get("download_url")
        if url:
            assets.append((url, record.get("started_at")))
    if len(assets) > 1:
        raise InvalidRecording("This bot produced multiple recordings. The combined playback is unavailable.")
    return assets[0] if assets else None


async def store_meeting_card_thumbnail(db, meeting_id, user_id: str, video: Path) -> None:
    """Save one frame for the meetings card, the same asset the Chrome extension uses."""
    try:
        existing = await db.meeting_sessions.find_one({"_id": meeting_id}, {"thumbnailS3Key": 1})
        if existing and existing.get("thumbnailS3Key"):
            return
        from services.meeting_extension.ffmpeg_audio import extract_video_thumbnail
        from services.meeting_extension.s3_keys import meeting_thumbnail_object_key, validate_meeting_object_key
        from services.storage.s3_audio_storage import get_s3_audio_storage

        best: Path | None = None
        best_size = 0
        for seek in (8, 30, 60, 1):
            created = await extract_video_thumbnail(video, video.with_name(f"thumbnail-{seek}.jpg"), seek_seconds=seek)
            if created is not None and created.stat().st_size > best_size:
                best = created
                best_size = created.stat().st_size
        if best is None:
            return
        thumbnail_key = meeting_thumbnail_object_key(user_id, str(meeting_id))
        validate_meeting_object_key(object_key=thumbnail_key, user_id=user_id, meeting_session_id=str(meeting_id))
        await get_s3_audio_storage().upload_file(best, thumbnail_key, content_type="image/jpeg")
        await db.meeting_sessions.update_one(
            {"_id": meeting_id, "$or": [{"thumbnailS3Key": None}, {"thumbnailS3Key": {"$exists": False}}]},
            {"$set": {"thumbnailS3Key": thumbnail_key, "updatedAt": now()}},
        )
    except Exception as exc:
        log.warning("meeting_bot_thumbnail_failed error=%s", type(exc).__name__)


def recorded_audio(run: dict, bot: dict) -> bool:
    return bool(run.get("startedAt") or any(
        change.get("code") == "in_call_recording" for change in bot.get("status_changes") or []
    ))


class LeaseExpired(Exception):
    pass


class WorkerRejected(Exception):
    pass


class MeetingBotWorker:
    def __init__(self, http: httpx.AsyncClient):
        self.http = http
        self.db = get_database()
        self.runs = self.db.meeting_bot_runs
        self.key = os.environ.get("MEETING_BOT_RECALL_API_KEY", "").strip()
        region = os.environ.get("MEETING_BOT_RECALL_REGION", "us-west-2").strip()
        if region not in {"us-west-2", "us-east-1", "eu-central-1", "ap-northeast-1"}:
            raise ValueError("Unsupported Recall region")
        self.recall_url = f"https://{region}.recall.ai/api/v1"
        self.node_url = os.environ.get("MEETING_BOT_NODE_API_URL", "").rstrip("/")
        self.token = os.environ.get("MEETING_BOT_WORKER_TOKEN", "")
        target = urlsplit(self.node_url)
        secure = target.scheme == "https" or (target.scheme == "http" and target.hostname in {"localhost", "127.0.0.1"})
        missing = []
        if not self.key:
            missing.append("MEETING_BOT_RECALL_API_KEY")
        if len(self.token) < 32:
            missing.append("MEETING_BOT_WORKER_TOKEN")
        if not self.node_url or not secure or target.username or target.password or target.query or target.fragment:
            missing.append("MEETING_BOT_NODE_API_URL")
        if missing:
            raise ValueError("Meeting bot worker configuration is invalid: " + ", ".join(missing))

    async def recall(self, method: str, path: str, **kwargs) -> httpx.Response:
        return await self.http.request(
            method, self.recall_url + path, headers={"Authorization": f"Token {self.key}"}, **kwargs,
        )

    async def node(self, run: dict, action: str, data: dict | None = None) -> dict:
        response = await self.http.post(
            f"{self.node_url}/meeting-bots/worker/{run['_id']}/{action}",
            json=data or {}, headers={"Authorization": f"Bearer {self.token}", "X-Bot-Lease": run["lease"]},
        )
        if response.status_code in {400, 403, 410, 413}:
            raise WorkerRejected(response.json().get("message") or "The recording could not be registered.")
        response.raise_for_status()
        return response.json()["data"]

    async def save(self, run: dict, **changes):
        result = await self.runs.update_one(
            {"_id": run["_id"], "lease": run["lease"], "leaseUntil": {"$gt": now()}},
            {"$set": {**changes, "updatedAt": now()}},
        )
        if not result.matched_count:
            raise LeaseExpired

    async def defer_attendance(self, run: dict, until: datetime):
        # A callback or stop request can make the run due while the worker reads Recall.
        await self.runs.update_one({
            "_id": run["_id"], "lease": run["lease"], "leaseUntil": {"$gt": now()},
            "providerStatusAt": run.get("providerStatusAt"), "stopRequested": bool(run.get("stopRequested")),
        }, {"$set": {"nextAt": until, "attempts": 0, "updatedAt": now()}})

    async def terminal(self, run: dict, state: str, message: str | None = None):
        await self.save(run, state=state, active=False, activeCapture=False, message=message or run.get("message"), url=None)
        if state == "failed" and run["state"] == "importing":
            await self.db.meeting_sessions.update_one(
                {"_id": run["_id"], "captureSource": "meeting_bot", "pipelineStopEnqueued": {"$ne": True},
                 "status": {"$in": ["RECORDING", "INTERRUPTED", "STOP_REQUESTED", "WAITING_FOR_UPLOADS"]}},
                {"$set": {"status": "FAILED", "lastErrorMessage": message, "updatedAt": now()}},
            )

    async def heartbeat(self, run: dict, operation: asyncio.Task):
        while True:
            await asyncio.sleep(LEASE_SECONDS / 3)
            try:
                await self.save(run, leaseUntil=now() + timedelta(seconds=LEASE_SECONDS))
            except Exception:
                operation.cancel()
                return

    async def dispatch(self, run: dict):
        if run.get("providerId"):
            await self.save(run, state="attendance", attempts=0, nextAt=now())
            return
        if run.get("dispatchStartedAt"):
            response = await self.recall("GET", "/bot/", params={"metadata__kukunotes_run": str(run["_id"])})
            response.raise_for_status()
            matches = response.json().get("results") or []
            match = next((bot for bot in matches if (bot.get("metadata") or {}).get("kukunotes_run") == str(run["_id"])), None)
            if match:
                await self.save(run, providerId=match["id"], state="attendance", attempts=0, nextAt=now())
            elif now() - as_utc(run["dispatchStartedAt"]) < timedelta(minutes=5):
                await self.save(run, nextAt=now() + timedelta(seconds=20))
            else:
                await self.terminal(run, "failed", "The recording service did not confirm the bot request.")
            return
        if run.get("stopRequested"):
            await self.terminal(run, "cancelled")
            return
        try:
            await self.save(run, activeCapture=True)
        except DuplicateKeyError:
            if now() > as_utc(run["joinAt"]) + timedelta(minutes=5):
                await self.terminal(run, "failed", "Another recording bot was active when this meeting started.")
            else:
                await self.save(run, nextAt=now() + timedelta(seconds=25))
            return
        try:
            checked = await self.node(run, "check")
        except WorkerRejected as error:
            await self.terminal(run, "cancelled" if run.get("eventId") else "failed", str(error))
            return
        join_at = as_utc(checked.get("joinAt") or run["joinAt"])
        if now() > join_at + timedelta(minutes=5):
            await self.terminal(run, "failed", "The scheduled join time has passed.")
            return
        if run.get("eventId") and join_at > now() + timedelta(seconds=5):
            await self.save(run, joinAt=join_at, nextAt=join_at, activeCapture=False, url=checked["url"], provider=checked["provider"])
            return
        latest = await self.runs.find_one({"_id": run["_id"], "lease": run["lease"]})
        if not latest:
            raise LeaseExpired
        if latest.get("stopRequested"):
            await self.terminal(run, "cancelled")
            return
        await self.save(run, dispatchStartedAt=now(), provider=checked.get("provider") or run["provider"], joinAt=join_at)
        payload = {
            "meeting_url": checked.get("url") or run["url"],
            "bot_name": "KukuNotes Recorder",
            "metadata": {"kukunotes_run": str(run["_id"])},
            "recording_config": {
                "video_mixed_layout": "gallery_view",
                "video_mixed_mp4": {},
                "video_mixed_participant_video_when_screenshare": "overlap",
            },
            "chat": {"on_bot_join": {"send_to": "everyone", "message": "KukuNotes is recording this meeting for a transcript and notes."}},
            "automatic_leave": {
                "waiting_room_timeout": 600, "noone_joined_timeout": 300,
                "in_call_not_recording_timeout": 300,
                "recording_permission_denied_timeout": 30, "in_call_recording_timeout": checked["maxSeconds"],
            },
        }
        if join_at > now() + timedelta(seconds=5):
            payload["join_at"] = join_at.isoformat()
        response = await self.recall("POST", "/bot/", json=payload)
        if response.status_code in {429, 507}:
            await self.save(run, dispatchStartedAt=None, activeCapture=False, attempts=run.get("attempts", 0) + 1,
                            nextAt=now() + timedelta(seconds=30))
            if run.get("attempts", 0) >= ATTEMPT_LIMIT:
                await self.terminal(run, "failed", "The recording service ran out of bot capacity.")
            return
        if 400 <= response.status_code < 500:
            await self.terminal(run, "failed", "The recording service rejected this join request.")
            return
        response.raise_for_status()
        await self.save(run, providerId=response.json()["id"], state="attendance", attempts=0, nextAt=now())

    async def attendance(self, run: dict):
        provider_id = run["providerId"]
        if not re.fullmatch(r"[a-fA-F0-9-]{36}", provider_id):
            raise InvalidRecording("The recording service returned an invalid bot ID.")
        path = f"/bot/{provider_id}/"
        join_at = as_utc(run["joinAt"])
        state = run.get("providerStatus")
        if run.get("stopRequested") and not run.get("stopSent") and state not in {"call_ended", "done", "fatal"}:
            if join_at > now() and state is None:
                response = await self.recall("DELETE", path)
                if response.is_success or response.status_code == 404:
                    await self.terminal(run, "cancelled")
                    return
                if response.status_code not in {400, 409}:
                    response.raise_for_status()
            response = await self.recall("POST", path + "leave_call/")
            if response.status_code == 404:
                await self.terminal(run, "cancelled")
                return
            if response.status_code not in {400, 409}:
                response.raise_for_status()
                await self.save(run, stopSent=True)
        if state == "done":
            await self.ensure_live_session(run, state)
            await self.save(run, state="importing", activeCapture=False, attempts=0, nextAt=now(),
                            sessionPublished=bool(run.get("sessionPublished")))
            return
        grace = as_utc(run.get("providerStatusAt") or join_at) + timedelta(seconds=RECOVERY_DELAY)
        # Until the dashboard row exists, poll Recall directly. Webhooks are not required.
        if now() < grace and not run.get("stopRequested") and run.get("sessionPublished"):
            await self.defer_attendance(run, grace)
            return
        response = await self.recall("GET", path)
        if response.status_code == 404:
            await self.terminal(run, "failed", "The recording service could not find the bot.")
            return
        response.raise_for_status()
        bot = response.json()
        statuses = bot.get("status_changes") or []
        last = (statuses[-1].get("code") if statuses else None) or state
        if last == "done":
            await self.ensure_live_session(run, last)
            await self.mark_call_ended(run)
            await self.save(run, state="importing", activeCapture=False, attempts=0, nextAt=now(),
                            providerStatus=last, providerStatusAt=now(),
                            sessionPublished=bool(run.get("sessionPublished")),
                            dashboardEnded=bool(run.get("dashboardEnded")))
        elif last == "fatal" or now() > join_at + timedelta(hours=5):
            if recording_media(bot):
                await self.ensure_live_session(run, "done")
                await self.save(run, state="importing", activeCapture=False, attempts=0, nextAt=now(),
                                sessionPublished=bool(run.get("sessionPublished")))
            elif last == "fatal" and recorded_audio(run, bot):
                await self.ensure_live_session(run, "call_ended")
                fatal_at = as_utc(run.get("fatalSeenAt") or (run.get("providerStatusAt") if state == "fatal" else None) or now())
                deadline = min(fatal_at + timedelta(minutes=15), join_at + timedelta(hours=5))
                if now() < deadline:
                    changes = {"sessionPublished": True} if run.get("sessionPublished") else {}
                    if not run.get("fatalSeenAt"):
                        changes["fatalSeenAt"] = fatal_at
                    if changes:
                        await self.save(run, **changes)
                    await self.defer_attendance(run, min(deadline, now() + timedelta(seconds=RECOVERY_DELAY)))
                else:
                    await self.terminal(run, "failed", "The bot stopped unexpectedly and no recording was available.")
            else:
                await self.terminal(run, "cancelled" if run.get("stopRequested") and not run.get("startedAt") else "failed",
                                    None if run.get("stopRequested") and not run.get("startedAt")
                                    else "The bot could not complete the meeting recording.")
        else:
            await self.ensure_live_session(run, last)
            if last == "call_ended":
                await self.mark_call_ended(run)
            if last in {"in_call_not_recording", "in_call_recording"}:
                delay = ATTENDANCE_POLL_SECONDS
            elif last == "call_ended":
                delay = 20
            elif not run.get("sessionPublished"):
                delay = ATTENDANCE_POLL_SECONDS
            elif run.get("stopRequested"):
                delay = 25
            else:
                delay = RECOVERY_DELAY
            changes = {
                "providerStatus": last,
                "providerStatusAt": now(),
                "nextAt": now() + timedelta(seconds=delay),
                "attempts": 0,
            }
            if run.get("sessionPublished"):
                changes["sessionPublished"] = True
            if run.get("dashboardEnded"):
                changes["dashboardEnded"] = True
            if last == "in_call_recording" and not run.get("startedAt"):
                changes["startedAt"] = now()
            await self.save(run, **changes)

    async def ensure_live_session(self, run: dict, status: str | None):
        if run.get("sessionPublished"):
            return
        if status not in LIVE_STATUSES and not run.get("startedAt"):
            return
        started = as_utc(run.get("startedAt") or now())
        await self.node(run, "live", {"startedAt": started.isoformat()})
        run["sessionPublished"] = True

    async def mark_call_ended(self, run: dict):
        """The call is over, so the meetings card must leave the live Recording state."""
        if run.get("dashboardEnded"):
            return
        started = as_utc(run.get("startedAt") or run.get("recordingStartedAt") or now())
        await self.node(run, "ended", {"startedAt": started.isoformat()})
        run["dashboardEnded"] = True

    async def import_recording(self, run: dict):
        await self.ensure_live_session(run, run.get("providerStatus") if run.get("providerStatus") in LIVE_STATUSES else "in_call_recording")
        await self.mark_call_ended(run)
        saved = {}
        if run.get("sessionPublished"):
            saved["sessionPublished"] = True
        if run.get("dashboardEnded"):
            saved["dashboardEnded"] = True
        if saved:
            await self.save(run, **saved)
        if run.get("importCompletion"):
            await self.finish_import(run, run["importCompletion"])
            return
        owner, run_id = str(run["userId"]), str(run["_id"])
        stored = bool(run.get("videoStored"))
        started = run.get("recordingStartedAt") or run.get("startedAt") or run["joinAt"]
        url = ""
        if not stored:
            response = await self.recall("GET", f"/bot/{run['providerId']}/")
            response.raise_for_status()
            bot = response.json()
            media = recording_media(bot)
            if not media:
                did_record = recorded_audio(run, bot)
                if run.get("stopRequested") and not did_record:
                    await self.terminal(run, "cancelled")
                    return
                if not did_record:
                    raise InvalidRecording("The bot did not record. Check host admission and recording permissions.")
                raise RuntimeError("Recording media is pending or no recording was permitted")
            url, recording_started = media
            if recording_started:
                started = as_utc(recording_started)
        started_at = as_utc(started)
        root = Path(os.environ.get("MEETING_BOT_TEMP_DIR", "/tmp/meeting-bots"))
        root.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=f"{run_id}-", dir=root) as name:
            directory = Path(name)
            video = directory / "meeting.mp4"
            await store_video(self.http, url, video, owner, run_id, stored)
            await self.save(run, videoStored=True, recordingStartedAt=started_at)
            limits = await self.node(run, "open", {"startedAt": started_at.isoformat()})
            try:
                await store_meeting_card_thumbnail(self.db, run["_id"], owner, video)
            except Exception as exc:
                log.warning("meeting_bot_thumbnail_failed error=%s", type(exc).__name__)
            seconds = run.get("chunkSeconds") or min(45, limits["chunkMaxDurationMs"] // 1000 - 1,
                                                     (limits["chunkMaxSizeBytes"] - 4096) // 32000 - 1)
            if seconds < 1:
                raise InvalidRecording("The meeting audio limits are incompatible with recording import.")
            await self.save(run, chunkSeconds=seconds)
            files = await extract_chunks(video, seconds, directory)
            offset = 0
            last_registered = int(run.get("registeredSequence") or 0)
            for sequence, audio in enumerate(files, 1):
                duration = audio_chunk_duration(audio)
                if duration <= 0:
                    raise InvalidRecording("The saved video contains an empty audio chunk.")
                if sequence > last_registered:
                    await self.wait_for_audio_capacity(run)
                    await store_audio_chunk(audio, owner, run_id, sequence)
                    await self.node(run, "chunk", {
                        "sequence": sequence, "sizeBytes": audio.stat().st_size,
                        "startOffsetMs": offset, "endOffsetMs": offset + duration, "durationMs": duration,
                    })
                    await self.save(run, registeredSequence=sequence)
                offset += duration
            completion = {
                "finalSequence": len(files), "durationMs": offset,
                "endedAt": (started_at + timedelta(milliseconds=offset)).isoformat(),
            }
            await self.save(run, importCompletion=completion)
        await self.finish_import(run, completion)

    async def finish_import(self, run: dict, completion: dict):
        await self.node(run, "finish", completion)
        await self.terminal(run, "processing")

    async def wait_for_audio_capacity(self, run: dict):
        async with asyncio.timeout(240):
            while await self.db.transcript_chunks.count_documents({
                "conversationId": run["_id"],
                "sttStatus": {"$nin": ["COMPLETED", "FAILED"]},
            }, limit=MAX_PENDING_AUDIO_CHUNKS) >= MAX_PENDING_AUDIO_CHUNKS:
                await asyncio.sleep(5)

    async def step(self, run: dict):
        try:
            action = {"queued": self.dispatch, "attendance": self.attendance, "importing": self.import_recording}[run["state"]]
            await action(run)
        except (InvalidRecording, WorkerRejected) as exc:
            await self.terminal(run, "failed", str(exc))
        except LeaseExpired:
            return
        except Exception as exc:
            attempts = int(run.get("attempts") or 0) + 1
            detail = type(exc).__name__
            response = getattr(exc, "response", None)
            if response is not None:
                detail = f"{detail}:{getattr(response, 'status_code', '')}"
                with suppress(Exception):
                    message = str((response.json() or {}).get("message") or "")[:160]
                    if message:
                        detail = f"{detail}:{message}"
            elif isinstance(exc, ModuleNotFoundError) and exc.name:
                detail = f"{detail}:{exc.name}"
            log.warning("meeting_bot_retry run_id=%s state=%s error=%s", run["_id"], run["state"], detail)
            if attempts >= ATTEMPT_LIMIT:
                await self.terminal(run, "failed", "The meeting bot could not complete this job after retrying.")
            else:
                await self.save(run, attempts=attempts, nextAt=now() + timedelta(seconds=min(300, 10 * 2**attempts)))

    async def loop(self, states: list[str]):
        while True:
            at = now()
            try:
                run = await self.runs.find_one_and_update(
                    {"state": {"$in": states}, "nextAt": {"$lte": at},
                     "$or": [{"leaseUntil": None}, {"leaseUntil": {"$lte": at}}]},
                    {"$set": {"lease": str(uuid4()), "leaseUntil": at + timedelta(seconds=LEASE_SECONDS)}},
                    sort=[("nextAt", 1)], return_document=ReturnDocument.AFTER,
                )
                if run is None:
                    await asyncio.sleep(2)
                    continue
                operation = asyncio.create_task(self.step(run))
                heartbeat = asyncio.create_task(self.heartbeat(run, operation))
                try:
                    await operation
                except asyncio.CancelledError:
                    if not heartbeat.done():
                        raise
                finally:
                    heartbeat.cancel()
                    with suppress(asyncio.CancelledError):
                        await heartbeat
                    await self.runs.update_one({"_id": run["_id"], "lease": run["lease"]},
                                               {"$unset": {"lease": "", "leaseUntil": ""}})
            except LeaseExpired:
                continue
            except Exception as exc:
                log.warning("meeting_bot_worker_retry error=%s", type(exc).__name__)
                await asyncio.sleep(5)


def load_local_env() -> None:
    """Docker injects the worker env. A local process needs the repo .env in os.environ."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(Path(__file__).resolve().parents[3] / ".env")


async def run_meeting_bot_worker() -> None:
    """Send queued bots into meetings. Safe to run beside the other worker loops."""
    load_local_env()
    logging.getLogger("httpx").setLevel(logging.WARNING)
    try:
        async with httpx.AsyncClient(timeout=25, follow_redirects=False) as http:
            worker = MeetingBotWorker(http)
            print("meeting bot joiner started", flush=True)
            await asyncio.gather(worker.loop(["queued", "attendance"]), worker.loop(["importing"]))
    except ValueError as error:
        print(f"meeting bot joiner disabled: {error}", flush=True)
        await asyncio.Event().wait()


async def main():
    logging.basicConfig(level=logging.INFO)
    await run_meeting_bot_worker()


if __name__ == "__main__":
    asyncio.run(main())
