from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from services.reminders.fcm import FcmPermanentError, FcmTransientError, raise_for_fcm_result


class ReminderFcmConfigError(RuntimeError):
    pass


class DryRunFcmSender:
    async def send(self, tokens: list[str], payload: dict[str, str], high_priority: bool) -> None:
        if not tokens:
            raise FcmPermanentError("no device tokens")
        print(
            "Reminder FCM dry-run: "
            f"tokenCount={len(tokens)} type={payload.get('type')} high_priority={high_priority}",
            flush=True,
        )


class FirebaseAdminFcmSender:
    def __init__(self, app, on_invalid_token=None):
        self.app = app
        self.on_invalid_token = on_invalid_token

    async def send(self, tokens: list[str], payload: dict[str, str], high_priority: bool) -> None:
        if not tokens:
            raise FcmPermanentError("no device tokens")
        try:
            from firebase_admin import messaging
        except ImportError as error:
            raise FcmPermanentError("firebase-admin is not installed") from error

        channel_id = payload.get("channelId") or "buddy_reminders"
        data_only = payload.get("type") in {"reminder_alarm", "ai_reminder_call"}
        android_kwargs: dict[str, Any] = {
            "priority": "high" if high_priority else "normal",
        }
        if not data_only:
            android_kwargs["notification"] = messaging.AndroidNotification(
                channel_id=channel_id,
                icon="ic_stat_buddy_mic",
                sound="default",
                default_sound=True,
                default_vibrate_timings=True,
            )
        android_config = messaging.AndroidConfig(**android_kwargs)
        errors = 0
        transient = 0
        sent = 0
        invalid_tokens: list[str] = []
        for token in tokens:
            message_kwargs: dict[str, Any] = {
                "data": {key: str(value) for key, value in payload.items() if value is not None},
                "token": token,
                "android": android_config,
            }
            if not data_only:
                message_kwargs["notification"] = messaging.Notification(
                    title=payload.get("title") or "Buddy",
                    body=payload.get("message") or "",
                )
            message = messaging.Message(**message_kwargs)
            try:
                message_id = await asyncio.to_thread(
                    lambda msg=message: messaging.send(msg, app=self.app)
                )
                sent += 1
                print(
                    f"Reminder FCM accepted: messageId={message_id} channelId={channel_id}",
                    flush=True,
                )
            except Exception as error:  # noqa: BLE001
                text = str(error).lower()
                print(f"Reminder FCM send failed: {error}", flush=True)
                if "not found" in text or "unregistered" in text or "invalid" in text:
                    errors += 1
                    invalid_tokens.append(token)
                else:
                    transient += 1
        if self.on_invalid_token:
            for token in invalid_tokens:
                try:
                    await self.on_invalid_token(token)
                    print("Reminder FCM pruned stale token", flush=True)
                except Exception as error:  # noqa: BLE001
                    print(f"Reminder FCM token prune failed: {error}", flush=True)
        print(
            f"Reminder FCM sent: tokenCount={len(tokens)} delivered={sent} "
            f"rejected={errors} type={payload.get('type')}",
            flush=True,
        )
        raise_for_fcm_result(sent, transient, errors)


def fcm_sender_mode(sender: Any) -> str:
    if isinstance(sender, FirebaseAdminFcmSender):
        return "firebase"
    if isinstance(sender, DryRunFcmSender):
        return "dry-run"
    return type(sender).__name__


def _credential_from_settings(settings: Any):
    from firebase_admin import credentials

    json_blob = str(getattr(settings, "FIREBASE_SERVICE_ACCOUNT_JSON", "") or "").strip()
    cred_path = str(getattr(settings, "GOOGLE_APPLICATION_CREDENTIALS", "") or "").strip()
    if json_blob.startswith("{"):
        return credentials.Certificate(json.loads(json_blob))
    path_value = json_blob or cred_path
    if not path_value:
        raise ReminderFcmConfigError("FCM_ENABLED but no Firebase credentials")
    resolved = _resolve_credential_file(path_value)
    return credentials.Certificate(str(resolved))


def _resolve_credential_file(path_value: str) -> Path:
    raw = Path(path_value)
    candidates = [raw]
    if not raw.is_absolute():
        repo_root = Path(__file__).resolve().parents[2]
        candidates.extend([Path.cwd() / raw, repo_root / raw])
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise ReminderFcmConfigError(f"Firebase credential file not found: {path_value}")


def build_fcm_sender(settings: Any, on_invalid_token=None):
    if not getattr(settings, "FCM_ENABLED", False):
        print(
            "Reminder FCM: FCM_ENABLED=false, using dry-run. "
            "Reminders will be marked delivered without reaching the phone.",
            flush=True,
        )
        return DryRunFcmSender()

    try:
        import firebase_admin
    except ImportError as error:
        raise ReminderFcmConfigError("firebase-admin is not installed") from error

    if firebase_admin._apps:
        return FirebaseAdminFcmSender(firebase_admin.get_app(), on_invalid_token)

    try:
        cred = _credential_from_settings(settings)
        app = firebase_admin.initialize_app(cred)
        return FirebaseAdminFcmSender(app, on_invalid_token)
    except ReminderFcmConfigError:
        raise
    except Exception as error:  # noqa: BLE001
        raise ReminderFcmConfigError(f"Firebase init failed: {error}") from error
