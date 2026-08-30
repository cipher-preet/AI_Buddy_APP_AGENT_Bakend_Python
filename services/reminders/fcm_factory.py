from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from services.reminders.fcm import FcmPermanentError, FcmTransientError


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
    def __init__(self, app):
        self.app = app

    async def send(self, tokens: list[str], payload: dict[str, str], high_priority: bool) -> None:
        if not tokens:
            raise FcmPermanentError("no device tokens")
        try:
            from firebase_admin import messaging
        except ImportError as error:
            raise FcmPermanentError("firebase-admin is not installed") from error

        android_config = messaging.AndroidConfig(
            priority="high" if high_priority else "normal",
        )
        errors = 0
        transient = 0
        sent = 0
        for token in tokens:
            message = messaging.Message(
                notification=messaging.Notification(
                    title=payload.get("title") or "Buddy",
                    body=payload.get("message") or "",
                ),
                data={key: str(value) for key, value in payload.items() if value is not None},
                token=token,
                android=android_config,
            )
            try:
                messaging.send(message, app=self.app)
                sent += 1
            except Exception as error:  # noqa: BLE001
                text = str(error).lower()
                if "not found" in text or "invalid" in text or "unregistered" in text:
                    errors += 1
                else:
                    transient += 1
        print(
            f"Reminder FCM sent: tokenCount={len(tokens)} delivered={sent} "
            f"rejected={errors} type={payload.get('type')}",
            flush=True,
        )
        if transient:
            raise FcmTransientError("fcm transient failure")
        if errors == len(tokens):
            raise FcmPermanentError("all device tokens rejected")


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
    resolved = Path(path_value)
    if not resolved.is_file():
        raise ReminderFcmConfigError(f"Firebase credential file not found: {resolved}")
    return credentials.Certificate(str(resolved))


def build_fcm_sender(settings: Any):
    if not getattr(settings, "FCM_ENABLED", False):
        return DryRunFcmSender()

    try:
        import firebase_admin
    except ImportError as error:
        raise ReminderFcmConfigError("firebase-admin is not installed") from error

    if firebase_admin._apps:
        return FirebaseAdminFcmSender(firebase_admin.get_app())

    try:
        cred = _credential_from_settings(settings)
        app = firebase_admin.initialize_app(cred)
        return FirebaseAdminFcmSender(app)
    except ReminderFcmConfigError:
        raise
    except Exception as error:  # noqa: BLE001
        raise ReminderFcmConfigError(f"Firebase init failed: {error}") from error
