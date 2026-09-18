from __future__ import annotations

import re

from apps.api_gateway.config.setting import settings
from services.storage.s3_audio_storage import PermanentS3StorageError, sanitize_key_part, validate_object_key

SOURCE_TYPE = "meeting_extension"


def meeting_chunk_object_key(
    user_id: str,
    meeting_session_id: str,
    sequence: int,
    extension: str = "webm",
    media_kind: str = "muxed",
) -> str:
    prefix = settings.MEETING_S3_PREFIX.strip().strip("/") or "meetings"
    seq = f"{int(sequence):06d}"
    ext = re.sub(r"[^a-z0-9]+", "", extension.lower()) or "webm"
    folder = "audio" if media_kind == "audio" else "video" if media_kind == "video" else "chunks"
    return f"{prefix}/{sanitize_key_part(user_id)}/{sanitize_key_part(meeting_session_id)}/{folder}/{seq}.{ext}"


def meeting_final_object_key(user_id: str, meeting_session_id: str) -> str:
    prefix = settings.MEETING_S3_PREFIX.strip().strip("/") or "meetings"
    return f"{prefix}/{sanitize_key_part(user_id)}/{sanitize_key_part(meeting_session_id)}/meeting.webm"


def validate_meeting_object_key(*, object_key: str, user_id: str, meeting_session_id: str) -> str:
    key = validate_object_key(object_key)
    prefix = (
        f"{settings.MEETING_S3_PREFIX.strip().strip('/') or 'meetings'}/"
        f"{sanitize_key_part(user_id)}/{sanitize_key_part(meeting_session_id)}/"
    )
    if not key.startswith(prefix):
        raise PermanentS3StorageError("S3 object key is outside the meeting recording scope")
    return key
