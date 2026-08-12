import asyncio
import shutil
from pathlib import Path

from services.queue.redis_queue import (
    get_job_result,
    mark_job_processing,
    mark_job_failed,
    pop_speech_job,
    push_completed_speech_job,
    save_job_result,
)

from services.speech.providers.sarvam_provider import (
    sarvam_transcribe_from_path,
)
from services.storage.s3_audio_storage import (
    PermanentS3StorageError,
    TemporaryS3StorageError,
    get_s3_audio_storage,
    safe_temp_audio_path,
    temp_audio_root,
    use_s3_storage,
)


async def process_speech_job(job: dict) -> None:
    job = _normalize_speech_job(job)
    job_id = job["job_id"]
    existing = await get_job_result(job_id)
    if existing and existing.get("status") == "completed":
        print("Speech job already completed:", job_id)
        return
    if existing:
        job = {**_normalize_speech_job(existing), **job}

    if _job_uses_s3(job):
        await _process_s3_speech_job(job)
        return

    await _process_local_speech_job(job)


async def _process_local_speech_job(job: dict) -> None:
    job_id = job["job_id"]
    try:
        print("Processing speech job:", job_id)
        _validate_local_job(job)

        await mark_job_processing(job_id)

        result = await sarvam_transcribe_from_path(
            file_path=job["file_path"],
            filename=job["filename"],
            content_type=job["content_type"],
        )

        await save_job_result(job_id, result)

        await push_completed_speech_job(job_id)

        print("Speech job completed:", job_id)
    except Exception as error:
        await mark_job_failed(job_id, str(error))
        raise


async def _process_s3_speech_job(job: dict) -> None:
    job_id = job["job_id"]
    try:
        _validate_s3_job(job)
    except ValueError as error:
        await mark_job_failed(job_id, str(error))
        raise
    local_path = safe_temp_audio_path(job)
    job_dir = local_path.parent
    s3_bucket = str(job["s3_bucket"])
    s3_object_key = str(job["s3_object_key"])
    succeeded = False

    try:
        _cleanup_job_dir(job_dir)
        print(
            "S3 audio download started:",
            {
                "job_id": job_id,
                "user_id": job.get("user_id"),
                "space_id": job.get("space_id"),
                "s3_bucket": s3_bucket,
                "s3_object_key": s3_object_key,
                "stage": "s3_download_started",
            },
        )
        downloaded = await get_s3_audio_storage().download_file(
            bucket=s3_bucket,
            object_key=s3_object_key,
            destination=local_path,
        )
        print(
            "S3 audio download completed:",
            {
                "job_id": job_id,
                "user_id": job.get("user_id"),
                "space_id": job.get("space_id"),
                "s3_bucket": s3_bucket,
                "s3_object_key": s3_object_key,
                "stage": "s3_download_completed",
            },
        )

        job_copy = dict(job)
        job_copy["file_path"] = str(downloaded)
        await _process_local_speech_job(job_copy)
        succeeded = True
    except PermanentS3StorageError as error:
        await mark_job_failed(job_id, str(error))
        raise ValueError(str(error)) from error
    except TemporaryS3StorageError as error:
        await mark_job_failed(job_id, str(error))
        raise RuntimeError(str(error)) from error
    finally:
        try:
            _cleanup_job_dir(job_dir)
            print(
                "Worker temporary audio cleanup completed:",
                {"job_id": job_id, "stage": "temporary_file_cleanup"},
            )
        except Exception as cleanup_error:
            print("Worker temporary audio cleanup failed:", str(cleanup_error))

    if succeeded and settings_delete_after_processing():
        try:
            await get_s3_audio_storage().delete_file(s3_bucket, s3_object_key)
            print(
                "S3 audio delete completed:",
                {
                    "job_id": job_id,
                    "s3_bucket": s3_bucket,
                    "s3_object_key": s3_object_key,
                    "stage": "s3_delete_completed",
                },
            )
        except TemporaryS3StorageError as error:
            raise RuntimeError(str(error)) from error
        except PermanentS3StorageError as error:
            print("S3 delete skipped after permanent cleanup error:", str(error))


def _job_uses_s3(job: dict) -> bool:
    provider = str(job.get("storage_provider") or "").strip().lower()
    has_s3_reference = bool(job.get("s3_bucket") or job.get("s3_object_key"))
    return provider == "s3" or has_s3_reference or (use_s3_storage() and not job.get("file_path"))


def _normalize_speech_job(job: dict) -> dict:
    normalized = dict(job)
    aliases = {
        "job_id": ("jobId",),
        "user_id": ("userId",),
        "space_id": ("spaceId",),
        "request_id": ("requestId",),
        "file_path": ("filePath",),
        "content_type": ("contentType",),
        "storage_provider": ("storageProvider",),
        "s3_bucket": ("s3Bucket",),
        "s3_object_key": ("s3ObjectKey", "s3_key", "s3Key", "object_key", "objectKey"),
    }
    for canonical, alternate_names in aliases.items():
        if normalized.get(canonical):
            continue
        for alternate in alternate_names:
            value = normalized.get(alternate)
            if value:
                normalized[canonical] = value
                break
    if not str(normalized.get("job_id") or "").strip():
        raise ValueError("Speech job payload is missing job_id")
    for field in ("job_id", "user_id", "space_id", "request_id", "file_path", "content_type", "storage_provider", "s3_bucket", "s3_object_key", "filename"):
        if normalized.get(field) is not None:
            normalized[field] = str(normalized[field]).strip()
    return normalized


def _validate_local_job(job: dict) -> None:
    for field in ("file_path", "filename", "content_type"):
        if not str(job.get(field) or "").strip():
            raise ValueError(f"Speech local job is missing {field}")


def _validate_s3_job(job: dict) -> None:
    if str(job.get("storage_provider") or "s3").lower() != "s3":
        raise ValueError("Unsupported storage_provider for speech job")
    for field in ("s3_bucket", "s3_object_key", "filename"):
        if not str(job.get(field) or "").strip():
            raise ValueError(f"Speech S3 job is missing {field}")


def _cleanup_job_dir(job_dir: Path) -> None:
    root = temp_audio_root()
    resolved = job_dir.resolve()
    if resolved == root:
        raise ValueError(f"Refusing to clean temporary audio root: {root}")
    if root not in resolved.parents:
        raise ValueError(f"Refusing to clean path outside temporary audio root: {root}")
    if resolved.exists():
        shutil.rmtree(resolved)


def settings_delete_after_processing() -> bool:
    from apps.api_gateway.config.setting import settings

    return settings.S3_DELETE_AFTER_PROCESSING


async def start_speech_consumer():
    print("Speech worker started...")

    while True:
        try:
            job = await pop_speech_job()

            if not job:
                await asyncio.sleep(1)
                continue

            await process_speech_job(job)

        except Exception as error:
            print("Speech worker error:", str(error))

            await asyncio.sleep(2)
