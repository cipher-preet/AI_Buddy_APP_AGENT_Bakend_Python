from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlparse

from apps.api_gateway.config.setting import settings


class S3StorageError(RuntimeError):
    pass


class TemporaryS3StorageError(S3StorageError):
    pass


class PermanentS3StorageError(S3StorageError):
    pass


@dataclass(frozen=True)
class S3ObjectReference:
    bucket: str
    object_key: str
    etag: str | None = None
    version_id: str | None = None
    size_bytes: int | None = None
    cloudfront_url: str | None = None


class S3AudioStorage:
    def __init__(self) -> None:
        self.bucket = settings.S3_AUDIO_BUCKET.strip()
        self.region = settings.AWS_REGION.strip() or "ap-south-1"
        self.prefix = settings.S3_AUDIO_PREFIX.strip().strip("/") or "buddy/audio"
        self._client: Any | None = None

    @property
    def client(self):
        if self._client is None:
            import boto3
            from botocore.config import Config

            access_key = settings.resolved_aws_access_key_id
            secret_key = settings.resolved_aws_secret_access_key
            if not self.bucket:
                raise PermanentS3StorageError("S3_AUDIO_BUCKET is required when STORAGE_PROVIDER=s3")
            if not access_key or not secret_key:
                raise PermanentS3StorageError("AWS S3 credentials are required")

            self._client = boto3.client(
                "s3",
                region_name=self.region,
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
                config=Config(
                    connect_timeout=settings.S3_UPLOAD_TIMEOUT_SECONDS,
                    read_timeout=max(settings.S3_UPLOAD_TIMEOUT_SECONDS, settings.S3_DOWNLOAD_TIMEOUT_SECONDS),
                    retries={"max_attempts": settings.S3_MAX_RETRIES, "mode": "standard"},
                ),
            )
        return self._client

    async def upload_file(
        self,
        local_path: str | Path,
        object_key: str,
        content_type: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> S3ObjectReference:
        path = Path(local_path)
        if not path.exists() or path.stat().st_size <= 0:
            raise PermanentS3StorageError("Cannot upload missing or empty audio file")

        safe_metadata = {str(k): str(v) for k, v in (metadata or {}).items() if v is not None}
        extra_args: dict[str, Any] = {
            "Metadata": safe_metadata,
            "Tagging": "temporary=true",
        }
        if content_type:
            extra_args["ContentType"] = content_type

        def _upload() -> S3ObjectReference:
            try:
                self.client.upload_file(str(path), self.bucket, object_key, ExtraArgs=extra_args)
                head = self.client.head_object(Bucket=self.bucket, Key=object_key)
                return S3ObjectReference(
                    bucket=self.bucket,
                    object_key=object_key,
                    etag=(head.get("ETag") or "").strip('"') or None,
                    version_id=head.get("VersionId"),
                    size_bytes=head.get("ContentLength"),
                    cloudfront_url=generate_cloudfront_signed_url(object_key),
                )
            except Exception as error:
                raise classify_s3_error(error) from error

        return await asyncio.wait_for(
            asyncio.to_thread(_upload),
            timeout=settings.S3_UPLOAD_TIMEOUT_SECONDS + 1,
        )

    async def download_file(
        self,
        bucket: str,
        object_key: str,
        destination: str | Path,
    ) -> Path:
        bucket = bucket.strip()
        object_key = validate_object_key(object_key)
        destination_path = Path(destination)
        destination_path.parent.mkdir(parents=True, exist_ok=True)

        def _download() -> Path:
            try:
                self.client.download_file(bucket, object_key, str(destination_path))
                return destination_path
            except Exception as error:
                raise classify_s3_error(error) from error

        downloaded = await asyncio.wait_for(
            asyncio.to_thread(_download),
            timeout=settings.S3_DOWNLOAD_TIMEOUT_SECONDS + 1,
        )
        if not downloaded.exists() or downloaded.stat().st_size <= 0:
            raise PermanentS3StorageError("Downloaded S3 audio file is missing or empty")
        return downloaded

    async def delete_file(self, bucket: str, object_key: str) -> None:
        bucket = bucket.strip()
        object_key = validate_object_key(object_key)

        def _delete() -> None:
            try:
                self.client.delete_object(Bucket=bucket, Key=object_key)
            except Exception as error:
                raise classify_s3_error(error) from error

        await asyncio.to_thread(_delete)


_storage: S3AudioStorage | None = None


def get_s3_audio_storage() -> S3AudioStorage:
    global _storage
    if _storage is None:
        _storage = S3AudioStorage()
    return _storage


def storage_provider() -> str:
    return settings.STORAGE_PROVIDER.strip().lower()


def use_s3_storage() -> bool:
    return storage_provider() == "s3"


def build_audio_object_key(
    *,
    user_id: str,
    space_id: str,
    session_id: str | None,
    job_id: str,
    filename: str,
) -> str:
    safe_filename = sanitize_filename(filename)
    parts = [
        get_s3_audio_storage().prefix,
        sanitize_key_part(user_id),
        sanitize_key_part(space_id),
        sanitize_key_part(session_id or "no-session"),
        sanitize_key_part(job_id),
        safe_filename,
    ]
    return "/".join(part.strip("/") for part in parts if part)


def sanitize_filename(filename: str | None) -> str:
    name = Path(filename or "audio").name.strip()
    if not name or name in {".", ".."}:
        return "audio"
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", Path(name).stem).strip(".-_") or "audio"
    suffix = re.sub(r"[^A-Za-z0-9.]+", "", Path(name).suffix)[:16]
    return f"{stem}{suffix}"


def sanitize_key_part(value: str | None) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._=-]+", "-", str(value or "unknown")).strip(".-/")
    return cleaned or "unknown"


def validate_object_key(object_key: str) -> str:
    key = str(object_key or "").strip().replace("\\", "/").lstrip("/")
    if not key or ".." in key.split("/"):
        raise PermanentS3StorageError("Invalid S3 object key")
    return key


def safe_temp_audio_path(job: dict) -> Path:
    job_id = sanitize_key_part(str(job.get("job_id") or "unknown"))
    filename = sanitize_filename(str(job.get("filename") or f"{job_id}.audio"))
    return Path("/tmp/buddy") / job_id / filename


def generate_cloudfront_signed_url(object_key: str | None, expires_in_seconds: int | None = None) -> str | None:
    base_url = settings.CLOUDFRONT_URL.strip().rstrip("/")
    key_pair_id = settings.CLOUDFRONT_KEY_PAIR_ID.strip()
    private_key = settings.CLOUDFRONT_PRIVATE_KEY.replace("\\n", "\n").strip()
    if not base_url or not object_key:
        return None

    normalized_key = _normalize_cloudfront_key(object_key)
    unsigned_url = f"{base_url}/{normalized_key}"
    if not key_pair_id or not private_key:
        return unsigned_url

    try:
        from botocore.signers import CloudFrontSigner
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import padding
    except ImportError:
        return unsigned_url

    def rsa_signer(message: bytes) -> bytes:
        key = serialization.load_pem_private_key(private_key.encode("utf-8"), password=None)
        return key.sign(message, padding.PKCS1v15(), hashes.SHA1())

    expires = expires_in_seconds or settings.CLOUDFRONT_SIGNED_URL_EXPIRES_SECONDS
    expire_at = datetime.now(timezone.utc) + timedelta(seconds=expires)
    return CloudFrontSigner(key_pair_id, rsa_signer).generate_presigned_url(
        unsigned_url,
        date_less_than=expire_at,
    )


def _normalize_cloudfront_key(object_key: str) -> str:
    value = str(object_key).strip()
    if value.startswith(("http://", "https://")):
        parsed = urlparse(value)
        value = parsed.path
    return "/".join(quote(part, safe="") for part in value.lstrip("/").split("/"))


def classify_s3_error(error: Exception) -> S3StorageError:
    try:
        from botocore.exceptions import (
            ClientError,
            ConnectTimeoutError,
            EndpointConnectionError,
            NoCredentialsError,
            PartialCredentialsError,
            ReadTimeoutError,
        )
    except ImportError:
        return TemporaryS3StorageError(str(error))

    if isinstance(error, (NoCredentialsError, PartialCredentialsError)):
        return PermanentS3StorageError("AWS S3 credentials are missing or incomplete")
    if isinstance(error, (ConnectTimeoutError, EndpointConnectionError, ReadTimeoutError, asyncio.TimeoutError)):
        return TemporaryS3StorageError(str(error))
    if isinstance(error, ClientError):
        code = str(error.response.get("Error", {}).get("Code", ""))
        status_code = int(error.response.get("ResponseMetadata", {}).get("HTTPStatusCode") or 0)
        if code in {"NoSuchKey", "NoSuchBucket", "AccessDenied", "InvalidAccessKeyId", "SignatureDoesNotMatch"}:
            return PermanentS3StorageError(f"S3 permanent error: {code}")
        if code in {"SlowDown", "RequestTimeout", "Throttling"} or status_code >= 500:
            return TemporaryS3StorageError(f"S3 temporary error: {code or status_code}")
        if status_code in {400, 403, 404}:
            return PermanentS3StorageError(f"S3 permanent error: {code or status_code}")
    return TemporaryS3StorageError(str(error))
