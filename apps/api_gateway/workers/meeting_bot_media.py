from __future__ import annotations

import asyncio
import ipaddress
import wave
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from services.meeting_extension.ffmpeg_audio import resolve_ffmpeg_bin
from services.meeting_extension.s3_keys import meeting_chunk_object_key, meeting_final_object_key
from services.storage.s3_audio_storage import get_s3_audio_storage

MAX_VIDEO_BYTES = 2 * 1024**3


class InvalidRecording(Exception):
    pass


def safe_download_url(url: str) -> bool:
    try:
        parsed = urlsplit(url)
        host = parsed.hostname or ""
        if parsed.scheme != "https" or not host or parsed.username or parsed.password or parsed.port not in (None, 443):
            return False
        try:
            ipaddress.ip_address(host)
            return False
        except ValueError:
            return host.endswith((".recall.ai", ".amazonaws.com", ".cloudfront.net"))
    except ValueError:
        return False


async def store_video(http: httpx.AsyncClient, url: str, path: Path, owner: str, run_id: str, stored: bool):
    storage = get_s3_audio_storage()
    key = meeting_final_object_key(owner, run_id)
    if stored:
        await asyncio.to_thread(storage.client.download_file, storage.bucket, key, str(path))
        return
    if not safe_download_url(url):
        raise InvalidRecording("The recording service returned an unsupported download URL.")
    downloaded = 0
    async with asyncio.timeout(900):
        async with http.stream(
            "GET", url, follow_redirects=False,
            timeout=httpx.Timeout(connect=30, read=900, write=30, pool=30),
        ) as response:
            response.raise_for_status()
            if int(response.headers.get("Content-Length", "0")) > MAX_VIDEO_BYTES:
                raise InvalidRecording("The video exceeds the import size limit.")
            with path.open("wb") as destination:
                async for block in response.aiter_bytes(chunk_size=1024 * 1024):
                    downloaded += len(block)
                    if downloaded > MAX_VIDEO_BYTES:
                        raise InvalidRecording("The video exceeds the import size limit.")
                    destination.write(block)
    with path.open("rb") as recording:
        header = recording.read(12)
    if len(header) < 12 or header[4:8] != b"ftyp":
        raise InvalidRecording("The recording service did not return an MP4 file.")
    # The complete recording is the playback asset. Do not apply the temporary audio tag.
    await asyncio.to_thread(
        storage.client.upload_file, str(path), storage.bucket, key,
        ExtraArgs={"ContentType": "video/mp4"},
    )


async def extract_chunks(video: Path, seconds: int, folder: Path) -> list[Path]:
    process = await asyncio.create_subprocess_exec(
        resolve_ffmpeg_bin(), "-nostdin", "-hide_banner", "-loglevel", "error",
        "-i", str(video), "-map", "0:a:0", "-vn", "-af", "aresample=async=1:first_pts=0",
        "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
        "-f", "segment", "-segment_time", str(seconds), "-segment_start_number", "1",
        str(folder / "%06d.wav"), stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE,
    )
    try:
        await asyncio.wait_for(process.communicate(), timeout=1200)
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()
    files = sorted(folder.glob("*.wav"))
    if process.returncode or not files:
        raise InvalidRecording("The saved video does not contain readable audio.")
    return files


def audio_chunk_duration(path: Path) -> int:
    with wave.open(str(path), "rb") as audio:
        return round(audio.getnframes() * 1000 / audio.getframerate())


async def store_audio_chunk(path: Path, owner: str, run_id: str, sequence: int):
    storage = get_s3_audio_storage()
    key = meeting_chunk_object_key(owner, run_id, sequence, extension="wav", media_kind="audio")
    await storage.upload_file(path, key, "audio/wav")
