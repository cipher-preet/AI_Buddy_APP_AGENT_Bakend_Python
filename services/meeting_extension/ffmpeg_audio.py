from __future__ import annotations

import asyncio
import re
from pathlib import Path

from apps.api_gateway.config.setting import settings
from services.observability.diagnostics import diag_log


class MeetingAudioExtractionError(RuntimeError):
    def __init__(self, message: str, *, corrupt: bool = False, stderr: str = "") -> None:
        super().__init__(message)
        self.corrupt = corrupt
        self.stderr = stderr


_CORRUPT_MARKERS = (
    "invalid data found",
    "does not contain any stream",
    "moov atom not found",
    "invalid argument",
    "not contain any stream",
    "ebml",
    "corrupt",
)


async def extract_audio_for_stt(input_path: str | Path, output_path: str | Path) -> Path:
    source = Path(input_path)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not source.exists() or source.stat().st_size <= 0:
        raise MeetingAudioExtractionError("Input video chunk is missing or empty", corrupt=True)

    copy_failed = await _run_ffmpeg(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(source),
            "-vn",
            "-acodec",
            "copy",
            str(destination.with_suffix(".opus")),
        ],
        timeout=settings.MEETING_FFMPEG_TIMEOUT_SECONDS,
        allow_failure=True,
    )
    copied = destination.with_suffix(".opus")
    if copy_failed is None and copied.exists() and copied.stat().st_size > 0:
        return copied

    wav_path = destination.with_suffix(".wav")
    error = await _run_ffmpeg(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(source),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(wav_path),
        ],
        timeout=settings.MEETING_FFMPEG_TIMEOUT_SECONDS,
        allow_failure=False,
    )
    if error:
        raise error
    if not wav_path.exists() or wav_path.stat().st_size <= 0:
        raise MeetingAudioExtractionError("FFmpeg produced an empty audio file", corrupt=True)
    return wav_path


async def concat_webm_chunks(chunk_paths: list[Path], output_path: Path) -> Path:
    if not chunk_paths:
        raise MeetingAudioExtractionError("No video chunks to concatenate", corrupt=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    list_file = output_path.parent / "concat.txt"
    list_file.write_text(
        "".join(f"file '{path.as_posix()}'\n" for path in chunk_paths),
        encoding="utf-8",
    )
    copy_error = await _run_ffmpeg(
        [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_file),
            "-c",
            "copy",
            str(output_path),
        ],
        timeout=settings.MEETING_MERGE_TIMEOUT_SECONDS,
        allow_failure=True,
    )
    if copy_error is None and output_path.exists() and output_path.stat().st_size > 0:
        return output_path
    reencode_error = await _run_ffmpeg(
        [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_file),
            "-c:v",
            "libvpx",
            "-b:v",
            "600k",
            "-c:a",
            "libopus",
            "-b:a",
            "48k",
            str(output_path),
        ],
        timeout=settings.MEETING_MERGE_TIMEOUT_SECONDS,
        allow_failure=False,
    )
    if reencode_error:
        raise reencode_error
    return output_path


async def _run_ffmpeg(
    command: list[str],
    *,
    timeout: float,
    allow_failure: bool,
) -> MeetingAudioExtractionError | None:
    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError as error:
        raise MeetingAudioExtractionError("ffmpeg is not installed", corrupt=False) from error
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
    except asyncio.TimeoutError as error:
        process.kill()
        await process.wait()
        raise MeetingAudioExtractionError("FFmpeg timed out", corrupt=False) from error

    stderr_text = (stderr or b"").decode("utf-8", errors="replace")[-4000:]
    if process.returncode == 0:
        return None
    corrupt = _is_corrupt(stderr_text)
    failure = MeetingAudioExtractionError(
        f"FFmpeg exited with code {process.returncode}",
        corrupt=corrupt,
        stderr=stderr_text,
    )
    diag_log(
        "meeting_ffmpeg_failed",
        exit_code=process.returncode,
        corrupt=corrupt,
        command0=command[0],
    )
    if allow_failure:
        return failure
    raise failure


def _is_corrupt(stderr: str) -> bool:
    lowered = stderr.lower()
    return any(marker in lowered for marker in _CORRUPT_MARKERS) or bool(
        re.search(r"invalid data", lowered)
    )
