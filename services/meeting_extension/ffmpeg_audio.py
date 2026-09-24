from __future__ import annotations

import asyncio
import os
import re
import shutil
from functools import lru_cache
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
    "not contain any stream",
    "corrupt",
)

WEBM_EBML_ID = bytes([0x1A, 0x45, 0xDF, 0xA3])
WEBM_CLUSTER_ID = bytes([0x1F, 0x43, 0xB6, 0x75])
_FFMPEG_INPUT_FLAGS = (
    "-fflags",
    "+genpts+discardcorrupt",
    "-err_detect",
    "ignore_err",
    "-analyzeduration",
    "20000000",
    "-probesize",
    "10000000",
)


def reset_ffmpeg_bin_cache() -> None:
    resolve_ffmpeg_bin.cache_clear()


def _ffmpeg_candidate(path_or_name: str) -> str | None:
    value = str(path_or_name or "").strip()
    if not value:
        return None
    path = Path(value)
    if path.is_file():
        return str(path)
    found = shutil.which(value)
    return found


@lru_cache(maxsize=1)
def resolve_ffmpeg_bin() -> str:
    candidates = [
        str(settings.MEETING_FFMPEG_BIN or "").strip(),
        str(os.environ.get("IMAGEIO_FFMPEG_EXE") or "").strip(),
        "/usr/bin/ffmpeg",
        "/usr/local/bin/ffmpeg",
        "ffmpeg",
        "ffmpeg.exe",
    ]
    seen: set[str] = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        found = _ffmpeg_candidate(candidate)
        if found:
            return found
    try:
        import imageio_ffmpeg

        bundled = imageio_ffmpeg.get_ffmpeg_exe()
        if bundled and Path(bundled).is_file():
            return bundled
    except Exception:
        pass
    raise MeetingAudioExtractionError(
        "ffmpeg is not installed. Install ffmpeg or set MEETING_FFMPEG_BIN.",
        corrupt=False,
    )


def has_webm_header(data: bytes) -> bool:
    return bool(data) and data.startswith(WEBM_EBML_ID)


def extract_webm_init(data: bytes) -> bytes:
    if not data:
        return b""
    cluster_at = data.find(WEBM_CLUSTER_ID)
    if cluster_at > 0:
        return data[:cluster_at]
    if has_webm_header(data):
        return data
    return b""


def write_standalone_webm(chunk_path: Path, init_bytes: bytes, output_path: Path | None = None) -> Path:
    payload = chunk_path.read_bytes()
    if has_webm_header(payload) or not init_bytes:
        return chunk_path
    destination = output_path or chunk_path.with_name(f"{chunk_path.stem}.standalone.webm")
    destination.write_bytes(init_bytes + payload)
    return destination


def probe_ffmpeg() -> str | None:
    if not settings.MEETING_EXTENSION_ENABLED:
        return None
    try:
        binary = resolve_ffmpeg_bin()
        diag_log("meeting_ffmpeg_ready", ffmpeg=binary)
        print(f"meeting ffmpeg: {binary}", flush=True)
        return binary
    except MeetingAudioExtractionError as error:
        diag_log("meeting_ffmpeg_missing", error=str(error))
        print(
            f"WARNING: {error}; meeting video chunks will fail until ffmpeg is available",
            flush=True,
        )
        return None


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
            *_FFMPEG_INPUT_FLAGS,
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
            *_FFMPEG_INPUT_FLAGS,
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
    normalized = _normalize_webm_fragments(chunk_paths)
    list_file = output_path.parent / "concat.txt"
    list_file.write_text(
        "".join(f"file '{path.as_posix()}'\n" for path in normalized),
        encoding="utf-8",
    )
    intermediate = output_path.with_name(f"{output_path.stem}.concat.webm")
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
            str(intermediate),
        ],
        timeout=settings.MEETING_MERGE_TIMEOUT_SECONDS,
        allow_failure=True,
    )
    if copy_error is not None or not intermediate.exists() or intermediate.stat().st_size <= 0:
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
                str(intermediate),
            ],
            timeout=settings.MEETING_MERGE_TIMEOUT_SECONDS,
            allow_failure=False,
        )
        if reencode_error:
            raise reencode_error

    # Remux to a browser-seekable MP4 when requested, otherwise rewrite WebM with cues.
    if output_path.suffix.lower() == ".mp4":
        remux_error = await _run_ffmpeg(
            [
                "ffmpeg",
                "-y",
                *_FFMPEG_INPUT_FLAGS,
                "-i",
                str(intermediate),
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "23",
                "-c:a",
                "aac",
                "-b:a",
                "128k",
                "-movflags",
                "+faststart",
                str(output_path),
            ],
            timeout=settings.MEETING_MERGE_TIMEOUT_SECONDS,
            allow_failure=True,
        )
        if remux_error is None and output_path.exists() and output_path.stat().st_size > 0:
            return output_path

    seekable_error = await _run_ffmpeg(
        [
            "ffmpeg",
            "-y",
            *_FFMPEG_INPUT_FLAGS,
            "-i",
            str(intermediate),
            "-c:v",
            "libvpx",
            "-b:v",
            "800k",
            "-deadline",
            "good",
            "-cpu-used",
            "4",
            "-auto-alt-ref",
            "0",
            "-c:a",
            "libopus",
            "-b:a",
            "64k",
            "-f",
            "webm",
            str(output_path if output_path.suffix.lower() == ".webm" else intermediate),
        ],
        timeout=settings.MEETING_MERGE_TIMEOUT_SECONDS,
        allow_failure=True,
    )
    target = output_path if output_path.suffix.lower() == ".webm" else intermediate
    if seekable_error is None and target.exists() and target.stat().st_size > 0:
        if target != output_path:
            output_path.write_bytes(target.read_bytes())
        return output_path

    # Do not ship a cue-less concat dump as "meeting.mp4" — Chromium then fails decode/seek.
    raise MeetingAudioExtractionError(
        "Could not produce a browser-playable recording (mp4/webm remux failed)",
        corrupt=True,
    )


async def _run_ffmpeg(
    command: list[str],
    *,
    timeout: float,
    allow_failure: bool,
) -> MeetingAudioExtractionError | None:
    try:
        binary = resolve_ffmpeg_bin()
        process = await asyncio.create_subprocess_exec(
            binary,
            *command[1:],
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError as error:
        raise MeetingAudioExtractionError(
            "ffmpeg is not installed. Install ffmpeg or set MEETING_FFMPEG_BIN.",
            corrupt=False,
        ) from error
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


def _normalize_webm_fragments(chunk_paths: list[Path]) -> list[Path]:
    first = chunk_paths[0].read_bytes()
    init = extract_webm_init(first) if has_webm_header(first) else b""
    normalized: list[Path] = []
    for index, path in enumerate(chunk_paths):
        data = path.read_bytes() if index else first
        if index == 0 or has_webm_header(data) or not init:
            normalized.append(path)
            continue
        repaired = path.with_name(f"{path.stem}.hdr.webm")
        repaired.write_bytes(init + data)
        normalized.append(repaired)
    return normalized


def _is_corrupt(stderr: str) -> bool:
    lowered = stderr.lower()
    return any(marker in lowered for marker in _CORRUPT_MARKERS) or bool(
        re.search(r"invalid data", lowered)
    )
