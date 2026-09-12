from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from collections import defaultdict, deque
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def diag_log(event: str, **fields: Any) -> None:
    payload = {"event": event, "timestamp": utc_timestamp()}
    for key, value in fields.items():
        if value is not None:
            payload[key] = value
    print(json.dumps(payload, default=str, ensure_ascii=False), flush=True)


class ActiveJobCounters:
    def __init__(self) -> None:
        self._counts: dict[str, int] = defaultdict(int)
        self._lock = threading.Lock()

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {
                "active_stt_jobs": self._counts.get("stt", 0),
                "active_audio_jobs": self._counts.get("audio", 0),
                "active_window_jobs": self._counts.get("window", 0),
                "active_briefing_jobs": self._counts.get("briefing", 0),
                "active_processing_jobs": self._counts.get("processing", 0),
            }

    def get(self, name: str) -> int:
        with self._lock:
            return int(self._counts.get(name, 0))

    def reset(self) -> None:
        with self._lock:
            self._counts.clear()

    @contextmanager
    def track(self, name: str) -> Iterator[None]:
        with self._lock:
            self._counts[name] += 1
        try:
            yield
        finally:
            with self._lock:
                self._counts[name] = max(0, self._counts[name] - 1)


class DuplicateBriefingTracker:
    def __init__(self) -> None:
        self._active: dict[tuple[str, str], str] = {}
        self._lock = threading.Lock()

    def begin(self, user_id: str, date_key: str, event_id: str) -> str | None:
        key = (user_id, date_key)
        with self._lock:
            existing = self._active.get(key)
            if existing is None:
                self._active[key] = event_id
                return None
            return existing

    def end(self, user_id: str, date_key: str, event_id: str) -> None:
        key = (user_id, date_key)
        with self._lock:
            if self._active.get(key) == event_id:
                self._active.pop(key, None)

    def reset(self) -> None:
        with self._lock:
            self._active.clear()


class ConversationRepublishTracker:
    def __init__(self) -> None:
        self._counts: dict[str, int] = defaultdict(int)
        self._lock = threading.Lock()

    def record(self, conversation_id: str) -> int:
        with self._lock:
            self._counts[conversation_id] += 1
            return self._counts[conversation_id]

    def get(self, conversation_id: str) -> int:
        with self._lock:
            return int(self._counts.get(conversation_id, 0))

    def reset(self) -> None:
        with self._lock:
            self._counts.clear()


class SupervisorRestartTracker:
    def __init__(self) -> None:
        self._history: dict[str, deque[float]] = defaultdict(deque)
        self._totals: dict[str, int] = defaultdict(int)
        self._lock = threading.Lock()

    def record(self, worker_name: str, exception_type: str) -> dict[str, Any]:
        now = time.monotonic()
        with self._lock:
            history = self._history[worker_name]
            history.append(now)
            while history and now - history[0] > 60:
                history.popleft()
            self._totals[worker_name] += 1
            restart_count = self._totals[worker_name]
            recent = len(history)
            storm = recent > 5
        return {
            "worker_name": worker_name,
            "exception_type": exception_type,
            "restart_count": restart_count,
            "restarts_last_60s": recent,
            "storm": storm,
        }

    def reset(self) -> None:
        with self._lock:
            self._history.clear()
            self._totals.clear()


class RetryRelayWindow:
    def __init__(self) -> None:
        self.messages_scanned = 0
        self.messages_due = 0
        self.messages_republished = 0
        self.delete_failures = 0
        self.invalid_messages = 0
        self._last_emit = time.monotonic()

    def maybe_emit(self, interval_seconds: float = 60.0) -> bool:
        now = time.monotonic()
        if now - self._last_emit < interval_seconds:
            return False
        diag_log(
            "retry_relay_health",
            messages_scanned=self.messages_scanned,
            messages_due=self.messages_due,
            messages_republished=self.messages_republished,
            delete_failures=self.delete_failures,
            invalid_messages=self.invalid_messages,
        )
        self.messages_scanned = 0
        self.messages_due = 0
        self.messages_republished = 0
        self.delete_failures = 0
        self.invalid_messages = 0
        self._last_emit = now
        return True


class _CpuSampler:
    def __init__(self) -> None:
        self._prev_proc: tuple[float, float] | None = None
        self._prev_sys: tuple[float, float] | None = None

    def sample(self) -> tuple[float | None, float | None]:
        proc = _process_cpu_times()
        sys_idle, sys_total = _system_cpu_times()
        now = time.monotonic()
        process_percent = None
        system_percent = None
        if proc is not None and self._prev_proc is not None:
            elapsed = now - self._prev_proc[0]
            if elapsed > 0:
                process_percent = round(max(0.0, (proc - self._prev_proc[1]) / elapsed * 100.0), 2)
        if sys_total is not None and sys_idle is not None and self._prev_sys is not None:
            total_delta = sys_total - self._prev_sys[1]
            idle_delta = sys_idle - self._prev_sys[0]
            if total_delta > 0:
                system_percent = round(max(0.0, (1.0 - idle_delta / total_delta) * 100.0), 2)
        if proc is not None:
            self._prev_proc = (now, proc)
        if sys_total is not None and sys_idle is not None:
            self._prev_sys = (sys_idle, sys_total)
        return process_percent, system_percent


def _process_cpu_times() -> float | None:
    try:
        with open("/proc/self/stat", encoding="utf-8") as handle:
            fields = handle.read().split()
        utime = int(fields[13])
        stime = int(fields[14])
        ticks = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100
        return (utime + stime) / float(ticks or 100)
    except Exception:
        return None


def _system_cpu_times() -> tuple[float | None, float | None]:
    try:
        with open("/proc/stat", encoding="utf-8") as handle:
            parts = handle.readline().split()
        values = [float(item) for item in parts[1:]]
        idle = values[3] + (values[4] if len(values) > 4 else 0.0)
        return idle, sum(values)
    except Exception:
        return None, None


def process_rss_mb() -> float | None:
    try:
        with open("/proc/self/status", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    kb = float(line.split()[1])
                    return round(kb / 1024.0, 2)
    except Exception:
        pass
    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if usage <= 0:
            return None
        # Linux ru_maxrss is KB; macOS is bytes.
        if usage > 10_000_000:
            return round(usage / (1024.0 * 1024.0), 2)
        return round(usage / 1024.0, 2)
    except Exception:
        return None


def system_memory_percent() -> float | None:
    try:
        meminfo: dict[str, float] = {}
        with open("/proc/meminfo", encoding="utf-8") as handle:
            for line in handle:
                parts = line.split()
                if len(parts) >= 2:
                    meminfo[parts[0].rstrip(":")] = float(parts[1])
        total = meminfo.get("MemTotal")
        available = meminfo.get("MemAvailable")
        if total and available is not None:
            return round((1.0 - available / total) * 100.0, 2)
    except Exception:
        return None
    return None


def asyncio_task_count() -> int:
    try:
        return len(asyncio.all_tasks())
    except Exception:
        return 0


def collect_process_health(cpu_sampler: _CpuSampler) -> dict[str, Any]:
    process_cpu, system_cpu = cpu_sampler.sample()
    return {
        "process_pid": os.getpid(),
        "process_cpu_percent": process_cpu,
        "process_rss_mb": process_rss_mb(),
        "system_cpu_percent": system_cpu,
        "system_memory_percent": system_memory_percent(),
        "asyncio_task_count": asyncio_task_count(),
        "thread_count": threading.active_count(),
        **active_jobs.snapshot(),
    }


async def mongo_health() -> dict[str, Any]:
    try:
        from services.db.mongo import get_mongo_client

        client = get_mongo_client()
        started = time.perf_counter()
        await client.admin.command("ping")
        return {
            "mongo_connected": True,
            "mongo_ping_ms": int((time.perf_counter() - started) * 1000),
        }
    except Exception:
        return {"mongo_connected": False, "mongo_ping_ms": None}


async def emit_redis_queue_health() -> None:
    from apps.api_gateway.config.setting import settings
    from services.queue.redis_queue import redis_client

    streams = (
        settings.REDIS_DAILY_BRIEFING_STREAM,
        settings.REDIS_STT_STREAM,
        settings.REDIS_PROCESSING_STREAM,
        settings.REDIS_RETRY_STREAM,
    )
    for stream in streams:
        try:
            length = int(await redis_client.xlen(stream))
            pending = None
            try:
                groups = await redis_client.xinfo_groups(stream)
                pending = sum(int(group.get("pending") or 0) for group in groups or [])
            except Exception:
                pending = None
            diag_log("redis_queue_health", stream=stream, length=length, pending=pending)
        except Exception as error:
            diag_log(
                "redis_queue_health",
                stream=stream,
                error_type=type(error).__name__,
            )


async def run_worker_heartbeat(interval_seconds: float = 30.0) -> None:
    sampler = _CpuSampler()
    while True:
        try:
            payload = collect_process_health(sampler)
            payload.update(await mongo_health())
            diag_log("worker_health", **payload)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            diag_log("worker_health", collection_error=type(error).__name__)
        await asyncio.sleep(interval_seconds)


async def run_event_loop_lag_monitor(interval_seconds: float = 8.0, threshold_ms: float = 250.0) -> None:
    loop = asyncio.get_running_loop()
    while True:
        started = loop.time()
        await asyncio.sleep(interval_seconds)
        lag_ms = int(max(0.0, (loop.time() - started - interval_seconds) * 1000))
        if lag_ms > threshold_ms:
            diag_log(
                "event_loop_lag",
                lag_ms=lag_ms,
                asyncio_task_count=asyncio_task_count(),
            )


async def run_redis_queue_health(interval_seconds: float = 60.0) -> None:
    while True:
        try:
            await emit_redis_queue_health()
        except asyncio.CancelledError:
            raise
        except Exception as error:
            diag_log("redis_queue_health", error_type=type(error).__name__)
        await asyncio.sleep(interval_seconds)


def note_stt_requeued(count: int) -> None:
    if count <= 0:
        return
    with _stt_requeue_lock:
        _stt_requeued_since_scan[0] += count


def drain_stt_requeued() -> int:
    with _stt_requeue_lock:
        value = _stt_requeued_since_scan[0]
        _stt_requeued_since_scan[0] = 0
        return value


def note_briefing_requeue(user_id: str, date_key: str) -> int:
    key = (user_id, date_key)
    with _briefing_requeue_lock:
        _briefing_requeue_counts[key] += 1
        return _briefing_requeue_counts[key]


active_jobs = ActiveJobCounters()
briefing_duplicates = DuplicateBriefingTracker()
conversation_republishes = ConversationRepublishTracker()
supervisor_restarts = SupervisorRestartTracker()
_stt_requeued_since_scan = [0]
_stt_requeue_lock = threading.Lock()
_briefing_requeue_counts: dict[tuple[str, str], int] = defaultdict(int)
_briefing_requeue_lock = threading.Lock()
