"""Local worker microbench. Mock delays are labeled and are not provider throughput.

Run from the repository root:

    python -m scripts.benchmark_worker_perf

The script writes only under the system temp directory and does not contact
Redis, MongoDB, S3, or a speech provider.
"""

from __future__ import annotations

import asyncio
import json
import statistics
import tempfile
import threading
import time
from pathlib import Path

from services.runtime.file_io import read_file_bytes, read_file_prefix


def _rss_bytes() -> dict[str, int] | None:
    try:
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(Counters), wintypes.DWORD]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        counters = Counters()
        counters.cb = ctypes.sizeof(Counters)
        if not psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            return None
        return {
            "rss": int(counters.WorkingSetSize),
            "peak_rss": int(counters.PeakWorkingSetSize),
        }
    except Exception:
        return None


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * fraction))))
    return ordered[index]


async def _lag_while(work, interval: float = 0.01) -> tuple[object, list[float]]:
    loop = asyncio.get_running_loop()
    gaps: list[float] = []
    stop = asyncio.Event()

    async def probe() -> None:
        while not stop.is_set():
            started = loop.time()
            await asyncio.sleep(interval)
            gaps.append(max(0.0, loop.time() - started - interval))

    watcher = asyncio.create_task(probe())
    await asyncio.sleep(0)
    try:
        result = await work
    finally:
        stop.set()
        watcher.cancel()
        try:
            await watcher
        except asyncio.CancelledError:
            pass
    return result, gaps


async def _blocking_reads(path: Path, repeats: int) -> list[float]:
    durations: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        path.read_bytes()
        durations.append(time.perf_counter() - started)
        await asyncio.sleep(0)
    return durations


async def _offloaded_reads(path: Path, repeats: int) -> list[float]:
    durations: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        await read_file_bytes(path)
        durations.append(time.perf_counter() - started)
    return durations


async def _jobs(path: Path, count: int, provider_delay: float, limit: int, *, offload: bool) -> list[dict[str, float]]:
    semaphore = asyncio.Semaphore(limit)
    rows: list[dict[str, float]] = []

    async def one(index: int) -> None:
        queued = time.perf_counter()
        async with semaphore:
            waited = time.perf_counter() - queued
            started = time.perf_counter()
            if offload:
                payload = await read_file_bytes(path)
            else:
                payload = path.read_bytes()
                await asyncio.sleep(0)
            read_done = time.perf_counter()
            await asyncio.sleep(provider_delay)
            rows.append(
                {
                    "index": float(index),
                    "queue_wait_s": waited,
                    "read_s": read_done - started,
                    "provider_s": provider_delay,
                    "total_s": time.perf_counter() - queued,
                    "bytes": float(len(payload)),
                }
            )

    await asyncio.gather(*(one(index) for index in range(count)))
    return rows


def _summary(rows: list[dict[str, float]], key: str) -> dict[str, float]:
    values = [row[key] for row in rows]
    return {
        "samples": float(len(values)),
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "mean": statistics.fmean(values) if values else 0.0,
    }


async def main() -> None:
    repeats = 8
    jobs = 12
    provider_delay = 0.02
    size = 8 * 1024 * 1024
    with tempfile.TemporaryDirectory(prefix="phase2d-") as directory:
        path = Path(directory) / "audio.bin"
        path.write_bytes(b"\x00" * size)
        memory_before = _rss_bytes()
        threads_before = threading.active_count()
        cpu_before = time.process_time()
        wall_started = time.perf_counter()

        _, blocking_gaps = await _lag_while(_blocking_reads(path, repeats))
        _, offload_gaps = await _lag_while(_offloaded_reads(path, repeats))
        stall = 0.2

        async def blocking_stall() -> None:
            for _ in range(6):
                await asyncio.sleep(0.02)
                time.sleep(stall)

        async def offloaded_stall() -> None:
            for _ in range(6):
                await asyncio.to_thread(time.sleep, stall)

        _, stall_blocking_gaps = await _lag_while(blocking_stall(), interval=0.02)
        _, stall_offload_gaps = await _lag_while(offloaded_stall(), interval=0.02)
        blocking_jobs = await _jobs(path, jobs, provider_delay, 1, offload=False)
        offload_jobs = await _jobs(path, jobs, provider_delay, 1, offload=True)
        concurrency = {}
        for limit in (1, 2, 4):
            started = time.perf_counter()
            rows = await _jobs(path, jobs, provider_delay, limit, offload=True)
            concurrency[str(limit)] = {
                "wall_s": time.perf_counter() - started,
                "total": _summary(rows, "total_s"),
                "queue_wait": _summary(rows, "queue_wait_s"),
                "read": _summary(rows, "read_s"),
            }
        prefix_started = time.perf_counter()
        header = await read_file_prefix(path, 4)
        prefix_s = time.perf_counter() - prefix_started
        memory_after = _rss_bytes()
        report = {
            "file_bytes": size,
            "repeats": repeats,
            "jobs": jobs,
            "simulated_provider_s": provider_delay,
            "provider_time_included_in_total": True,
            "wall_s": time.perf_counter() - wall_started,
            "cpu_process_s": time.process_time() - cpu_before,
            "threads_before": threads_before,
            "threads_after": threading.active_count(),
            "rss": {"before": memory_before, "after": memory_after},
            "blocking_loop_lag_s": {
                "samples": len(blocking_gaps),
                "p50": _percentile(blocking_gaps, 0.50),
                "p95": _percentile(blocking_gaps, 0.95),
            },
            "offloaded_loop_lag_s": {
                "samples": len(offload_gaps),
                "p50": _percentile(offload_gaps, 0.50),
                "p95": _percentile(offload_gaps, 0.95),
            },
            "synthetic_stall_s": stall,
            "synthetic_stall_repeats": 6,
            "synthetic_blocking_loop_lag_s": {
                "samples": len(stall_blocking_gaps),
                "p50": _percentile(stall_blocking_gaps, 0.50),
                "p95": _percentile(stall_blocking_gaps, 0.95),
            },
            "synthetic_offloaded_loop_lag_s": {
                "samples": len(stall_offload_gaps),
                "p50": _percentile(stall_offload_gaps, 0.50),
                "p95": _percentile(stall_offload_gaps, 0.95),
            },
            "blocking_job_total_s": _summary(blocking_jobs, "total_s"),
            "offloaded_job_total_s": _summary(offload_jobs, "total_s"),
            "blocking_read_s": _summary(blocking_jobs, "read_s"),
            "offloaded_read_s": _summary(offload_jobs, "read_s"),
            "blocking_queue_wait_s": _summary(blocking_jobs, "queue_wait_s"),
            "offloaded_queue_wait_s": _summary(offload_jobs, "queue_wait_s"),
            "benchmark_concurrency": concurrency,
            "prefix_read_s": prefix_s,
            "prefix_bytes": len(header),
            "production_speech_concurrency": 1,
            "production_vector_concurrency": 1,
        }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
