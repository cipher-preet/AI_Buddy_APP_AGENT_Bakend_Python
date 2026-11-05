"""Move blocking filesystem reads off the asyncio event loop.

The default executor is process-wide and bounded by asyncio. These helpers do
not open a new pool per job. Reading bytes in a thread does not reduce the
memory required when a provider still needs the full buffer.
"""

from __future__ import annotations

import asyncio
from pathlib import Path


async def read_file_bytes(path: str | Path) -> bytes:
    """Read a whole file without stalling lease heartbeats and other tasks."""
    return await asyncio.to_thread(Path(path).read_bytes)


def _read_prefix(path: Path, size: int) -> bytes:
    with path.open("rb") as handle:
        return handle.read(size)


async def read_file_prefix(path: str | Path, size: int) -> bytes:
    """Read a short header. The rest of the file stays on disk."""
    if size <= 0:
        return b""
    return await asyncio.to_thread(_read_prefix, Path(path), int(size))
