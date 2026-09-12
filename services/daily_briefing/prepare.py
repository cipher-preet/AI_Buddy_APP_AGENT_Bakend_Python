from __future__ import annotations

import re
from datetime import datetime

from services.conversation.transcript import estimate_tokens
from services.daily_briefing.schemas import TranscriptWindow

_WHITESPACE = re.compile(r"\s+")


def _normalize_text(value: str) -> str:
    return _WHITESPACE.sub(" ", (value or "").strip()).lower()


def _timestamp(item: dict) -> datetime:
    value = item.get("createdAt") or item.get("timestamp")
    if isinstance(value, datetime):
        return value
    return datetime.min


def filter_and_order_transcripts(items: list[dict]) -> list[dict]:
    useful: list[dict] = []
    seen: set[str] = set()
    for item in sorted(items, key=_timestamp):
        text = str(item.get("text") or "").strip()
        if len(text) < 8:
            continue
        key = _normalize_text(text)
        if not key or key in seen:
            continue
        seen.add(key)
        useful.append(item)
    return useful


def build_windows(items: list[dict], target_tokens: int, max_tokens: int) -> list[TranscriptWindow]:
    if not items:
        return []
    windows: list[TranscriptWindow] = []
    current: list[dict] = []
    current_tokens = 0
    index = 0
    for item in items:
        tokens = max(1, estimate_tokens(str(item.get("text") or "")))
        if current and current_tokens + tokens > max(target_tokens, 1) and current_tokens + tokens > max_tokens * 0.9:
            windows.append(TranscriptWindow(index=index, items=current, tokenCount=current_tokens))
            index += 1
            current = []
            current_tokens = 0
        if tokens > max_tokens and not current:
            windows.append(TranscriptWindow(index=index, items=[item], tokenCount=tokens))
            index += 1
            continue
        current.append(item)
        current_tokens += tokens
        if current_tokens >= max_tokens:
            windows.append(TranscriptWindow(index=index, items=current, tokenCount=current_tokens))
            index += 1
            current = []
            current_tokens = 0
    if current:
        windows.append(TranscriptWindow(index=index, items=current, tokenCount=current_tokens))
    return windows
