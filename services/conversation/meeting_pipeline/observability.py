"""Structured tracing for the meeting extract-consolidate-verify path."""

from __future__ import annotations

from typing import Any


def log_pipeline(payload: dict[str, Any]) -> None:
    safe = {key: value for key, value in payload.items() if "key" not in key.lower() and "secret" not in key.lower()}
    _safe_print("Meeting pipeline:", safe)


def log_artifact(payload: dict[str, Any]) -> None:
    _safe_print("Meeting artifact provenance:", payload)


def _safe_print(*parts: Any) -> None:
    try:
        print(*parts)
    except UnicodeEncodeError:
        text = " ".join(str(part) for part in parts)
        print(text.encode("ascii", "backslashreplace").decode("ascii"))
