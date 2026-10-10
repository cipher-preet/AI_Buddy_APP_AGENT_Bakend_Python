from __future__ import annotations

from typing import Any


def document_log(event: str, **fields: Any) -> None:
    payload = {key: value for key, value in fields.items() if value is not None}
    print(f"document.{event}:", payload, flush=True)
