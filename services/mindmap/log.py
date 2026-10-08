from __future__ import annotations

from typing import Any


def mindmap_log(event: str, **fields: Any) -> None:
    payload = {key: value for key, value in fields.items() if value is not None}
    print(f"mindmap.{event}:", payload, flush=True)
