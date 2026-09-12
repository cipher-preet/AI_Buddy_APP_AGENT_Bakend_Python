from __future__ import annotations

import json
from typing import Any


def briefing_log(event: str, **fields: Any) -> None:
    payload = {"event": event, **{key: value for key, value in fields.items() if value is not None}}
    print(json.dumps(payload, default=str, ensure_ascii=False), flush=True)
