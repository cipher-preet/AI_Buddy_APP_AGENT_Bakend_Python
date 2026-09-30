from __future__ import annotations

import json


def extraction_log(event: str, **fields) -> None:
    print(json.dumps({"event": event, **fields}, default=str), flush=True)
