"""Live probe for OpenRouter Nemotron free JSON / reasoning / tools behavior."""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_env(path: Path) -> None:
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_env(ROOT / ".env.aws")
_load_env(ROOT / ".env")

KEY = (os.environ.get("OPENROUTER_API_KEY") or "").strip()
BASE = (os.environ.get("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1").rstrip("/")
MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"


def _request(method: str, path: str, payload: dict | None = None) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE}{path}",
        data=data,
        method=method,
        headers={
            "Authorization": f"Bearer {KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://kukunotes.local",
            "X-Title": "KukuNotes-probe",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {error.code}: {body[:800]}") from error


def chat(label: str, payload: dict) -> None:
    try:
        data = _request("POST", "/chat/completions", payload)
    except Exception as error:
        print(f"\n{label} FAILED {error}")
        return
    choice = (data.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    print(
        f"\n{label}",
        json.dumps(
            {
                "finish_reason": choice.get("finish_reason"),
                "content_len": len(str(msg.get("content") or "")),
                "content_preview": str(msg.get("content") or "")[:200],
                "reasoning_len": len(str(msg.get("reasoning") or msg.get("reasoning_content") or "")),
                "reasoning_preview": str(msg.get("reasoning") or msg.get("reasoning_content") or "")[:120],
                "tool_calls": msg.get("tool_calls"),
                "usage": data.get("usage"),
                "error": data.get("error"),
            },
            indent=2,
        ),
    )


def main() -> int:
    if not KEY:
        print("OPENROUTER_API_KEY missing", file=sys.stderr)
        return 1

    models = _request("GET", "/models").get("data") or []
    item = next((m for m in models if m.get("id") == MODEL), None)
    print(
        "MODEL_META",
        json.dumps(
            {
                "id": item and item.get("id"),
                "supported_parameters": item and item.get("supported_parameters"),
                "reasoning": item and item.get("reasoning"),
            },
            indent=2,
        ),
    )

    msgs = [{"role": "user", "content": 'Return ONLY this JSON object: {"ok":true,"n":1}'}]
    chat(
        "A_medium_capped",
        {
            "model": MODEL,
            "messages": msgs,
            "temperature": 0,
            "max_tokens": 4096,
            "reasoning": {"max_tokens": 1024},
        },
    )
    chat(
        "B_tool_call_medium",
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": "Fill document_json with ok=true and n=1"}],
            "temperature": 0,
            "max_tokens": 4096,
            "reasoning": {"max_tokens": 1024},
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "document_json",
                        "description": "Return structured document fields",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "ok": {"type": "boolean"},
                                "n": {"type": "integer"},
                            },
                            "required": ["ok", "n"],
                            "additionalProperties": False,
                        },
                    },
                }
            ],
            "tool_choice": {"type": "function", "function": {"name": "document_json"}},
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
