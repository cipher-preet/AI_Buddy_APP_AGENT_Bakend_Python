"""Live verification for conversation-intelligence models.

Loads .env.aws, then:
  1. Confirms Krutrim catalogue IDs and context_length
  2. Sends one tiny chat completion per production model
  3. Runs COMPLEX_MEETING (and the short two-hour fixture) through Buddy

This does not run in pytest. Never prints API keys or transcript bodies.

  py -3 scripts/verify_conversation_llm.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_env_aws() -> None:
    path = ROOT / ".env.aws"
    if not path.is_file():
        raise SystemExit(f"missing {path}")
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ[key.strip()] = value.strip().strip('"').strip("'")
    os.environ["SERVICE_ROLE"] = "api"
    os.environ.setdefault("LLM_TIMEOUT_SECONDS", "180")


_load_env_aws()

import httpx  # noqa: E402

from tests.eval.conversations import BENCHMARK_CASES  # noqa: E402
from tests.fixtures.conversation_meetings import COMPLEX_MEETING_TRANSCRIPT  # noqa: E402

EXPECTED_KRUTRIM = {
    "gemma-4-31b-it": 131072,
    "gpt-oss-120b": 65536,
    "gpt-oss-20b": 131072,
}
SMOKE_MODELS = [
    ("krutrim", "gemma-4-31b-it"),
    ("krutrim", "gpt-oss-120b"),
    ("krutrim", "gpt-oss-20b"),
    ("mistral", "ministral-14b-latest"),
]


def _secret(name: str) -> str:
    return str(os.environ.get(name) or "").strip()


def _provider_conf(name: str) -> tuple[str, str]:
    if name == "krutrim":
        return _secret("KRUTRIM_BASE_URL") or "https://cloud.olakrutrim.com/v1", _secret("KRUTRIM_API_KEY")
    return _secret("MISTRAL_BASE_URL") or "https://api.mistral.ai/v1", _secret("MISTRAL_API_KEY")


def _headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}


def _model_record(item: dict) -> dict:
    params = item.get("supported_parameters") or item.get("supported_params") or []
    if isinstance(params, str):
        params = [params]
    return {
        "id": item.get("id") or item.get("name"),
        "context_length": item.get("context_length") or item.get("max_model_len") or item.get("context_window"),
        "supported_parameters": list(params),
        "response_format_listed": "response_format" in {str(p) for p in params},
    }


def verify_krutrim_catalogue() -> dict:
    base, key = _provider_conf("krutrim")
    if not key:
        return {"ok": False, "error": "KRUTRIM_API_KEY is empty"}
    started = time.perf_counter()
    with httpx.Client(timeout=30.0) as client:
        response = client.get(f"{base.rstrip('/')}/models", headers=_headers(key))
    latency_ms = int((time.perf_counter() - started) * 1000)
    payload = {}
    try:
        payload = response.json()
    except Exception:
        payload = {}
    models = [_model_record(item) for item in (payload.get("data") or []) if isinstance(item, dict)]
    by_id = {item["id"]: item for item in models if item.get("id")}
    mismatches = []
    for model_id, expected in EXPECTED_KRUTRIM.items():
        actual = by_id.get(model_id)
        if actual is None:
            mismatches.append({"id": model_id, "error": "not_in_catalogue"})
            continue
        if int(actual.get("context_length") or 0) != expected:
            mismatches.append(
                {
                    "id": model_id,
                    "expected_context_length": expected,
                    "actual_context_length": actual.get("context_length"),
                }
            )
    extras = sorted(mid for mid in by_id if mid not in EXPECTED_KRUTRIM)
    return {
        "ok": response.status_code == 200 and not mismatches,
        "status": response.status_code,
        "latencyMs": latency_ms,
        "verified": [by_id.get(model_id) for model_id in EXPECTED_KRUTRIM],
        "mismatches": mismatches,
        "otherCatalogueIds": extras,
    }


def smoke_chat(provider: str, model: str) -> dict:
    base, key = _provider_conf(provider)
    if not key:
        return {"provider": provider, "model": model, "ok": False, "error": f"{provider} API key is empty"}
    timeout = 120.0 if "120b" in model else 60.0
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with the single word pong."}],
        "max_tokens": 256,
        "temperature": 0,
    }
    started = time.perf_counter()
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.post(
                f"{base.rstrip('/')}/chat/completions",
                headers=_headers(key),
                json=payload,
            )
        latency_ms = int((time.perf_counter() - started) * 1000)
        body = {}
        try:
            body = response.json()
        except Exception:
            body = {}
        choice = (body.get("choices") or [{}])[0]
        message = (choice.get("message") or {})
        content = str(message.get("content") or "")
        reasoning = str(message.get("reasoning_content") or message.get("reasoning") or "")
        usage = body.get("usage") or {}
        error = body.get("error") if isinstance(body.get("error"), dict) else None
        completion_tokens = int(usage.get("completion_tokens") or 0)
        return {
            "provider": provider,
            "model": model,
            "ok": response.status_code == 200 and bool(content.strip() or reasoning.strip() or completion_tokens > 0),
            "status": response.status_code,
            "latencyMs": latency_ms,
            "finishReason": choice.get("finish_reason"),
            "contentChars": len(content),
            "reasoningChars": len(reasoning),
            "completionTokens": completion_tokens,
            "messageKeys": sorted(str(key) for key in message.keys()),
            "errorType": (error or {}).get("type") or (error or {}).get("code"),
            "errorMessage": (error or {}).get("message") or (None if response.status_code == 200 else str(body)[:240]),
        }
    except Exception as exc:
        return {
            "provider": provider,
            "model": model,
            "ok": False,
            "latencyMs": int((time.perf_counter() - started) * 1000),
            "errorType": type(exc).__name__,
            "errorMessage": str(exc)[:240],
        }


def _artifact_rows(items) -> list[dict]:
    rows = []
    for item in items:
        rows.append(
            {
                "title": item.title,
                "ownerText": getattr(item, "ownerText", None),
                "dueDateText": getattr(item, "dueDateText", None),
                "confidence": getattr(item, "confidence", None),
            }
        )
    return rows


def _diag_subset(diagnostics: dict) -> dict:
    keys = (
        "finalSynthesisInvoked",
        "finalSynthesisProvider",
        "finalSynthesisModel",
        "finalSynthesisVerdict",
        "finalSynthesisRawTaskCount",
        "finalSynthesisRawNoteCount",
        "qualityAcceptedTaskCount",
        "qualityAcceptedNoteCount",
        "validatedSemanticUnitCount",
        "dropStage",
        "parsingOutcome",
        "structuredOutputOutcome",
        "extractionProvider",
        "extractionModel",
        "taskCoverageConflict",
        "validatedActionableUnitCount",
        "qualityRepairAttempted",
        "publishVerdictOverridden",
        "unitDispositions",
    )
    return {key: diagnostics.get(key) for key in keys if key in diagnostics}


async def run_buddy(case_id: str, transcript: str) -> dict:
    from services.conversation import agents
    from services.llm import router as llm_router

    llm_router._router = None
    started = time.perf_counter()
    try:
        result, provider, model = await agents.extract_from_raw_transcript(
            llm_router.get_llm_router(),
            case_id,
            "verify-user",
            "verify-space",
            transcript,
            {},
        )
        diagnostics = dict(result.extractionDiagnostics or {})
        synthesis_reached = bool(diagnostics.get("finalSynthesisInvoked")) or int(diagnostics.get("validatedSemanticUnitCount") or 0) > 0
        produced_task = bool(result.tasks)
        if case_id == "complex-meeting":
            passed = produced_task
        elif case_id == "long-meeting-two-hour":
            passed = produced_task or synthesis_reached
        else:
            passed = produced_task or bool(result.notes)
        return {
            "ok": result.extractionOutcome.value in {"SUCCESS", "VALID_EMPTY_EXTRACTION"} and passed,
            "caseId": case_id,
            "latencyMs": int((time.perf_counter() - started) * 1000),
            "extractionOutcome": result.extractionOutcome.value,
            "extractionError": result.extractionError,
            "semanticProvider": provider,
            "semanticModel": model,
            "taskCount": len(result.tasks),
            "noteCount": len(result.notes),
            "semanticUnitCount": len(result.semanticUnits),
            "tasks": _artifact_rows(result.tasks),
            "notes": _artifact_rows(result.notes),
            "diagnostics": _diag_subset(diagnostics),
        }
    except Exception as exc:
        return {
            "ok": False,
            "caseId": case_id,
            "latencyMs": int((time.perf_counter() - started) * 1000),
            "errorType": type(exc).__name__,
            "errorMessage": str(exc)[:400],
        }


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--skip-buddy", action="store_true")
    parser.add_argument("--buddy-only", action="store_true")
    args = parser.parse_args()
    catalogue = {"ok": True, "skipped": True} if args.buddy_only else verify_krutrim_catalogue()
    smokes = [] if args.buddy_only else [smoke_chat(provider, model) for provider, model in SMOKE_MODELS]
    buddy = []
    if args.buddy_only or (not args.smoke_only and not args.skip_buddy):
        long_case = next(item for item in BENCHMARK_CASES if item["id"] == "long-meeting-two-hour")
        buddy = [
            await run_buddy("complex-meeting", COMPLEX_MEETING_TRANSCRIPT),
            await run_buddy(long_case["id"], long_case["transcript"]),
        ]
    report = {
        "krutrimCatalogue": catalogue,
        "smoke": smokes,
        "buddy": buddy,
    }
    print(json.dumps(report, indent=2))
    catalogue_ok = True if args.buddy_only else bool(catalogue.get("ok"))
    smoke_ok = True if args.buddy_only else all(item.get("ok") for item in smokes)
    buddy_ok = True if args.smoke_only or args.skip_buddy else all(item.get("ok") for item in buddy)
    if not (catalogue_ok and smoke_ok and buddy_ok):
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
