"""Dry-run real-model evaluation for the frozen meeting pipeline.

Uses configured SEMANTIC_EXTRACTION, FINAL_SYNTHESIS, and VALIDATION routes.
Does not persist tasks/notes to production collections.

  py -3 -m services.conversation.eval_real_models
  py -3 -m services.conversation.eval_real_models --suite lenskart
  py -3 -m services.conversation.eval_real_models --suite atomic
  py -3 -m services.conversation.eval_real_models --suite long
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from collections import Counter
from typing import Any

from datetime import datetime, timezone

from apps.api_gateway.config.setting import settings
from services.conversation.eval_metrics import (
    CaseScore,
    PredictedItem,
    _as_gold_item,
    _is_required_gold,
    predicted_from_extraction,
    score_case,
    score_corpus,
    semantic_align_score,
)
from services.conversation.eval_semantic import embedding_scores_for_meanings
from services.conversation.meeting_pipeline.pipeline import run_meeting_pipeline
from services.conversation.models import STTStatus, TranscriptChunkDocument
from services.llm.router import LLMCapability, get_llm_router

_ALIGN = 0.42
_LINE_RE = re.compile(r"^\[(\d+)\]\s*(.*)$")
_EVAL_MEETING_AT = datetime(2026, 8, 28, tzinfo=timezone.utc)


def _meeting_at(case: dict[str, Any]) -> datetime:
    raw = case.get("meetingTimestamp") or case.get("meeting_at")
    if isinstance(raw, datetime):
        return raw
    if raw:
        try:
            return datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        except ValueError:
            pass
    return _EVAL_MEETING_AT


def chunks_from_transcript(
    conversation_id: str,
    transcript: str,
    user_id: str = "eval-user",
    space_id: str = "eval-space",
) -> list[TranscriptChunkDocument]:
    chunks: list[TranscriptChunkDocument] = []
    for raw_line in (transcript or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = _LINE_RE.match(line)
        if match:
            sequence = int(match.group(1))
            text = match.group(2).strip()
        else:
            sequence = len(chunks)
            text = line
        chunks.append(
            TranscriptChunkDocument(
                conversationId=conversation_id,
                userId=user_id,
                spaceId=space_id,
                chunkId=f"{conversation_id}:{sequence}",
                sequenceNumber=sequence,
                rawText=text,
                normalizedText=text,
                sttStatus=STTStatus.COMPLETED,
            )
        )
    return chunks


def normalize_case(case: dict[str, Any]) -> dict[str, Any]:
    payload = dict(case)
    payload["expectedTasks"] = payload.get("expectedTasks") or payload.get("goldTasks") or []
    payload["expectedNotes"] = payload.get("expectedNotes") or payload.get("goldNotes") or []
    payload["goldTasks"] = payload["expectedTasks"]
    payload["goldNotes"] = payload["expectedNotes"]
    payload["forbiddenArtifacts"] = payload.get("forbiddenArtifacts") or []
    payload["expectedEvidence"] = payload.get("expectedEvidence") or [
        {"id": item.get("id"), "kind": item.get("kind"), "evidenceSequences": item.get("evidenceSequences") or []}
        for item in [*payload["expectedTasks"], *payload["expectedNotes"]]
    ]
    return payload


def classify_first_loss(case: dict[str, Any], result, score: CaseScore) -> str | None:
    if result is None:
        return "MODEL_PROVIDER_FAILURE"
    predicted = predicted_from_extraction(result)
    candidates = [
        PredictedItem(kind="note", meaning=item.meaning, evidenceSequences=list(item.evidenceSequences))
        for item in (result.candidates or [])
    ]
    claims = [
        PredictedItem(kind=item.kind, meaning=f"{item.title}\n{item.body}".strip(), evidenceSequences=list(item.evidenceSequences))
        for item in (result.claims or [])
    ]
    rejected = [
        PredictedItem(kind=item.kind, meaning=f"{item.title}\n{item.body}".strip(), evidenceSequences=list(item.evidenceSequences))
        for item in (result.rejected or [])
    ]

    def _retained(gold: dict) -> bool:
        blob = "\n".join(item.meaning for item in predicted)
        if semantic_align_score(gold.get("meaning") or "", blob) >= _ALIGN:
            return True
        gold_id = str(gold.get("id") or "")
        return gold_id in (score.details or {}).get("retainedGoldIds") or []

    def _hit(gold: dict, pool: list[PredictedItem]) -> PredictedItem | None:
        best = None
        best_score = 0.0
        gold_seqs = set(gold.get("evidenceSequences") or [])
        for item in pool:
            overlap = len(gold_seqs & set(item.evidenceSequences)) if gold_seqs else 0
            aligned = semantic_align_score(gold.get("meaning") or "", item.meaning)
            ranked = aligned + (0.15 if overlap else 0.0)
            if ranked > best_score:
                best_score = ranked
                best = item
        if best is None or best_score < _ALIGN:
            return None
        return best

    def _required(items: list[dict]) -> list[dict]:
        required = []
        for gold in items:
            status = str(gold.get("reviewStatus") or "REQUIRED").upper()
            if status in {"", "REQUIRED"}:
                required.append(gold)
        return required

    for forbidden in case.get("forbiddenArtifacts") or []:
        blob_match = [
            item
            for item in predicted
            if semantic_align_score(str(forbidden), item.meaning) >= _ALIGN or str(forbidden).casefold() in item.meaning.casefold()
        ]
        if not blob_match:
            continue
        if any(
            semantic_align_score(str(forbidden), item.meaning) >= _ALIGN or str(forbidden).casefold() in item.meaning.casefold()
            for item in candidates
        ):
            return "FALSE_POSITIVE"
        return "FALSE_POSITIVE"

    if (score.backgroundFalsePositiveRate or 0) > 0:
        return "FALSE_POSITIVE"

    missing = []
    for gold in [*_required(case.get("goldTasks") or []), *_required(case.get("goldNotes") or [])]:
        if _hit(gold, predicted) or _retained(gold):
            continue
        missing.append(gold)
        if _hit(gold, rejected) or (
            _hit(gold, claims) and any(
                semantic_align_score(gold.get("meaning") or "", item.meaning) >= _ALIGN for item in rejected
            )
        ):
            return "VERIFIER_FALSE_REJECT"
        if not _hit(gold, candidates) and not _hit(gold, claims):
            return "REAL_INFORMATION_LOSS"
        if _hit(gold, claims) and not predicted:
            return "VERIFIER_FALSE_REJECT"
        return "REAL_INFORMATION_LOSS"

    if missing:
        return "REAL_INFORMATION_LOSS"
    if (score.duplicateRate or 0) > 0.03 and (score.duplicateCount or 0) > 0:
        return "DUPLICATE_PRESENTATION"
    if score.ownerAccuracy is not None and score.ownerAccuracy < 1:
        return "OWNER_FIELD_ERROR"
    if score.deadlineAccuracy is not None and score.deadlineAccuracy < 1:
        return "DEADLINE_FIELD_ERROR"
    if (score.taskRecall if score.taskRecall is not None else 1) < 1 or (
        score.noteRecall if score.noteRecall is not None else 1
    ) < 1:
        return "ARTIFACT_POLICY_DIFFERENCE"
    return None


def _usage_by_stage(usage: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for item in usage or []:
        stage = str(item.get("stage") or "unknown")
        bucket = grouped.setdefault(stage, {"calls": 0, "inputTokens": 0, "outputTokens": 0, "latencyMs": 0, "fallback": 0})
        bucket["calls"] += 1
        bucket["inputTokens"] += int(item.get("inputTokens") or 0)
        bucket["outputTokens"] += int(item.get("outputTokens") or 0)
        bucket["latencyMs"] += int(item.get("latencyMs") or 0)
        bucket["fallback"] += 1 if item.get("fallback") else 0
    return grouped


def _atomic_coverage(case: dict[str, Any], result) -> dict[str, Any] | None:
    expected = [item.get("meaning") for item in (case.get("goldCandidates") or []) if item.get("meaning")]
    if not expected:
        return None
    candidates = list(result.candidates or [])
    hits = []
    misses = []
    for meaning in expected:
        best = 0.0
        matched = None
        for item in candidates:
            score = semantic_align_score(meaning, item.meaning)
            if score > best:
                best = score
                matched = item.meaning
        if best >= _ALIGN:
            hits.append({"expected": meaning, "matched": matched, "score": best})
        else:
            misses.append({"expected": meaning, "bestScore": best, "bestCandidate": matched})
    return {
        "expectedCount": len(expected),
        "hitCount": len(hits),
        "recall": (len(hits) / len(expected)) if expected else 1.0,
        "hits": hits,
        "misses": misses,
        "extractorMeanings": [item.meaning for item in candidates],
    }


def _trace(case: dict[str, Any], result) -> dict[str, Any]:
    obs = result.observability or {}
    return {
        "conversation_id": case["id"],
        "window_ranges": obs.get("window_sequence_ranges") or [],
        "extraction_window_records": obs.get("extraction_window_records") or [],
        "emptyCandidateWindowRate": obs.get("emptyCandidateWindowRate"),
        "usefulWindowsWithZeroCandidates": obs.get("usefulWindowsWithZeroCandidates"),
        "extractor_output_per_window": [
            {
                "windowId": window.window_id,
                "start": window.sequence_start,
                "end": window.sequence_end,
                "candidates": [
                    {
                        "candidateId": item.candidateId,
                        "kind": item.kind.value if hasattr(item.kind, "value") else str(item.kind),
                        "meaning": item.meaning,
                        "evidenceSequences": item.evidenceSequences,
                    }
                    for item in result.candidates
                    if item.sourceWindowId == window.window_id
                ],
            }
            for window in result.windows
        ],
        "candidate_ledger": result.ledgerPayload,
        "consolidator_output": [
            {
                "artifactKey": item.artifactKey,
                "kind": item.kind,
                "title": item.title,
                "body": item.body,
                "owner": item.owner,
                "dueDate": item.dueDate,
                "sourceCandidateIds": item.sourceCandidateIds,
                "evidenceSequences": item.evidenceSequences,
            }
            for item in result.claims
        ],
        "verifier_verdicts": [
            {
                "artifactKey": item.artifactKey,
                "kind": item.kind,
                "title": item.title,
                "verdict": item.verdict.value,
                "owner": item.owner,
                "dueDate": item.dueDate,
                "evidenceSequences": item.evidenceSequences,
                "fieldSupport": item.fieldSupport.model_dump() if item.fieldSupport else {},
                "reason": item.reason,
            }
            for item in [*result.verified, *result.rejected]
        ],
        "rejected_artifacts": [
            {"title": item.title, "kind": item.kind, "reason": item.reason, "verdict": item.verdict.value}
            for item in result.rejected
        ],
        "final_output": {
            "tasks": [
                {
                    "title": item.title,
                    "body": item.body,
                    "owner": item.ownerText,
                    "dueDate": item.dueDateText,
                    "evidence": [span.sequenceStart for span in item.evidence],
                    "confidence": item.confidence,
                    "evidenceVerified": (item.changes or {}).get("evidenceVerified"),
                    "verificationVerdict": (item.changes or {}).get("verificationVerdict"),
                }
                for item in result.tasks
            ],
            "notes": [
                {
                    "title": item.title,
                    "body": item.body,
                    "evidence": [span.sequenceStart for span in item.evidence],
                    "confidence": item.confidence,
                    "evidenceVerified": (item.debug or {}).get("evidenceVerified"),
                    "verificationVerdict": (item.debug or {}).get("verificationVerdict"),
                }
                for item in result.notes
            ],
        },
        "actual_models": {
            "extractor": {"provider": obs.get("extractor_provider"), "model": obs.get("extractor_model")},
            "consolidator": {"provider": obs.get("consolidator_provider"), "model": obs.get("consolidator_model")},
            "verifier": {"provider": obs.get("verifier_provider"), "model": obs.get("verifier_model")},
        },
        "tokens": {
            "input": obs.get("input_tokens"),
            "output": obs.get("output_tokens"),
            "byStage": _usage_by_stage(result.usage),
            "calls": result.usage,
        },
        "latency_ms": obs.get("processing_duration_ms"),
        "persisted": False,
        "dryRun": True,
    }


async def run_case(router, case: dict[str, Any]) -> dict[str, Any]:
    case = normalize_case(case)
    chunks = chunks_from_transcript(case["id"], case["transcript"])
    original = None
    if case.get("forceSmallWindows"):
        original = (
            settings.EXTRACTION_WINDOW_TARGET_TOKENS,
            settings.EXTRACTION_WINDOW_MAX_TOKENS,
            settings.EXTRACTION_WINDOW_OVERLAP_RATIO,
        )
        settings.EXTRACTION_WINDOW_TARGET_TOKENS = 200
        settings.EXTRACTION_WINDOW_MAX_TOKENS = 280
        settings.EXTRACTION_WINDOW_OVERLAP_RATIO = 0.15
    started = time.perf_counter()
    try:
        result = await run_meeting_pipeline(
            chunks,
            case["id"],
            "eval-user",
            "eval-space",
            router=router,
            meeting_at=_meeting_at(case),
        )
        error = None
    except Exception as exc:
        result = None
        error = f"{type(exc).__name__}: {exc}"
    finally:
        if original is not None:
            (
                settings.EXTRACTION_WINDOW_TARGET_TOKENS,
                settings.EXTRACTION_WINDOW_MAX_TOKENS,
                settings.EXTRACTION_WINDOW_OVERLAP_RATIO,
            ) = original
    if result is None:
        return {
            "id": case["id"],
            "category": case.get("category"),
            "error": error,
            "firstLoss": "MODEL_PROVIDER_FAILURE",
            "latency_ms": int((time.perf_counter() - started) * 1000),
            "dryRun": True,
            "persisted": False,
        }
    predicted = predicted_from_extraction(result)
    candidates = [
        PredictedItem(kind="note", meaning=item.meaning, evidenceSequences=list(item.evidenceSequences))
        for item in result.candidates
    ]
    score = score_case(case, predicted, predicted_candidates=candidates)
    if (score.meaningRetentionRecall or 1) < 1:
        required = [
            gold
            for gold in [
                _as_gold_item(item)
                for item in [*(case.get("goldTasks") or []), *(case.get("goldNotes") or [])]
            ]
            if _is_required_gold([gold], gold.id)
        ]
        embed_scores = await embedding_scores_for_meanings(required, predicted)
        if embed_scores:
            score = score_case(
                case,
                predicted,
                predicted_candidates=candidates,
                meaning_embedding_scores=embed_scores,
            )
    first_loss = classify_first_loss(case, result, score)
    usage_by_stage = _usage_by_stage(result.usage)
    cited = int(result.observability.get("consolidator_cited_sequence_count") or 0)
    total_seq = int(result.observability.get("transcript_sequence_count") or len(chunks))
    payload = {
        "id": case["id"],
        "category": case.get("category"),
        "probe": case.get("probe"),
        "taskRecall": score.taskRecall,
        "taskPrecision": score.taskPrecision,
        "noteRecall": score.noteRecall,
        "notePrecision": score.notePrecision,
        "candidateRecall": score.candidateRecall,
        "evidencePrecision": score.evidenceAccuracy,
        "unsupportedArtifactRate": score.unsupportedArtifactRate,
        "backgroundFalsePositiveRate": score.backgroundFalsePositiveRate,
        "duplicateRate": score.duplicateRate,
        "ownerAccuracy": score.ownerAccuracy,
        "deadlineAccuracy": score.deadlineAccuracy,
        "meaningRetentionRecall": score.meaningRetentionRecall,
        "firstLoss": first_loss,
        "score": score,
        "models": {
            "extractor": result.observability.get("extractor_model"),
            "consolidator": result.observability.get("consolidator_model"),
            "verifier": result.observability.get("verifier_model"),
            "extractorProvider": result.observability.get("extractor_provider"),
            "consolidatorProvider": result.observability.get("consolidator_provider"),
            "verifierProvider": result.observability.get("verifier_provider"),
        },
        "stats": {
            "windowCount": result.observability.get("window_count"),
            "candidateCount": result.observability.get("total_candidate_count"),
            "ledgerSize": result.observability.get("ledger_size"),
            "taskCount": len(result.tasks),
            "noteCount": len(result.notes),
            "modelCalls": result.observability.get("model_calls"),
            "extractorCalls": result.observability.get("extractor_calls"),
            "consolidatorCalls": result.observability.get("consolidator_calls"),
            "verifierCalls": result.observability.get("verifier_calls"),
            "repairCalls": result.observability.get("repair_calls"),
            "maxInflight": result.observability.get("max_extraction_inflight"),
            "inputTokens": result.observability.get("input_tokens"),
            "outputTokens": result.observability.get("output_tokens"),
            "extractorInputTokens": (usage_by_stage.get("extractor") or {}).get("inputTokens"),
            "consolidatorInputTokens": (usage_by_stage.get("consolidator") or {}).get("inputTokens"),
            "verifierInputTokens": (usage_by_stage.get("verifier") or {}).get("inputTokens"),
            "latencyMs": result.observability.get("processing_duration_ms"),
            "transcriptSequenceCount": total_seq,
            "consolidatorCitedSequenceCount": cited,
            "consolidatorReceivedFullTranscript": bool(total_seq and cited >= total_seq and total_seq > 8),
            "emptyCandidateWindowRate": result.observability.get("emptyCandidateWindowRate"),
            "usefulWindowsWithZeroCandidates": result.observability.get("usefulWindowsWithZeroCandidates"),
            "providerFallback": any(item.get("fallback") for item in result.usage),
        },
        "atomicCoverage": _atomic_coverage(case, result),
        "longMeeting": _long_meeting_survival(case, result) if case.get("probe") == "long" else None,
        "trace": _trace(case, result),
        "dryRun": True,
        "persisted": False,
    }
    return payload


def build_long_meeting(sequences: int = 60) -> dict[str, Any]:
    def _chatter(index: int) -> str:
        block = (
            f"We are still walking through routing option {index}. No one is assigned on this stretch. "
            "Quotas look the same as last week and there is no new commitment here. "
            "People are repeating the same status without deciding a next action. "
        )
        return (block * 2).strip()

    mid = sequences // 2
    end = sequences - 1
    lines = [f"[0] START: Mira will implement the drain gate in week one. {_chatter(0)}"]
    for index in range(1, mid):
        lines.append(f"[{index}] {_chatter(index)}")
    lines.append(f"[{mid}] MIDPOINT: Kabir will add the retry budget dashboard this month. {_chatter(mid)}")
    for index in range(mid + 1, end):
        lines.append(f"[{index}] {_chatter(index)}")
    lines.append(f"[{end}] END: Neha will publish the meeting notes after STOP drain completes. {_chatter(end)}")
    return normalize_case(
        {
            "id": "long-meeting-production",
            "category": "long_meeting",
            "transcript": "\n".join(lines),
            "goldTasks": [
                {"id": "t-begin", "kind": "task", "meaning": "Mira will implement the drain gate in week one", "evidenceSequences": [0], "ownerText": "Mira"},
                {"id": "t-mid", "kind": "task", "meaning": "Kabir will add the retry budget dashboard this month", "evidenceSequences": [mid], "ownerText": "Kabir"},
                {"id": "t-end", "kind": "task", "meaning": "Neha will publish the meeting notes after STOP drain", "evidenceSequences": [end], "ownerText": "Neha"},
            ],
            "goldNotes": [],
            "probe": "long",
        }
    )


def _count_first_loss(rows: list[dict[str, Any]]) -> Counter:
    return Counter(str(row.get("firstLoss")) for row in rows if row.get("firstLoss"))


def _long_meeting_survival(case: dict[str, Any], result) -> dict[str, Any]:
    predicted = predicted_from_extraction(result)
    records = list((result.observability or {}).get("extraction_window_records") or [])
    planted = [
        ("begin", case["goldTasks"][0] if case.get("goldTasks") else None),
        ("mid", case["goldTasks"][1] if len(case.get("goldTasks") or []) > 1 else None),
        ("end", case["goldTasks"][2] if len(case.get("goldTasks") or []) > 2 else None),
    ]
    rows = []
    for label, gold in planted:
        if not gold:
            continue
        sequence = int((gold.get("evidenceSequences") or [None])[0])
        source_window = None
        extractor_candidate = None
        for window in result.windows or []:
            if sequence in (window.owned_sequence_ids or window.sequence_ids or []):
                source_window = {
                    "windowId": window.window_id,
                    "start": window.sequence_start,
                    "end": window.sequence_end,
                    "owned": sequence in (window.owned_sequence_ids or []),
                    "candidateCount": sum(1 for item in result.candidates if item.sourceWindowId == window.window_id),
                }
                if sequence in (window.owned_sequence_ids or []):
                    break
        for item in result.candidates or []:
            if sequence in item.evidenceSequences or semantic_align_score(gold["meaning"], item.meaning) >= _ALIGN:
                extractor_candidate = {"meaning": item.meaning, "evidenceSequences": item.evidenceSequences, "windowId": item.sourceWindowId}
                break
        artifact = None
        for item in predicted:
            if semantic_align_score(gold["meaning"], item.meaning) >= _ALIGN:
                artifact = {"kind": item.kind, "meaning": item.meaning, "evidence": item.evidenceSequences, "owner": item.ownerText}
                break
        in_input = any(sequence in (record.get("sequenceIds") or []) for record in records)
        if not in_input:
            in_input = any(sequence in (window.sequence_ids or []) for window in result.windows or [])
        rows.append(
            {
                "position": label,
                "sequenceId": sequence,
                "presentInWindowInput": in_input,
                "sourceWindow": source_window,
                "extractorCandidate": extractor_candidate,
                "finalArtifact": artifact,
                "found": artifact is not None,
            }
        )
    window_counts = [
        {
            "windowId": window.window_id,
            "start": window.sequence_start,
            "end": window.sequence_end,
            "candidateCount": sum(1 for item in result.candidates if item.sourceWindowId == window.window_id),
            "containsEndSequence": (case["goldTasks"][-1]["evidenceSequences"][0] in window.sequence_ids) if case.get("goldTasks") else False,
        }
        for window in result.windows or []
    ]
    return {
        "begin": next((row["found"] for row in rows if row["position"] == "begin"), False),
        "middle": next((row["found"] for row in rows if row["position"] == "mid"), False),
        "end": next((row["found"] for row in rows if row["position"] == "end"), False),
        "positions": rows,
        "windows": window_counts,
        "usefulWindowsWithZeroCandidates": (result.observability or {}).get("usefulWindowsWithZeroCandidates"),
        "emptyCandidateWindowRate": (result.observability or {}).get("emptyCandidateWindowRate"),
    }


def _print_case(row: dict[str, Any]) -> None:
    loss = row.get("firstLoss") or "ok"
    print(
        f"{row['id']:40} taskR={row.get('taskRecall')} noteR={row.get('noteRecall')} "
        f"evid={row.get('evidencePrecision')} bgFP={row.get('backgroundFalsePositiveRate')} "
        f"dup={row.get('duplicateRate')} loss={loss}"
    )


def configured_routes(router) -> dict[str, Any]:
    payload = {}
    for capability in (LLMCapability.SEMANTIC_EXTRACTION, LLMCapability.FINAL_SYNTHESIS, LLMCapability.VALIDATION):
        provider, model = router.route(capability)
        payload[capability.value] = {
            "provider": getattr(provider, "name", str(provider)),
            "model": model,
        }
    return payload


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _jsonable(row: dict[str, Any]) -> dict[str, Any]:
    payload = dict(row)
    payload.pop("score", None)
    return payload


async def async_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Dry-run real-model evaluation of the meeting pipeline")
    parser.add_argument("--suite", default="all", choices=("all", "gold", "lenskart", "atomic", "long", "noisy", "task_note", "cross_window", "tail"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)

    from tests.eval.meeting_gold import gold_cases, tail_position_cases
    from services.llm.router import llm_provider_status

    router = get_llm_router()
    routes = configured_routes(router)
    providers = llm_provider_status()
    print("Configured routes:", json.dumps(routes, indent=2))
    print("Provider status:", json.dumps(providers, indent=2))
    print("dryRun=true persisted=false")

    missing = []
    ready = {item["provider"]: item["configured"] for item in providers}
    for role, spec in routes.items():
        name = str(spec.get("provider") or "")
        if name and not ready.get(name, True):
            missing.append({"role": role, "provider": name, "model": spec.get("model")})
    if missing:
        summary = {
            "suite": args.suite,
            "dryRun": True,
            "persisted": False,
            "routes": routes,
            "providerStatus": providers,
            "failures": [
                {
                    "id": "preflight",
                    "firstLoss": "MODEL_PROVIDER_FAILURE",
                    "error": "routed provider is not configured: "
                    + ", ".join(f"{item['role']}={item['provider']}/{item['model']}" for item in missing),
                }
            ],
            "metrics": None,
            "caseCount": 0,
        }
        print("\n=== SUMMARY ===")
        print(json.dumps(summary, indent=2))
        if args.out:
            with open(args.out, "w", encoding="utf-8") as handle:
                json.dump(summary, handle, ensure_ascii=False, indent=2)
            print("Wrote", args.out)
        print(
            "Real-model evaluation did not call LLMs because the routed extractor/consolidator "
            "provider is not configured. Add the provider API key and rerun; do not substitute a different provider in this runner."
        )
        return 1

    cases = gold_cases()
    if args.suite == "lenskart":
        cases = [item for item in cases if item["id"] == "lenskart-hrms-meeting"]
    elif args.suite == "atomic":
        cases = [item for item in cases if item.get("probe") == "atomic"]
    elif args.suite == "noisy":
        cases = [item for item in cases if item.get("probe") == "noisy_number"]
    elif args.suite == "task_note":
        cases = [item for item in cases if item.get("probe") == "task_note"]
    elif args.suite == "cross_window":
        cases = [item for item in cases if item.get("probe") == "cross_window"]
    elif args.suite == "long":
        cases = [build_long_meeting()]
    elif args.suite == "tail":
        cases = tail_position_cases()
    elif args.suite == "gold":
        pass
    else:
        cases = [*cases, *tail_position_cases(), build_long_meeting()]
    if args.limit:
        cases = cases[: args.limit]

    rows: list[dict[str, Any]] = []
    for case in cases:
        print(f"\n=== {case['id']} ===")
        row = await run_case(router, case)
        _print_case(row)
        if row.get("error"):
            print("ERROR:", row["error"])
        elif row.get("trace"):
            compact = {
                "models": row.get("models"),
                "stats": row.get("stats"),
                "atomicCoverage": row.get("atomicCoverage"),
                "longMeeting": row.get("longMeeting"),
                "final_output": row["trace"]["final_output"],
                "rejected": row["trace"]["rejected_artifacts"],
            }
            print(json.dumps(compact, ensure_ascii=True, indent=2, default=str))
        rows.append(row)

    scored_rows = [row for row in rows if row.get("score") is not None]
    corpus = score_corpus([row["score"] for row in scored_rows]) if scored_rows else None
    failures = [row for row in rows if row.get("firstLoss")]
    summary = {
        "suite": args.suite,
        "caseCount": len(rows),
        "goldConversationCount": len(gold_cases()),
        "dryRun": True,
        "persisted": False,
        "routes": routes,
        "metrics": None
        if corpus is None
        else {
            "taskRecall": corpus.taskRecall,
            "taskPrecision": corpus.taskPrecision,
            "noteRecall": corpus.noteRecall,
            "notePrecision": corpus.notePrecision,
            "candidateRecall": corpus.candidateRecall,
            "meaningRetentionRecall": corpus.meaningRetentionRecall,
            "evidencePrecision": corpus.evidenceAccuracy,
            "unsupportedArtifactRate": corpus.unsupportedArtifactRate,
            "backgroundFalsePositiveRate": corpus.backgroundFalsePositiveRate,
            "duplicateRate": corpus.duplicateRate,
            "ownerAccuracy": corpus.ownerAccuracy,
            "deadlineAccuracy": corpus.deadlineAccuracy,
        },
        "averages": {
            "extractorCandidatesPerWindow": _mean(
                [
                    (row["stats"]["candidateCount"] / max(row["stats"]["windowCount"] or 1, 1))
                    for row in scored_rows
                    if row.get("stats")
                ]
            ),
            "finalTasks": _mean([row["stats"]["taskCount"] for row in scored_rows if row.get("stats")]),
            "finalNotes": _mean([row["stats"]["noteCount"] for row in scored_rows if row.get("stats")]),
            "inputTokens": _mean([row["stats"]["inputTokens"] or 0 for row in scored_rows if row.get("stats")]),
            "outputTokens": _mean([row["stats"]["outputTokens"] or 0 for row in scored_rows if row.get("stats")]),
            "modelCalls": _mean([row["stats"]["modelCalls"] or 0 for row in scored_rows if row.get("stats")]),
            "emptyCandidateWindowRate": _mean(
                [row["stats"].get("emptyCandidateWindowRate") or 0 for row in scored_rows if row.get("stats")]
            ),
            "usefulWindowsWithZeroCandidates": _mean(
                [row["stats"].get("usefulWindowsWithZeroCandidates") or 0 for row in scored_rows if row.get("stats")]
            ),
        },
        "firstLossCounts": dict(_count_first_loss(failures)),
        "informationLossCounts": dict(
            _count_first_loss([row for row in failures if row.get("firstLoss") == "REAL_INFORMATION_LOSS"])
        ),
        "failures": [{"id": row["id"], "firstLoss": row.get("firstLoss"), "error": row.get("error")} for row in failures],
        "cases": [_jsonable(row) for row in rows],
    }
    print("\n=== SUMMARY ===")
    print(json.dumps({key: summary[key] for key in ("caseCount", "metrics", "averages", "firstLossCounts", "informationLossCounts", "failures", "routes") if key in summary}, indent=2, default=str, ensure_ascii=True))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2, default=str)
        print("Wrote", args.out)
    return 0 if not failures else 1


def main() -> None:
    raise SystemExit(asyncio.run(async_main(sys.argv[1:])))


if __name__ == "__main__":
    main()
