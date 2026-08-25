"""Opt-in live accuracy benchmark against the current conversation pipeline.

This does not run in pytest. It calls the configured LLM router.

  py -3 scripts/run_conversation_eval.py
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from services.conversation import agents
from services.conversation.artifacts import artifacts_from_window
from services.conversation.eval_metrics import predicted_from_extraction, score_case, score_corpus
from services.llm.router import get_llm_router
from tests.eval.conversations import BENCHMARK_CASES


def _print_score(score) -> None:
    print(
        f"{score.caseId:40} cat={score.category:22} "
        f"taskR={score.taskRecall:.2f} taskP={score.taskPrecision:.2f} "
        f"noteR={score.noteRecall:.2f} dup={score.duplicateRate:.2f} "
        f"falseT={score.falseTaskRate:.2f} owner={score.ownerAccuracy} "
        f"due={score.deadlineAccuracy} evid={score.evidenceAccuracy} "
        f"xwin={score.crossWindowUpdateAccuracy}"
    )


async def _score_live(case: dict):
    router = get_llm_router()
    result, provider, model = await agents.extract_from_raw_transcript(
        router,
        case["id"],
        "eval-user",
        "eval-space",
        case["transcript"],
        {},
    )
    window = SimpleNamespace(
        text=case["transcript"],
        conversationId=case["id"],
        userId="eval-user",
        spaceId="eval-space",
        id=f"{case['id']}:raw",
        sequenceStart=0,
        sequenceEnd=0,
        windowIndex=0,
    )
    artifacts = artifacts_from_window(window, result)
    predicted = predicted_from_extraction(result)
    score = score_case(case, predicted)
    score.details["provider"] = provider
    score.details["model"] = model
    score.details["artifactCount"] = len(artifacts)
    return score


async def main() -> None:
    scores = []
    for case in BENCHMARK_CASES:
        score = await _score_live(case)
        _print_score(score)
        scores.append(score)
    corpus = score_corpus(scores)
    print(
        json.dumps(
            {
                "cases": len(scores),
                "taskRecall": corpus.taskRecall,
                "taskPrecision": corpus.taskPrecision,
                "noteRecall": corpus.noteRecall,
                "notePrecision": corpus.notePrecision,
                "duplicateRate": corpus.duplicateRate,
                "falseTaskRate": corpus.falseTaskRate,
                "ownerAccuracy": corpus.ownerAccuracy,
                "deadlineAccuracy": corpus.deadlineAccuracy,
                "evidenceAccuracy": corpus.evidenceAccuracy,
                "crossWindowUpdateAccuracy": corpus.crossWindowUpdateAccuracy,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
