"""Scale meeting transcripts for end-to-end count reports."""

from __future__ import annotations

from tests.fixtures.long_meeting_gold import FILLERS, build_gold_transcript
from services.conversation.models import STTStatus, TranscriptChunkDocument


TOPIC_SNIPPETS = [
    "S3 frontend tak nahi aa raha",
    "bucket ka response UI ko receive nahi ho raha",
    "Please create the server ID",
    "Pricing should start around 200",
    "database server connection string missing hai",
    "Play Store issue exist karta hai abhi",
    "old keys are currently in use",
    "microphone access required hai",
    "use GPT and OpenCV for coordinate extraction",
    "create meeting page banana hai dashboard pe",
    "Connection is insecure.",
    "kal testing karenge is flow ki",
    "PNB setup alag topic hai",
    "port tracking bhi pending dikh raha",
]


def build_scale_meeting(count: int) -> dict:
    if count <= 221:
        gold = build_gold_transcript()
        chunks = gold["chunks"][:count] if count < len(gold["chunks"]) else gold["chunks"]
        return {
            "id": f"scale-{count}",
            "chunks": chunks,
            "goldComplete": count >= 220,
            "goldTasks": gold["goldTasks"] if count >= 220 else [],
            "goldNotes": gold["goldNotes"] if count >= 220 else [],
            "validAdditionalNotes": gold.get("validAdditionalNotes") if count >= 220 else [],
            "events": gold["events"] if count >= 220 else None,
            "goldThreads": gold.get("goldThreads") if count >= 220 else None,
        }
    gold = build_gold_transcript()
    lines = dict(gold["lines"])
    for sequence in range(221, count):
        if sequence % 23 == 0:
            lines[sequence] = TOPIC_SNIPPETS[sequence % len(TOPIC_SNIPPETS)] + f" {sequence}"
        else:
            lines[sequence] = FILLERS[sequence % len(FILLERS)] + f" {sequence}"
    chunks = [
        TranscriptChunkDocument(
            conversationId=f"scale-{count}",
            userId="user_1",
            spaceId="space_1",
            chunkId=f"chunk_{sequence}",
            sequenceNumber=sequence,
            rawText=text,
            sttStatus=STTStatus.COMPLETED,
        )
        for sequence, text in sorted(lines.items())
        if sequence < count
    ]
    return {
        "id": f"scale-{count}",
        "chunks": chunks,
        "goldComplete": False,
        "goldTasks": [],
        "goldNotes": [],
        "events": None,
        "goldThreads": None,
    }
