"""Gold long-meeting fixture: 220+ noisy Hindi/Hinglish/English STT chunks.

Expected artifacts are concept-level, not exact wording. Negative assertions
cover the production bugs: generic tasks and mixed-thread evidence.
"""

from __future__ import annotations

from services.conversation.event_pipeline.schemas import ActionSignal, AtomicEvent, EventKind, MemorySignal
from services.conversation.models import EvidenceSpan, STTStatus, TranscriptChunkDocument


FILLERS = [
    "haan",
    "ok wait",
    "theek hai",
    "umm actually",
    "so yeah",
    "ek second",
    "hello hello",
    "mic is rustling",
    "background mein noise hai",
    "achha achha",
]


def _event(event_id: str, kind: EventKind, meaning: str, sequences: list[int], texts: dict[int, str], **kwargs) -> AtomicEvent:
    evidence = [
        EvidenceSpan(sequenceStart=seq, sequenceEnd=seq, text=texts[seq])
        for seq in sequences
        if seq in texts
    ]
    return AtomicEvent(
        eventId=event_id,
        topicId=kwargs.get("topicId", "T1"),
        kind=kind,
        meaning=meaning,
        actor=kwargs.get("actor"),
        object=kwargs.get("object"),
        timeExpression=kwargs.get("timeExpression"),
        entities=kwargs.get("entities") or [],
        evidence=evidence,
        sequenceIds=list(sequences),
        sourceIds=[f"chunk_{seq}" for seq in sequences],
        conversationId="gold-long-meeting",
        userId="user_1",
        spaceId="space_1",
        actionSignal=kwargs.get("actionSignal"),
        memorySignal=kwargs.get("memorySignal"),
        fieldEvidence=kwargs.get("fieldEvidence"),
    )


def build_gold_transcript() -> dict:
    lines: dict[int, str] = {}

    def put(sequence: int, text: str) -> None:
        lines[sequence] = text

    for sequence in range(0, 220):
        put(sequence, FILLERS[sequence % len(FILLERS)] + f" {sequence}")

    put(0, "")
    put(1, "   ")
    put(2, "null")
    put(3, "...")
    put(4, "haan haan meeting start")
    put(20, "S3 is not reaching frontend")
    put(21, "We need to fix that tomorrow")
    put(22, "configuration change pending hai")
    put(40, "Pricing should start around 200")
    put(41, "free plan pe usage limit discuss karte hain")
    put(42, "Monday meeting mein pricing finalize karenge")
    put(60, "database server connection string missing hai")
    put(61, "network parameter information nahi mil rahi")
    put(62, "Connection is insecure.")
    put(63, "PNB setup alag topic hai")
    put(64, "port tracking bhi pending dikh raha")
    put(80, "Play Store issue exist karta hai abhi")
    put(90, "old keys are currently in use")
    put(91, "generated notes were reviewed and improved")
    put(92, "master-prompt output requirements document karo")
    put(105, "S3 configuration was changed")
    put(110, "Server ID create karna hai")
    put(111, "please create the server ID")
    put(130, "create meeting page banana hai dashboard pe")
    put(150, "use GPT and OpenCV for coordinate extraction")
    put(160, "microphone access required hai, currently available nahi")
    put(161, "please request microphone access")
    put(170, "kal testing karenge is flow ki")
    put(171, "we will test this tomorrow")
    put(180, "notes generate hue the unko improve kiya")
    put(200, "unrelated lunch discussion about biryani")
    put(220, "S3 still does not reach frontend")

    chunks = [
        TranscriptChunkDocument(
            conversationId="gold-long-meeting",
            userId="user_1",
            spaceId="space_1",
            chunkId=f"chunk_{sequence}",
            sequenceNumber=sequence,
            rawText=text,
            sttStatus=STTStatus.COMPLETED,
        )
        for sequence, text in sorted(lines.items())
    ]
    events = [
        _event("e-s3-issue", EventKind.ISSUE, "S3 is not reaching the frontend.", [20], lines, object="S3 frontend integration", entities=["S3", "frontend"], memorySignal=MemorySignal(isMemoryWorthy=True, importance="HIGH")),
        _event("e-s3-commit", EventKind.COMMITMENT, "The S3 issue will be fixed tomorrow.", [21], lines, object="S3 issue", timeExpression="tomorrow", entities=["S3"], actionSignal=ActionSignal(isActionable=True, role="COMMITMENT", actionStrength="EXPLICIT", verb="fix", object="S3 issue", objectGroundingType="LOCAL_COREFERENCE", deadline="tomorrow")),
        _event("e-pricing", EventKind.PROPOSAL, "Pricing should start around 200.", [40], lines, object="pricing", entities=["Pricing"], memorySignal=MemorySignal(isMemoryWorthy=True, importance="MEDIUM")),
        _event("e-monday", EventKind.FACT, "Monday meeting was mentioned for pricing finalization.", [42], lines, object="Monday meeting", timeExpression="Monday", entities=["Monday"], memorySignal=MemorySignal(isMemoryWorthy=True, importance="MEDIUM")),
        _event("e-pricing-finalize", EventKind.COMMITMENT, "The team will finalize pricing in the Monday meeting.", [42], lines, object="pricing", entities=["Monday", "pricing"], actionSignal=ActionSignal(isActionable=True, role="COMMITMENT", actionStrength="EXPLICIT", verb="finalize", object="pricing", objectGroundingType="EXPLICIT", deadline="Monday")),
        _event("e-conn-string", EventKind.ISSUE, "Database server connection string is missing.", [60], lines, object="connection string", entities=["database"], memorySignal=MemorySignal(isMemoryWorthy=True, importance="HIGH")),
        _event("e-network", EventKind.ISSUE, "Network parameter information is missing.", [61], lines, object="network parameter", entities=["network"], memorySignal=MemorySignal(isMemoryWorthy=True, importance="HIGH")),
        _event("e-insecure", EventKind.STATE, "Connection is reported as insecure.", [62], lines, object="connection security", entities=["Connection"], memorySignal=MemorySignal(isMemoryWorthy=True, importance="HIGH")),
        _event("e-playstore", EventKind.ISSUE, "A Play Store issue exists.", [80], lines, object="Play Store issue", entities=["Play", "Store"], memorySignal=MemorySignal(isMemoryWorthy=True, importance="HIGH")),
        _event("e-keys", EventKind.STATE, "Old keys are currently in use.", [90], lines, object="old keys", entities=["keys"], memorySignal=MemorySignal(isMemoryWorthy=True, importance="HIGH")),
        _event("e-notes-review", EventKind.RESULT, "Generated notes were reviewed and improved.", [91], lines, object="generated notes", entities=["notes"], memorySignal=MemorySignal(isMemoryWorthy=True, importance="MEDIUM")),
        _event("e-master-prompt", EventKind.REQUIREMENT, "Master-prompt output requirements should be documented.", [92], lines, object="master-prompt output requirements", entities=["master"], actionSignal=ActionSignal(isActionable=True, role="REQUEST", actionStrength="EXPLICIT", verb="document", object="master-prompt output requirements", objectGroundingType="EXPLICIT"), memorySignal=MemorySignal(isMemoryWorthy=True, importance="HIGH")),
        _event("e-s3-config", EventKind.FACT, "S3 configuration was changed.", [105], lines, object="S3 configuration", entities=["S3"], memorySignal=MemorySignal(isMemoryWorthy=True, importance="HIGH")),
        _event("e-server-id", EventKind.REQUEST, "Create server ID.", [110, 111], lines, object="server ID", entities=["Server", "ID"], actionSignal=ActionSignal(isActionable=True, role="REQUEST", actionStrength="EXPLICIT", verb="create", object="server ID", objectGroundingType="EXPLICIT")),
        _event("e-meeting-page", EventKind.REQUEST, "Create meeting page on the dashboard.", [130], lines, object="meeting page", entities=["meeting"], actionSignal=ActionSignal(isActionable=True, role="REQUEST", actionStrength="EXPLICIT", verb="create", object="meeting page", objectGroundingType="EXPLICIT")),
        _event("e-opencv", EventKind.REQUEST, "Use GPT and OpenCV for coordinate extraction.", [150], lines, object="coordinate extraction", entities=["GPT", "OpenCV"], actionSignal=ActionSignal(isActionable=True, role="INSTRUCTION", actionStrength="EXPLICIT", verb="use", object="GPT and OpenCV for coordinate extraction", objectGroundingType="EXPLICIT")),
        _event("e-mic-issue", EventKind.ISSUE, "Microphone access is required and not currently available.", [160], lines, object="microphone access", entities=["microphone"], memorySignal=MemorySignal(isMemoryWorthy=True, importance="HIGH")),
        _event("e-mic-action", EventKind.REQUEST, "Request microphone access.", [161], lines, object="microphone access", entities=["microphone"], actionSignal=ActionSignal(isActionable=True, role="REQUEST", actionStrength="EXPLICIT", verb="request", object="microphone access", objectGroundingType="EXPLICIT")),
        _event("e-tomorrow-test", EventKind.COMMITMENT, "This flow will be tested tomorrow.", [170, 171], lines, object="flow testing", timeExpression="tomorrow", entities=["testing"], actionSignal=ActionSignal(isActionable=True, role="COMMITMENT", actionStrength="EXPLICIT", verb="test", object="flow testing", objectGroundingType="EXPLICIT", deadline="tomorrow")),
        _event("e-s3-still", EventKind.STATE, "S3 still does not reach frontend.", [220], lines, object="S3 frontend integration", entities=["S3", "frontend"], memorySignal=MemorySignal(isMemoryWorthy=True, importance="HIGH")),
    ]
    gold_tasks = [
        {"id": "t-meeting-page", "kind": "task", "meaning": "Create meeting page", "evidenceSequences": [130]},
        {"id": "t-server-id", "kind": "task", "meaning": "Create server ID", "evidenceSequences": [110, 111]},
        {"id": "t-opencv", "kind": "task", "meaning": "Use GPT and OpenCV for coordinate extraction", "evidenceSequences": [150]},
        {"id": "t-mic", "kind": "task", "meaning": "Request microphone access", "evidenceSequences": [161]},
        {"id": "t-tomorrow", "kind": "task", "meaning": "Test this flow tomorrow", "evidenceSequences": [170, 171]},
        {"id": "t-s3", "kind": "task", "meaning": "Fix the S3 issue tomorrow", "evidenceSequences": [21]},
    ]
    gold_notes = [
        {"id": "n-keys", "kind": "note", "meaning": "Old keys are currently in use", "evidenceSequences": [90]},
        {"id": "n-monday", "kind": "note", "meaning": "Monday meeting mentioned", "evidenceSequences": [42], "reviewStatus": "OPTIONAL_VALID"},
        {"id": "n-notes", "kind": "note", "meaning": "Generated notes were reviewed and improved", "evidenceSequences": [91]},
        {"id": "n-master", "kind": "note", "meaning": "Master-prompt output requirements", "evidenceSequences": [92]},
        {"id": "n-play", "kind": "note", "meaning": "Play Store issue exists", "evidenceSequences": [80]},
        {"id": "n-insecure", "kind": "note", "meaning": "Connection reported insecure", "evidenceSequences": [62]},
        {"id": "n-network", "kind": "note", "meaning": "Network parameter information missing", "evidenceSequences": [61]},
        {"id": "n-s3", "kind": "note", "meaning": "S3 frontend integration problem", "evidenceSequences": [20, 220]},
    ]
    return {
        "id": "gold-long-meeting",
        "chunks": chunks,
        "lines": lines,
        "events": events,
        "goldTasks": gold_tasks,
        "goldNotes": gold_notes,
        "validAdditionalNotes": [
            {"id": "n-pricing", "kind": "note", "meaning": "Pricing should start around 200", "evidenceSequences": [40]},
            {"id": "n-conn", "kind": "note", "meaning": "Database server connection string is missing", "evidenceSequences": [60]},
            {"id": "n-s3-config", "kind": "note", "meaning": "S3 configuration was changed", "evidenceSequences": [105]},
            {"id": "n-s3-still", "kind": "note", "meaning": "S3 still does not reach frontend", "evidenceSequences": [220]},
            {"id": "n-mic-issue", "kind": "note", "meaning": "Microphone access is required and not currently available", "evidenceSequences": [160]},
            {"id": "n-pnb", "kind": "note", "meaning": "PNB setup is a separate topic", "evidenceSequences": [63]},
        ],
        "validAdditionalTasks": [
            {"id": "t-master-prompt", "kind": "task", "meaning": "Document master-prompt output requirements", "evidenceSequences": [92]},
            {"id": "t-pricing-finalize", "kind": "task", "meaning": "Finalize pricing in the Monday meeting", "evidenceSequences": [42]},
        ],
        "goldThreads": [
            ["e-s3-issue", "e-s3-commit", "e-s3-config", "e-s3-still"],
            ["e-pricing", "e-monday", "e-pricing-finalize"],
            ["e-conn-string"],
            ["e-playstore"],
            ["e-server-id"],
        ],
        "originalActionableEventIds": [
            "e-s3-commit",
            "e-server-id",
            "e-meeting-page",
            "e-opencv",
            "e-mic-action",
            "e-tomorrow-test",
        ],
        "reviewedActionableEventIds": [
            "e-s3-commit",
            "e-server-id",
            "e-meeting-page",
            "e-opencv",
            "e-mic-action",
            "e-tomorrow-test",
            "e-pricing-finalize",
            "e-master-prompt",
        ],
        "forbiddenTaskTitles": ["Complete pending task", "Fix issue", "Handle problem", "Do this"],
        "serverIdForbiddenSequences": {60, 61, 62, 63, 64},
    }
