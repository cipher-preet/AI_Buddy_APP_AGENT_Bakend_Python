"""Reviewed multi-meeting fixtures. Distinct from the 221-sequence gold transcript."""

from __future__ import annotations

from services.conversation.event_pipeline.schemas import ActionSignal, AtomicEvent, EventKind, MemorySignal
from services.conversation.models import EvidenceSpan, STTStatus, TranscriptChunkDocument

FILLERS = [
    "umm",
    "yeah yeah",
    "thoda wait",
    "checking",
    "ok ok",
    "suno",
    "one sec",
    "background rustle",
]


def _chunk(meeting_id: str, sequence: int, text: str) -> TranscriptChunkDocument:
    return TranscriptChunkDocument(
        conversationId=meeting_id,
        userId="user_1",
        spaceId="space_1",
        chunkId=f"{meeting_id}_{sequence}",
        sequenceNumber=sequence,
        rawText=text,
        sttStatus=STTStatus.COMPLETED,
    )


def _event(event_id: str, kind: EventKind, meaning: str, sequences: list[int], lines: dict[int, str], **kwargs) -> AtomicEvent:
    evidence = [
        EvidenceSpan(sequenceStart=seq, sequenceEnd=seq, text=lines[seq])
        for seq in sequences
        if seq in lines
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
        conversationId=kwargs.get("conversationId", "meeting"),
        userId="user_1",
        spaceId="space_1",
        actionSignal=kwargs.get("actionSignal"),
        memorySignal=kwargs.get("memorySignal"),
    )


def _pad(lines: dict[int, str], count: int) -> dict[int, str]:
    filled = dict(lines)
    for sequence in range(count):
        if sequence not in filled:
            filled[sequence] = FILLERS[sequence % len(FILLERS)] + f" {sequence}"
    return {key: filled[key] for key in range(count)}


def _meeting(meeting_id: str, count: int, content: dict[int, str], events: list[AtomicEvent], gold_tasks: list[dict], gold_notes: list[dict], **extra) -> dict:
    lines = _pad(content, count)
    chunks = [_chunk(meeting_id, sequence, text) for sequence, text in sorted(lines.items())]
    return {
        "id": meeting_id,
        "chunks": chunks,
        "lines": lines,
        "events": events,
        "goldTasks": gold_tasks,
        "goldNotes": gold_notes,
        "validAdditionalNotes": extra.get("validAdditionalNotes") or [],
        "validAdditionalTasks": extra.get("validAdditionalTasks") or [],
        "goldThreads": extra.get("goldThreads") or [],
        "originalActionableEventIds": extra.get("originalActionableEventIds") or [],
        "reviewedActionableEventIds": extra.get("reviewedActionableEventIds") or [],
        "goldComplete": True,
        "size": count,
    }


def build_meeting_a() -> dict:
    """~50 chunks: standup. Hindi/English, one action, issue without action, decision."""
    content = {
        2: "stand-up start karte hain",
        8: "login bug kal fix kar dena",
        14: "payment timeout ho raha hai, koi action nahi liya",
        22: "Redis hi rakhte hain, that is the decision",
        30: "analytics dashboard sirf status update tha",
        40: "no owner assigned for the timeout",
    }
    lines = _pad(content, 50)
    events = [
        _event("a-login", EventKind.REQUEST, "Fix the login bug tomorrow.", [8], lines, object="login bug", entities=["login"], actionSignal=ActionSignal(isActionable=True, role="REQUEST", actionStrength="EXPLICIT", verb="fix", object="login bug", objectGroundingType="EXPLICIT", deadline="kal"), conversationId="meeting-a"),
        _event("a-timeout", EventKind.ISSUE, "Payment timeout is happening.", [14], lines, object="payment timeout", entities=["payment"], memorySignal=MemorySignal(isMemoryWorthy=True, importance="HIGH"), conversationId="meeting-a"),
        _event("a-redis", EventKind.DECISION, "Keep Redis.", [22], lines, object="Redis", entities=["Redis"], memorySignal=MemorySignal(isMemoryWorthy=True, importance="HIGH"), conversationId="meeting-a"),
    ]
    return _meeting(
        "meeting-a",
        50,
        content,
        events,
        [{"id": "t-login", "kind": "task", "meaning": "Fix the login bug", "evidenceSequences": [8]}],
        [
            {"id": "n-timeout", "kind": "note", "meaning": "Payment timeout is happening", "evidenceSequences": [14]},
            {"id": "n-redis", "kind": "note", "meaning": "Keep Redis", "evidenceSequences": [22]},
        ],
        originalActionableEventIds=["a-login"],
        reviewedActionableEventIds=["a-login"],
        goldThreads=[["a-timeout"], ["a-login"], ["a-redis"]],
    )


def build_meeting_b() -> dict:
    """~150 chunks: topic switch, returning billing, pronoun, action without owner."""
    content = {
        10: "billing retry limit 3 pe rakhna hai",
        18: "onboarding checklist incomplete hai",
        40: "dark mode experiment ko hold karo",
        41: "isko production pe mat dalna",
        70: "docs update karna hai, owner nahi bola",
        110: "billing retry wapas discuss kiya, limit same rahegi",
        130: "onboarding copy still wrong hai",
    }
    lines = _pad(content, 150)
    events = [
        _event("b-billing", EventKind.REQUIREMENT, "Keep billing retry limit at 3.", [10], lines, object="billing retry limit", entities=["billing"], actionSignal=ActionSignal(isActionable=True, role="INSTRUCTION", actionStrength="EXPLICIT", verb="keep", object="billing retry limit", objectGroundingType="EXPLICIT"), conversationId="meeting-b"),
        _event("b-onboarding", EventKind.ISSUE, "Onboarding checklist is incomplete.", [18], lines, object="onboarding checklist", entities=["onboarding"], memorySignal=MemorySignal(isMemoryWorthy=True, importance="HIGH"), conversationId="meeting-b"),
        _event("b-dark", EventKind.REQUEST, "Do not put dark mode in production.", [40, 41], lines, object="dark mode", entities=["dark"], actionSignal=ActionSignal(isActionable=True, role="INSTRUCTION", actionStrength="EXPLICIT", verb="hold", object="dark mode", objectGroundingType="LOCAL_COREFERENCE"), conversationId="meeting-b"),
        _event("b-docs", EventKind.REQUEST, "Update the docs.", [70], lines, object="docs", entities=["docs"], actionSignal=ActionSignal(isActionable=True, role="REQUEST", actionStrength="EXPLICIT", verb="update", object="docs", objectGroundingType="EXPLICIT"), conversationId="meeting-b"),
        _event("b-billing-return", EventKind.DECISION, "Billing retry limit stays the same.", [110], lines, object="billing retry limit", entities=["billing"], memorySignal=MemorySignal(isMemoryWorthy=True, importance="HIGH"), conversationId="meeting-b"),
        _event("b-copy", EventKind.ISSUE, "Onboarding copy is still wrong.", [130], lines, object="onboarding copy", entities=["onboarding"], memorySignal=MemorySignal(isMemoryWorthy=True, importance="HIGH"), conversationId="meeting-b"),
    ]
    return _meeting(
        "meeting-b",
        150,
        content,
        events,
        [
            {"id": "t-billing", "kind": "task", "meaning": "Keep billing retry limit at 3", "evidenceSequences": [10], "reviewStatus": "REQUIRED"},
            {"id": "t-dark", "kind": "task", "meaning": "Do not put dark mode in production", "evidenceSequences": [40, 41], "reviewStatus": "REQUIRED"},
            {"id": "t-docs", "kind": "task", "meaning": "Update the docs", "evidenceSequences": [70], "reviewStatus": "REQUIRED"},
        ],
        [
            {"id": "n-onboarding", "kind": "note", "meaning": "Onboarding checklist is incomplete", "evidenceSequences": [18]},
            {"id": "n-billing-stay", "kind": "note", "meaning": "Billing retry limit stays the same", "evidenceSequences": [110]},
            {"id": "n-copy", "kind": "note", "meaning": "Onboarding copy is still wrong", "evidenceSequences": [130]},
        ],
        originalActionableEventIds=["b-billing", "b-dark", "b-docs"],
        reviewedActionableEventIds=["b-billing", "b-dark", "b-docs"],
        goldThreads=[["b-billing", "b-billing-return"], ["b-onboarding", "b-copy"], ["b-dark"]],
    )


def build_meeting_c() -> dict:
    """~300 chunks: returning invoice PDF thread, implicit follow-up, noise."""
    content = {
        20: "invoice PDF generation fail ho rahi hai",
        21: "kal isko check kar lena",
        80: "webhook retries too aggressive hain",
        140: "invoice PDF wapas fail hua after retry",
        200: "follow up on the PDF after deploy",
        250: "GST field optional rakhne ka decision ho gaya",
    }
    lines = _pad(content, 300)
    events = [
        _event("c-pdf-issue", EventKind.ISSUE, "Invoice PDF generation is failing.", [20], lines, object="invoice PDF", entities=["invoice", "PDF"], memorySignal=MemorySignal(isMemoryWorthy=True, importance="HIGH"), conversationId="meeting-c"),
        _event("c-pdf-action", EventKind.REQUEST, "Check the invoice PDF tomorrow.", [21], lines, object="invoice PDF", entities=["invoice"], actionSignal=ActionSignal(isActionable=True, role="REQUEST", actionStrength="EXPLICIT", verb="check", object="invoice PDF", objectGroundingType="LOCAL_COREFERENCE", deadline="kal"), conversationId="meeting-c"),
        _event("c-webhook", EventKind.ISSUE, "Webhook retries are too aggressive.", [80], lines, object="webhook retries", entities=["webhook"], memorySignal=MemorySignal(isMemoryWorthy=True, importance="HIGH"), conversationId="meeting-c"),
        _event("c-pdf-still", EventKind.STATE, "Invoice PDF failed again after retry.", [140], lines, object="invoice PDF", entities=["invoice", "PDF"], memorySignal=MemorySignal(isMemoryWorthy=True, importance="HIGH"), conversationId="meeting-c"),
        _event("c-follow", EventKind.FOLLOW_UP, "Follow up on the PDF after deploy.", [200], lines, object="invoice PDF", entities=["PDF"], actionSignal=ActionSignal(isActionable=True, role="FOLLOW_UP", actionStrength="EXPLICIT", verb="follow up", object="invoice PDF", objectGroundingType="EXPLICIT"), conversationId="meeting-c"),
        _event("c-gst", EventKind.DECISION, "GST field will remain optional.", [250], lines, object="GST field", entities=["GST"], memorySignal=MemorySignal(isMemoryWorthy=True, importance="HIGH"), conversationId="meeting-c"),
    ]
    return _meeting(
        "meeting-c",
        300,
        content,
        events,
        [
            {"id": "t-pdf", "kind": "task", "meaning": "Check the invoice PDF", "evidenceSequences": [21]},
            {"id": "t-follow", "kind": "task", "meaning": "Follow up on the PDF after deploy", "evidenceSequences": [200]},
        ],
        [
            {"id": "n-pdf", "kind": "note", "meaning": "Invoice PDF generation is failing", "evidenceSequences": [20]},
            {"id": "n-webhook", "kind": "note", "meaning": "Webhook retries are too aggressive", "evidenceSequences": [80]},
            {"id": "n-pdf-still", "kind": "note", "meaning": "Invoice PDF failed again after retry", "evidenceSequences": [140]},
            {"id": "n-gst", "kind": "note", "meaning": "GST field will remain optional", "evidenceSequences": [250]},
        ],
        originalActionableEventIds=["c-pdf-action", "c-follow"],
        reviewedActionableEventIds=["c-pdf-action", "c-follow"],
        goldThreads=[["c-pdf-issue", "c-pdf-action", "c-pdf-still", "c-follow"], ["c-webhook"], ["c-gst"]],
    )


def build_meeting_d() -> dict:
    """~500 chunks: noisy all-hands with Hindi/Hinglish/English and several threads."""
    content = {
        25: "vendor contract Friday tak sign karna hai",
        90: "office wifi unstable hai, no action",
        160: "hiring freeze continue, that is a constraint",
        240: "please share the Q3 numbers",
        320: "canteen card recharge process alag topic hai",
        400: "vendor contract still unsigned",
        460: "Q3 numbers shared, result is 12 percent growth",
    }
    lines = _pad(content, 500)
    events = [
        _event("d-contract", EventKind.REQUEST, "Sign the vendor contract by Friday.", [25], lines, object="vendor contract", entities=["vendor"], actionSignal=ActionSignal(isActionable=True, role="REQUEST", actionStrength="EXPLICIT", verb="sign", object="vendor contract", objectGroundingType="EXPLICIT", deadline="Friday"), conversationId="meeting-d"),
        _event("d-wifi", EventKind.ISSUE, "Office wifi is unstable.", [90], lines, object="office wifi", entities=["wifi"], memorySignal=MemorySignal(isMemoryWorthy=True, importance="MEDIUM"), conversationId="meeting-d"),
        _event("d-hiring", EventKind.CONSTRAINT, "Hiring freeze continues.", [160], lines, object="hiring freeze", entities=["hiring"], memorySignal=MemorySignal(isMemoryWorthy=True, importance="HIGH"), conversationId="meeting-d"),
        _event("d-numbers", EventKind.REQUEST, "Share the Q3 numbers.", [240], lines, object="Q3 numbers", entities=["Q3"], actionSignal=ActionSignal(isActionable=True, role="REQUEST", actionStrength="EXPLICIT", verb="share", object="Q3 numbers", objectGroundingType="EXPLICIT"), conversationId="meeting-d"),
        _event("d-canteen", EventKind.FACT, "Canteen card recharge is a separate topic.", [320], lines, object="canteen card", entities=["canteen"], memorySignal=MemorySignal(isMemoryWorthy=True, importance="LOW"), conversationId="meeting-d"),
        _event("d-unsigned", EventKind.STATE, "Vendor contract is still unsigned.", [400], lines, object="vendor contract", entities=["vendor"], memorySignal=MemorySignal(isMemoryWorthy=True, importance="HIGH"), conversationId="meeting-d"),
        _event("d-growth", EventKind.RESULT, "Q3 numbers show 12 percent growth.", [460], lines, object="Q3 numbers", entities=["Q3"], memorySignal=MemorySignal(isMemoryWorthy=True, importance="HIGH"), conversationId="meeting-d"),
    ]
    return _meeting(
        "meeting-d",
        500,
        content,
        events,
        [
            {"id": "t-contract", "kind": "task", "meaning": "Sign the vendor contract by Friday", "evidenceSequences": [25]},
            {"id": "t-q3", "kind": "task", "meaning": "Share the Q3 numbers", "evidenceSequences": [240]},
        ],
        [
            {"id": "n-wifi", "kind": "note", "meaning": "Office wifi is unstable", "evidenceSequences": [90]},
            {"id": "n-hiring", "kind": "note", "meaning": "Hiring freeze continues", "evidenceSequences": [160]},
            {"id": "n-unsigned", "kind": "note", "meaning": "Vendor contract is still unsigned", "evidenceSequences": [400]},
            {"id": "n-growth", "kind": "note", "meaning": "Q3 numbers show 12 percent growth", "evidenceSequences": [460]},
        ],
        originalActionableEventIds=["d-contract", "d-numbers"],
        reviewedActionableEventIds=["d-contract", "d-numbers"],
        goldThreads=[["d-contract", "d-unsigned"], ["d-numbers", "d-growth"], ["d-hiring"]],
    )


def all_reviewed_meetings() -> list[dict]:
    return [build_meeting_a(), build_meeting_b(), build_meeting_c(), build_meeting_d()]
