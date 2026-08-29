"""Lenskart background + HRMS meeting fixture for meeting-pipeline regressions."""

from services.conversation.models import STTStatus, TranscriptChunkDocument

LENSKART_BACKGROUND = {
    0: "Lenskart gold membership ab 21 percent growth de raha hai international markets mein.",
    1: "UAE expansion and franchising conversation, sixty to seventy percent margin.",
    2: "Membership target discuss ho raha hai China aur franchise ke saath.",
    3: "Revenue from Lenskart stores and gold membership.",
    4: "Franchise partners in international markets, Lenskart branding.",
    5: "Lenskart / China / franchise discussion continues in the background.",
}

HRMS_MEETING = {
    6: "अब हम meeting start करते हैं. We are building HRMS software, especially candidate onboarding.",
    7: "Employee page par ek action ya button add karenge. Clicking it generates a candidate link. Candidate us link se information enter karega.",
    8: "",
    9: "Automation se HR ka manual work kam hoga because candidate khud details bhar dega.",
    10: "Isse HR ko baar baar information update nahi karni padegi.",
    11: "Payroll module leave, leave cancellation, salary deduction aur PF calculate karega.",
    12: "Expense tracker company expenses aur payment tracking ke liye chahiye.",
    13: "Second project ek AI assistant hai jo conversations sunega aur important information record karega.",
    14: "It should divide conversations into chunks and create useful notes from experimentation.",
    15: "Website credit discussion 1 80 crore 01:21 percent something unclear noisy speech.",
}

FORBIDDEN_BACKGROUND_TITLES = [
    "Gold membership",
    "21% growth",
    "UAE expansion",
    "Lenskart franchising",
    "60–70% margin",
    "membership target",
]

REQUIRED_HRMS_MEANINGS = [
    "HRMS project",
    "candidate onboarding",
    "generated candidate link",
    "candidate submits information through link",
    "reduce manual HR work",
    "payroll",
    "leave handling",
    "salary deduction",
    "PF calculation",
    "expense tracking",
    "AI assistant project",
    "conversation capture",
    "chunking/recording",
    "structured notes",
]

DENSE_ONBOARDING = (
    "We'll add an action on employee page. Clicking it generates a candidate link. "
    "We'll send that link to the candidate. Candidate enters their details there so HR doesn't need "
    "to manually update the information repeatedly."
)

ATOMIC_ONBOARDING_MEANINGS = [
    "employee-page onboarding action",
    "generated candidate link",
    "candidate receives/uses link",
    "candidate submits information",
    "reduces manual HR work",
]


def _chunk(sequence: int, text: str, speaker: int | None = None) -> TranscriptChunkDocument:
    raw = text
    if speaker is not None and text:
        raw = f"Speaker {speaker}: {text}"
    return TranscriptChunkDocument(
        conversationId="lenskart-hrms",
        userId="user_1",
        spaceId="space_1",
        chunkId=f"chunk_{sequence}",
        sequenceNumber=sequence,
        rawText=raw,
        normalizedText=raw,
        sttStatus=STTStatus.COMPLETED,
    )


def lenskart_hrms_chunks() -> list[TranscriptChunkDocument]:
    chunks = []
    for sequence, text in sorted({**LENSKART_BACKGROUND, **HRMS_MEETING}.items()):
        speaker = 0 if sequence <= 5 else 1
        chunks.append(_chunk(sequence, text, speaker=speaker))
    return chunks


def lenskart_hrms_case() -> dict:
    lines = {**LENSKART_BACKGROUND, **HRMS_MEETING}
    transcript = "\n".join(f"[{sequence}] {text}" for sequence, text in sorted(lines.items()) if text)
    return {
        "id": "lenskart-hrms-meeting",
        "category": "technical_meeting",
        "transcript": transcript,
        "backgroundSequences": [0, 1, 2, 3, 4, 5],
        "forbiddenArtifacts": FORBIDDEN_BACKGROUND_TITLES,
        "goldTasks": [
            {
                "id": "t-hrms-onboarding",
                "kind": "task",
                "meaning": "Build HRMS candidate onboarding with an employee-page action that generates a candidate link",
                "evidenceSequences": [6, 7],
            },
            {
                "id": "t-payroll",
                "kind": "task",
                "meaning": "Build payroll including leave, salary deduction and PF",
                "evidenceSequences": [11],
            },
            {
                "id": "t-expense",
                "kind": "task",
                "meaning": "Build expense tracker for company expenses and payment tracking",
                "evidenceSequences": [12],
            },
            {
                "id": "t-ai-assistant",
                "kind": "task",
                "meaning": "Build AI assistant that captures conversations, chunks them, and creates notes",
                "evidenceSequences": [13, 14],
            },
        ],
        "goldNotes": [
            {
                "id": "n-link-flow",
                "kind": "note",
                "meaning": "Candidate submits information through the generated link, reducing manual HR work",
                "evidenceSequences": [7, 9, 10],
            },
            {
                "id": "n-payroll-pf",
                "kind": "note",
                "meaning": "Payroll should calculate PF deductions and handle leave cancellation salary deduction",
                "evidenceSequences": [11],
            },
        ],
        "goldCandidates": [
            {"id": "c-hrms", "meaning": "HRMS software project", "evidenceSequences": [6]},
            {"id": "c-onboarding", "meaning": "candidate onboarding", "evidenceSequences": [6]},
            {"id": "c-link", "meaning": "generate candidate link from employee page", "evidenceSequences": [7]},
            {"id": "c-submit", "meaning": "candidate enters information through link", "evidenceSequences": [7]},
            {"id": "c-manual", "meaning": "reduce HR manual work", "evidenceSequences": [9, 10]},
            {"id": "c-payroll", "meaning": "payroll module", "evidenceSequences": [11]},
            {"id": "c-leave", "meaning": "leave handling", "evidenceSequences": [11]},
            {"id": "c-pf", "meaning": "PF calculation", "evidenceSequences": [11]},
            {"id": "c-expense", "meaning": "expense tracker", "evidenceSequences": [12]},
            {"id": "c-ai", "meaning": "AI assistant project", "evidenceSequences": [13]},
            {"id": "c-chunks", "meaning": "divide conversations into chunks", "evidenceSequences": [14]},
            {"id": "c-notes", "meaning": "create useful notes", "evidenceSequences": [14]},
        ],
        "nonTaskSequences": [0, 1, 2, 3, 4, 5, 15],
    }
