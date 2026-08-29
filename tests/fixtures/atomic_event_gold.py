"""Gold atomic events for noisy multilingual segments. Independent of task/note scoring."""

from __future__ import annotations

from services.conversation.event_pipeline.schemas import EventKind

SEGMENTS: list[dict] = [
    {
        "id": "s3-issue-commitment",
        "text": "S3 नहीं आ रहा front end पे, कल उसको solve करेंगे.",
        "expected": [
            {"kind": EventKind.ISSUE, "meaning": "S3 is not reaching the frontend."},
            {"kind": EventKind.COMMITMENT, "meaning": "The S3 issue will be addressed tomorrow."},
        ],
        "forbidden": [{"kind": EventKind.REQUEST, "meaning": "Complete pending task"}],
    },
    {
        "id": "insecure-no-task",
        "text": "Connection insecure है",
        "expected": [{"kind": EventKind.STATE, "meaning": "Connection is insecure."}],
        "forbidden": [{"kind": "TASK", "meaning": "Fix connection security"}],
        "mustNotCreateTask": True,
    },
    {
        "id": "kal-kar-denge-ambiguous",
        "text": "कल कर देंगे",
        "expected": [{"kind": EventKind.COMMITMENT, "meaning": "It will be done tomorrow.", "uncertainty": ["missing_object"]}],
        "forbidden": [{"meaning": "Complete pending task"}],
        "preserveAmbiguity": True,
    },
    {
        "id": "server-id-request",
        "text": "Server ID create karna hai, please create the server ID",
        "expected": [{"kind": EventKind.REQUEST, "meaning": "Create the server ID.", "object": "server ID"}],
        "forbidden": [],
    },
    {
        "id": "pricing-proposal",
        "text": "Pricing should start around 200, Monday meeting mein finalize karenge",
        "expected": [
            {"kind": EventKind.PROPOSAL, "meaning": "Pricing should start around 200."},
            {"kind": EventKind.FACT, "meaning": "Pricing will be finalized in the Monday meeting."},
        ],
        "forbidden": [{"kind": EventKind.REQUEST, "meaning": "Set pricing to 200"}],
        "mustNotCreateTaskFrom": ["Pricing should start around 200"],
    },
    {
        "id": "pricing-rakh-sakte",
        "text": "pricing around 200 rakh sakte hain",
        "expected": [{"kind": EventKind.PROPOSAL, "meaning": "Pricing can be around 200."}],
        "mustNotCreateTask": True,
    },
    {
        "id": "pricing-final-kar-lena",
        "text": "pricing kal final kar lena",
        "expected": [{"kind": EventKind.REQUEST, "meaning": "Finalize pricing tomorrow.", "object": "pricing"}],
        "mustCreateTask": True,
    },
    {
        "id": "gpt-kar-sakte",
        "text": "GPT use kar sakte hain",
        "expected": [{"kind": EventKind.PROPOSAL, "meaning": "GPT can be used."}],
        "mustNotCreateTask": True,
    },
    {
        "id": "gpt-use-karna-hai",
        "text": "GPT coordinate extraction ke liye use karna hai",
        "expected": [{"kind": EventKind.REQUIREMENT, "meaning": "Use GPT for coordinate extraction.", "object": "GPT"}],
        "mustCreateTask": True,
    },
    {
        "id": "connection-fix-kal",
        "text": "connection kal fix kar dena",
        "expected": [{"kind": EventKind.REQUEST, "meaning": "Fix the connection tomorrow.", "object": "connection"}],
        "mustCreateTask": True,
    },
    {
        "id": "play-store-issue",
        "text": "Play Store issue exist karta hai abhi",
        "expected": [{"kind": EventKind.ISSUE, "meaning": "A Play Store issue currently exists."}],
        "mustNotCreateTask": True,
    },
    {
        "id": "mic-issue-and-request",
        "text": "microphone access required hai, currently available nahi. please request microphone access",
        "expected": [
            {"kind": EventKind.ISSUE, "meaning": "Microphone access is required and not available."},
            {"kind": EventKind.REQUEST, "meaning": "Request microphone access."},
        ],
        "forbidden": [],
    },
    {
        "id": "old-keys-state",
        "text": "old keys are currently in use",
        "expected": [{"kind": EventKind.STATE, "meaning": "Old keys are currently in use."}],
        "mustNotCreateTask": True,
    },
    {
        "id": "connection-string-issue",
        "text": "database server connection string missing hai",
        "expected": [{"kind": EventKind.ISSUE, "meaning": "The database server connection string is missing."}],
        "mustNotCreateTask": True,
    },
    {
        "id": "complete-it-tomorrow",
        "text": "Complete it tomorrow",
        "expected": [{"kind": EventKind.COMMITMENT, "meaning": "Complete it tomorrow.", "uncertainty": ["missing_object"]}],
        "forbidden": [{"meaning": "Complete pending task"}],
        "preserveAmbiguity": True,
        "mustNotCreateTask": True,
    },
]
