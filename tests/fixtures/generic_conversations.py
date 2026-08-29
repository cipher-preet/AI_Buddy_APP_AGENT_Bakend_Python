"""Regression fixtures for personal, family, study, travel, work, casual, and code-switching conversations.

Examples are regression-only. Pipeline logic must stay domain-agnostic.
"""

from __future__ import annotations

from services.conversation.event_pipeline.schemas import ActionSignal, AtomicEvent, EventKind, MemorySignal
from services.conversation.models import EvidenceSpan, STTStatus, TranscriptChunkDocument


def _chunk(conversation_id: str, sequence: int, text: str) -> TranscriptChunkDocument:
    return TranscriptChunkDocument(
        conversationId=conversation_id,
        userId="user_1",
        spaceId="space_1",
        chunkId=f"{conversation_id}_{sequence}",
        sequenceNumber=sequence,
        rawText=text,
        sttStatus=STTStatus.COMPLETED,
    )


def _event(conversation_id: str, event_id: str, kind: EventKind, meaning: str, sequence: int, text: str, **kwargs) -> AtomicEvent:
    return AtomicEvent(
        eventId=event_id,
        topicId=kwargs.get("topicId", "T1"),
        kind=kind,
        meaning=meaning,
        object=kwargs.get("object"),
        entities=kwargs.get("entities") or [],
        evidence=[EvidenceSpan(sequenceStart=sequence, sequenceEnd=sequence, text=text)],
        sequenceIds=[sequence],
        sourceIds=[f"{conversation_id}_{sequence}"],
        conversationId=conversation_id,
        userId="user_1",
        spaceId="space_1",
        actionSignal=kwargs.get("actionSignal"),
        memorySignal=kwargs.get("memorySignal"),
        timeExpression=kwargs.get("timeExpression"),
    )


def personal_planning() -> dict:
    cid = "generic-personal"
    lines = {
        0: "Tomorrow dad's train reaches at 7.",
        1: "I need to pick him up.",
    }
    return {
        "id": cid,
        "chunks": [_chunk(cid, seq, text) for seq, text in lines.items()],
        "events": [
            _event(
                cid,
                "e-train",
                EventKind.FACT,
                "Dad's train reaches at 7 tomorrow.",
                0,
                lines[0],
                object="dad's train",
                entities=["dad", "train"],
                memorySignal=MemorySignal(isMemoryWorthy=True, importance="HIGH", reason="FACT"),
            ),
            _event(
                cid,
                "e-pick",
                EventKind.COMMITMENT,
                "Pick dad up from the train.",
                1,
                lines[1],
                object="dad",
                entities=["dad"],
                actionSignal=ActionSignal(
                    isActionable=True,
                    role="COMMITMENT",
                    actionStrength="EXPLICIT",
                    verb="pick up",
                    object="dad",
                    objectGroundingType="LOCAL_COREFERENCE",
                ),
            ),
        ],
        "expectNoteSubstrings": ["7", "train"],
        "expectTaskSubstrings": ["pick"],
        "forbidGenericTask": True,
    }


def family_decision() -> dict:
    cid = "generic-family"
    lines = {0: "We decided to visit grandma Sunday."}
    return {
        "id": cid,
        "chunks": [_chunk(cid, 0, lines[0])],
        "events": [
            _event(
                cid,
                "e-visit",
                EventKind.DECISION,
                "Visit grandma on Sunday.",
                0,
                lines[0],
                object="grandma",
                entities=["grandma"],
                memorySignal=MemorySignal(isMemoryWorthy=True, importance="HIGH", reason="DECISION"),
            )
        ],
        "expectNoteSubstrings": ["grandma"],
        "expectNoTask": True,
    }


def study_status() -> dict:
    cid = "generic-study"
    lines = {
        0: "DBMS is complete.",
        1: "Networking chapter 4 is still pending.",
    }
    return {
        "id": cid,
        "chunks": [_chunk(cid, seq, text) for seq, text in lines.items()],
        "events": [
            _event(
                cid,
                "e-dbms",
                EventKind.RESULT,
                "DBMS revision is complete.",
                0,
                lines[0],
                object="DBMS",
                entities=["DBMS"],
                memorySignal=MemorySignal(isMemoryWorthy=True, importance="HIGH", reason="RESULT"),
            ),
            _event(
                cid,
                "e-net",
                EventKind.STATE,
                "Networking chapter 4 is still pending.",
                1,
                lines[1],
                object="Networking chapter 4",
                entities=["Networking"],
                memorySignal=MemorySignal(isMemoryWorthy=True, importance="HIGH", reason="STATUS"),
            ),
        ],
        "expectNoteSubstrings": ["DBMS", "Networking"],
        "expectNoTask": True,
        "forbidGenericTask": True,
    }


def travel_change() -> dict:
    cid = "generic-travel"
    lines = {
        0: "We changed the hotel to one near the airport.",
        1: "Book the airport cab tomorrow.",
    }
    return {
        "id": cid,
        "chunks": [_chunk(cid, seq, text) for seq, text in lines.items()],
        "events": [
            _event(
                cid,
                "e-hotel",
                EventKind.DECISION,
                "Hotel changed to one near the airport.",
                0,
                lines[0],
                object="hotel",
                entities=["hotel", "airport"],
                memorySignal=MemorySignal(isMemoryWorthy=True, importance="HIGH", reason="DECISION"),
            ),
            _event(
                cid,
                "e-cab",
                EventKind.REQUEST,
                "Book the airport cab tomorrow.",
                1,
                lines[1],
                object="airport cab",
                entities=["airport", "cab"],
                timeExpression="tomorrow",
                actionSignal=ActionSignal(
                    isActionable=True,
                    role="REQUEST",
                    actionStrength="EXPLICIT",
                    verb="book",
                    object="airport cab",
                    objectGroundingType="EXPLICIT",
                    deadline="tomorrow",
                ),
            ),
        ],
        "expectNoteSubstrings": ["hotel"],
        "expectTaskSubstrings": ["cab"],
    }


def work_followup() -> dict:
    cid = "generic-work"
    lines = {
        0: "Customer approval hasn't arrived.",
        1: "Follow up Friday.",
    }
    return {
        "id": cid,
        "chunks": [_chunk(cid, seq, text) for seq, text in lines.items()],
        "events": [
            _event(
                cid,
                "e-approval",
                EventKind.ISSUE,
                "Customer approval has not arrived.",
                0,
                lines[0],
                object="customer approval",
                entities=["customer"],
                memorySignal=MemorySignal(isMemoryWorthy=True, importance="HIGH", reason="ISSUE"),
            ),
            _event(
                cid,
                "e-follow",
                EventKind.FOLLOW_UP,
                "Follow up on customer approval Friday.",
                1,
                lines[1],
                object="customer approval",
                entities=["customer"],
                timeExpression="Friday",
                actionSignal=ActionSignal(
                    isActionable=True,
                    role="FOLLOW_UP",
                    actionStrength="EXPLICIT",
                    verb="follow up",
                    object="customer approval",
                    objectGroundingType="LOCAL_COREFERENCE",
                    deadline="Friday",
                ),
            ),
        ],
        "expectNoteSubstrings": ["approval"],
        "expectTaskSubstrings": ["follow"],
    }


def casual_noise() -> dict:
    cid = "generic-casual"
    lines = {
        0: "haan haan",
        1: "ok wait",
        2: "theek hai theek hai",
        3: "umm actually",
        4: "hello hello",
        5: "so yeah",
    }
    return {
        "id": cid,
        "chunks": [_chunk(cid, seq, text) for seq, text in lines.items()],
        "events": [
            _event(
                cid,
                f"e-noise-{seq}",
                EventKind.NOISE,
                text,
                seq,
                text,
                memorySignal=MemorySignal(isMemoryWorthy=False, importance="LOW", reason="FILLER"),
            )
            for seq, text in lines.items()
        ],
        "expectNoTask": True,
        "expectNoNote": True,
    }


def code_switching() -> dict:
    cid = "generic-hinglish"
    lines = {
        0: "Kal mom ka appointment 4 baje hai.",
        1: "Remind me to take her there.",
    }
    return {
        "id": cid,
        "chunks": [_chunk(cid, seq, text) for seq, text in lines.items()],
        "events": [
            _event(
                cid,
                "e-appt",
                EventKind.FACT,
                "Mom's appointment is at 4 PM.",
                0,
                lines[0],
                object="mom's appointment",
                entities=["mom"],
                memorySignal=MemorySignal(isMemoryWorthy=True, importance="HIGH", reason="FACT"),
            ),
            _event(
                cid,
                "e-remind",
                EventKind.REQUEST,
                "Remind me to take mom to the appointment.",
                1,
                lines[1],
                object="mom's appointment",
                entities=["mom"],
                actionSignal=ActionSignal(
                    isActionable=True,
                    role="REQUEST",
                    actionStrength="EXPLICIT",
                    verb="remind",
                    object="mom's appointment",
                    objectGroundingType="LOCAL_COREFERENCE",
                ),
            ),
        ],
        "expectNoteSubstrings": ["4"],
        "expectTaskSubstrings": ["remind", "take", "appointment"],
    }


def personal_hinglish_pickup() -> dict:
    cid = "generic-personal-hinglish"
    lines = {0: "Kal papa ko station se pick karna hai."}
    return {
        "id": cid,
        "chunks": [_chunk(cid, 0, lines[0])],
        "events": [
            _event(
                cid,
                "e-pick",
                EventKind.COMMITMENT,
                "Pick dad up from the station tomorrow.",
                0,
                lines[0],
                object="papa",
                entities=["papa", "station"],
                actionSignal=ActionSignal(
                    isActionable=True,
                    role="COMMITMENT",
                    actionStrength="EXPLICIT",
                    verb="pick up",
                    object="papa",
                    objectGroundingType="EXPLICIT",
                ),
            )
        ],
        "expectTaskSubstrings": ["pick", "papa", "station"],
        "forbidGenericTask": True,
    }


def study_revise_intent() -> dict:
    cid = "generic-study-revise"
    lines = {0: "Graphs chapter revise karna hai."}
    return {
        "id": cid,
        "chunks": [_chunk(cid, 0, lines[0])],
        "events": [
            _event(
                cid,
                "e-revise",
                EventKind.COMMITMENT,
                "Revise the graphs chapter.",
                0,
                lines[0],
                object="Graphs chapter",
                entities=["Graphs"],
                actionSignal=ActionSignal(
                    isActionable=True,
                    role="COMMITMENT",
                    actionStrength="EXPLICIT",
                    verb="revise",
                    object="Graphs chapter",
                    objectGroundingType="EXPLICIT",
                ),
            )
        ],
        "expectTaskSubstrings": ["graph", "revise"],
        "forbidGenericTask": True,
    }


def travel_hotel_and_cab() -> dict:
    cid = "generic-travel-hotel-cab"
    lines = {
        0: "Hotel final ho gaya.",
        1: "Cab kal book karenge.",
    }
    return {
        "id": cid,
        "chunks": [_chunk(cid, seq, text) for seq, text in lines.items()],
        "events": [
            _event(
                cid,
                "e-hotel",
                EventKind.DECISION,
                "The hotel is finalized.",
                0,
                lines[0],
                object="hotel",
                entities=["Hotel"],
                memorySignal=MemorySignal(isMemoryWorthy=True, importance="HIGH", reason="DECISION"),
            ),
            _event(
                cid,
                "e-cab",
                EventKind.COMMITMENT,
                "Book the cab tomorrow.",
                1,
                lines[1],
                object="cab",
                entities=["Cab"],
                timeExpression="kal",
                actionSignal=ActionSignal(
                    isActionable=True,
                    role="COMMITMENT",
                    actionStrength="EXPLICIT",
                    verb="book",
                    object="cab",
                    objectGroundingType="EXPLICIT",
                    deadline="kal",
                ),
            ),
        ],
        "expectNoteSubstrings": ["hotel"],
        "expectTaskSubstrings": ["cab", "book"],
        "forbidGenericTask": True,
    }


def family_sunday_plan() -> dict:
    cid = "generic-family-sunday"
    lines = {0: "Sunday grandma ke ghar jayenge."}
    return {
        "id": cid,
        "chunks": [_chunk(cid, 0, lines[0])],
        "events": [
            _event(
                cid,
                "e-visit",
                EventKind.DECISION,
                "Visit grandma's home on Sunday.",
                0,
                lines[0],
                object="grandma",
                entities=["grandma", "Sunday"],
                memorySignal=MemorySignal(isMemoryWorthy=True, importance="HIGH", reason="PLAN"),
            )
        ],
        "expectNoteSubstrings": ["grandma"],
        "expectNoTask": True,
    }


def work_invoice_and_update() -> dict:
    cid = "generic-work-invoice"
    lines = {0: "Invoice issue check karna hai aur client ko Friday update bhejna hai."}
    return {
        "id": cid,
        "chunks": [_chunk(cid, 0, lines[0])],
        "events": [
            _event(
                cid,
                "e-check",
                EventKind.REQUEST,
                "Check the invoice issue.",
                0,
                lines[0],
                object="invoice issue",
                entities=["Invoice"],
                actionSignal=ActionSignal(
                    isActionable=True,
                    role="REQUEST",
                    actionStrength="EXPLICIT",
                    verb="check",
                    object="invoice issue",
                    objectGroundingType="EXPLICIT",
                ),
            ),
            _event(
                cid,
                "e-update",
                EventKind.REQUEST,
                "Send the client an update on Friday.",
                0,
                lines[0],
                object="client update",
                entities=["client", "Friday"],
                timeExpression="Friday",
                actionSignal=ActionSignal(
                    isActionable=True,
                    role="REQUEST",
                    actionStrength="EXPLICIT",
                    verb="send",
                    object="client update",
                    objectGroundingType="EXPLICIT",
                    deadline="Friday",
                ),
            ),
        ],
        "expectTaskSubstrings": ["invoice", "client"],
        "forbidGenericTask": True,
    }


def garbled_noise() -> dict:
    cid = "generic-garbled-noise"
    lines = {0: "asdf qop relief el que la hmm ÃÂ"}
    return {
        "id": cid,
        "chunks": [_chunk(cid, 0, lines[0])],
        "events": [
            _event(
                cid,
                "e-noise",
                EventKind.NOISE,
                lines[0],
                0,
                lines[0],
                memorySignal=MemorySignal(isMemoryWorthy=False, importance="LOW", reason="NOISE"),
            )
        ],
        "expectNoTask": True,
        "expectNoNote": True,
    }


def all_generic_conversations() -> list[dict]:
    return [
        personal_planning(),
        family_decision(),
        study_status(),
        travel_change(),
        work_followup(),
        casual_noise(),
        code_switching(),
        personal_hinglish_pickup(),
        study_revise_intent(),
        travel_hotel_and_cab(),
        family_sunday_plan(),
        work_invoice_and_update(),
        garbled_noise(),
    ]


def personal_multi_meaning() -> dict:
    cid = "multi-personal"
    text = "Doctor appointment 4 baje hai, reports leke jana hai, aur medicine list bhi doctor ko dikhani hai."
    return {
        "id": cid,
        "chunks": [_chunk(cid, 0, text)],
        "firstPassEvents": [
            _event(
                cid,
                "e-appt-only",
                EventKind.FACT,
                "Doctor appointment is at 4.",
                0,
                text,
                object="doctor appointment",
                entities=["Doctor"],
                memorySignal=MemorySignal(isMemoryWorthy=True, importance="HIGH", reason="FACT"),
            )
        ],
        "repairEvents": [
            _event(
                cid,
                "e-reports",
                EventKind.COMMITMENT,
                "Bring reports to the appointment.",
                0,
                text,
                object="reports",
                entities=["reports"],
                actionSignal=ActionSignal(
                    isActionable=True,
                    role="COMMITMENT",
                    actionStrength="EXPLICIT",
                    verb="bring",
                    object="reports",
                    objectGroundingType="EXPLICIT",
                ),
                memorySignal=MemorySignal(isMemoryWorthy=True, importance="HIGH", reason="PLAN"),
            ),
            _event(
                cid,
                "e-meds",
                EventKind.COMMITMENT,
                "Show the medicine list to the doctor.",
                0,
                text,
                object="medicine list",
                entities=["medicine"],
                actionSignal=ActionSignal(
                    isActionable=True,
                    role="COMMITMENT",
                    actionStrength="EXPLICIT",
                    verb="show",
                    object="medicine list",
                    objectGroundingType="EXPLICIT",
                ),
                memorySignal=MemorySignal(isMemoryWorthy=True, importance="HIGH", reason="PLAN"),
            ),
        ],
        "expectedUnits": [
            {"meaning": "Doctor appointment is at 4.", "kind": "FACT", "sequenceStart": 0, "sequenceEnd": 0, "evidenceText": text},
            {"meaning": "Bring reports to the appointment.", "kind": "COMMITMENT", "sequenceStart": 0, "sequenceEnd": 0, "evidenceText": text},
            {"meaning": "Show the medicine list to the doctor.", "kind": "COMMITMENT", "sequenceStart": 0, "sequenceEnd": 0, "evidenceText": text},
        ],
        "expectNoteSubstrings": ["4"],
        "expectTaskSubstrings": ["report", "medicine"],
        "minEvents": 3,
    }


def travel_multi_meaning() -> dict:
    cid = "multi-travel"
    text = "Hotel airport ke paas final hai. Cab kal book karni hai. Check-in 2 PM hai."
    return {
        "id": cid,
        "chunks": [_chunk(cid, 0, text)],
        "firstPassEvents": [
            _event(
                cid,
                "e-hotel-only",
                EventKind.DECISION,
                "The hotel near the airport is finalized.",
                0,
                text,
                object="hotel",
                entities=["Hotel"],
                memorySignal=MemorySignal(isMemoryWorthy=True, importance="HIGH", reason="DECISION"),
            )
        ],
        "repairEvents": [
            _event(
                cid,
                "e-cab-book",
                EventKind.COMMITMENT,
                "Book the cab tomorrow.",
                0,
                text,
                object="cab",
                entities=["Cab"],
                timeExpression="kal",
                actionSignal=ActionSignal(
                    isActionable=True,
                    role="COMMITMENT",
                    actionStrength="EXPLICIT",
                    verb="book",
                    object="cab",
                    objectGroundingType="EXPLICIT",
                    deadline="kal",
                ),
            ),
            _event(
                cid,
                "e-checkin",
                EventKind.FACT,
                "Check-in is at 2 PM.",
                0,
                text,
                object="check-in",
                entities=["Check-in"],
                memorySignal=MemorySignal(isMemoryWorthy=True, importance="HIGH", reason="FACT"),
            ),
        ],
        "expectedUnits": [
            {"meaning": "The hotel near the airport is finalized.", "kind": "DECISION", "sequenceStart": 0, "sequenceEnd": 0, "evidenceText": text},
            {"meaning": "Book the cab tomorrow.", "kind": "COMMITMENT", "sequenceStart": 0, "sequenceEnd": 0, "evidenceText": text},
            {"meaning": "Check-in is at 2 PM.", "kind": "FACT", "sequenceStart": 0, "sequenceEnd": 0, "evidenceText": text},
        ],
        "expectNoteSubstrings": ["hotel", "2"],
        "expectTaskSubstrings": ["cab"],
        "minEvents": 3,
    }


def study_multi_meaning() -> dict:
    cid = "multi-study"
    text = "Arrays complete hain. Graphs pending hain. Friday ko mock test dena hai."
    return {
        "id": cid,
        "chunks": [_chunk(cid, 0, text)],
        "firstPassEvents": [
            _event(
                cid,
                "e-arrays-only",
                EventKind.RESULT,
                "Arrays are complete.",
                0,
                text,
                object="Arrays",
                entities=["Arrays"],
                memorySignal=MemorySignal(isMemoryWorthy=True, importance="HIGH", reason="RESULT"),
            )
        ],
        "repairEvents": [
            _event(
                cid,
                "e-graphs",
                EventKind.STATE,
                "Graphs are pending.",
                0,
                text,
                object="Graphs",
                entities=["Graphs"],
                memorySignal=MemorySignal(isMemoryWorthy=True, importance="HIGH", reason="STATUS"),
            ),
            _event(
                cid,
                "e-mock",
                EventKind.COMMITMENT,
                "Take the mock test on Friday.",
                0,
                text,
                object="mock test",
                entities=["Friday"],
                timeExpression="Friday",
                actionSignal=ActionSignal(
                    isActionable=True,
                    role="COMMITMENT",
                    actionStrength="EXPLICIT",
                    verb="take",
                    object="mock test",
                    objectGroundingType="EXPLICIT",
                    deadline="Friday",
                ),
            ),
        ],
        "expectedUnits": [
            {"meaning": "Arrays are complete.", "kind": "RESULT", "sequenceStart": 0, "sequenceEnd": 0, "evidenceText": text},
            {"meaning": "Graphs are pending.", "kind": "STATE", "sequenceStart": 0, "sequenceEnd": 0, "evidenceText": text},
            {"meaning": "Take the mock test on Friday.", "kind": "COMMITMENT", "sequenceStart": 0, "sequenceEnd": 0, "evidenceText": text},
        ],
        "expectNoteSubstrings": ["array", "graph"],
        "expectTaskSubstrings": ["mock"],
        "minEvents": 3,
    }


def work_multi_meaning() -> dict:
    cid = "multi-work"
    text = "Invoice retry fail hua. Client ko Friday update bhejna hai. Finance team ne GST field optional kar di."
    return {
        "id": cid,
        "chunks": [_chunk(cid, 0, text)],
        "firstPassEvents": [
            _event(
                cid,
                "e-invoice-only",
                EventKind.ISSUE,
                "Invoice retry failed.",
                0,
                text,
                object="invoice retry",
                entities=["Invoice"],
                memorySignal=MemorySignal(isMemoryWorthy=True, importance="HIGH", reason="ISSUE"),
            )
        ],
        "repairEvents": [
            _event(
                cid,
                "e-client-update",
                EventKind.REQUEST,
                "Send the client an update on Friday.",
                0,
                text,
                object="client update",
                entities=["Client", "Friday"],
                timeExpression="Friday",
                actionSignal=ActionSignal(
                    isActionable=True,
                    role="REQUEST",
                    actionStrength="EXPLICIT",
                    verb="send",
                    object="client update",
                    objectGroundingType="EXPLICIT",
                    deadline="Friday",
                ),
            ),
            _event(
                cid,
                "e-gst",
                EventKind.DECISION,
                "Finance made the GST field optional.",
                0,
                text,
                object="GST field",
                entities=["GST", "Finance"],
                memorySignal=MemorySignal(isMemoryWorthy=True, importance="HIGH", reason="DECISION"),
            ),
        ],
        "expectedUnits": [
            {"meaning": "Invoice retry failed.", "kind": "ISSUE", "sequenceStart": 0, "sequenceEnd": 0, "evidenceText": text},
            {"meaning": "Send the client an update on Friday.", "kind": "REQUEST", "sequenceStart": 0, "sequenceEnd": 0, "evidenceText": text},
            {"meaning": "Finance made the GST field optional.", "kind": "DECISION", "sequenceStart": 0, "sequenceEnd": 0, "evidenceText": text},
        ],
        "expectNoteSubstrings": ["invoice", "gst"],
        "expectTaskSubstrings": ["client"],
        "minEvents": 3,
    }


def filler_does_not_invent_units() -> dict:
    cid = "multi-filler"
    lines = {
        0: "haan haan",
        1: "ok wait",
        2: "theek hai theek hai",
        3: "umm actually",
    }
    return {
        "id": cid,
        "chunks": [_chunk(cid, seq, text) for seq, text in lines.items()],
        "firstPassEvents": [
            _event(cid, f"e-fill-{seq}", EventKind.NOISE, text, seq, text, memorySignal=MemorySignal(isMemoryWorthy=False, importance="LOW", reason="FILLER"))
            for seq, text in lines.items()
        ],
        "repairEvents": [],
        "expectedUnits": [],
        "expectNoTask": True,
        "expectNoNote": True,
        "minEvents": 0,
    }


def related_distinct_same_subject() -> dict:
    cid = "multi-related-distinct"
    lines = {
        0: "The candidate link is generated.",
        1: "The candidate link opens a form.",
    }
    return {
        "id": cid,
        "chunks": [_chunk(cid, seq, text) for seq, text in lines.items()],
        "events": [
            _event(
                cid,
                "e-link-gen",
                EventKind.REQUIREMENT,
                "The candidate link is generated.",
                0,
                lines[0],
                object="candidate link",
                entities=["candidate", "link"],
                memorySignal=MemorySignal(isMemoryWorthy=True, importance="HIGH", reason="REQUIREMENT"),
            ),
            _event(
                cid,
                "e-link-form",
                EventKind.REQUIREMENT,
                "The candidate link opens a form.",
                1,
                lines[1],
                object="candidate link",
                entities=["candidate", "link", "form"],
                memorySignal=MemorySignal(isMemoryWorthy=True, importance="HIGH", reason="REQUIREMENT"),
            ),
        ],
        "expectNoteSubstrings": ["generated", "form"],
        "expectNoTask": True,
    }


def multi_meaning_conversations() -> list[dict]:
    return [
        personal_multi_meaning(),
        travel_multi_meaning(),
        study_multi_meaning(),
        work_multi_meaning(),
        filler_does_not_invent_units(),
    ]
