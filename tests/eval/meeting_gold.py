"""Additional production-style gold conversations for the meeting pipeline.

Reuses BENCHMARK_CASES and adds harder multilingual, noisy, background, and
task/note fixtures. Exact wording is not required; evaluation is semantic.
"""

from __future__ import annotations

from tests.eval.conversations import BENCHMARK_CASES, _t
from tests.fixtures.lenskart_hrms_meeting import ATOMIC_ONBOARDING_MEANINGS, DENSE_ONBOARDING, lenskart_hrms_case


def _case(**kwargs) -> dict:
    case = dict(kwargs)
    case.setdefault("goldTasks", case.get("expectedTasks") or [])
    case.setdefault("goldNotes", case.get("expectedNotes") or [])
    case.setdefault("expectedTasks", case["goldTasks"])
    case.setdefault("expectedNotes", case["goldNotes"])
    case.setdefault("expectedEvidence", [
        {"id": item.get("id"), "kind": item.get("kind"), "evidenceSequences": item.get("evidenceSequences") or []}
        for item in [*case["goldTasks"], *case["goldNotes"]]
    ])
    case.setdefault("forbiddenArtifacts", [])
    return case


ADDITIONAL_CASES: list[dict] = [
    _case(
        id="atomic-dense-onboarding",
        category="technical_meeting",
        transcript=_t(f"[7] {DENSE_ONBOARDING}"),
        goldTasks=[
            {
                "id": "t-onboarding",
                "kind": "task",
                "meaning": "Add an employee-page action that generates a candidate link, send it to the candidate, and let the candidate submit details to reduce manual HR updates",
                "evidenceSequences": [7],
            }
        ],
        goldNotes=[
            {"id": "n-manual", "kind": "note", "meaning": "The flow reduces HR repeatedly updating candidate information manually", "evidenceSequences": [7]},
        ],
        goldCandidates=[{"id": f"c{index}", "meaning": meaning, "evidenceSequences": [7]} for index, meaning in enumerate(ATOMIC_ONBOARDING_MEANINGS)],
        probe="atomic",
    ),
    _case(
        id="cross-window-onboarding",
        category="technical_meeting",
        transcript=_t(
            "[40] We need candidate onboarding in the HRMS project before payroll. " + ("context " * 80),
            "[41] Generate the candidate link from the employee page. " + ("detail " * 80),
            "[42] Candidate will submit details using it so HR does not type the same data. " + ("followup " * 80),
        ),
        goldTasks=[
            {
                "id": "t1",
                "kind": "task",
                "meaning": "Implement candidate onboarding by generating a candidate link from the employee page so the candidate can submit details",
                "evidenceSequences": [40, 41, 42],
            }
        ],
        goldNotes=[
            {"id": "n1", "kind": "note", "meaning": "Candidates submit details through the generated link so HR does not retype data", "evidenceSequences": [42]},
        ],
        probe="cross_window",
        forceSmallWindows=True,
    ),
    _case(
        id="noisy-1-80-crore",
        category="vague",
        transcript=_t(
            "[0] Website credit discussion 1 80 crore 01:21 percent something unclear.",
            "[1] Do not treat that as a confirmed number.",
            "[2] Mira will send the cleaned transcript to finance tomorrow.",
        ),
        goldTasks=[
            {
                "id": "t1",
                "kind": "task",
                "meaning": "Mira will send the cleaned transcript to finance tomorrow",
                "evidenceSequences": [2],
                "ownerText": "Mira",
                "dueDateText": "tomorrow",
            }
        ],
        goldNotes=[
            {"id": "n1", "kind": "note", "meaning": "A website/credit figure was mentioned with damaged STT and is not a confirmed amount", "evidenceSequences": [0, 1]},
        ],
        forbiddenArtifacts=["180 members", "180 crore confirmed", "1.21 percent growth"],
        probe="noisy_number",
    ),
    _case(
        id="noisy-around-twenty-thirty",
        category="vague",
        transcript=_t(
            "[0] Headcount is around twenty... maybe thirty, I am not sure.",
            "[1] Let's not lock a number until HR shares the roster.",
        ),
        goldTasks=[],
        goldNotes=[
            {"id": "n1", "kind": "note", "meaning": "Headcount was estimated around twenty or thirty and is not locked until HR shares the roster", "evidenceSequences": [0, 1]},
        ],
        forbiddenArtifacts=["exactly 20 employees", "exactly 30 employees", "25 employees"],
        probe="noisy_number",
    ),
    _case(
        id="task-note-payroll-pf",
        category="technical_meeting",
        transcript=_t(
            "[0] We need to build payroll.",
            "[1] Payroll should calculate PF deductions.",
            "[2] Leave cancellation should also adjust salary.",
        ),
        goldTasks=[
            {"id": "t1", "kind": "task", "meaning": "Build payroll", "evidenceSequences": [0]},
        ],
        goldNotes=[
            {"id": "n1", "kind": "note", "meaning": "Payroll should calculate PF deductions", "evidenceSequences": [1]},
            {"id": "n2", "kind": "note", "meaning": "Leave cancellation should adjust salary", "evidenceSequences": [2]},
        ],
        probe="task_note",
    ),
    _case(
        id="idea-not-task-analytics",
        category="vague",
        transcript=_t(
            "[0] Maybe later we can add analytics.",
            "[1] Nobody is committing to it this quarter.",
        ),
        goldTasks=[],
        goldNotes=[
            {"id": "n1", "kind": "note", "meaning": "Analytics was mentioned as a possible later idea, not a current commitment", "evidenceSequences": [0, 1]},
        ],
        nonTaskSequences=[0, 1],
        probe="task_note",
    ),
    _case(
        id="rahul-integrate-tomorrow",
        category="implicit_assignment",
        transcript=_t(
            "[0] Rahul will integrate the API tomorrow.",
            "[1] He already has the staging credentials.",
        ),
        goldTasks=[
            {
                "id": "t1",
                "kind": "task",
                "meaning": "Rahul will integrate the API tomorrow",
                "evidenceSequences": [0],
                "ownerText": "Rahul",
                "dueDateText": "tomorrow",
            }
        ],
        goldNotes=[
            {"id": "n1", "kind": "note", "meaning": "Rahul already has the staging credentials", "evidenceSequences": [1]},
        ],
        probe="task_note",
    ),
    _case(
        id="reduce-hr-manual-rationale",
        category="technical_meeting",
        transcript=_t(
            "[0] Candidate onboarding link is already in progress.",
            "[1] This will reduce HR manual work because candidates fill their own details.",
        ),
        goldTasks=[],
        goldNotes=[
            {"id": "n1", "kind": "note", "meaning": "The candidate onboarding link is intended to reduce HR manual work because candidates fill their own details", "evidenceSequences": [0, 1]},
        ],
        probe="task_note",
    ),
    _case(
        id="background-podcast-then-meeting",
        category="casual_speech",
        transcript=_t(
            "[0] Podcast host: today's episode is about cricket world cup tickets.",
            "[1] Advertisement for a loan app at nine percent.",
            "[2] Okay team, ignore that, meeting started.",
            "[3] Priya will share the sprint board before standup.",
        ),
        goldTasks=[
            {
                "id": "t1",
                "kind": "task",
                "meaning": "Priya will share the sprint board before standup",
                "evidenceSequences": [3],
                "ownerText": "Priya",
            }
        ],
        goldNotes=[],
        backgroundSequences=[0, 1],
        forbiddenArtifacts=["cricket world cup tickets", "loan app at nine percent"],
        nonTaskSequences=[0, 1, 2],
    ),
    _case(
        id="hinglish-leave-payroll",
        category="hindi_hinglish",
        transcript=_t(
            "[0] Payroll module mein leave aur leave cancellation hona chahiye.",
            "[1] Salary deduction leave cancel ke baad adjust ho.",
            "[2] PF calculation bhi usi module mein rahe.",
            "[3] Dev, yeh payroll flow kal tak likh dena.",
        ),
        goldTasks=[
            {
                "id": "t1",
                "kind": "task",
                "meaning": "Dev will write the payroll flow covering leave, leave cancellation, salary deduction and PF by tomorrow",
                "evidenceSequences": [0, 1, 2, 3],
                "ownerText": "Dev",
                "dueDateText": "kal",
            }
        ],
        goldNotes=[
            {"id": "n1", "kind": "note", "meaning": "Payroll should handle leave, leave cancellation, salary deduction and PF calculation", "evidenceSequences": [0, 1, 2]},
        ],
    ),
    _case(
        id="pre-meeting-chatter",
        category="casual_speech",
        transcript=_t(
            "[0] Wait, the mic is still catching the TV.",
            "[1] Can someone close the door?",
            "[2] Okay. Aisha will update the grey banner before review.",
        ),
        goldTasks=[
            {
                "id": "t1",
                "kind": "task",
                "meaning": "Aisha will update the grey banner before review",
                "evidenceSequences": [2],
                "ownerText": "Aisha",
            }
        ],
        goldNotes=[],
        nonTaskSequences=[0, 1],
        forbiddenArtifacts=["close the door as a project task"],
    ),
    _case(
        id="product-requirement-decision",
        category="technical_meeting",
        transcript=_t(
            "[0] Expense tracker should show company expenses and payment status.",
            "[1] We decided against a separate finance app.",
            "[2] Kabir will add the expense tracker to HRMS this sprint.",
        ),
        goldTasks=[
            {
                "id": "t1",
                "kind": "task",
                "meaning": "Kabir will add the expense tracker to HRMS this sprint",
                "evidenceSequences": [2],
                "ownerText": "Kabir",
                "dueDateText": "this sprint",
            }
        ],
        goldNotes=[
            {"id": "n1", "kind": "note", "meaning": "Expense tracker should show company expenses and payment status", "evidenceSequences": [0]},
            {"id": "n2", "kind": "note", "meaning": "The team decided against a separate finance app", "evidenceSequences": [1]},
        ],
    ),
    _case(
        id="cancelled-then-replaced",
        category="changed_decision",
        transcript=_t(
            "[0] Let's hire an outside STT vendor.",
            "[1] Cancel that. We will keep Deepgram.",
            "[2] Omar will renew the Deepgram contract this week.",
        ),
        goldTasks=[
            {
                "id": "t1",
                "kind": "task",
                "meaning": "Omar will renew the Deepgram contract this week",
                "evidenceSequences": [2],
                "ownerText": "Omar",
                "dueDateText": "this week",
            }
        ],
        goldNotes=[
            {"id": "n1", "kind": "note", "meaning": "Hiring an outside STT vendor was cancelled; the team will keep Deepgram", "evidenceSequences": [0, 1]},
        ],
        forbiddenArtifacts=["hire an outside STT vendor as an active task"],
    ),
    _case(
        id="code-switch-onboarding-button",
        category="hindi_hinglish",
        transcript=_t(
            "[0] Employee page pe ek action button add karna hai.",
            "[1] Click karte hi candidate-specific link generate hoga.",
            "[2] Candidate us link se form fill karega.",
        ),
        goldTasks=[
            {
                "id": "t1",
                "kind": "task",
                "meaning": "Add an employee-page action that generates a candidate-specific link for form fill",
                "evidenceSequences": [0, 1, 2],
            }
        ],
        goldNotes=[
            {"id": "n1", "kind": "note", "meaning": "The candidate fills the form through the generated link", "evidenceSequences": [2]},
        ],
        goldCandidates=[
            {"id": "c0", "meaning": "employee-page action button", "evidenceSequences": [0]},
            {"id": "c1", "meaning": "candidate-specific generated link", "evidenceSequences": [1]},
            {"id": "c2", "meaning": "candidate submits form through link", "evidenceSequences": [2]},
        ],
    ),
    _case(
        id="multi-speaker-explicit-split",
        category="multiple_people",
        transcript=_t(
            "[0] Speaker 1: I will write the API contract tonight.",
            "[1] Speaker 2: I will implement the employee page button tomorrow.",
            "[2] Speaker 0: Those are two different tasks, keep them separate.",
        ),
        goldTasks=[
            {"id": "t1", "kind": "task", "meaning": "Write the API contract tonight", "evidenceSequences": [0], "dueDateText": "tonight"},
            {"id": "t2", "kind": "task", "meaning": "Implement the employee page button tomorrow", "evidenceSequences": [1], "dueDateText": "tomorrow"},
        ],
        goldNotes=[],
        expectedArtifactCount=2,
    ),
    _case(
        id="hindi-only-leave-cancellation",
        category="hindi_hinglish",
        transcript=_t(
            "[0] हमें लीव मॉड्यूल में कैंसलेशन जोड़ना है।",
            "[1] अंशु कल तक स्पेक लिख देंगी।",
        ),
        goldTasks=[
            {
                "id": "t1",
                "kind": "task",
                "meaning": "Anshu will write the leave-module cancellation spec by tomorrow",
                "evidenceSequences": [1],
                "ownerText": "अंशु",
                "dueDateText": "कल",
            }
        ],
        goldNotes=[
            {"id": "n1", "kind": "note", "meaning": "The leave module needs cancellation support", "evidenceSequences": [0]},
        ],
    ),
    _case(
        id="clean-stt-explicit-payroll-api",
        category="technical_meeting",
        transcript=_t(
            "[0] Ship the payroll API on Friday.",
            "[1] Priya owns it.",
            "[2] Leave handling is already covered in the spec.",
        ),
        goldTasks=[
            {
                "id": "t1",
                "kind": "task",
                "meaning": "Priya will ship the payroll API on Friday",
                "evidenceSequences": [0, 1],
                "ownerText": "Priya",
                "dueDateText": "Friday",
            }
        ],
        goldNotes=[
            {"id": "n1", "kind": "note", "meaning": "Leave handling is already covered in the spec", "evidenceSequences": [2]},
        ],
    ),
    _case(
        id="noisy-timestamp-percent",
        category="vague",
        transcript=_t(
            "[0] Growth was 01:21 percent according to the damaged recording.",
            "[1] Do not treat that timestamp as a growth rate.",
            "[2] Kabir will request the original audio from STT.",
        ),
        goldTasks=[
            {
                "id": "t1",
                "kind": "task",
                "meaning": "Kabir will request the original audio from STT",
                "evidenceSequences": [2],
                "ownerText": "Kabir",
            }
        ],
        goldNotes=[
            {"id": "n1", "kind": "note", "meaning": "A damaged recording mentioned 01:21 percent and should not be treated as a confirmed growth rate", "evidenceSequences": [0, 1]},
        ],
        forbiddenArtifacts=["1.21 percent growth", "21 percent growth confirmed"],
        probe="noisy_number",
    ),
]


def gold_cases() -> list[dict]:
    seen: set[str] = set()
    cases: list[dict] = []
    for item in [*BENCHMARK_CASES, *ADDITIONAL_CASES, lenskart_hrms_case()]:
        case = _case(**item)
        if case["id"] in seen:
            continue
        seen.add(case["id"])
        cases.append(case)
    return cases


def tail_position_cases() -> list[dict]:
    filler = "We are still reviewing status. No one is assigned on this stretch. There is no new commitment in this status update."
    n = 20

    def _lines(index: int, commitment: str) -> str:
        rows = [f"[{i}] {filler}" for i in range(n)]
        rows[index] = f"[{index}] {commitment}"
        return _t(*rows)

    specs = [
        ("tail-en-first-10", 1, "Mira will close the drain ticket today.", "Mira will close the drain ticket today", "Mira"),
        ("tail-en-middle", 10, "Kabir will add the retry budget dashboard this month.", "Kabir will add the retry budget dashboard this month", "Kabir"),
        ("tail-en-last-25", 15, "Omar will restore staging seed data tonight.", "Omar will restore staging seed data tonight", "Omar"),
        ("tail-en-last-10", 18, "Priya will share the sprint board before standup.", "Priya will share the sprint board before standup", "Priya"),
        ("tail-en-final-sequence", 19, "Neha will publish the meeting notes after STOP drain completes.", "Neha will publish the meeting notes after STOP drain", "Neha"),
        ("tail-hi-final-sequence", 19, "अंशु कल तक स्पेक लिख देंगी।", "Anshu will write the spec by tomorrow", "अंशु"),
        ("tail-hinglish-final-sequence", 19, "Rahul kal tak API integrate kar dega.", "Rahul will integrate the API by tomorrow", "Rahul"),
    ]
    cases = []
    for case_id, index, spoken, meaning, owner in specs:
        cases.append(
            _case(
                id=case_id,
                category="tail_position",
                transcript=_lines(index, spoken),
                goldTasks=[
                    {
                        "id": "t1",
                        "kind": "task",
                        "meaning": meaning,
                        "evidenceSequences": [index],
                        "ownerText": owner,
                    }
                ],
                goldNotes=[],
                probe="tail_position",
            )
        )
    return cases
