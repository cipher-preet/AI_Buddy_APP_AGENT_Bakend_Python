from tests.fixtures.lenskart_hrms_meeting import lenskart_hrms_case

REQUIRED_CATEGORIES = {
    "casual_speech",
    "technical_meeting",
    "no_task",
    "implicit_assignment",
    "multiple_people",
    "changed_decision",
    "task_completion_later",
    "hindi_hinglish",
    "interruptions",
    "vague",
    "long_meeting",
}


def _t(*rows: str) -> str:
    return "\n".join(rows)


BENCHMARK_CASES: list[dict] = [
    {
        "id": "casual-weekend-plans",
        "category": "casual_speech",
        "transcript": _t(
            "[0] How was your Saturday?",
            "[1] Pretty slow, we just walked around the lake and got chai.",
            "[2] Same, I slept till noon and then watched a match.",
            "[3] We should hang out like that more often, just saying.",
        ),
        "goldTasks": [],
        "goldNotes": [
            {"id": "n1", "kind": "note", "meaning": "Saturday was spent walking around the lake and getting chai.", "evidenceSequences": [1], "reviewStatus": "OPTIONAL"},
        ],
        "nonTaskSequences": [0, 2, 3],
    },
    {
        "id": "casual-family-call",
        "category": "casual_speech",
        "transcript": _t(
            "[0] Mom asked if I am eating properly.",
            "[1] I told her the office pantry has food this week.",
            "[2] She laughed and said the dog is chewing shoes again.",
        ),
        "goldTasks": [],
        "goldNotes": [
            {"id": "n1", "kind": "note", "meaning": "A family call mentioned pantry food and the dog chewing shoes.", "evidenceSequences": [0, 1, 2]},
        ],
        "nonTaskSequences": [0, 1, 2],
    },
    {
        "id": "technical-stt-drain",
        "category": "technical_meeting",
        "transcript": _t(
            "[0] The worker still marks READY_FOR_PROCESSING while sequence 14 is in flight.",
            "[1] That is why STOP can skip a pending Deepgram job.",
            "[2] Mira, please add a drain gate that waits for every expected sequence before finalization.",
            "[3] Put it on Thursday evening so we can ship Friday.",
            "[4] Rahul already published the current race notes in the channel.",
        ),
        "goldTasks": [
            {
                "id": "t1",
                "kind": "task",
                "meaning": "Mira will add a drain gate that waits for every expected sequence before finalization by Thursday evening.",
                "evidenceSequences": [2, 3],
                "ownerText": "Mira",
                "dueDateText": "Thursday evening",
            }
        ],
        "goldNotes": [
            {"id": "n1", "kind": "note", "meaning": "STOP can skip a pending Deepgram job because READY_FOR_PROCESSING is marked while sequence 14 is in flight.", "evidenceSequences": [0, 1]},
            {"id": "n2", "kind": "note", "meaning": "Rahul already published the race notes.", "evidenceSequences": [4]},
        ],
        "nonTaskSequences": [0, 1, 4],
    },
    {
        "id": "technical-quota-and-schema",
        "category": "technical_meeting",
        "transcript": _t(
            "[0] Groq is returning 429s after about twenty four requests.",
            "[1] The structured schema retry is also burning tokens.",
            "[2] Can we keep json_schema first and only fall back to json_object after a 422?",
            "[3] I will patch the provider this afternoon.",
        ),
        "goldTasks": [
            {
                "id": "t1",
                "kind": "task",
                "meaning": "Patch the provider so json_schema is tried first and json_object is used only after a 422.",
                "evidenceSequences": [2, 3],
            }
        ],
        "goldNotes": [
            {"id": "n1", "kind": "note", "meaning": "Groq is returning 429s after about twenty four requests and schema retries are burning tokens.", "evidenceSequences": [0, 1]},
        ],
        "nonTaskSequences": [0, 1],
    },
    {
        "id": "no-task-status-sync",
        "category": "no_task",
        "transcript": _t(
            "[0] Just a sync, nothing to pick up.",
            "[1] The staging deploy from last night is healthy.",
            "[2] Error rate stayed under half a percent.",
            "[3] We can talk about the next milestone next week if we want.",
        ),
        "goldTasks": [],
        "goldNotes": [
            {"id": "n1", "kind": "note", "meaning": "Staging deploy from last night is healthy with error rate under half a percent.", "evidenceSequences": [1, 2]},
        ],
        "nonTaskSequences": [0, 3],
    },
    {
        "id": "no-task-research-chat",
        "category": "no_task",
        "transcript": _t(
            "[0] I read that soral membranes soften shock at the edge.",
            "[1] If that is true the centre would stay stable during a drop.",
            "[2] Interesting, but nobody is asking anyone to change the design today.",
        ),
        "goldTasks": [],
        "goldNotes": [
            {"id": "n1", "kind": "note", "meaning": "Soral membranes may soften shock at the edge so the centre stays stable during a drop.", "evidenceSequences": [0, 1]},
        ],
        "nonTaskSequences": [2],
    },
    {
        "id": "implicit-owner-shift",
        "category": "implicit_assignment",
        "transcript": _t(
            "[0] The banner copy is still wrong on login.",
            "[1] Design will hate us if it ships like this.",
            "[2] Aisha, you are already in Figma, yeah?",
            "[3] Yeah I can take a pass before standup tomorrow.",
        ),
        "goldTasks": [
            {
                "id": "t1",
                "kind": "task",
                "meaning": "Aisha will revise the login banner copy before standup tomorrow.",
                "evidenceSequences": [0, 2, 3],
                "ownerText": "Aisha",
                "dueDateText": "before standup tomorrow",
            }
        ],
        "goldNotes": [],
        "nonTaskSequences": [1],
    },
    {
        "id": "implicit-we-should",
        "category": "implicit_assignment",
        "transcript": _t(
            "[0] Customers keep asking for export to CSV.",
            "[1] We should just do it in the reports page.",
            "[2] Kabir, since you own reports, that is you unless you push back.",
            "[3] Fine, I will put a spike on it this sprint.",
        ),
        "goldTasks": [
            {
                "id": "t1",
                "kind": "task",
                "meaning": "Kabir will spike CSV export on the reports page this sprint.",
                "evidenceSequences": [1, 2, 3],
                "ownerText": "Kabir",
                "dueDateText": "this sprint",
            }
        ],
        "goldNotes": [
            {"id": "n1", "kind": "note", "meaning": "Customers keep asking for export to CSV.", "evidenceSequences": [0]},
        ],
        "nonTaskSequences": [],
    },
    {
        "id": "multiple-people-split-work",
        "category": "multiple_people",
        "transcript": _t(
            "[0] Three things before the demo.",
            "[1] Neha will record the walkthrough.",
            "[2] Omar will restore the staging seed data.",
            "[3] I will send the customer the join link tonight.",
            "[4] Please do not mix these up, they are separate.",
        ),
        "goldTasks": [
            {"id": "t1", "kind": "task", "meaning": "Neha will record the walkthrough.", "evidenceSequences": [1], "ownerText": "Neha"},
            {"id": "t2", "kind": "task", "meaning": "Omar will restore the staging seed data.", "evidenceSequences": [2], "ownerText": "Omar"},
            {"id": "t3", "kind": "task", "meaning": "Send the customer the join link tonight.", "evidenceSequences": [3], "dueDateText": "tonight"},
        ],
        "goldNotes": [],
        "nonTaskSequences": [0, 4],
    },
    {
        "id": "multiple-people-handoff",
        "category": "multiple_people",
        "transcript": _t(
            "[0] Priya started the vendor email but did not finish.",
            "[1] Dev, can you close it because Priya is on leave?",
            "[2] Yes, I will send the vendor email today.",
            "[3] Keep Priya in cc so she sees it next week.",
        ),
        "goldTasks": [
            {
                "id": "t1",
                "kind": "task",
                "meaning": "Dev will send the vendor email today and keep Priya in cc.",
                "evidenceSequences": [1, 2, 3],
                "ownerText": "Dev",
                "dueDateText": "today",
            }
        ],
        "goldNotes": [
            {"id": "n1", "kind": "note", "meaning": "Priya started the vendor email and is on leave.", "evidenceSequences": [0, 1]},
        ],
        "nonTaskSequences": [0],
    },
    {
        "id": "changed-decision-color",
        "category": "changed_decision",
        "transcript": _t(
            "[0] Let's ship the banner in green.",
            "[1] Wait, legal said green looks like a success state.",
            "[2] Cancel green. Use the neutral grey instead.",
            "[3] I will update the banner to grey before review.",
        ),
        "goldTasks": [
            {
                "id": "t1",
                "kind": "task",
                "meaning": "Update the banner to neutral grey before review; green is cancelled.",
                "evidenceSequences": [2, 3],
                "state": "proposed",
            }
        ],
        "goldNotes": [
            {"id": "n1", "kind": "note", "meaning": "Legal said green looks like a success state, so green is cancelled.", "evidenceSequences": [1, 2]},
        ],
        "nonTaskSequences": [0],
        "expectedUpdates": [{"goldId": "t1", "state": "proposed"}],
    },
    {
        "id": "changed-decision-owner",
        "category": "changed_decision",
        "transcript": _t(
            "[0] Rahul will write the drain-safe finalizer notes.",
            "[1] Actually Rahul is on the quota work.",
            "[2] Mira will write the drain-safe finalizer notes instead.",
        ),
        "goldTasks": [
            {
                "id": "t1",
                "kind": "task",
                "meaning": "Mira will write the drain-safe finalizer notes.",
                "evidenceSequences": [2],
                "ownerText": "Mira",
            }
        ],
        "goldNotes": [
            {"id": "n1", "kind": "note", "meaning": "Rahul will not write the notes because he is on quota work.", "evidenceSequences": [0, 1]},
        ],
        "nonTaskSequences": [0],
    },
    {
        "id": "completion-later-window",
        "category": "task_completion_later",
        "transcript": _t(
            "[0] Rahul will prepare the pricing proposal.",
            "[1] We can send it after legal looks at the discount table.",
            "[80] Rahul already sent the pricing proposal to the customer.",
        ),
        "windows": [
            {"id": "w1", "transcript": "[0] Rahul will prepare the pricing proposal.\n[1] We can send it after legal looks at the discount table."},
            {"id": "w2", "transcript": "[80] Rahul already sent the pricing proposal to the customer."},
        ],
        "goldTasks": [
            {
                "id": "t1",
                "kind": "task",
                "meaning": "Rahul prepared and sent the pricing proposal to the customer.",
                "evidenceSequences": [0, 80],
                "ownerText": "Rahul",
                "state": "completed",
            }
        ],
        "goldNotes": [
            {"id": "n1", "kind": "note", "meaning": "The proposal could be sent after legal looks at the discount table.", "evidenceSequences": [1]},
        ],
        "nonTaskSequences": [1],
        "expectedUpdates": [{"goldId": "t1", "state": "completed"}],
        "expectedArtifactCount": 1,
    },
    {
        "id": "cancel-later-window",
        "category": "task_completion_later",
        "transcript": _t(
            "[0] Please book the offsite for next Friday.",
            "[1] Keep it in Pune if rooms exist.",
            "[40] Cancel the offsite booking; leadership moved it to next month.",
        ),
        "windows": [
            {"id": "w1", "transcript": "[0] Please book the offsite for next Friday.\n[1] Keep it in Pune if rooms exist."},
            {"id": "w2", "transcript": "[40] Cancel the offsite booking; leadership moved it to next month."},
        ],
        "goldTasks": [
            {
                "id": "t1",
                "kind": "task",
                "meaning": "The offsite booking is cancelled because leadership moved it to next month.",
                "evidenceSequences": [0, 40],
                "state": "cancelled",
            }
        ],
        "goldNotes": [
            {"id": "n1", "kind": "note", "meaning": "The offsite had been planned for Pune next Friday.", "evidenceSequences": [0, 1]},
        ],
        "expectedUpdates": [{"goldId": "t1", "state": "cancelled"}],
        "expectedArtifactCount": 1,
    },
    {
        "id": "hinglish-zafran-lens",
        "category": "hindi_hinglish",
        "transcript": _t(
            "[0] Kal zafran-lens ko andhere mein rakhna hai.",
            "[1] Keep it away from the bright lamps overnight.",
            "[2] Vireli pollen is only a fact, nobody is asking for a task there.",
        ),
        "goldTasks": [
            {
                "id": "t1",
                "kind": "task",
                "meaning": "Keep the zafran-lens in darkness overnight, away from bright lamps.",
                "evidenceSequences": [0, 1],
            }
        ],
        "goldNotes": [
            {"id": "n1", "kind": "note", "meaning": "Vireli pollen was mentioned as a fact, not a request.", "evidenceSequences": [2]},
        ],
        "nonTaskSequences": [2],
    },
    {
        "id": "hindi-ticket-handoff",
        "category": "hindi_hinglish",
        "transcript": _t(
            "[0] Sequence wait wala ticket abhi open nahi hua.",
            "[1] Mira, yeh ticket Thursday evening tak close kar do.",
            "[2] Rahul mat dena, quota pe hai.",
        ),
        "goldTasks": [
            {
                "id": "t1",
                "kind": "task",
                "meaning": "Mira will close the sequence-wait ticket by Thursday evening.",
                "evidenceSequences": [0, 1],
                "ownerText": "Mira",
                "dueDateText": "Thursday evening",
            }
        ],
        "goldNotes": [
            {"id": "n1", "kind": "note", "meaning": "Rahul should not own it because he is on quota work.", "evidenceSequences": [2]},
        ],
        "nonTaskSequences": [2],
    },
    {
        "id": "interruption-mid-commitment",
        "category": "interruptions",
        "transcript": _t(
            "[0] I will send the",
            "[1] wait, someone is at the door.",
            "[2] okay I am back.",
            "[3] I will send the invoice to finance before six.",
            "[4] Sorry, the dog started barking in the middle.",
        ),
        "goldTasks": [
            {
                "id": "t1",
                "kind": "task",
                "meaning": "Send the invoice to finance before six.",
                "evidenceSequences": [3],
                "dueDateText": "before six",
            }
        ],
        "goldNotes": [
            {"id": "n1", "kind": "note", "meaning": "The speaker was interrupted by the door and a barking dog.", "evidenceSequences": [1, 4], "reviewStatus": "OPTIONAL"},
        ],
        "nonTaskSequences": [0, 1, 2, 4],
    },
    {
        "id": "interruption-correction",
        "category": "interruptions",
        "transcript": _t(
            "[0] Please page Rahul for the staging outage.",
            "[1] No, stop, Rahul is not on call.",
            "[2] Page Sana instead, she has the pager.",
        ),
        "goldTasks": [
            {
                "id": "t1",
                "kind": "task",
                "meaning": "Page Sana for the staging outage; do not page Rahul.",
                "evidenceSequences": [1, 2],
                "ownerText": "Sana",
            }
        ],
        "goldNotes": [
            {"id": "n1", "kind": "note", "meaning": "Rahul is not on call.", "evidenceSequences": [1], "reviewStatus": "OPTIONAL"},
        ],
        "nonTaskSequences": [0],
    },
    {
        "id": "vague-maybe-later",
        "category": "vague",
        "transcript": _t(
            "[0] We might want a nicer settings page someday.",
            "[1] Not sure who would own it.",
            "[2] Let's not make it a task until someone actually asks.",
        ),
        "goldTasks": [],
        "goldNotes": [
            {"id": "n1", "kind": "note", "meaning": "A nicer settings page was mentioned as a someday idea, not a current task.", "evidenceSequences": [0, 2]},
        ],
        "nonTaskSequences": [0, 1, 2],
    },
    {
        "id": "vague-unfinished-thought",
        "category": "vague",
        "transcript": _t(
            "[0] The reports feel slow-ish.",
            "[1] Could be the warehouse, could be the chart library, who knows.",
            "[2] Maybe we look later if customers complain.",
        ),
        "goldTasks": [],
        "goldNotes": [
            {"id": "n1", "kind": "note", "meaning": "Reports feel slow; cause is unknown and action is deferred unless customers complain.", "evidenceSequences": [0, 1, 2]},
        ],
        "nonTaskSequences": [0, 1, 2],
    },
    {
        "id": "long-meeting-two-hour",
        "category": "long_meeting",
        "transcript": _t(
            "[0] Hour one: we walked through the drain race on STOP.",
            "[1] Agreement was that pending STT must block READY_FOR_PROCESSING.",
            "[2] Mira will implement the drain gate.",
            "[3] Hour two: quota errors on Groq showed up in the same review.",
            "[4] Kabir will add a retry budget dashboard this week.",
            "[5] Remaining thirty minutes were hallway chat about lunch.",
        ),
        "windows": [
            {"id": "h1", "transcript": "[0] Hour one: we walked through the drain race on STOP.\n[1] Agreement was that pending STT must block READY_FOR_PROCESSING.\n[2] Mira will implement the drain gate."},
            {"id": "h2", "transcript": "[3] Hour two: quota errors on Groq showed up in the same review.\n[4] Kabir will add a retry budget dashboard this week.\n[5] Remaining thirty minutes were hallway chat about lunch."},
        ],
        "goldTasks": [
            {"id": "t1", "kind": "task", "meaning": "Mira will implement the drain gate so pending STT blocks READY_FOR_PROCESSING.", "evidenceSequences": [1, 2], "ownerText": "Mira"},
            {"id": "t2", "kind": "task", "meaning": "Kabir will add a retry budget dashboard this week.", "evidenceSequences": [4], "ownerText": "Kabir", "dueDateText": "this week"},
        ],
        "goldNotes": [
            {"id": "n1", "kind": "note", "meaning": "Hour one reviewed the STOP drain race.", "evidenceSequences": [0]},
            {"id": "n2", "kind": "note", "meaning": "Hour two reviewed Groq quota errors.", "evidenceSequences": [3]},
        ],
        "nonTaskSequences": [5],
        "expectedArtifactCount": 2,
    },
    {
        "id": "long-meeting-four-hour",
        "category": "long_meeting",
        "transcript": _t(
            "[0] Hour one: Neha will record the customer walkthrough.",
            "[1] Hour two: Omar will restore staging seed data.",
            "[2] Hour three: legal rejected the green banner.",
            "[3] Use grey, and Aisha will update the banner.",
            "[4] Hour four: Neha already uploaded the walkthrough recording.",
            "[5] Omar still needs to restore staging before the demo.",
        ),
        "windows": [
            {"id": "h1", "transcript": "[0] Hour one: Neha will record the customer walkthrough."},
            {"id": "h2", "transcript": "[1] Hour two: Omar will restore staging seed data."},
            {"id": "h3", "transcript": "[2] Hour three: legal rejected the green banner.\n[3] Use grey, and Aisha will update the banner."},
            {"id": "h4", "transcript": "[4] Hour four: Neha already uploaded the walkthrough recording.\n[5] Omar still needs to restore staging before the demo."},
        ],
        "goldTasks": [
            {"id": "t1", "kind": "task", "meaning": "Neha recorded and uploaded the customer walkthrough.", "evidenceSequences": [0, 4], "ownerText": "Neha", "state": "completed"},
            {"id": "t2", "kind": "task", "meaning": "Omar will restore staging seed data before the demo.", "evidenceSequences": [1, 5], "ownerText": "Omar"},
            {"id": "t3", "kind": "task", "meaning": "Aisha will update the banner to grey after legal rejected green.", "evidenceSequences": [2, 3], "ownerText": "Aisha"},
        ],
        "goldNotes": [
            {"id": "n1", "kind": "note", "meaning": "Legal rejected the green banner.", "evidenceSequences": [2]},
        ],
        "expectedUpdates": [{"goldId": "t1", "state": "completed"}],
        "expectedArtifactCount": 3,
    },
    {
        "id": "shared-vocab-distinct-actions",
        "category": "technical_meeting",
        "transcript": _t(
            "[0] Rahul needs to deploy the backend.",
            "[1] Rahul also needs to test the backend.",
            "[2] Those are two different jobs, do not merge them.",
        ),
        "goldTasks": [
            {"id": "t1", "kind": "task", "meaning": "Rahul needs to deploy the backend.", "evidenceSequences": [0], "ownerText": "Rahul"},
            {"id": "t2", "kind": "task", "meaning": "Rahul needs to test the backend.", "evidenceSequences": [1], "ownerText": "Rahul"},
        ],
        "goldNotes": [],
        "nonTaskSequences": [2],
        "expectedArtifactCount": 2,
    },
    {
        "id": "paraphrase-same-task-two-windows",
        "category": "task_completion_later",
        "transcript": _t(
            "[0] Mira will file the drain-safe finalizer notes.",
            "[40] Mira still owns writing those drain-safe finalizer notes.",
        ),
        "windows": [
            {"id": "w1", "transcript": "[0] Mira will file the drain-safe finalizer notes."},
            {"id": "w2", "transcript": "[40] Mira still owns writing those drain-safe finalizer notes."},
        ],
        "goldTasks": [
            {
                "id": "t1",
                "kind": "task",
                "meaning": "Mira will file the drain-safe finalizer notes.",
                "evidenceSequences": [0, 40],
                "ownerText": "Mira",
            }
        ],
        "goldNotes": [],
        "expectedArtifactCount": 1,
    },
    lenskart_hrms_case(),
]
