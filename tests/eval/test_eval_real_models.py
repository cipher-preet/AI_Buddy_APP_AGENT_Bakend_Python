from types import SimpleNamespace

from services.conversation.eval_metrics import PredictedItem, score_case
from services.conversation.eval_real_models import (
    classify_first_loss,
    chunks_from_transcript,
    normalize_case,
)
from services.conversation.meeting_pipeline.schemas import CandidateKind, MeetingCandidate, VerifiedArtifact, VerifierVerdict


def test_chunks_from_transcript_keep_sequence_ids():
    chunks = chunks_from_transcript("c1", "[7] Generate the candidate link.\n[8] Candidate fills details.")
    assert [item.sequenceNumber for item in chunks] == [7, 8]
    assert chunks[0].rawText == "Generate the candidate link."


def test_first_loss_extraction_miss():
    case = normalize_case(
        {
            "id": "miss",
            "transcript": "[0] We need to build payroll.",
            "goldTasks": [{"id": "t1", "kind": "task", "meaning": "Build payroll", "evidenceSequences": [0]}],
            "goldNotes": [],
        }
    )
    result = SimpleNamespace(
        candidates=[],
        claims=[],
        rejected=[],
        tasks=[],
        notes=[],
    )
    predicted = []
    score = score_case(case, predicted, predicted_candidates=[])
    assert classify_first_loss(case, result, score) == "REAL_INFORMATION_LOSS"


def test_first_loss_verifier_false_reject():
    case = normalize_case(
        {
            "id": "reject",
            "transcript": "[0] We need to build payroll.",
            "goldTasks": [{"id": "t1", "kind": "task", "meaning": "Build payroll", "evidenceSequences": [0]}],
            "goldNotes": [],
        }
    )
    candidate = MeetingCandidate(
        candidateId="c1",
        kind=CandidateKind.ACTION,
        meaning="Build payroll",
        evidenceSequences=[0],
        sourceWindowId="w0",
    )
    claim = SimpleNamespace(
        kind="task",
        title="Build payroll",
        body="We need to build payroll.",
        evidenceSequences=[0],
        sourceCandidateIds=["c1"],
    )
    rejected = VerifiedArtifact(
        kind="task",
        title="Build payroll",
        body="We need to build payroll.",
        evidenceSequences=[0],
        verdict=VerifierVerdict.UNSUPPORTED,
        reason="false_reject",
        artifactKey="task:0",
    )
    result = SimpleNamespace(candidates=[candidate], claims=[claim], rejected=[rejected], tasks=[], notes=[])
    score = score_case(case, [], predicted_candidates=[PredictedItem(kind="note", meaning="Build payroll", evidenceSequences=[0])])
    assert classify_first_loss(case, result, score) == "VERIFIER_FALSE_REJECT"


def test_first_loss_artifact_policy_when_note_is_inside_task():
    case = normalize_case(
        {
            "id": "policy",
            "transcript": "[0] Build payroll including PF.",
            "goldTasks": [{"id": "t1", "kind": "task", "meaning": "Build payroll", "evidenceSequences": [0]}],
            "goldNotes": [{"id": "n1", "kind": "note", "meaning": "Payroll includes PF", "evidenceSequences": [0]}],
        }
    )
    predicted = [
        PredictedItem(kind="task", meaning="Build payroll\nImplement payroll with PF.", evidenceSequences=[0]),
    ]
    task = SimpleNamespace(
        title="Build payroll",
        body="Implement payroll with PF.",
        evidence=[SimpleNamespace(sequenceStart=0, sequenceEnd=0)],
        ownerText=None,
        dueDateText=None,
        dueDateResolved=None,
        operation="CREATE",
        artifactId=None,
    )
    result = SimpleNamespace(candidates=[], claims=[], rejected=[], tasks=[task], notes=[])
    score = score_case(case, predicted, predicted_candidates=[])
    assert score.meaningRetentionRecall == 1
    assert classify_first_loss(case, result, score) == "ARTIFACT_POLICY_DIFFERENCE"


def test_first_loss_duplicate_presentation_when_meaning_kept():
    case = normalize_case(
        {
            "id": "dup",
            "transcript": "[0] Mira will file the notes.\n[40] Mira still owns writing those notes.",
            "goldTasks": [{"id": "t1", "kind": "task", "meaning": "Mira will file the notes", "evidenceSequences": [0, 40], "ownerText": "Mira"}],
            "goldNotes": [],
        }
    )
    predicted = [
        PredictedItem(kind="task", meaning="Mira will file the notes", evidenceSequences=[0], ownerText="Mira"),
        PredictedItem(kind="task", meaning="Mira still owns the notes", evidenceSequences=[40], ownerText="Mira"),
    ]
    task = SimpleNamespace(title="Mira will file the notes", body="Mira will file the notes", evidence=[SimpleNamespace(sequenceStart=0, sequenceEnd=0)], ownerText="Mira", dueDateText=None, dueDateResolved=None, operation="CREATE", artifactId=None)
    result = SimpleNamespace(candidates=[], claims=[], rejected=[], tasks=[task, task], notes=[])
    score = score_case(case, predicted)
    assert score.meaningRetentionRecall == 1
    assert classify_first_loss(case, result, score) == "DUPLICATE_PRESENTATION"


def test_first_loss_owner_field_error_when_meaning_kept():
    case = normalize_case(
        {
            "id": "owner",
            "transcript": "[0] Rahul will integrate the API.",
            "goldTasks": [{"id": "t1", "kind": "task", "meaning": "Rahul will integrate the API", "evidenceSequences": [0], "ownerText": "Rahul"}],
            "goldNotes": [],
        }
    )
    predicted = [PredictedItem(kind="task", meaning="Rahul will integrate the API", evidenceSequences=[0], ownerText=None)]
    task = SimpleNamespace(title="Integrate API", body="Rahul will integrate the API", evidence=[SimpleNamespace(sequenceStart=0, sequenceEnd=0)], ownerText=None, dueDateText=None, dueDateResolved=None, operation="CREATE", artifactId=None)
    result = SimpleNamespace(candidates=[], claims=[], rejected=[], tasks=[task], notes=[])
    score = score_case(case, predicted)
    assert score.meaningRetentionRecall == 1
    assert classify_first_loss(case, result, score) == "OWNER_FIELD_ERROR"



def test_eval_runner_does_not_import_publish():
    import inspect
    import services.conversation.eval_real_models as module

    source = inspect.getsource(module)
    assert "publish_outputs" not in source
    assert "dryRun" in source
