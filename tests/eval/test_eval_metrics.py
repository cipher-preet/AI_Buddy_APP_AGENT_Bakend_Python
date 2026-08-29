from services.conversation.eval_metrics import PredictedItem, score_case, score_corpus
from tests.eval.conversations import BENCHMARK_CASES, REQUIRED_CATEGORIES


def test_perfect_task_and_note_prediction_scores_one():
    case = next(item for item in BENCHMARK_CASES if item["id"] == "technical-stt-drain")
    predicted = [
        PredictedItem(kind="task", meaning="Mira drain gate", evidenceSequences=[2, 3], ownerText="Mira", dueDateText="Thursday evening"),
        PredictedItem(kind="note", meaning="STOP skip", evidenceSequences=[0, 1]),
        PredictedItem(kind="note", meaning="Rahul notes", evidenceSequences=[4]),
    ]
    score = score_case(case, predicted)
    assert score.taskRecall == 1
    assert score.taskPrecision == 1
    assert score.noteRecall == 1
    assert score.ownerAccuracy == 1
    assert score.deadlineAccuracy == 1
    assert score.evidenceAccuracy == 1
    assert score.falseTaskCount == 0
    assert score.duplicateCount == 0


def test_duplicate_predictions_against_one_gold_task_are_counted():
    case = next(item for item in BENCHMARK_CASES if item["id"] == "paraphrase-same-task-two-windows")
    predicted = [
        PredictedItem(kind="task", meaning="Mira files notes", evidenceSequences=[0], ownerText="Mira"),
        PredictedItem(kind="task", meaning="Mira still owns notes", evidenceSequences=[40], ownerText="Mira"),
    ]
    score = score_case(case, predicted)
    assert score.matchedTasks == 1
    assert score.duplicateCount == 1
    assert score.duplicateRate > 0


def test_non_task_span_emitted_as_task_is_a_false_task():
    case = next(item for item in BENCHMARK_CASES if item["id"] == "vague-maybe-later")
    predicted = [
        PredictedItem(kind="task", meaning="Build a nicer settings page", evidenceSequences=[0]),
    ]
    score = score_case(case, predicted)
    assert score.goldTaskCount == 0
    assert score.falseTaskCount == 1
    assert score.falseTaskRate == 1
    assert score.taskPrecision == 0


def test_cross_window_completion_uses_predicted_state():
    case = next(item for item in BENCHMARK_CASES if item["id"] == "completion-later-window")
    predicted = [
        PredictedItem(
            kind="task",
            meaning="Rahul sent the proposal",
            evidenceSequences=[0, 80],
            ownerText="Rahul",
            state="completed",
            artifactId="same-artifact",
        ),
        PredictedItem(kind="note", meaning="legal discount table", evidenceSequences=[1]),
    ]
    score = score_case(case, predicted, expected_artifact_ids={"t1": "same-artifact"})
    assert score.taskRecall == 1
    assert score.crossWindowUpdateAccuracy == 1


def test_corpus_covers_required_conversation_types():
    categories = {item["category"] for item in BENCHMARK_CASES}
    assert REQUIRED_CATEGORIES <= categories
    assert len(BENCHMARK_CASES) >= 20


def test_corpus_mean_metrics_are_defined_for_synthetic_perfect_run():
    scores = []
    for case in BENCHMARK_CASES:
        predicted = []
        for gold in case["goldTasks"]:
            predicted.append(
                PredictedItem(
                    kind="task",
                    meaning=gold["meaning"],
                    evidenceSequences=gold["evidenceSequences"],
                    ownerText=gold.get("ownerText"),
                    dueDateText=gold.get("dueDateText"),
                    state=gold.get("state"),
                )
            )
        for gold in case["goldNotes"]:
            predicted.append(
                PredictedItem(kind="note", meaning=gold["meaning"], evidenceSequences=gold["evidenceSequences"])
            )
        scores.append(score_case(case, predicted))
    corpus = score_corpus(scores)
    assert corpus.taskRecall == 1
    assert corpus.noteRecall == 1
    assert corpus.falseTaskRate == 0
    assert corpus.duplicateRate == 0


def test_optional_gold_note_is_excluded_from_required_recall():
    case = {
        "id": "optional-note",
        "goldTasks": [{"id": "t1", "kind": "task", "meaning": "Create server ID", "evidenceSequences": [1]}],
        "goldNotes": [
            {"id": "n1", "kind": "note", "meaning": "Old keys in use", "evidenceSequences": [2]},
            {
                "id": "n-opt",
                "kind": "note",
                "meaning": "Monday meeting mentioned",
                "evidenceSequences": [3],
                "reviewStatus": "OPTIONAL_VALID",
            },
        ],
    }
    predicted = [
        PredictedItem(kind="task", meaning="Create server ID", evidenceSequences=[1]),
        PredictedItem(kind="note", meaning="Old keys in use", evidenceSequences=[2]),
    ]
    score = score_case(case, predicted)
    assert score.taskRecall == 1
    assert score.noteRecall == 1
    assert score.details["requiredNoteRecall"] == 1


def test_paraphrased_task_matches_without_sequence_overlap():
    case = {
        "id": "paraphrase-task",
        "goldTasks": [
            {"id": "t-billing", "kind": "task", "meaning": "Keep billing retry limit at 3", "evidenceSequences": [10]}
        ],
        "goldNotes": [],
    }
    predicted = [
        PredictedItem(
            kind="task",
            meaning="Preserve the retry limit for billing attempts",
            evidenceSequences=[110],
        )
    ]
    score = score_case(case, predicted)
    assert score.taskRecall == 1
    assert score.matchedTasks == 1


def test_background_false_positive_and_candidate_recall_are_reported():
    case = {
        "id": "background-fp",
        "transcript": "[0] Lenskart margin\n[7] generate candidate link",
        "backgroundSequences": [0],
        "goldTasks": [{"id": "t1", "kind": "task", "meaning": "generate candidate link", "evidenceSequences": [7]}],
        "goldNotes": [],
        "goldCandidates": [{"id": "c1", "meaning": "generate candidate link", "evidenceSequences": [7]}],
    }
    predicted = [
        PredictedItem(kind="task", meaning="generate candidate link", evidenceSequences=[7]),
        PredictedItem(kind="note", meaning="Lenskart margin", evidenceSequences=[0]),
    ]
    score = score_case(
        case,
        predicted,
        predicted_candidates=[PredictedItem(kind="note", meaning="generate candidate link", evidenceSequences=[7])],
    )
    assert score.backgroundFalsePositiveRate == 0.5
    assert score.candidateRecall == 1
    assert score.unsupportedArtifactRate == 0.5


def test_eval_matcher_treats_submit_information_as_enter_details():
    from services.conversation.eval_metrics import semantic_align_score

    score = semantic_align_score(
        "candidate submits information",
        "Candidates should be able to enter their details via the link.",
    )
    assert score >= 0.42
    assert semantic_align_score("generated candidate link", "Send the generated link to the candidate.") >= 0.42
    assert semantic_align_score("candidate receives/uses link", "Send the generated link to the candidate.") >= 0.42
    assert semantic_align_score(
        "reduces manual HR work",
        "Allowing candidates to enter their own details removes the need for HR to manually update information repeatedly.",
    ) >= 0.42


def test_eval_dates_match_relative_and_resolved_iso():
    from services.conversation.eval_metrics import _dates_match

    assert _dates_match("tomorrow", "tomorrow")
    assert _dates_match("tomorrow", "2026-08-29")
    assert _dates_match("Friday", "2026-08-28")


def test_meaning_retention_counts_requirements_inside_task_description():
    from services.conversation.eval_metrics import PredictedItem, score_case

    case = {
        "id": "payroll-pf",
        "transcript": "[0] Build payroll. [1] Payroll includes PF and leave.",
        "goldTasks": [{"id": "t1", "kind": "task", "meaning": "Build payroll", "evidenceSequences": [0]}],
        "goldNotes": [
            {"id": "n1", "kind": "note", "meaning": "Payroll includes PF", "evidenceSequences": [1]},
            {"id": "n2", "kind": "note", "meaning": "Payroll includes leave", "evidenceSequences": [1]},
        ],
    }
    predicted = [
        PredictedItem(
            kind="task",
            meaning="Build payroll\nImplement payroll with PF and leave handling.",
            evidenceSequences=[0, 1],
        )
    ]
    score = score_case(case, predicted)
    assert score.meaningRetentionRecall == 1
    assert score.noteRecall == 0
    assert score.taskRecall == 1


def test_meaning_retention_enter_details_matches_submit_information():
    from services.conversation.eval_metrics import semantic_align_score
    from services.conversation.eval_semantic import meaning_matched

    assert semantic_align_score("enter details", "submit information") >= 0.42
    result = meaning_matched("Candidates enter their details", "Candidate submits information through the generated link")
    assert result["matched"] is True


def test_meaning_retention_embedding_score_can_rescue_paraphrase():
    from services.conversation.eval_metrics import PredictedItem, score_case

    case = {
        "id": "embed",
        "transcript": "[0] Candidate apni details fill karega.",
        "goldTasks": [],
        "goldNotes": [{"id": "n1", "kind": "note", "meaning": "Candidate submits information through the link", "evidenceSequences": [0]}],
    }
    predicted = [PredictedItem(kind="task", meaning="Candidate fills the form via the generated URL.", evidenceSequences=[0])]
    lexical = score_case(case, predicted)
    rescued = score_case(case, predicted, meaning_embedding_scores={"n1": 0.91})
    assert rescued.meaningRetentionRecall == 1
    assert lexical.meaningRetentionRecall is not None


def test_eval_owner_aliases_anshu_devanagari():
    from services.conversation.eval_metrics import _names_match

    assert _names_match("अंशु", "Anshu")
    assert not _names_match("अंशु", None)



