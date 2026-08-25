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
