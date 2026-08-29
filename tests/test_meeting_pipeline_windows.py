from services.conversation.meeting_pipeline.schemas import TranscriptTurn
from services.conversation.meeting_pipeline.windows import (
    build_extraction_windows,
    covered_sequence_ids,
    format_window_line,
    turns_from_chunks,
    useful_token_count,
    window_coverage_error,
)
from services.conversation.models import TranscriptChunkDocument


def _chunk(sequence: int, text: str, speaker: int | None = None) -> TranscriptChunkDocument:
    raw = f"Speaker {speaker}: {text}" if speaker is not None and text else text
    return TranscriptChunkDocument(
        conversationId="conv",
        userId="user",
        spaceId="space",
        chunkId=f"c{sequence}",
        sequenceNumber=sequence,
        rawText=raw,
        normalizedText=raw,
    )


def _turns(*rows: tuple[int, str]) -> list[TranscriptTurn]:
    return [TranscriptTurn(sequence_id=sequence, speaker="Speaker 1", raw_text=text) for sequence, text in rows]


def test_short_transcript_is_one_window():
    turns = _turns((0, "We will ship the drain notes tomorrow."))
    windows = build_extraction_windows(turns, conversation_id="short", target_tokens=8000, max_tokens=12000, overlap_ratio=0.12)
    assert len(windows) == 1
    assert windows[0].owned_sequence_ids == [0]
    assert windows[0].overlap_sequence_ids == []


def test_long_transcript_makes_multiple_overlapping_windows():
    turns = _turns(*[(index, f"Useful speech block {index} " + ("word " * 40)) for index in range(40)])
    windows = build_extraction_windows(turns, conversation_id="long", target_tokens=80, max_tokens=120, overlap_ratio=0.15)
    assert len(windows) >= 2
    for left, right in zip(windows, windows[1:]):
        assert set(left.sequence_ids) & set(right.sequence_ids), (left.sequence_ids, right.sequence_ids)
    owned = covered_sequence_ids(windows)
    assert owned == [turn.sequence_id for turn in turns]


def test_empty_sequences_do_not_consume_token_budget():
    useful = TranscriptTurn(sequence_id=1, speaker="Speaker 1", raw_text="Keep the drain notes.")
    empty = TranscriptTurn(sequence_id=2, speaker="Speaker 1", raw_text="   ")
    more = TranscriptTurn(sequence_id=3, speaker="Speaker 0", raw_text="Ship the ticket today.")
    windows = build_extraction_windows([useful, empty, more], conversation_id="empty", target_tokens=8000)
    assert covered_sequence_ids(windows) == [1, 3]
    assert 2 not in windows[0].sequence_ids
    assert windows[0].token_count == useful_token_count(useful) + useful_token_count(more)


def test_ordering_and_speaker_boundaries_are_preserved():
    turns = _turns(
        (4, "Speaker four starts the topic."),
        (5, "Speaker five continues."),
        (6, "Speaker six closes."),
    )
    windows = build_extraction_windows(turns, conversation_id="order")
    assert [turn.sequence_id for turn in turns] == windows[0].sequence_ids
    assert format_window_line(turns[0]).startswith("[4][Speaker 1]")
    assert windows[0].text.splitlines()[0].startswith("[4]")


def test_very_large_single_sequence_is_handled():
    huge = TranscriptTurn(sequence_id=9, speaker="Speaker 1", raw_text=" ".join(f"token{index}" for index in range(5000)))
    windows = build_extraction_windows([huge], conversation_id="huge", target_tokens=50, max_tokens=80, overlap_ratio=0.1)
    assert windows
    assert all(9 in window.sequence_ids for window in windows)
    assert covered_sequence_ids(windows) == [9]


def test_turns_from_chunks_skip_empty_text_but_keep_order():
    chunks = [
        _chunk(2, "Second useful line", speaker=1),
        _chunk(0, "First useful line", speaker=0),
        _chunk(1, "   "),
    ]
    turns = turns_from_chunks(chunks)
    assert [turn.sequence_id for turn in turns] == [0, 1, 2]
    windows = build_extraction_windows(turns)
    assert covered_sequence_ids(windows) == [0, 2]


def test_union_of_window_sequences_equals_all_useful_ids_including_last():
    turns = _turns(*[(index, f"Useful speech block {index} " + ("word " * 25)) for index in range(30)])
    windows = build_extraction_windows(turns, conversation_id="cover", target_tokens=80, max_tokens=120, overlap_ratio=0.12)
    useful = [turn.sequence_id for turn in turns]
    union = {sequence for window in windows for sequence in window.sequence_ids}
    assert union == set(useful)
    assert useful[-1] in windows[-1].sequence_ids
    assert window_coverage_error(windows, useful) is None


def test_long_meeting_fixture_puts_final_sequence_in_last_window_text():
    from services.conversation.eval_real_models import build_long_meeting, chunks_from_transcript
    from services.conversation.meeting_pipeline.windows import turns_from_chunks

    case = build_long_meeting()
    turns = turns_from_chunks(chunks_from_transcript(case["id"], case["transcript"]))
    windows = build_extraction_windows(turns, conversation_id=case["id"], target_tokens=5000, max_tokens=7000, overlap_ratio=0.12)
    assert len(windows) >= 2
    assert 59 in windows[-1].sequence_ids
    assert "Neha" in windows[-1].text
    assert "END:" in windows[-1].text
    assert window_coverage_error(windows, [turn.sequence_id for turn in turns if (turn.raw_text or "").strip()]) is None


def test_tail_position_commitments_remain_in_window_input():
    from services.conversation.eval_real_models import chunks_from_transcript
    from tests.eval.meeting_gold import tail_position_cases

    for case in tail_position_cases():
        sequence = case["goldTasks"][0]["evidenceSequences"][0]
        turns = turns_from_chunks(chunks_from_transcript(case["id"], case["transcript"]))
        windows = build_extraction_windows(turns, conversation_id=case["id"], target_tokens=5000, max_tokens=7000)
        assert windows
        union = {item for window in windows for item in window.sequence_ids}
        assert sequence in union
        assert sequence in windows[-1].sequence_ids or sequence in windows[0].sequence_ids
        spoken = case["transcript"].splitlines()[sequence]
        assert any(spoken.split("]", 1)[-1].strip()[:20] in window.text or str(sequence) in window.text for window in windows)


def test_overlap_is_input_context_not_owned_evidence():
    turns = _turns(*[(index, "content " * 20) for index in range(12)])
    windows = build_extraction_windows(turns, conversation_id="overlap", target_tokens=40, max_tokens=70, overlap_ratio=0.25)
    assert len(windows) >= 2
    overlap = set(windows[1].overlap_sequence_ids)
    owned = set(windows[1].owned_sequence_ids)
    assert overlap
    assert overlap.isdisjoint(owned)
    assert overlap <= set(windows[0].sequence_ids)
