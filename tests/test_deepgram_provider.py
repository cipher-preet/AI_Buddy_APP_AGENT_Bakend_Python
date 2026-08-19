from services.speech.providers import deepgram_provider


SAMPLE_RESPONSE = {
    "metadata": {"request_id": "req-1", "language": "multi"},
    "results": {
        "channels": [
            {
                "alternatives": [
                    {
                        "transcript": "We need to finish the API tomorrow. I will complete testing tonight.",
                        "confidence": 0.91,
                        "words": [
                            {
                                "word": "we",
                                "punctuated_word": "We",
                                "confidence": 0.99,
                                "speaker": 0,
                                "speaker_confidence": 0.88,
                            },
                            {
                                "word": "need",
                                "punctuated_word": "need",
                                "confidence": 0.97,
                                "speaker": 0,
                                "speaker_confidence": 0.88,
                            },
                        ],
                    }
                ]
            }
        ],
        "utterances": [
            {
                "transcript": "We need to finish the API tomorrow.",
                "confidence": 0.94,
                "speaker": 0,
                "words": [
                    {
                        "word": "we",
                        "punctuated_word": "We",
                        "confidence": 0.99,
                        "speaker": 0,
                        "speaker_confidence": 0.88,
                    }
                ],
            },
            {
                "transcript": "I will complete testing tonight.",
                "confidence": 0.93,
                "speaker": 1,
                "words": [
                    {
                        "word": "i",
                        "punctuated_word": "I",
                        "confidence": 0.96,
                        "speaker": 1,
                        "speaker_confidence": 0.81,
                    }
                ],
            },
        ],
    },
}


def test_listen_params_keep_nova3_meeting_options_without_keyterms():
    params = deepgram_provider.build_deepgram_listen_params("multi")

    assert params["model"] == "nova-3"
    assert params["language"] == "multi"
    assert params["smart_format"] is True
    assert params["detect_language"] is False
    assert params["diarize_model"] == "latest"
    assert params["utterances"] is True
    assert params["utt_split"] == 0.8
    assert "keyterm" not in params
    assert "diarize" not in params


def test_listen_params_include_keyterms_only_when_present():
    params = deepgram_provider.build_deepgram_listen_params(
        "multi",
        ["API", "  Buddy  ", "API"],
    )

    assert params["keyterm"] == ["API", "Buddy"]


def test_keyterms_are_resolved_from_existing_space_context():
    keyterms = deepgram_provider.resolve_deepgram_keyterms(
        None,
        {"terminology": ["Qdrant", "semantic window"]},
    )

    assert keyterms == ["Qdrant", "semantic window"]
    assert deepgram_provider.resolve_deepgram_keyterms(None, {}) == []


def test_reconstruct_transcript_from_utterances_and_speakers():
    transcript = deepgram_provider.reconstruct_speaker_transcript(SAMPLE_RESPONSE)

    assert transcript == (
        "[Speaker 0] We need to finish the API tomorrow.\n"
        "[Speaker 1] I will complete testing tonight."
    )


def test_reconstruct_transcript_from_words_when_utterances_missing():
    response = {
        "results": {
            "channels": [
                {
                    "alternatives": [
                        {
                            "transcript": "hello there thanks",
                            "words": [
                                {
                                    "word": "hello",
                                    "punctuated_word": "Hello",
                                    "speaker": 0,
                                    "confidence": 0.9,
                                    "speaker_confidence": 0.8,
                                },
                                {
                                    "word": "there",
                                    "punctuated_word": "there.",
                                    "speaker": 0,
                                    "confidence": 0.91,
                                    "speaker_confidence": 0.8,
                                },
                                {
                                    "word": "thanks",
                                    "punctuated_word": "Thanks.",
                                    "speaker": 1,
                                    "confidence": 0.92,
                                    "speaker_confidence": 0.7,
                                },
                            ],
                        }
                    ]
                }
            ]
        }
    }

    assert deepgram_provider.reconstruct_speaker_transcript(response) == (
        "[Speaker 0] Hello there.\n[Speaker 1] Thanks."
    )


def test_falls_back_to_plain_transcript_when_speakers_are_missing():
    result = deepgram_provider._finalize_transcription_result(
        {
            "results": {
                "channels": [
                    {
                        "alternatives": [
                            {"transcript": "plain transcript", "words": [{"word": "plain", "confidence": 0.99}]}
                        ]
                    }
                ]
            }
        },
        "multi",
    )

    assert result["transcript"] == "plain transcript"
    assert result["is_empty_transcript"] is False
    assert result["is_uncertain_transcript"] is False


def test_low_confidence_chunks_are_marked_uncertain_but_kept():
    result = deepgram_provider._finalize_transcription_result(
        {
            "results": {
                "channels": [
                    {
                        "alternatives": [
                            {
                                "transcript": "maybe this is garbled",
                                "words": [
                                    {"word": "maybe", "confidence": 0.21, "speaker": 0, "speaker_confidence": 0.4},
                                    {"word": "this", "confidence": 0.22, "speaker": 0, "speaker_confidence": 0.4},
                                ],
                            }
                        ]
                    }
                ],
                "utterances": [
                    {
                        "transcript": "maybe this is garbled",
                        "confidence": 0.22,
                        "speaker": 0,
                        "words": [
                            {"word": "maybe", "confidence": 0.21, "speaker": 0, "speaker_confidence": 0.4},
                            {"word": "this", "confidence": 0.22, "speaker": 0, "speaker_confidence": 0.4},
                        ],
                    }
                ],
            }
        },
        "multi",
    )

    assert result["transcript"] == "[Speaker 0] maybe this is garbled"
    assert result["is_empty_transcript"] is False
    assert result["is_uncertain_transcript"] is True
    assert result["transcript_quality"]["uncertain"] is True
    assert result["transcript_quality"]["min_confidence"] == 0.21


def test_single_low_confidence_word_does_not_mark_chunk_uncertain():
    words = [{"word": f"word{index}", "confidence": 0.95, "speaker": 0} for index in range(9)]
    words.append({"word": "garbled", "confidence": 0.21, "speaker": 0})
    result = deepgram_provider._finalize_transcription_result(
        {
            "results": {
                "channels": [{"alternatives": [{"transcript": "ok", "words": words}]}],
                "utterances": [{"transcript": "ok", "speaker": 0, "confidence": 0.94, "words": words}],
            }
        },
        "multi",
    )

    assert result["is_empty_transcript"] is False
    assert result["is_uncertain_transcript"] is False


def test_malformed_deepgram_payload_does_not_crash():
    result = deepgram_provider._finalize_transcription_result({"results": ["bad"]}, "multi")

    assert result["transcript"] == ""
    assert result["is_empty_transcript"] is True
    assert result["is_uncertain_transcript"] is False


def test_invalid_keyterm_types_and_long_phrases_are_ignored():
    keyterms = deepgram_provider.resolve_deepgram_keyterms(
        [{"not": "a term"}, True, None, "API"],
        {"terminology": "this is a whole sentence that should not become a keyterm"},
    )

    assert keyterms == ["API"]


def test_reconstruct_groups_int_and_float_speaker_ids():
    transcript = deepgram_provider.reconstruct_speaker_transcript(
        {
            "results": {
                "channels": [
                    {
                        "alternatives": [
                            {
                                "words": [
                                    {"punctuated_word": "Hello", "speaker": 0},
                                    {"punctuated_word": "there.", "speaker": 0.0},
                                    {"punctuated_word": "Thanks.", "speaker": 1},
                                ]
                            }
                        ]
                    }
                ]
            }
        }
    )

    assert transcript == "[Speaker 0] Hello there.\n[Speaker 1] Thanks."
