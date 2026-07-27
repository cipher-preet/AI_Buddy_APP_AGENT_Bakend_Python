import asyncio
import os

os.environ.setdefault("REDIS_URL", "redis://localhost:6379")
os.environ.setdefault("SARVAM_API_KEY", "test")
os.environ.setdefault("OPENAI_API_KEY", "test")
os.environ.setdefault("QDRANT_API_KEY", "")
os.environ.setdefault("QDRANT_URL", "http://localhost:6333")
os.environ.setdefault("QDRANT_COLLECTION", "test")
os.environ.setdefault("EMBEDDING_MODEL", "text-embedding-3-small")
os.environ.setdefault("VECTOR_SIZE", "1536")
os.environ.setdefault("MONGO_URL", "mongodb://localhost:27017")
os.environ.setdefault("MONGO_DB_NAME", "test")

from apps.agent_runtime.rag.vectorstores.qdrant_store import is_eligible_speech_payload
from apps.agent_runtime.llms.prompts.transcript_analysis_prompt import (
    TRANSCRIPT_ANALYSIS_COVERAGE_REPAIR_SYSTEM_PROMPT,
    TRANSCRIPT_ANALYSIS_TASK_REPAIR_SYSTEM_PROMPT,
    TRANSCRIPT_ANALYSIS_REPAIR_SYSTEM_PROMPT,
    TRANSCRIPT_ANALYSIS_SYSTEM_PROMPT,
)
from apps.agent_runtime.services.transcript_analysis_service import (
    MemoryVector,
    _build_analysis_window,
    _ensure_meaningful_summary,
    _estimate_future_work_count,
    _needs_detail_repair,
    _normalize_generated_operations,
    _published_point_ids_for_output,
    _promote_summary_to_note_if_needed,
    _repair_empty_analysis_output_if_needed,
    _repair_missing_tasks_if_needed,
    normalize_text,
    task_fingerprint,
    window_id_for_chunks,
)
import apps.agent_runtime.services.transcript_analysis_service as transcript_service
from packages.schemas.transcript_analysis_schema import NoteOperation, TranscriptAnalysisOutput


def test_split_sentence_window_id_is_one_batch():
    chunk_ids = ["chunk-1", "chunk-2", "chunk-3"]

    assert window_id_for_chunks("user", "space", chunk_ids) == window_id_for_chunks(
        "user", "space", chunk_ids
    )
    assert window_id_for_chunks("user", "space", chunk_ids) != window_id_for_chunks(
        "user", "other-space", chunk_ids
    )


def test_analysis_window_accepts_ids_with_accidental_whitespace():
    user_id = " 6a21be267be2c45e7960c4ab"
    space_id = " 6a21be267be2c45e7960c4ac "

    window = _build_analysis_window(
        user_id=user_id,
        space_id=space_id,
        chunks=[
            {
                "point_id": "point-1",
                "chunkId": "chunk-1",
                "text": "Remember to create a note from this conversation.",
                "request_id": "request-1",
                "createdAt": "2026-07-27T10:00:00+00:00",
                "chunkIndex": 0,
            }
        ],
    )

    assert str(window["user_id"]) == user_id.strip()
    assert str(window["space_id"]) == space_id.strip()


def test_task_fingerprint_dedupes_repeated_task():
    first = task_fingerprint(
        user_id="user",
        space_id="space",
        title="Call Rahul about pending payment",
        description="Ask Rahul tomorrow morning.",
        due_at="2026-07-28T09:00:00+05:30",
    )
    second = task_fingerprint(
        user_id="user",
        space_id="space",
        title="call rahul about pending payment",
        description="Ask Rahul tomorrow morning!",
        due_at="2026-07-28T18:00:00+05:30",
    )

    assert first == second


def test_damaged_transcript_is_ineligible():
    assert not is_eligible_speech_payload(
        {
            "sourceType": "speech",
            "isDamaged": True,
            "isPublish": False,
            "isUseful": True,
            "chunkStatus": "active",
        },
        require_unpublished=True,
    )


def test_another_space_changes_fingerprint():
    base = {
        "user_id": "user",
        "title": "Call Rahul",
        "description": "Pending payment",
        "due_at": None,
    }

    assert task_fingerprint(space_id="space-a", **base) != task_fingerprint(
        space_id="space-b", **base
    )


def test_multilingual_text_normalizes_without_being_empty():
    marathi_text = "\u0930\u093e\u0939\u0941\u0932\u0932\u093e \u0909\u0926\u094d\u092f\u093e \u0915\u0949\u0932 \u0915\u0930\u093e\u092f\u091a\u093e \u0906\u0939\u0947"
    assert normalize_text(marathi_text)


def test_older_reference_schema_can_resolve_him_to_rahul():
    output = TranscriptAnalysisOutput.model_validate(
        {
            "is_complete_enough": True,
            "requires_more_context": False,
            "context_resolution": {
                "resolved_entities": {"him": "Rahul"},
                "confidence": 0.94,
            },
            "task_operations": [
                {
                    "operation": "create",
                    "title": "Call Rahul",
                    "description": "Call Rahul tomorrow.",
                    "due_at": "2026-07-28T09:00:00+05:30",
                    "status": "open",
                    "confidence": 0.9,
                    "source_chunk_ids": ["new", "old"],
                }
            ],
            "note_operations": [],
            "summary_update": {"should_update": False, "updated_summary": ""},
        }
    )

    assert output.context_resolution.resolved_entities["him"] == "Rahul"
    assert output.task_operations[0].source_chunk_ids == ["new", "old"]


def test_task_completed_operation_requires_existing_id_shape():
    output = TranscriptAnalysisOutput.model_validate(
        {
            "is_complete_enough": True,
            "requires_more_context": False,
            "context_resolution": {"resolved_entities": {}, "confidence": 1},
            "task_operations": [
                {
                    "operation": "complete",
                    "existing_task_id": "64f000000000000000000001",
                    "title": "Send invoice to client",
                    "description": "",
                    "due_at": None,
                    "status": "completed",
                    "confidence": 0.95,
                    "source_chunk_ids": ["chunk"],
                }
            ],
            "note_operations": [],
            "summary_update": {"should_update": False, "updated_summary": ""},
        }
    )

    assert output.task_operations[0].operation == "complete"
    assert output.task_operations[0].existing_task_id == "64f000000000000000000001"


def test_incomplete_speech_can_require_more_context_without_operations():
    output = TranscriptAnalysisOutput.model_validate(
        {
            "is_complete_enough": False,
            "requires_more_context": True,
            "context_resolution": {"resolved_entities": {}, "confidence": 0.2},
            "task_operations": [],
            "note_operations": [],
            "summary_update": {"should_update": False, "updated_summary": ""},
        }
    )

    assert output.requires_more_context
    assert not output.task_operations


def test_note_only_output_has_no_task_operations():
    output = TranscriptAnalysisOutput.model_validate(
        {
            "is_complete_enough": True,
            "requires_more_context": False,
            "context_resolution": {"resolved_entities": {}, "confidence": 0.9},
            "task_operations": [],
            "note_operations": [
                {
                    "operation": "create",
                    "title": "Client report preference",
                    "content": "The client prefers weekly reports in PDF format.",
                    "confidence": 0.9,
                    "source_chunk_ids": ["chunk"],
                }
            ],
            "summary_update": {"should_update": True, "updated_summary": "Client prefers weekly PDF reports."},
        }
    )

    assert not output.task_operations
    assert output.note_operations[0].title == "Client report preference"


def test_duplicate_redis_job_uses_same_window_id():
    chunk_ids = ["chunk-a", "chunk-b"]

    first = window_id_for_chunks("user", "space", chunk_ids)
    second = window_id_for_chunks("user", "space", list(chunk_ids))

    assert first == second


def test_transcript_prompt_requires_english_synthesized_memory():
    assert "clear English" in TRANSCRIPT_ANALYSIS_SYSTEM_PROMPT
    assert "do not paste the raw transcript" in TRANSCRIPT_ANALYSIS_SYSTEM_PROMPT
    assert "English synthesis" in TRANSCRIPT_ANALYSIS_SYSTEM_PROMPT
    assert "Do not paste the raw transcript" in TRANSCRIPT_ANALYSIS_REPAIR_SYSTEM_PROMPT
    assert "any language" in TRANSCRIPT_ANALYSIS_TASK_REPAIR_SYSTEM_PROMPT
    assert "Do not require the word \"task\"" in TRANSCRIPT_ANALYSIS_TASK_REPAIR_SYSTEM_PROMPT
    assert "Hindi/Hinglish" not in TRANSCRIPT_ANALYSIS_TASK_REPAIR_SYSTEM_PROMPT
    assert "vague umbrella outputs" in TRANSCRIPT_ANALYSIS_COVERAGE_REPAIR_SYSTEM_PROMPT
    assert "completed work versus remaining work" in TRANSCRIPT_ANALYSIS_SYSTEM_PROMPT


def test_low_detail_generic_output_triggers_detail_repair():
    hindi_project_updates = """
    तो अभी प्रॉपर्टी लिस्टिंग वाला काम लगभग पूरा हो गया है। अब अगला काम सर्च को थोड़ा तेज़ करना है क्योंकि ज़्यादा डेटा आने पर रिज़ल्ट लेट आ रहे हैं। उसके बाद हम इसे फ्रंटएंड से जोड़कर पूरा फ्लो टेस्ट करेंगे।
    मैंने लॉगिन और रजिस्ट्रेशन का काम पूरा कर लिया है। अब मैं पासवर्ड रीसेट वाला फीचर शुरू करूँगा। उसके बाद अगर समय मिला तो गूगल लॉगिन की इंटीग्रेशन भी पूरी कर दूँगा।
    कल क्लाइंट से बात हुई थी। उन्होंने कहा कि प्रॉपर्टी की फोटो थोड़ी जल्दी लोड होनी चाहिए और मोबाइल पर गैलरी का डिज़ाइन भी बेहतर होना चाहिए।
    लीड मैनेजमेंट का काम लगभग तैयार है। अब बस नोटिफिकेशन जोड़ने हैं ताकि जैसे ही कोई नई पूछताछ आए, एजेंट को तुरंत जानकारी मिल जाए।
    आज का पहला काम API की परफॉर्मेंस देखना है। अगर रिस्पॉन्स टाइम सही रहा तो शाम तक हम पूरा मॉड्यूल स्टेजिंग सर्वर पर डिप्लॉय कर देंगे।
    पेमेंट गेटवे की इंटीग्रेशन हो गई है, लेकिन एक दिक्कत अभी भी है। अगर पेमेंट फेल हो जाती है तो यूज़र को सही मैसेज नहीं दिख रहा। पहले इसे ठीक करते हैं, फिर दोबारा पूरा फ्लो चेक करेंगे।
    अभी हमारे पास तीन मुख्य काम बचे हैं। एक टीम नोटिफिकेशन सर्विस पूरी करेगी, दूसरी CRM इंटीग्रेशन पर काम करेगी और मैं रिपोर्टिंग डैशबोर्ड देख लेता हूँ।
    ठीक है, मैं अभी सर्च वाले बग को ठीक करता हूँ। तुम प्रॉपर्टी डिटेल पेज पूरा कर लो। उसके बाद दोनों चीज़ें मर्ज करके एक बार पूरी एप्लिकेशन टेस्ट कर लेते हैं।
    """
    output = TranscriptAnalysisOutput.model_validate(
        {
            "is_complete_enough": True,
            "requires_more_context": False,
            "context_resolution": {"resolved_entities": {}, "confidence": 0.9},
            "task_operations": [
                {
                    "operation": "create",
                    "title": "Finalize the testing for the application",
                    "description": "Ensure all features are functioning as expected.",
                    "due_at": None,
                    "status": "open",
                    "confidence": 0.9,
                    "source_chunk_ids": ["chunk-1"],
                },
                {
                    "operation": "create",
                    "title": "Start integrating the new features",
                    "description": "Start integrating the new features into the existing application framework.",
                    "due_at": None,
                    "status": "open",
                    "confidence": 0.9,
                    "source_chunk_ids": ["chunk-1"],
                },
            ],
            "note_operations": [
                {
                    "operation": "create",
                    "title": "Project updates",
                    "content": "The project is progressing with the integration of new features and ongoing testing.",
                    "confidence": 0.9,
                    "source_chunk_ids": ["chunk-1"],
                }
            ],
            "summary_update": {"should_update": True, "updated_summary": "Project work is progressing."},
        }
    )

    assert _estimate_future_work_count(hindi_project_updates) >= 8
    assert _needs_detail_repair(
        window={"combined_text": hindi_project_updates, "chunk_ids": ["chunk-1"]},
        output=output,
    )


def test_detailed_specific_output_does_not_trigger_detail_repair():
    text = "अब सर्च को तेज़ करना है। उसके बाद फ्रंटएंड से जोड़कर पूरा फ्लो टेस्ट करेंगे।"
    output = TranscriptAnalysisOutput.model_validate(
        {
            "is_complete_enough": True,
            "requires_more_context": False,
            "context_resolution": {"resolved_entities": {}, "confidence": 0.9},
            "task_operations": [
                {
                    "operation": "create",
                    "title": "Improve property search performance",
                    "description": "Optimize search so results stay fast when larger data volumes are added.",
                    "due_at": None,
                    "status": "open",
                    "confidence": 0.9,
                    "source_chunk_ids": ["chunk-1"],
                },
                {
                    "operation": "create",
                    "title": "Connect property search to the frontend",
                    "description": "Integrate the optimized search with the frontend before full-flow testing.",
                    "due_at": None,
                    "status": "open",
                    "confidence": 0.9,
                    "source_chunk_ids": ["chunk-1"],
                },
                {
                    "operation": "create",
                    "title": "Test the complete property search flow",
                    "description": "Run the full property search flow after frontend integration is complete.",
                    "due_at": None,
                    "status": "open",
                    "confidence": 0.9,
                    "source_chunk_ids": ["chunk-1"],
                },
            ],
            "note_operations": [],
            "summary_update": {"should_update": True, "updated_summary": "Property search needs performance work, frontend integration, and full-flow testing."},
        }
    )

    assert not _needs_detail_repair(
        window={"combined_text": text, "chunk_ids": ["chunk-1"]},
        output=output,
    )


def test_empty_analysis_uses_repaired_english_note_and_summary(monkeypatch):
    initial = TranscriptAnalysisOutput.model_validate(
        {
            "is_complete_enough": True,
            "requires_more_context": False,
            "context_resolution": {"resolved_entities": {}, "confidence": 0.8},
            "task_operations": [],
            "note_operations": [],
            "summary_update": {"should_update": False, "updated_summary": ""},
        }
    )
    repaired = TranscriptAnalysisOutput.model_validate(
        {
            "is_complete_enough": True,
            "requires_more_context": False,
            "context_resolution": {"resolved_entities": {}, "confidence": 0.9},
            "task_operations": [],
            "note_operations": [
                {
                    "operation": "create",
                    "title": "Payment follow-up context",
                    "content": "Topic: Payment follow-up\nKey details:\n- Rahul is the person connected to the pending payment.",
                    "confidence": 0.9,
                    "source_chunk_ids": ["chunk"],
                }
            ],
            "summary_update": {
                "should_update": True,
                "updated_summary": "Rahul is connected to an active payment follow-up.",
            },
        }
    )

    async def fake_parse_chat_completion(*args, **kwargs):
        return repaired

    monkeypatch.setattr(
        transcript_service,
        "parse_chat_completion",
        fake_parse_chat_completion,
    )

    result = asyncio.run(
        _repair_empty_analysis_output_if_needed(
            window={
                "window_id": "window",
                "combined_text": "Rahul payment pending follow up tomorrow",
                "chunk_ids": ["chunk"],
            },
            context_package={
                "user_id": "user",
                "space_id": "space",
                "current_datetime": "2026-07-27T10:00:00+05:30",
                "timezone": "Asia/Calcutta",
                "analysis_window": {
                    "window_id": "window",
                    "combined_text": "Rahul payment pending follow up tomorrow",
                    "chunk_ids": ["chunk"],
                },
                "recent_transcripts": [],
                "relevant_older_context": [],
                "running_summary": "",
                "existing_tasks": [],
                "existing_notes": [],
            },
            output=initial,
        )
    )

    assert result.note_operations[0].title == "Payment follow-up context"
    assert "Topic:" in result.note_operations[0].content
    assert result.summary_update.updated_summary != result.note_operations[0].content


def test_missing_tasks_repair_adds_general_future_work_tasks(monkeypatch):
    initial = TranscriptAnalysisOutput.model_validate(
        {
            "is_complete_enough": True,
            "requires_more_context": False,
            "context_resolution": {"resolved_entities": {}, "confidence": 0.8},
            "task_operations": [],
            "note_operations": [
                {
                    "operation": "create",
                    "title": "Property search work status",
                    "content": "Property search is complete and needs follow-up work.",
                    "confidence": 0.8,
                    "source_chunk_ids": ["chunk"],
                }
            ],
            "summary_update": {
                "should_update": True,
                "updated_summary": "Property search is complete, with future performance and QA work discussed.",
            },
        }
    )
    repaired = TranscriptAnalysisOutput.model_validate(
        {
            "is_complete_enough": True,
            "requires_more_context": False,
            "context_resolution": {"resolved_entities": {}, "confidence": 0.9},
            "task_operations": [
                {
                    "operation": "create",
                    "title": "Improve property search performance",
                    "description": "Optimize property search because response time is slow with larger data volumes.",
                    "due_at": None,
                    "status": "open",
                    "confidence": 0.9,
                    "source_chunk_ids": ["chunk"],
                },
                {
                    "operation": "create",
                    "title": "Send property search flow to QA",
                    "description": "Hand the flow to QA so the complete flow can be tested.",
                    "due_at": None,
                    "status": "open",
                    "confidence": 0.9,
                    "source_chunk_ids": ["chunk"],
                },
            ],
            "note_operations": [],
            "summary_update": {
                "should_update": True,
                "updated_summary": "Property search is complete; performance optimization and QA handoff are open next steps.",
            },
        }
    )

    async def fake_parse_chat_completion(*args, **kwargs):
        return repaired

    monkeypatch.setattr(
        transcript_service,
        "parse_chat_completion",
        fake_parse_chat_completion,
    )

    result = asyncio.run(
        _repair_missing_tasks_if_needed(
            window={
                "window_id": "window",
                "combined_text": "Property search work is complete. The next work is performance improvement, then QA handoff.",
                "chunk_ids": ["chunk"],
            },
            context_package={
                "user_id": "user",
                "space_id": "space",
                "current_datetime": "2026-07-27T10:00:00+05:30",
                "timezone": "Asia/Calcutta",
                "analysis_window": {
                    "window_id": "window",
                    "combined_text": "Property search work is complete. The next work is performance improvement, then QA handoff.",
                    "chunk_ids": ["chunk"],
                },
                "recent_transcripts": [],
                "relevant_older_context": [],
                "running_summary": "",
                "existing_tasks": [],
                "existing_notes": [],
            },
            output=initial,
        )
    )

    assert [task.title for task in result.task_operations] == [
        "Improve property search performance",
        "Send property search flow to QA",
    ]
    assert result.note_operations[0].title == "Property search work status"


def test_generated_create_tasks_get_required_persistence_metadata():
    output = TranscriptAnalysisOutput.model_validate(
        {
            "is_complete_enough": True,
            "requires_more_context": False,
            "context_resolution": {"resolved_entities": {}, "confidence": 0.9},
            "task_operations": [
                {
                    "operation": "update",
                    "existing_task_id": None,
                    "title": "Improve the next workflow step",
                    "description": "The conversation contains a clear next action.",
                    "due_at": None,
                    "status": "open",
                    "confidence": 0.0,
                    "source_chunk_ids": [],
                }
            ],
            "note_operations": [],
            "summary_update": {"should_update": False, "updated_summary": ""},
        }
    )

    result = _normalize_generated_operations(
        window={"chunk_ids": ["chunk-1"]},
        output=output,
    )

    assert result.task_operations[0].operation == "create"
    assert result.task_operations[0].confidence == 0.75
    assert result.task_operations[0].source_chunk_ids == ["chunk-1"]


def test_meaningful_summary_is_created_from_structured_operations():
    output = TranscriptAnalysisOutput.model_validate(
        {
            "is_complete_enough": True,
            "requires_more_context": False,
            "context_resolution": {"resolved_entities": {}, "confidence": 0.9},
            "task_operations": [
                {
                    "operation": "create",
                    "title": "Call Rahul about the payment",
                    "description": "Follow up with Rahul about the pending payment.",
                    "due_at": "2026-07-28T09:00:00+05:30",
                    "status": "open",
                    "confidence": 0.9,
                    "source_chunk_ids": ["chunk"],
                }
            ],
            "note_operations": [],
            "summary_update": {"should_update": False, "updated_summary": ""},
        }
    )

    result = _ensure_meaningful_summary(output)

    assert result.summary_update.should_update
    assert "Task: Call Rahul about the payment" in result.summary_update.updated_summary
    assert "Due: 2026-07-28T09:00:00+05:30" in result.summary_update.updated_summary


def test_summary_only_analysis_is_promoted_to_note():
    output = TranscriptAnalysisOutput.model_validate(
        {
            "is_complete_enough": True,
            "requires_more_context": False,
            "context_resolution": {"resolved_entities": {}, "confidence": 0.9},
            "task_operations": [],
            "note_operations": [],
            "summary_update": {
                "should_update": True,
                "updated_summary": "The user wants Mate to analyze conversations and save appropriate tasks and durable notes.",
            },
        }
    )

    result = _promote_summary_to_note_if_needed(
        window={"chunk_ids": ["chunk-1"]},
        output=output,
    )

    assert len(result.note_operations) == 1
    assert result.note_operations[0].operation == "create"
    assert result.note_operations[0].source_chunk_ids == ["chunk-1"]
    assert "Mate" in result.note_operations[0].content


def test_summary_promotion_keeps_existing_note_operations_unchanged():
    note = NoteOperation(
        operation="create",
        title="Assistant memory behavior",
        content="Mate should save durable context as notes.",
        confidence=0.9,
        source_chunk_ids=["chunk-1"],
    )
    output = TranscriptAnalysisOutput.model_validate(
        {
            "is_complete_enough": True,
            "requires_more_context": False,
            "context_resolution": {"resolved_entities": {}, "confidence": 0.9},
            "task_operations": [],
            "note_operations": [note.model_dump()],
            "summary_update": {
                "should_update": True,
                "updated_summary": "Mate should save durable context as notes.",
            },
        }
    )

    result = _promote_summary_to_note_if_needed(
        window={"chunk_ids": ["chunk-1"]},
        output=output,
    )

    assert len(result.note_operations) == 1
    assert result.note_operations[0].title == "Assistant memory behavior"


def test_no_change_operations_do_not_block_summary_promotion():
    output = TranscriptAnalysisOutput.model_validate(
        {
            "is_complete_enough": True,
            "requires_more_context": False,
            "context_resolution": {"resolved_entities": {}, "confidence": 0.9},
            "task_operations": [],
            "note_operations": [
                {
                    "operation": "no_change",
                    "title": "",
                    "content": "",
                    "confidence": 0.9,
                    "source_chunk_ids": [],
                }
            ],
            "summary_update": {
                "should_update": True,
                "updated_summary": "The user clarified that Mate should behave like a human assistant with space and conversation context.",
            },
        }
    )

    result = _promote_summary_to_note_if_needed(
        window={"chunk_ids": ["chunk-1"]},
        output=output,
    )

    created_notes = [
        operation
        for operation in result.note_operations
        if operation.operation == "create"
    ]
    assert len(created_notes) == 1
    assert "human assistant" in created_notes[0].content


def test_task_only_analysis_is_promoted_to_note_when_summary_exists():
    output = TranscriptAnalysisOutput.model_validate(
        {
            "is_complete_enough": True,
            "requires_more_context": False,
            "context_resolution": {"resolved_entities": {}, "confidence": 0.9},
            "task_operations": [
                {
                    "operation": "create",
                    "title": "Review Mate memory note creation",
                    "description": "Fix the issue where tasks are created but ai_memory_notes stays empty.",
                    "due_at": None,
                    "status": "open",
                    "confidence": 0.9,
                    "source_chunk_ids": ["chunk-1"],
                }
            ],
            "note_operations": [],
            "summary_update": {
                "should_update": True,
                "updated_summary": "The user reported that task creation works, but ai_memory_notes is not receiving the durable conversation note.",
            },
        }
    )

    output = _ensure_meaningful_summary(output)
    result = _promote_summary_to_note_if_needed(
        window={"chunk_ids": ["chunk-1"]},
        output=output,
    )

    assert len(result.task_operations) == 1
    assert len(result.note_operations) == 1
    assert result.note_operations[0].operation == "create"
    assert "ai_memory_notes" in result.note_operations[0].content


def test_multi_chunk_window_requires_explicit_source_coverage_before_full_publish():
    output = TranscriptAnalysisOutput.model_validate(
        {
            "is_complete_enough": True,
            "requires_more_context": False,
            "context_resolution": {"resolved_entities": {}, "confidence": 0.9},
            "task_operations": [
                {
                    "operation": "create",
                    "title": "Follow up on payment",
                    "description": "Check the payment status tomorrow.",
                    "due_at": None,
                    "status": "open",
                    "confidence": 0.9,
                    "source_chunk_ids": [],
                }
            ],
            "note_operations": [],
            "summary_update": {"should_update": False, "updated_summary": ""},
        }
    )

    window = {"chunk_ids": ["chunk-1", "chunk-2", "chunk-3"]}
    normalized = _normalize_generated_operations(window=window, output=output)
    vectors = [
        MemoryVector(
            point_id=f"point-{index}",
            text=f"text {index}",
            request_id=None,
            payload={"chunkId": chunk_id},
        )
        for index, chunk_id in enumerate(window["chunk_ids"], start=1)
    ]

    assert normalized.task_operations[0].source_chunk_ids == []
    assert (
        _published_point_ids_for_output(
            vectors=vectors,
            window=window,
            output=normalized,
        )
        == []
    )
