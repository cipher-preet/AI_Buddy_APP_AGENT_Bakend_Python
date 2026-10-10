from __future__ import annotations

from services.document_it.generator import DocumentGenerateError, _extract_json_dict
from services.document_it.schemas import DocumentGenerateSchema, get_template_spec
from services.llm.openai_compatible import _assistant_message_text
from services.llm.schema_adapter import (
    build_structured_plan,
    openrouter_json_reasoning_body,
    openrouter_prefers_plain_json,
    structured_capabilities,
    structured_modes_for,
)


def test_openrouter_nemotron_free_prefers_plain_json():
    model = "nvidia/nemotron-3-ultra-550b-a55b:free"
    assert openrouter_prefers_plain_json(model) is True
    caps = structured_capabilities("openrouter", model)
    assert caps.supports_json_schema is False
    assert caps.supports_json_object is False
    assert structured_modes_for("openrouter", model) == ["plain_json_prompt"]
    plan = build_structured_plan("openrouter", model, DocumentGenerateSchema, DocumentGenerateSchema.__name__)
    assert len(plan.attempts) == 1
    assert plan.attempts[0].mode == "plain_json_prompt"
    assert plan.attempts[0].response_format is None
    body = openrouter_json_reasoning_body()
    assert plan.attempts[0].extra_body == body
    # OpenRouter: only one of effort/max_tokens; never exclude (hides empty answers).
    assert "effort" not in body["reasoning"]
    assert "exclude" not in body["reasoning"]
    assert body["reasoning"]["max_tokens"] == 2048


def test_assistant_message_text_uses_tool_call_arguments():
    message = {
        "content": "",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "emit_document",
                    "arguments": '{"title":"Meeting Recap","sections":[{"heading":"Summary","body":"Done"}]}',
                },
            }
        ],
    }
    text = _assistant_message_text(message)
    assert "Meeting Recap" in text
    assert "Summary" in text


def test_openrouter_embedded_error_payload_raises():
    from services.llm.errors import LLMProviderError
    from services.llm.openai_compatible import _raise_if_openrouter_error_payload

    try:
        _raise_if_openrouter_error_payload(
            {
                "error": {
                    "message": "Upstream error from Nvidia: Service temporarily overloaded",
                    "code": 503,
                }
            },
            provider="openrouter",
            model="nvidia/nemotron-3-ultra-550b-a55b:free",
        )
        assert False, "expected LLMProviderError"
    except LLMProviderError as error:
        assert error.status_code == 503
        assert "overloaded" in str(error).lower()


def test_openrouter_upstream_outage_skips_plain_retry():
    from services.document_it.generator import _is_openrouter_upstream_outage
    from services.llm.errors import LLMProviderError

    assert _is_openrouter_upstream_outage(
        LLMProviderError(
            "openrouter:nvidia/nemotron-3-ultra-550b-a55b:free upstream error: "
            "Upstream error from Nvidia: Internal server error",
            retryable=True,
            status_code=502,
        )
    )
    assert not _is_openrouter_upstream_outage(DocumentGenerateError("Plain JSON generation returned non-JSON content."))


def test_extract_json_dict_strips_reasoning_and_fences():
    raw = """
    <think>planning the document</think>
    ```json
    {"title": "Weekly Review", "subtitle": "", "metaLines": [], "sections": [{"heading": "Completed", "body": "Shipped.", "bullets": [], "tableHeaders": [], "tableRows": []}]}
    ```
    """
    payload = _extract_json_dict(raw)
    assert payload is not None
    assert payload["title"] == "Weekly Review"
    assert payload["sections"][0]["heading"] == "Completed"


def test_extract_json_dict_accepts_array_wrapper():
    raw = '[{"title": "Doc", "sections": [{"heading": "Summary", "body": "Hi"}]}]'
    payload = _extract_json_dict(raw)
    assert payload is not None
    assert payload["title"] == "Doc"


def test_template_spec_available_for_meeting_recap():
    spec = get_template_spec("meeting-recap")
    assert "Summary" in spec.requiredSections
