from __future__ import annotations

import json
import re
from typing import Any

from apps.api_gateway.config.setting import settings
from services.document_it.log import document_log
from services.document_it.schemas import (
    DocumentGenerateSchema,
    DocumentSection,
    DocumentSectionLLM,
    DocumentTable,
    GeneratedDocumentContent,
    TemplateSpec,
    get_template_spec,
)
from services.llm.errors import LLMProviderError, StructuredOutputError
from services.llm.models import LLMMessage, LLMRequest, StructuredLLMRequest
from services.llm.openai_compatible import (
    _close_truncated_json,
    _extract_json_object,
    _sanitize_json_text,
    parse_structured_content,
)
from services.llm.router import get_llm_router
from services.llm.schema_adapter import (
    openrouter_json_reasoning_body,
    openrouter_prefers_plain_json,
)
from services.prompts.loader import load_prompt
from services.schedule_extraction.llm import parse_model_route


class DocumentGenerateError(RuntimeError):
    pass


_THINK_RE = re.compile(r"<think>[\s\S]*?</think>", re.IGNORECASE)
_REASONING_FENCE_RE = re.compile(r"```(?:reasoning|think)[\s\S]*?```", re.IGNORECASE)
_REDACTED_REASONING_RE = re.compile(
    r"<\|?(?:redacted_)?reasoning\|?>[\s\S]*?<\|?/(?:redacted_)?reasoning\|?>",
    re.IGNORECASE,
)
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)

_PLAIN_JSON_INSTRUCTION = (
    "Return ONLY a single JSON object matching the required document shape. "
    "No markdown fences, no prose, no reasoning, no commentary before or after the JSON. "
    "Use metaLines as a string array. Use tableRows as pipe-separated strings. "
    'Required keys: "title", "subtitle", "metaLines", "sections".'
)


def _route() -> list[tuple[Any, str]]:
    router = get_llm_router()
    spec = settings.DOCUMENT_MODELS or settings.MINDMAP_MODELS
    route: list[tuple[Any, str]] = []
    for name, model in parse_model_route(spec):
        provider = router.providers.get(name)
        if provider is None or getattr(provider, "configured", True) is False:
            continue
        route.append((provider, model))
    if not route:
        raise DocumentGenerateError(
            "Document generation LLM route is not configured (set OPENROUTER_API_KEY)."
        )
    return route


def _model_rejects_response_format(provider_name: str, model: str) -> bool:
    provider = str(provider_name or "").casefold()
    return provider == "openrouter" and openrouter_prefers_plain_json(model)


def _split_table_row(line: str) -> list[str]:
    text = str(line or "").strip()
    if not text:
        return []
    if "|" in text:
        return [part.strip() for part in text.split("|")]
    if "\t" in text:
        return [part.strip() for part in text.split("\t")]
    return [text]


def _coerce_section(raw: Any) -> DocumentSection | None:
    if isinstance(raw, DocumentSection):
        return raw
    if isinstance(raw, DocumentSectionLLM):
        headers = [str(item).strip() for item in (raw.tableHeaders or []) if str(item).strip()]
        rows = [_split_table_row(item) for item in (raw.tableRows or [])]
        rows = [row for row in rows if any(cell.strip() for cell in row)]
        heading = str(raw.heading or "").strip()
        if not heading:
            return None
        return DocumentSection(
            heading=heading,
            body=str(raw.body or "").strip() or None,
            bullets=[str(item).strip() for item in (raw.bullets or []) if str(item).strip()],
            table=DocumentTable(headers=headers, rows=rows) if headers else None,
        )
    if not isinstance(raw, dict):
        return None

    heading = str(raw.get("heading") or raw.get("title") or "").strip()
    if not heading:
        return None

    body = raw.get("body")
    bullets_raw = raw.get("bullets") or raw.get("items") or []
    bullets = (
        [str(item).strip() for item in bullets_raw if str(item).strip()]
        if isinstance(bullets_raw, list)
        else []
    )

    headers: list[str] = []
    rows: list[list[str]] = []
    table = raw.get("table")
    if isinstance(table, dict):
        headers = [str(item).strip() for item in (table.get("headers") or []) if str(item).strip()]
        for row in table.get("rows") or []:
            if isinstance(row, list):
                cells = [str(cell).strip() for cell in row]
            else:
                cells = _split_table_row(str(row))
            if any(cells):
                rows.append(cells)
    else:
        headers = [
            str(item).strip()
            for item in (raw.get("tableHeaders") or raw.get("headers") or [])
            if str(item).strip()
        ]
        for row in raw.get("tableRows") or raw.get("rows") or []:
            if isinstance(row, list):
                cells = [str(cell).strip() for cell in row]
            else:
                cells = _split_table_row(str(row))
            if any(cells):
                rows.append(cells)

    return DocumentSection(
        heading=heading,
        body=str(body).strip() if body else None,
        bullets=bullets,
        table=DocumentTable(headers=headers, rows=rows) if headers else None,
    )


def _meta_from_payload(payload: dict[str, Any]) -> dict[str, str]:
    meta: dict[str, str] = {}
    raw_meta = payload.get("meta")
    if isinstance(raw_meta, dict):
        for key, value in raw_meta.items():
            if value is None:
                continue
            meta[str(key)] = str(value)
    lines = payload.get("metaLines") or payload.get("meta_lines") or []
    if isinstance(lines, list):
        for line in lines:
            text = str(line or "").strip()
            if not text:
                continue
            if ":" in text:
                key, value = text.split(":", 1)
                meta[key.strip() or "info"] = value.strip()
            else:
                meta[f"note{len(meta) + 1}"] = text
    return meta


def coerce_document_payload(payload: Any, *, template: TemplateSpec) -> DocumentGenerateSchema:
    if isinstance(payload, DocumentGenerateSchema):
        return payload
    data = payload if isinstance(payload, dict) else {}
    sections_raw = data.get("sections") if isinstance(data.get("sections"), list) else []
    sections: list[DocumentSectionLLM] = []
    for item in sections_raw:
        section = _coerce_section(item)
        if section is None:
            continue
        table_rows = [
            " | ".join(str(cell) for cell in row)
            for row in (section.table.rows if section.table else [])
        ]
        sections.append(
            DocumentSectionLLM(
                heading=section.heading,
                body=section.body or "",
                bullets=section.bullets,
                tableHeaders=section.table.headers if section.table else [],
                tableRows=table_rows,
            )
        )

    title = str(data.get("title") or template.title or "Document").strip() or template.title
    subtitle = str(data.get("subtitle") or "").strip()
    meta_lines = data.get("metaLines") if isinstance(data.get("metaLines"), list) else []
    if not meta_lines and isinstance(data.get("meta"), dict):
        meta_lines = [f"{key}: {value}" for key, value in data["meta"].items() if value is not None]

    return DocumentGenerateSchema(
        title=title,
        subtitle=subtitle,
        metaLines=[str(item).strip() for item in meta_lines if str(item).strip()],
        sections=sections,
    )


def _ensure_required_sections(
    content: DocumentGenerateSchema,
    template: TemplateSpec,
) -> GeneratedDocumentContent:
    ordered_sections: list[DocumentSection] = []
    by_heading: dict[str, DocumentSection] = {}

    for raw in content.sections or []:
        section = _coerce_section(raw)
        if section is None:
            continue
        by_heading[section.heading.strip().lower()] = section

    for required in template.requiredSections:
        key = required.lower()
        existing = by_heading.pop(key, None)
        if existing:
            ordered_sections.append(existing)
            continue
        hint_headers = template.tableHints.get(required)
        ordered_sections.append(
            DocumentSection(
                heading=required,
                body="Not enough evidence in the workspace context to populate this section yet.",
                bullets=[],
                table=DocumentTable(headers=hint_headers, rows=[]) if hint_headers else None,
            )
        )

    for section in by_heading.values():
        ordered_sections.append(section)

    meta = _meta_from_payload({"metaLines": content.metaLines})
    return GeneratedDocumentContent(
        title=content.title or template.title,
        subtitle=content.subtitle or None,
        meta=meta,
        sections=ordered_sections,
    )


def _strip_non_json_wrappers(text: str) -> str:
    value = str(text or "")
    value = _THINK_RE.sub("", value)
    value = _REDACTED_REASONING_RE.sub("", value)
    value = _REASONING_FENCE_RE.sub("", value)
    return _sanitize_json_text(value)


def _extract_json_dict(text: str) -> dict[str, Any] | None:
    raw = str(text or "")
    cleaned = _strip_non_json_wrappers(raw)
    if not cleaned and not raw:
        return None

    candidates: list[str] = []
    if cleaned:
        candidates.append(cleaned)
    for match in _JSON_FENCE_RE.finditer(raw):
        fenced = _strip_non_json_wrappers(match.group(1) or "")
        if fenced:
            candidates.append(fenced)
    for source in list(candidates):
        extracted = _extract_json_object(source)
        if extracted:
            candidates.append(extracted)
        repaired = _close_truncated_json(source)
        if repaired:
            candidates.append(repaired)
        if extracted:
            repaired_extracted = _close_truncated_json(extracted)
            if repaired_extracted:
                candidates.append(repaired_extracted)

    seen: set[str] = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            soft = re.sub(r",\s*([}\]])", r"\1", candidate)
            try:
                payload = json.loads(soft)
            except json.JSONDecodeError:
                continue
        if isinstance(payload, dict):
            return payload
        if isinstance(payload, list) and payload and isinstance(payload[0], dict):
            # Some models wrap the document object in a one-item array.
            return payload[0]
    return None


def _document_tool_schema() -> dict[str, Any]:
    """OpenRouter tool schema — Nemotron free supports tools, not response_format."""
    return {
        "type": "function",
        "function": {
            "name": "emit_document",
            "description": "Emit the finished workspace document as structured fields.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "subtitle": {"type": "string"},
                    "metaLines": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": 'Lines like "Space: Name" or "Date: 2026-10-09"',
                    },
                    "sections": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "heading": {"type": "string"},
                                "body": {"type": "string"},
                                "bullets": {"type": "array", "items": {"type": "string"}},
                                "tableHeaders": {"type": "array", "items": {"type": "string"}},
                                "tableRows": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": 'Pipe-separated cells, e.g. "Task | Owner | Fri"',
                                },
                            },
                            "required": ["heading"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["title", "sections"],
                "additionalProperties": False,
            },
        },
    }


async def _generate_via_tool_call(
    provider: Any,
    *,
    model: str,
    messages: list[LLMMessage],
    max_tokens: int,
    template: TemplateSpec,
) -> DocumentGenerateSchema:
    """Preferred path for OpenRouter free/Nemotron — tools are supported, response_format is not."""
    tool_messages = [
        *messages,
        LLMMessage(
            role="system",
            content=(
                "Call the emit_document tool exactly once with the finished document. "
                "Do not write prose outside the tool arguments. "
                f"Include every required section heading: {', '.join(template.requiredSections)}."
            ),
        ),
    ]
    extra_body: dict[str, Any] = {
        **openrouter_json_reasoning_body(),
        "tools": [_document_tool_schema()],
        "tool_choice": {"type": "function", "function": {"name": "emit_document"}},
    }
    # Reasoning max_tokens (2k) + document body need a large shared budget.
    request = LLMRequest(
        messages=tool_messages,
        model=model,
        temperature=0.1,
        max_tokens=max(max_tokens, 10000),
        metadata={"stage": "document_generate_tool_call", "extra_body": extra_body},
    )
    response = await provider.generate(request)
    raw_content = response.content or ""
    payload = _extract_json_dict(raw_content)
    if payload is None:
        raise DocumentGenerateError(
            "Tool-call document generation returned empty/non-JSON content."
            f" finishReason={response.finishReason!r}"
            f" preview={str(raw_content)[:160]!r}"
        )
    return coerce_document_payload(payload, template=template)


async def _generate_via_plain_json(
    provider: Any,
    *,
    model: str,
    messages: list[LLMMessage],
    max_tokens: int,
    template: TemplateSpec,
    attempt: int = 1,
) -> DocumentGenerateSchema:
    """Fallback path for free OpenRouter/Nemotron — never send response_format."""
    instruction = _PLAIN_JSON_INSTRUCTION
    if attempt > 1:
        instruction = (
            f"{_PLAIN_JSON_INSTRUCTION} "
            "Your previous reply was empty or not valid JSON. Reply with the raw JSON object only."
        )
    plain_messages = [
        *messages,
        LLMMessage(role="system", content=instruction),
    ]
    extra_body: dict[str, Any] = {}
    if openrouter_prefers_plain_json(model):
        # Official Nemotron free efforts are only high/medium — never effort=none / exclude.
        extra_body.update(openrouter_json_reasoning_body())
    request = LLMRequest(
        messages=plain_messages,
        model=model,
        temperature=0.1 if attempt == 1 else 0.0,
        max_tokens=max(max_tokens, 10000),
        metadata={
            "stage": "document_generate_plain_json",
            "extra_body": extra_body,
        },
    )
    response = await provider.generate(request)
    raw_content = response.content or ""
    payload = _extract_json_dict(raw_content)
    if payload is None and attempt < 2:
        document_log(
            "plain_json_retry",
            provider=getattr(provider, "name", None),
            model=model,
            finishReason=response.finishReason,
            preview=str(raw_content)[:160],
        )
        return await _generate_via_plain_json(
            provider,
            model=model,
            messages=messages,
            max_tokens=max(max_tokens, 12000),
            template=template,
            attempt=attempt + 1,
        )
    if payload is None:
        raise DocumentGenerateError(
            "Plain JSON generation returned non-JSON content."
            f" finishReason={response.finishReason!r}"
            f" preview={str(raw_content)[:160]!r}"
        )
    return coerce_document_payload(payload, template=template)


def _is_openrouter_upstream_outage(error: BaseException) -> bool:
    """Nvidia free often returns HTTP 200 + embedded 502/503 (OpenRouter free-model capacity)."""
    message = str(error).casefold()
    markers = (
        "upstream error",
        "provider_unavailable",
        "provider_overloaded",
        "internal server error",
        "temporarily overloaded",
        "returned no choices",
        "returned empty assistant content",
        "service temporarily",
    )
    if any(token in message for token in markers):
        return True
    if isinstance(error, LLMProviderError) and error.status_code in {500, 502, 503, 504, 529}:
        return True
    return False


async def _generate_for_openrouter_free(
    provider: Any,
    *,
    model: str,
    messages: list[LLMMessage],
    max_tokens: int,
    template: TemplateSpec,
) -> DocumentGenerateSchema:
    """OpenRouter free/Nemotron: tools first, then plain JSON — unless upstream Nvidia is down."""
    tool_error: Exception | None = None
    try:
        return await _generate_via_tool_call(
            provider,
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            template=template,
        )
    except Exception as error:
        tool_error = error
        # Same Nvidia endpoint cannot serve plain JSON if it just 502'd — skip to next model.
        if _is_openrouter_upstream_outage(error):
            document_log(
                "openrouter_upstream_skip_plain",
                provider=getattr(provider, "name", None),
                model=model,
                error=type(error).__name__,
                detail=str(error)[:180],
            )
            raise DocumentGenerateError(
                f"openrouter_upstream: {type(error).__name__}: {str(error)[:160]}"
            ) from error
        document_log(
            "tool_call_failed_fallback_plain",
            provider=getattr(provider, "name", None),
            model=model,
            error=type(error).__name__,
            detail=str(error)[:180],
        )
    try:
        return await _generate_via_plain_json(
            provider,
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            template=template,
        )
    except Exception as plain_error:
        raise DocumentGenerateError(
            f"tool: {type(tool_error).__name__}: {str(tool_error)[:120]} | "
            f"plain: {type(plain_error).__name__}: {str(plain_error)[:120]}"
        ) from plain_error


async def generate_document_content(
    context_pack: dict[str, Any],
    *,
    template: TemplateSpec,
    space_name: str,
) -> tuple[GeneratedDocumentContent, str]:
    messages = [
        LLMMessage(role="system", content=load_prompt("document-generate-v1")),
        LLMMessage(
            role="user",
            content=json.dumps(
                {
                    **context_pack,
                    "spaceName": space_name,
                    "instructions": {
                        "templateCode": template.code,
                        "templateTitle": template.title,
                        "requiredSections": template.requiredSections,
                        "guidance": template.guidance,
                        "tableHints": template.tableHints,
                    },
                },
                ensure_ascii=False,
                default=str,
            ),
        ),
    ]
    # Give free/reasoning models enough room for the full document JSON body.
    max_tokens = max(int(settings.DOCUMENT_MAX_OUTPUT_TOKENS), 6000)
    errors: list[str] = []

    for provider, model in _route():
        result: DocumentGenerateSchema | None = None
        # OpenRouter docs: Nemotron free does not support response_format — always plain JSON.
        prefer_plain = _model_rejects_response_format(getattr(provider, "name", ""), model)

        if prefer_plain:
            document_log(
                "openrouter_free_primary",
                provider=getattr(provider, "name", None),
                model=model,
                reason="tools_then_plain_json_no_response_format",
            )
            try:
                result = await _generate_for_openrouter_free(
                    provider,
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    template=template,
                )
            except Exception as error:
                errors.append(f"{provider.name}:{model}: free:{type(error).__name__}: {str(error)[:220]}")
                document_log(
                    "llm_failed",
                    provider=getattr(provider, "name", None),
                    model=model,
                    error=type(error).__name__,
                    detail=str(error)[:180],
                )
                continue
        else:
            request = StructuredLLMRequest(
                messages=messages,
                model=model,
                temperature=0.1,
                max_tokens=max_tokens,
                metadata={"stage": "document_generate", "extra_body": {}},
                schema_name=DocumentGenerateSchema.__name__,
            )
            try:
                raw = await provider.generate_structured(request, DocumentGenerateSchema)
                result = coerce_document_payload(raw, template=template)
            except StructuredOutputError as error:
                document_log(
                    "structured_failed_fallback",
                    provider=getattr(provider, "name", None),
                    model=model,
                    error=str(error)[:180],
                )
                diagnostics = getattr(provider, "last_structured_diagnostics", {}) or {}
                raw_content = str(diagnostics.get("rawContent") or "")
                if raw_content:
                    payload = _extract_json_dict(raw_content)
                    if payload is not None:
                        result = coerce_document_payload(payload, template=template)
                    else:
                        try:
                            parsed, _ = parse_structured_content(DocumentGenerateSchema, raw_content)
                            result = coerce_document_payload(parsed, template=template)
                        except Exception:
                            result = None

                if result is None:
                    try:
                        result = await _generate_via_plain_json(
                            provider,
                            model=model,
                            messages=messages,
                            max_tokens=max_tokens,
                            template=template,
                        )
                    except Exception as fallback_error:
                        errors.append(
                            f"{provider.name}:{model}: {type(error).__name__}: {str(error)[:100]} | "
                            f"fallback: {type(fallback_error).__name__}: {str(fallback_error)[:120]}"
                        )
                        continue
            except Exception as error:
                errors.append(f"{provider.name}:{model}: {type(error).__name__}: {str(error)[:180]}")
                document_log(
                    "llm_failed",
                    provider=getattr(provider, "name", None),
                    model=model,
                    error=type(error).__name__,
                )
                continue

        if result is None:
            errors.append(f"{model}: empty_result")
            continue

        # Template fills any missing required sections — do not fail on thin LLM output.
        content = _ensure_required_sections(result, template)
        if not content.sections:
            errors.append(f"{model}: empty_sections")
            continue

        document_log(
            "llm_ok",
            provider=getattr(provider, "name", None),
            model=model,
            sectionCount=len(content.sections),
            template=template.code,
        )
        return content, f"{getattr(provider, 'name', 'llm')}:{model}"

    raise DocumentGenerateError("; ".join(errors) or "Document generation failed.")


resolve_template_spec = get_template_spec
