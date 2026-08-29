from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ValidationError


QUOTA_UNAVAILABLE = "QUOTA_UNAVAILABLE"
RATE_LIMITED = "RATE_LIMITED"
HTTP_ERROR = "HTTP_ERROR"
STRUCTURED_SCHEMA_UNSUPPORTED = "STRUCTURED_SCHEMA_UNSUPPORTED"
MALFORMED_JSON = "MALFORMED_JSON"
SCHEMA_VALIDATION_FAILED = "SCHEMA_VALIDATION_FAILED"
SCHEMA_ECHO = "SCHEMA_ECHO"
INCOMPLETE_STRUCTURED_OUTPUT = "INCOMPLETE_STRUCTURED_OUTPUT"
PROVIDER_TIMEOUT = "PROVIDER_TIMEOUT"
ASYNC_LIFECYCLE_ERROR = "ASYNC_LIFECYCLE_ERROR"
PROVIDER_FAILURE = "PROVIDER_FAILURE"
RATE_LIMIT = "RATE_LIMIT"
TIMEOUT = "TIMEOUT"
STRUCTURED_OUTPUT_FAILURE = "STRUCTURED_OUTPUT_FAILURE"

# Backward-compatible aliases used by extraction diagnostics and existing tests.
STRUCTURED_SCHEMA_ECHO = "STRUCTURED_SCHEMA_ECHO"
MALFORMED_STRUCTURED_OUTPUT = "MALFORMED_STRUCTURED_OUTPUT"
PARSED_INSTANCE = "PARSED_INSTANCE"

STRING_LIST_FIELDS = (
    "topics",
    "decisions",
    "problems",
    "solutions",
    "commitments",
    "requests",
    "followUps",
    "deadlines",
    "owners",
    "dependencies",
    "requirements",
    "constraints",
    "risks",
    "importantFacts",
    "ideas",
    "unresolvedQuestions",
    "nextSteps",
    "openQuestions",
)

WIRE_REQUIRED_COLLECTIONS = {
    "FinalSynthesisLLMResponse": ("tasks", "notes"),
    "MeetingCandidateExtractorResponse": ("candidates",),
    "MeetingVerifierResponse": ("items",),
}

_INSTANCE_KEYS = {
    "semanticUnits",
    "units",
    "semantic_units",
    "semanticUnit",
    "tasks",
    "notes",
    "decisions",
    "issues",
    "supportedUnitVerdict",
    "unitVerdict",
    "emptyExtractionVerdict",
    "extractionVerdict",
    "summary",
    "narrative",
    "rejectedCandidates",
    "publishVerdict",
    "importantFacts",
    "topics",
    "openQuestions",
    "problems",
    "requirements",
    "commitments",
}
_SCHEMA_MARKER_KEYS = {"$defs", "$schema", "$ref", "$id", "additionalProperties", "definitions"}

_PROVIDER_LOCAL_RECOVERY_REASONS = {
    STRUCTURED_SCHEMA_UNSUPPORTED,
    SCHEMA_VALIDATION_FAILED,
    MALFORMED_JSON,
    SCHEMA_ECHO,
    STRUCTURED_SCHEMA_ECHO,
    INCOMPLETE_STRUCTURED_OUTPUT,
    MALFORMED_STRUCTURED_OUTPUT,
}


@dataclass(frozen=True)
class StructuredOutputCapability:
    supports_json_schema: bool
    supports_json_object: bool
    supports_plain_json_prompt: bool
    structured_output_reliability: str


@dataclass
class StructuredAttemptPlan:
    mode: str
    response_format: dict[str, Any] | None
    extra_body: dict[str, Any] = field(default_factory=dict)
    schema: dict[str, Any] = field(default_factory=dict)
    instruction: str = ""
    temperature: float = 0.0
    parsing_strategy: str = "canonical_pydantic"


@dataclass
class StructuredProviderPlan:
    provider: str
    model: str
    schema_name: str
    canonical_schema: dict[str, Any]
    attempts: list[StructuredAttemptPlan]


def structured_capabilities(provider: str, model: str) -> StructuredOutputCapability:
    provider_name = str(provider or "").strip().casefold()
    if provider_name in {"openai", "krutrim"}:
        return StructuredOutputCapability(True, True, True, "high")
    if provider_name in {"mistral", "groq", "gemini", "sarvam"}:
        return StructuredOutputCapability(True, True, True, "medium")
    return StructuredOutputCapability(False, True, True, "low")


def structured_modes_for(provider: str, model: str) -> list[str]:
    plan = build_structured_plan(provider, model, None, "Schema")
    return [attempt.mode for attempt in plan.attempts]


def build_structured_plan(
    provider: str,
    model: str,
    response_schema: type[BaseModel] | None,
    schema_name: str,
) -> StructuredProviderPlan:
    canonical = canonical_json_schema(response_schema, schema_name)
    adapter = _adapter_for(provider)
    return adapter.plan(provider, model, schema_name, canonical)


def canonical_json_schema(response_schema: type[BaseModel] | None, schema_name: str) -> dict[str, Any]:
    if response_schema is None:
        return {"type": "object", "properties": {}, "required": [], "additionalProperties": False}
    schema = response_schema.model_json_schema()
    schema = _inline_refs(schema)
    schema.pop("$defs", None)
    schema.pop("definitions", None)
    _normalize_schema_node(schema)
    _force_string_array_items(schema)
    required_extra = list(WIRE_REQUIRED_COLLECTIONS.get(schema_name or getattr(response_schema, "__name__", ""), ()))
    if schema_name == "ConversationUnderstandingResponse" or getattr(response_schema, "__name__", "") == "ConversationUnderstandingResponse":
        required_extra.extend(STRING_LIST_FIELDS)
    properties = schema.get("properties") or {}
    required = [str(name) for name in (schema.get("required") or [])]
    for name in required_extra:
        if name in properties and name not in required:
            required.append(name)
    if required:
        schema["required"] = required
    return schema


class StructuredSchemaAdapter:
    def plan(self, provider: str, model: str, schema_name: str, canonical_schema: dict[str, Any]) -> StructuredProviderPlan:
        raise NotImplementedError

    def _json_schema_attempt(
        self,
        schema_name: str,
        schema: dict[str, Any],
        *,
        strict: bool | None = None,
        extra_body: dict[str, Any] | None = None,
        instruction: str | None = None,
    ) -> StructuredAttemptPlan:
        json_schema: dict[str, Any] = {"name": schema_name, "schema": schema}
        if strict is not None:
            json_schema["strict"] = strict
        return StructuredAttemptPlan(
            mode="json_schema",
            response_format={"type": "json_schema", "json_schema": json_schema},
            extra_body=dict(extra_body or {}),
            schema=schema,
            instruction=instruction or _schema_instruction(schema_name, schema),
            temperature=0.0,
        )

    def _json_object_attempt(self, schema_name: str, schema: dict[str, Any]) -> StructuredAttemptPlan:
        return StructuredAttemptPlan(
            mode="json_object",
            response_format={"type": "json_object"},
            schema=schema,
            instruction=_schema_instruction(schema_name, schema, recovery=True),
            temperature=0.0,
        )


class GroqStructuredAdapter(StructuredSchemaAdapter):
    def plan(self, provider: str, model: str, schema_name: str, canonical_schema: dict[str, Any]) -> StructuredProviderPlan:
        strict_schema, incompatible = groq_strict_schema(canonical_schema)
        if strict_schema is not None:
            attempts = [
                self._json_schema_attempt(schema_name, strict_schema, strict=True),
                self._json_object_attempt(schema_name, canonical_schema),
            ]
        else:
            fallback = self._json_object_attempt(schema_name, canonical_schema)
            fallback.instruction = f"{fallback.instruction} Groq strict json_schema was not used: {incompatible}."
            attempts = [fallback]
        return StructuredProviderPlan(provider, model, schema_name, canonical_schema, attempts)


class GeminiStructuredAdapter(StructuredSchemaAdapter):
    def plan(self, provider: str, model: str, schema_name: str, canonical_schema: dict[str, Any]) -> StructuredProviderPlan:
        gemini_schema = gemini_response_schema(canonical_schema)
        # OpenAI-compatible Gemini accepts response_format.json_schema. Native
        # generateContent fields (response_mime_type, response_json_schema) are
        # not valid extra_body.google keys and cause HTTP 400.
        attempts = [
            self._json_schema_attempt(
                schema_name,
                gemini_schema,
                instruction=_schema_instruction(schema_name, gemini_schema),
            ),
            self._json_object_attempt(schema_name, gemini_schema),
        ]
        return StructuredProviderPlan(provider, model, schema_name, canonical_schema, attempts)


class MistralStructuredAdapter(StructuredSchemaAdapter):
    def plan(self, provider: str, model: str, schema_name: str, canonical_schema: dict[str, Any]) -> StructuredProviderPlan:
        mistral_schema = mistral_json_schema(canonical_schema)
        attempts = [
            self._json_schema_attempt(schema_name, mistral_schema, strict=True),
            self._json_object_attempt(schema_name, mistral_schema),
        ]
        return StructuredProviderPlan(provider, model, schema_name, canonical_schema, attempts)


class SarvamStructuredAdapter(StructuredSchemaAdapter):
    def plan(self, provider: str, model: str, schema_name: str, canonical_schema: dict[str, Any]) -> StructuredProviderPlan:
        schema = dict(canonical_schema)
        capability = structured_capabilities(provider, model)
        attempts: list[StructuredAttemptPlan] = []
        if capability.supports_json_schema:
            attempts.append(self._json_schema_attempt(schema_name, schema, strict=False))
        attempts.append(self._json_object_attempt(schema_name, schema))
        return StructuredProviderPlan(provider, model, schema_name, canonical_schema, attempts[:2])


class KrutrimStructuredAdapter(StructuredSchemaAdapter):
    def plan(self, provider: str, model: str, schema_name: str, canonical_schema: dict[str, Any]) -> StructuredProviderPlan:
        # Transport may use json_schema, json_object, or a JSON prompt. Every
        # attempt still carries the canonical schema and is pydantic-validated.
        attempts = [
            self._json_schema_attempt(schema_name, canonical_schema, strict=False),
            self._json_object_attempt(schema_name, canonical_schema),
            StructuredAttemptPlan(
                mode="plain_json_prompt",
                response_format=None,
                schema=canonical_schema,
                instruction=_schema_instruction(schema_name, canonical_schema, recovery=True),
                temperature=0.0,
            ),
        ]
        return StructuredProviderPlan(provider, model, schema_name, canonical_schema, attempts)


class DefaultStructuredAdapter(StructuredSchemaAdapter):
    def plan(self, provider: str, model: str, schema_name: str, canonical_schema: dict[str, Any]) -> StructuredProviderPlan:
        capability = structured_capabilities(provider, model)
        attempts: list[StructuredAttemptPlan] = []
        if capability.supports_json_schema:
            attempts.append(self._json_schema_attempt(schema_name, canonical_schema, strict=False))
        if capability.supports_json_object:
            attempts.append(self._json_object_attempt(schema_name, canonical_schema))
        return StructuredProviderPlan(provider, model, schema_name, canonical_schema, attempts[:2] or [
            StructuredAttemptPlan(
                mode="plain_json_prompt",
                response_format=None,
                schema=canonical_schema,
                instruction=_schema_instruction(schema_name, canonical_schema),
            )
        ])


def _adapter_for(provider: str) -> StructuredSchemaAdapter:
    name = str(provider or "").strip().casefold()
    if name == "groq":
        return GroqStructuredAdapter()
    if name == "gemini":
        return GeminiStructuredAdapter()
    if name == "mistral":
        return MistralStructuredAdapter()
    if name == "sarvam":
        return SarvamStructuredAdapter()
    if name == "krutrim":
        return KrutrimStructuredAdapter()
    return DefaultStructuredAdapter()


def groq_strict_schema(schema: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    try:
        cloned = _deep_copy(schema)
        _normalize_schema_node(cloned)
        reason = _groq_incompatibility(cloned)
        if reason:
            return None, reason
        _apply_groq_strict(cloned)
        reason = _groq_incompatibility(cloned)
        if reason:
            return None, reason
        return cloned, None
    except Exception as error:
        return None, str(error)[:200]


def gemini_response_schema(schema: dict[str, Any]) -> dict[str, Any]:
    cloned = _deep_copy(schema)
    _normalize_schema_node(cloned)
    _strip_schema_metadata_keys(cloned, {"$schema", "$id", "$defs", "definitions", "title", "default", "examples"})
    return cloned


def mistral_json_schema(schema: dict[str, Any]) -> dict[str, Any]:
    # Official Mistral custom structured output requires additionalProperties: false
    # on every object. Do not strip property names such as title.
    cloned = _deep_copy(schema)
    _normalize_schema_node(cloned)
    _force_string_array_items(cloned)
    _apply_object_closed(cloned, nullable_optional=False)
    return cloned


def is_schema_echo(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    keys = set(value.keys())
    if keys & _INSTANCE_KEYS:
        return False
    if keys & _SCHEMA_MARKER_KEYS:
        return True
    type_value = value.get("type")
    if isinstance(type_value, list):
        type_name = " ".join(str(item) for item in type_value).casefold()
    else:
        type_name = str(type_value or "").casefold()
    has_properties = isinstance(value.get("properties"), dict)
    if type_name == "object" and has_properties:
        return True
    if has_properties and isinstance(value.get("required"), list):
        return True
    return False


def normalize_failure_reason(reason: str | None) -> str:
    value = str(reason or "").strip()
    if value == STRUCTURED_SCHEMA_ECHO:
        return SCHEMA_ECHO
    if value == MALFORMED_STRUCTURED_OUTPUT:
        return MALFORMED_JSON
    return value or HTTP_ERROR


def structured_outcome_from_error(error: Exception) -> str | None:
    from services.llm.errors import StructuredOutputError

    if isinstance(error, StructuredOutputError):
        return error.outcome
    classified = classify_llm_failure(error)
    if classified == SCHEMA_ECHO:
        return STRUCTURED_SCHEMA_ECHO
    if classified in {MALFORMED_JSON, SCHEMA_VALIDATION_FAILED, INCOMPLETE_STRUCTURED_OUTPUT}:
        return classified if classified != MALFORMED_JSON else MALFORMED_STRUCTURED_OUTPUT
    message = str(error or "")
    if STRUCTURED_SCHEMA_ECHO in message or SCHEMA_ECHO in message:
        return STRUCTURED_SCHEMA_ECHO
    if MALFORMED_STRUCTURED_OUTPUT in message or "Structured response validation failed" in message:
        return MALFORMED_STRUCTURED_OUTPUT
    if INCOMPLETE_STRUCTURED_OUTPUT in message:
        return INCOMPLETE_STRUCTURED_OUTPUT
    if SCHEMA_VALIDATION_FAILED in message:
        return SCHEMA_VALIDATION_FAILED
    return None


def drop_stage_for_structured_outcome(outcome: str | None) -> str:
    if outcome in {STRUCTURED_SCHEMA_ECHO, SCHEMA_ECHO}:
        return "structured_schema_echo"
    if outcome == INCOMPLETE_STRUCTURED_OUTPUT:
        return "incomplete_structured_output"
    if outcome in {SCHEMA_VALIDATION_FAILED, MALFORMED_STRUCTURED_OUTPUT, MALFORMED_JSON}:
        return "malformed_structured_output"
    return "provider_or_parser_failure"


def classify_failure_class(error: Exception | None = None, reason: str | None = None) -> str:
    from services.llm.async_runtime import is_async_lifecycle_error

    if error is not None and is_async_lifecycle_error(error):
        return ASYNC_LIFECYCLE_ERROR
    value = normalize_failure_reason(reason or (classify_llm_failure(error) if error is not None else None))
    if value == ASYNC_LIFECYCLE_ERROR:
        return ASYNC_LIFECYCLE_ERROR
    if value == RATE_LIMITED:
        return RATE_LIMIT
    if value == PROVIDER_TIMEOUT:
        return TIMEOUT
    if value in {
        MALFORMED_JSON,
        SCHEMA_VALIDATION_FAILED,
        SCHEMA_ECHO,
        STRUCTURED_SCHEMA_ECHO,
        INCOMPLETE_STRUCTURED_OUTPUT,
        MALFORMED_STRUCTURED_OUTPUT,
        STRUCTURED_SCHEMA_UNSUPPORTED,
    }:
        return STRUCTURED_OUTPUT_FAILURE
    return PROVIDER_FAILURE


def classify_llm_failure(error: Exception) -> str:
    from services.llm.async_runtime import is_async_lifecycle_error
    from services.llm.errors import LLMProviderError, StructuredOutputError

    if is_async_lifecycle_error(error):
        return ASYNC_LIFECYCLE_ERROR
    if isinstance(error, StructuredOutputError):
        return normalize_failure_reason(error.outcome)
    if isinstance(error, LLMProviderError):
        if error.failure_reason:
            return normalize_failure_reason(error.failure_reason)
        message = str(error).casefold()
        if "quota guard" in message or "tpm limit" in message or "rpd limit" in message or "rpm limit" in message or "tpd limit" in message:
            return QUOTA_UNAVAILABLE
        if error.status_code == 429 or "rate limit" in message:
            return RATE_LIMITED
        if error.status_code in {408} or "timeout" in message:
            return PROVIDER_TIMEOUT
        if error.status_code in {400, 422} and any(
            token in message for token in ("json_schema", "response_format", "schema", "strict")
        ):
            return STRUCTURED_SCHEMA_UNSUPPORTED
        if error.status_code:
            return HTTP_ERROR
    name = type(error).__name__.casefold()
    message = str(error or "").casefold()
    if is_async_lifecycle_error(error):
        return ASYNC_LIFECYCLE_ERROR
    if "timeout" in name or "timeout" in message:
        return PROVIDER_TIMEOUT
    if INCOMPLETE_STRUCTURED_OUTPUT.casefold() in message:
        return INCOMPLETE_STRUCTURED_OUTPUT
    if SCHEMA_VALIDATION_FAILED.casefold() in message:
        return SCHEMA_VALIDATION_FAILED
    return HTTP_ERROR


def provider_local_recovery_eligible(error: Exception) -> bool:
    return classify_llm_failure(error) in _PROVIDER_LOCAL_RECOVERY_REASONS


def classify_validation_error(response_schema: type[BaseModel], payload: Any, error: ValidationError) -> str:
    schema_name = getattr(response_schema, "__name__", "")
    required = WIRE_REQUIRED_COLLECTIONS.get(schema_name, ())
    if isinstance(payload, dict):
        if any(field not in payload or payload.get(field) is None for field in required):
            return INCOMPLETE_STRUCTURED_OUTPUT
    message = str(error)
    if INCOMPLETE_STRUCTURED_OUTPUT in message:
        return INCOMPLETE_STRUCTURED_OUTPUT
    for item in error.errors():
        if item.get("type") == "missing":
            loc = item.get("loc") or ()
            if loc and loc[0] in required:
                return INCOMPLETE_STRUCTURED_OUTPUT
        if INCOMPLETE_STRUCTURED_OUTPUT in str(item.get("msg") or ""):
            return INCOMPLETE_STRUCTURED_OUTPUT
    return SCHEMA_VALIDATION_FAILED


def _schema_instruction(schema_name: str, schema: dict[str, Any], recovery: bool = False) -> str:
    properties = schema.get("properties") or {}
    required = [str(name) for name in (schema.get("required") or [])]
    field_lines: list[str] = []
    for name, spec in list(properties.items())[:40]:
        field_lines.append(f"- {name}: {_type_label(spec)}")
    fields = "\n".join(field_lines) or f"- {schema_name}"
    required_text = ", ".join(required[:24]) or "none"
    extra = ""
    if schema_name == "ConversationUnderstandingResponse":
        extra = (
            " String-list fields such as problems, requirements, decisions, commitments, "
            "deadlines, importantFacts, nextSteps, owners, and unresolvedQuestions must be "
            'arrays of strings. Correct: "problems": ["Duplicate outlet appeared twice"]. '
            'Incorrect: "problems": [{"description": "..."}].'
        )
    if schema_name == "FinalSynthesisLLMResponse":
        extra = (
            " tasks and notes are required arrays of objects. Each task and note MUST have "
            "string fields title and body. Do not use content, description, or text in place of "
            "title/body. Keep the JSON compact and complete; do not emit chain-of-thought. "
            "A legitimate empty synthesis is "
            '{"publishVerdict":"NO_PUBLISHABLE_ARTIFACTS","tasks":[],"notes":[]}.'
        )
    if schema_name == "WindowExtractionLLMResponse":
        extra = (
            " issues items must include title, kind (blocker|risk|open_question|missing_information), "
            "confidence, and evidence spans with sequenceStart, sequenceEnd, and text. "
            "Do not use description in place of title. "
            "decisions items must include title, status, confidence, and evidence spans."
        )
    if schema_name == "ExtractionQualityReviewResponse":
        extra = (
            " decisions is an array of objects with kind (task|note), index (integer), keep (boolean), "
            "and reason (string). Do not use strings in decisions. "
            "missingActionable and missingNotes must be arrays of strings, not objects. "
            'Correct: "missingActionable": ["Fix duplicate tasks today"]. '
            'Incorrect: "missingActionable": [{"meaning": "..."}].'
        )
    recovery_text = " This is a recovery attempt. Follow the field types exactly." if recovery else ""
    return (
        f"Return only a JSON instance for {schema_name}. "
        "Do not return JSON Schema, $defs, $schema, properties, required, or type definitions. "
        f"Required fields: {required_text}.{extra}{recovery_text}\n"
        f"Field types:\n{fields}"
    )


def _type_label(spec: Any) -> str:
    if not isinstance(spec, dict):
        return "any"
    if "$ref" in spec:
        return str(spec["$ref"]).rsplit("/", 1)[-1]
    types = spec.get("type")
    if isinstance(types, list):
        type_name = "|".join(str(item) for item in types)
    else:
        type_name = str(types or spec.get("anyOf") and "union" or "any")
    if type_name == "array" or spec.get("items") is not None:
        return f"array<{_type_label(spec.get('items') or {})}>"
    return type_name


def _inline_refs(schema: dict[str, Any]) -> dict[str, Any]:
    defs = dict(schema.get("$defs") or schema.get("definitions") or {})
    return _resolve_refs(schema, defs, seen=set())


def _resolve_refs(node: Any, defs: dict[str, Any], seen: set[str]) -> Any:
    if isinstance(node, list):
        return [_resolve_refs(item, defs, seen) for item in node]
    if not isinstance(node, dict):
        return node
    if "$ref" in node:
        name = str(node["$ref"]).rsplit("/", 1)[-1]
        if name in seen:
            return {"type": "object", "properties": {}, "additionalProperties": False, "required": []}
        resolved = defs.get(name)
        if not isinstance(resolved, dict):
            return {key: value for key, value in node.items() if key != "$ref"}
        merged = {**_resolve_refs(resolved, defs, seen | {name}), **{key: value for key, value in node.items() if key != "$ref"}}
        return _resolve_refs(merged, defs, seen | {name})
    return {key: _resolve_refs(value, defs, seen) for key, value in node.items()}


def _normalize_schema_node(node: Any) -> None:
    if isinstance(node, list):
        for item in node:
            _normalize_schema_node(item)
        return
    if not isinstance(node, dict):
        return
    if "const" in node and "type" not in node:
        node["type"] = _json_type(node["const"])
    if "enum" in node and "type" not in node and node["enum"]:
        node["type"] = _json_type(node["enum"][0])
    any_of = node.get("anyOf") or node.get("oneOf")
    if isinstance(any_of, list):
        _flatten_nullable_union(node, any_of)
    if node.get("type") == "array" or "items" in node:
        items = node.get("items")
        if items is None:
            node["items"] = {"type": "string"}
        elif isinstance(items, list):
            node["items"] = items[0] if items else {"type": "string"}
        elif not isinstance(items, dict):
            node["items"] = {"type": "string"}
    properties = node.get("properties")
    if isinstance(properties, dict) or node.get("type") == "object":
        node.setdefault("type", "object")
        node.setdefault("properties", properties if isinstance(properties, dict) else {})
        node.setdefault("additionalProperties", False)
    for value in list(node.values()):
        _normalize_schema_node(value)


def _flatten_nullable_union(node: dict[str, Any], options: list[Any]) -> None:
    types: list[str] = []
    remainder: list[dict[str, Any]] = []
    nullable = False
    for option in options:
        if not isinstance(option, dict):
            continue
        option_type = option.get("type")
        if option_type == "null":
            nullable = True
            continue
        remainder.append(option)
        if isinstance(option_type, list):
            types.extend(str(item) for item in option_type if item != "null")
        elif option_type:
            types.append(str(option_type))
    if remainder and all(item.get("type") and not (set(item) - {"type", "enum", "const"}) for item in remainder):
        unique = list(dict.fromkeys(types))
        if nullable:
            unique.append("null")
        if unique:
            node["type"] = unique if len(unique) > 1 else unique[0]
            node.pop("anyOf", None)
            node.pop("oneOf", None)
            if remainder[0].get("enum"):
                node["enum"] = remainder[0]["enum"]
            return
    if len(remainder) == 1:
        merged = dict(remainder[0])
        node.clear()
        node.update(merged)
        if nullable:
            current = node.get("type")
            if isinstance(current, list):
                if "null" not in current:
                    node["type"] = [*current, "null"]
            elif current:
                node["type"] = [current, "null"]
            else:
                node["type"] = ["object", "null"]
        _normalize_schema_node(node)


def _force_string_array_items(node: Any, field_name: str | None = None) -> None:
    if isinstance(node, list):
        for item in node:
            _force_string_array_items(item)
        return
    if not isinstance(node, dict):
        return
    properties = node.get("properties")
    if isinstance(properties, dict):
        for name, spec in properties.items():
            if isinstance(spec, dict) and name in STRING_LIST_FIELDS and not _items_are_objects(spec):
                spec["type"] = "array"
                spec["items"] = {"type": "string"}
            _force_string_array_items(spec, name)
    if node.get("type") == "array" or "items" in node:
        items = node.get("items")
        if field_name in STRING_LIST_FIELDS and not _items_are_objects(node):
            node["items"] = {"type": "string"}
        elif isinstance(items, dict):
            _force_string_array_items(items)
    for key, value in list(node.items()):
        if key in {"properties", "items"}:
            continue
        _force_string_array_items(value)


def _items_are_objects(spec: dict[str, Any]) -> bool:
    items = spec.get("items")
    if not isinstance(items, dict):
        return False
    if items.get("type") == "object" or isinstance(items.get("properties"), dict):
        return True
    for option in list(items.get("anyOf") or []) + list(items.get("oneOf") or []):
        if isinstance(option, dict) and (option.get("type") == "object" or isinstance(option.get("properties"), dict)):
            return True
    return False


def _apply_groq_strict(node: Any) -> None:
    if isinstance(node, list):
        for item in node:
            _apply_groq_strict(item)
        return
    if not isinstance(node, dict):
        return
    properties = node.get("properties")
    if isinstance(properties, dict) or node.get("type") == "object":
        props = properties if isinstance(properties, dict) else {}
        previously_required = {str(name) for name in (node.get("required") or [])}
        node["type"] = "object" if node.get("type") in {None, "object"} else node.get("type")
        if node.get("type") == "object" or props:
            node["properties"] = props
            node["additionalProperties"] = False
            for name, spec in props.items():
                if name not in previously_required:
                    _make_nullable(spec)
            node["required"] = list(props.keys())
    items = node.get("items")
    if node.get("type") == "array" and not isinstance(items, dict):
        node["items"] = {"type": "string"}
    for value in list(node.values()):
        _apply_groq_strict(value)


def _make_nullable(spec: Any) -> None:
    if not isinstance(spec, dict):
        return
    current = spec.get("type")
    if current == "null" or (isinstance(current, list) and "null" in current):
        return
    if isinstance(current, list):
        spec["type"] = [*current, "null"]
        return
    if current:
        spec["type"] = [current, "null"]
        return
    if "anyOf" in spec or "oneOf" in spec:
        options = list(spec.get("anyOf") or spec.get("oneOf") or [])
        if not any(isinstance(item, dict) and item.get("type") == "null" for item in options):
            options.append({"type": "null"})
            spec["anyOf"] = options
            spec.pop("oneOf", None)


def _apply_object_closed(node: Any, nullable_optional: bool) -> None:
    if isinstance(node, list):
        for item in node:
            _apply_object_closed(item, nullable_optional)
        return
    if not isinstance(node, dict):
        return
    properties = node.get("properties")
    if isinstance(properties, dict):
        node["type"] = "object"
        node["additionalProperties"] = False
        existing_required = [str(name) for name in (node.get("required") or [])]
        node["required"] = list(properties.keys()) if nullable_optional else (existing_required or list(properties.keys()))
    if node.get("type") == "array" and not isinstance(node.get("items"), dict):
        node["items"] = {"type": "string"}
    for value in list(node.values()):
        _apply_object_closed(value, nullable_optional)


def _groq_incompatibility(node: Any) -> str | None:
    if isinstance(node, list):
        for item in node:
            reason = _groq_incompatibility(item)
            if reason:
                return reason
        return None
    if not isinstance(node, dict):
        return None
    if node.get("additionalProperties") not in {None, False} and isinstance(node.get("additionalProperties"), dict):
        return "open additionalProperties"
    if node.get("additionalProperties") is True:
        return "additionalProperties true"
    if node.get("type") == "object":
        props = node.get("properties") or {}
        if not props and node.get("additionalProperties") is not False:
            return "free-form object"
    if "patternProperties" in node or "unevaluatedProperties" in node:
        return "unconstrained object properties"
    if node.get("type") == "array" and "items" not in node:
        return "array missing items"
    if "$ref" in node:
        return "unresolved $ref"
    for value in node.values():
        reason = _groq_incompatibility(value)
        if reason:
            return reason
    return None


def _strip_schema_metadata_keys(node: Any, keys: set[str]) -> None:
    """Strip JSON Schema metadata without deleting property names such as `title`."""
    if isinstance(node, list):
        for item in node:
            _strip_schema_metadata_keys(item, keys)
        return
    if not isinstance(node, dict):
        return
    properties = node.get("properties")
    for key in list(node.keys()):
        if key in keys:
            node.pop(key, None)
    if isinstance(properties, dict):
        for spec in properties.values():
            _strip_schema_metadata_keys(spec, keys)
    items = node.get("items")
    if isinstance(items, (dict, list)):
        _strip_schema_metadata_keys(items, keys)
    additional = node.get("additionalProperties")
    if isinstance(additional, dict):
        _strip_schema_metadata_keys(additional, keys)
    for option_key in ("anyOf", "oneOf", "allOf"):
        _strip_schema_metadata_keys(node.get(option_key), keys)


def _json_type(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int) and not isinstance(value, bool):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    if value is None:
        return "null"
    return "string"


def _deep_copy(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _deep_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_deep_copy(item) for item in value]
    return value
