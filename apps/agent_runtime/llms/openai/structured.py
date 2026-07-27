"""Compatibility helpers for OpenAI structured chat output."""

import json
from typing import TypeVar

from openai import AsyncOpenAI
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def _message_content(response: object) -> str:
    choice = response.choices[0]
    message = choice.message
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(message, dict):
        value = message.get("content")
        if isinstance(value, str):
            return value
    raise ValueError("OpenAI returned no message content.")


def _json_payload(content: str) -> str:
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end >= start:
        return text[start : end + 1]
    return text


async def parse_chat_completion(
    client: AsyncOpenAI,
    *,
    model: str,
    messages: list[dict[str, str]],
    response_model: type[T],
    temperature: float = 0,
) -> T:
    """Return a Pydantic model with support for old and new OpenAI SDKs."""
    parse = getattr(client.chat.completions, "parse", None)
    if parse is None and hasattr(client, "beta"):
        parse = getattr(client.beta.chat.completions, "parse", None)

    if parse is not None:
        try:
            response = await parse(
                model=model,
                temperature=temperature,
                response_format=response_model,
                messages=messages,
            )
            parsed = response.choices[0].message.parsed
            if parsed:
                return parsed
            raise ValueError(
                f"OpenAI returned no parsed {response_model.__name__} output."
            )
        except Exception:
            # Some SDK/model combinations reject otherwise-valid Pydantic schemas.
            # Fall through to JSON mode and validate locally.
            pass

    schema = response_model.model_json_schema()
    try:
        response = await client.chat.completions.create(
            model=model,
            temperature=temperature,
            response_format={"type": "json_object"},
            messages=[
                *messages,
                {
                    "role": "system",
                    "content": (
                        "Return only valid JSON matching this JSON Schema: "
                        f"{json.dumps(schema, ensure_ascii=True)}"
                    ),
                },
            ],
        )
    except Exception:
        raise

    return response_model.model_validate_json(_json_payload(_message_content(response)))
