import asyncio

from pydantic import BaseModel

from services.llm.errors import LLMProviderError, provider_pressure
from services.llm.pressure import pause_remaining_seconds, reset_provider_pressure
from services.queue.streams import retry_delay
from services.llm.fallback import FallbackLLMProvider, LLMRouteCandidate
from services.llm.models import LLMMessage, StructuredLLMRequest
from services.llm.openai_compatible import OpenAICompatibleProvider


class _Ok(BaseModel):
    ok: bool = True


class _Calls:
    def __init__(self, name: str, *, outage: bool = False):
        self.name = name
        self.outage = outage
        self.calls = 0

    async def generate_structured(self, request, schema):
        del request, schema
        self.calls += 1
        if self.outage:
            raise LLMProviderError(
                "krutrim error 503: db_unavailable verifying your key",
                retryable=True,
                status_code=503,
                provider_outage=True,
            )
        return _Ok()


def test_model_failure_still_tries_the_next_model_on_the_same_provider():
    class _Once:
        name = "krutrim"
        calls = 0

        async def generate_structured(self, request, schema):
            del request, schema
            self.calls += 1
            raise LLMProviderError("krutrim error 429: slow down", retryable=True, status_code=429)

    once = _Once()
    second = _Calls("krutrim")
    wrapper = FallbackLLMProvider(
        "krutrim",
        [
            LLMRouteCandidate(once, "gpt-oss-120b"),
            LLMRouteCandidate(second, "gemma-4-31b-it"),
        ],
    )
    request = StructuredLLMRequest(
        model="gpt-oss-120b",
        schema_name="Ok",
        messages=[LLMMessage(role="user", content="consolidate")],
    )
    parsed = asyncio.run(wrapper.generate_structured(request, _Ok))
    assert parsed.ok is True
    assert once.calls == 1
    assert second.calls == 1


def test_provider_outage_skips_sibling_models_and_uses_the_other_provider():
    first = _Calls("krutrim", outage=True)
    second = _Calls("krutrim", outage=True)
    mistral = _Calls("mistral")
    wrapper = FallbackLLMProvider(
        "krutrim",
        [
            LLMRouteCandidate(first, "gpt-oss-120b"),
            LLMRouteCandidate(second, "gemma-4-31b-it"),
            LLMRouteCandidate(mistral, "zai-glm-5-3-flash"),
        ],
    )
    request = StructuredLLMRequest(
        model="gpt-oss-120b",
        schema_name="Ok",
        messages=[LLMMessage(role="user", content="consolidate")],
    )
    parsed = asyncio.run(wrapper.generate_structured(request, _Ok))
    assert parsed.ok is True
    assert first.calls == 1
    assert second.calls == 0
    assert mistral.calls == 1
    assert wrapper.last_successful_model == "zai-glm-5-3-flash"


def test_key_database_outage_is_not_retried_on_the_same_model():
    reset_provider_pressure()
    provider = OpenAICompatibleProvider(
        name="krutrim",
        api_key="test",
        base_url="http://localhost",
        default_model="gpt-oss-120b",
        timeout_seconds=1,
        max_retries=3,
        max_concurrency=1,
    )
    posts = {"count": 0}

    class _Response:
        status_code = 503
        text = (
            '{"error":{"message":"The service is briefly overloaded verifying your key. '
            'Please retry in a moment.","type":"api_error","code":"db_unavailable"}}'
        )
        headers: dict[str, str] = {}

    class _Client:
        async def post(self, path, headers, json):
            del path, headers, json
            posts["count"] += 1
            return _Response()

    async def client():
        return _Client()

    provider._http_client = client  # type: ignore[method-assign]
    try:
        asyncio.run(provider._post_with_retries("/chat/completions", {"model": "gpt-oss-120b"}))
    except LLMProviderError as error:
        assert error.provider_outage is True
        assert error.retryable is True
    else:
        raise AssertionError("provider outage should fail over immediately")
    assert posts["count"] == 1


def test_rate_limit_is_not_retried_on_the_same_model():
    reset_provider_pressure()
    provider = OpenAICompatibleProvider(
        name="mistral",
        api_key="test",
        base_url="http://localhost",
        default_model="zai-glm-5-3-flash",
        timeout_seconds=1,
        max_retries=3,
        max_concurrency=1,
    )
    posts = {"count": 0}

    class _Response:
        status_code = 429
        text = '{"message":"too many requests"}'
        headers: dict[str, str] = {}

    class _Client:
        async def post(self, path, headers, json):
            del path, headers, json
            posts["count"] += 1
            return _Response()

    async def client():
        return _Client()

    provider._http_client = client  # type: ignore[method-assign]
    try:
        asyncio.run(provider._post_with_retries("/chat/completions", {"model": "zai-glm-5-3-flash"}))
    except LLMProviderError as error:
        assert error.status_code == 429
        assert error.provider_outage is False
        assert provider_pressure(error) is True
    else:
        raise AssertionError("a rate limit should move to the next model")
    assert posts["count"] == 1


def test_mistral_capacity_refusal_is_one_call_then_the_model_stays_paused():
    reset_provider_pressure()
    provider = OpenAICompatibleProvider(
        name="mistral",
        api_key="test",
        base_url="http://localhost",
        default_model="zai-glm-5-3-flash",
        timeout_seconds=1,
        max_retries=3,
        max_concurrency=1,
    )
    posts = {"count": 0}

    class _Response:
        status_code = 429
        text = (
            '{"object":"error","message":"Not enough capacity available for this request, please retry later.",'
            '"type":"backend_out_of_capacity","code":"3505","raw_status_code":429}'
        )
        headers: dict[str, str] = {}

    class _Client:
        async def post(self, path, headers, json):
            del path, headers, json
            posts["count"] += 1
            return _Response()

    async def client():
        return _Client()

    provider._http_client = client  # type: ignore[method-assign]

    async def one_call():
        try:
            await provider._post_with_retries("/chat/completions", {"model": "zai-glm-5-3-flash"})
        except LLMProviderError:
            return

    asyncio.run(one_call())
    asyncio.run(one_call())
    assert posts["count"] == 1
    assert pause_remaining_seconds("mistral") >= 120


def test_overloaded_provider_accepts_one_call_then_the_queue_holds_the_rest():
    reset_provider_pressure()
    provider = OpenAICompatibleProvider(
        name="mistral",
        api_key="test",
        base_url="http://localhost",
        default_model="zai-glm-5-3-flash",
        timeout_seconds=1,
        max_retries=3,
        max_concurrency=4,
    )
    posts = {"count": 0}
    started = asyncio.Event()

    class _Response:
        status_code = 429
        text = '{"message":"too many requests"}'
        headers: dict[str, str] = {}

    class _Client:
        async def post(self, path, headers, json):
            del path, headers, json
            posts["count"] += 1
            started.set()
            await asyncio.sleep(0.05)
            return _Response()

    async def client():
        return _Client()

    provider._http_client = client  # type: ignore[method-assign]

    async def one_call():
        try:
            await provider._post_with_retries("/chat/completions", {"model": "zai-glm-5-3-flash"})
        except LLMProviderError:
            return

    async def run():
        first = asyncio.create_task(one_call())
        await started.wait()
        second = asyncio.create_task(one_call())
        await asyncio.gather(first, second)

    asyncio.run(run())
    assert posts["count"] == 1


def test_provider_refusal_waits_before_running_the_whole_job_again():
    error = LLMProviderError(
        "all structured LLM fallbacks failed: krutrim error 503: db_unavailable",
        retryable=True,
        status_code=503,
        provider_outage=True,
    )
    assert retry_delay(0, error) >= 30
    assert retry_delay(0, RuntimeError("schema mismatch")) < 30
