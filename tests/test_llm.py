"""LLM client hardening: output cap, retries, and failure semantics."""
from __future__ import annotations

import json

import httpx
import pytest

from backend.config import Settings
from backend.llm import LLMClient, LLMError


def make_settings(**overrides) -> Settings:
    s = Settings()
    s.llm_api_key = "test-key"
    s.llm_base_url = "https://llm.test/v1"
    s.llm_model = "test-model"
    s.llm_max_tokens = 400
    s.llm_max_retries = 2
    for key, value in overrides.items():
        setattr(s, key, value)
    return s


@pytest.fixture(autouse=True)
def fast_retries(monkeypatch):
    """Skip the real backoff delays in tests."""
    import backend.llm as llm_module

    async def no_sleep(_seconds):
        return None

    monkeypatch.setattr(llm_module.asyncio, "sleep", no_sleep)


async def test_request_includes_max_tokens():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "hello"}}]}
        )

    client = LLMClient(make_settings(), transport=httpx.MockTransport(handler))
    answer = await client.complete("system", "user")
    await client.close()

    assert answer == "hello"
    assert seen["max_tokens"] == 400
    assert seen["temperature"] == 0.1
    assert seen["messages"][0]["role"] == "system"


async def test_retries_on_429_then_succeeds():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) < 3:
            return httpx.Response(429, json={"error": "rate limited"})
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "recovered"}}]}
        )

    client = LLMClient(make_settings(), transport=httpx.MockTransport(handler))
    answer = await client.complete("s", "u")
    await client.close()

    assert answer == "recovered"
    assert len(calls) == 3


async def test_does_not_retry_auth_errors():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(401, json={"error": "bad key"})

    client = LLMClient(make_settings(), transport=httpx.MockTransport(handler))
    with pytest.raises(LLMError):
        await client.complete("s", "u")
    await client.close()

    assert len(calls) == 1, "401 must not be retried"


async def test_gives_up_after_max_retries():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(503, json={"error": "unavailable"})

    client = LLMClient(make_settings(llm_max_retries=1),
                       transport=httpx.MockTransport(handler))
    with pytest.raises(LLMError):
        await client.complete("s", "u")
    await client.close()

    assert len(calls) == 2  # initial attempt + 1 retry


async def test_empty_completion_is_an_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "  "}}]}
        )

    client = LLMClient(make_settings(), transport=httpx.MockTransport(handler))
    with pytest.raises(LLMError):
        await client.complete("s", "u")
    await client.close()


async def test_provider_error_envelope_is_surfaced():
    """Some providers return HTTP 200 with an error object (e.g. retired model)."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"error": {"message": "model retired", "code": 404}},
        )

    client = LLMClient(make_settings(), transport=httpx.MockTransport(handler))
    with pytest.raises(LLMError, match="model retired"):
        await client.complete("s", "u")
    await client.close()


async def test_retries_provider_429_envelope():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(
                200, json={"error": {"message": "busy", "code": 429}}
            )
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "ok"}}]}
        )

    client = LLMClient(make_settings(), transport=httpx.MockTransport(handler))
    assert await client.complete("s", "u") == "ok"
    await client.close()
    assert len(calls) == 2

