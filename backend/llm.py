"""Minimal client for any OpenAI-compatible chat-completions endpoint.

Works with OpenAI, Groq, Together, Mistral, OpenRouter, or a local server
(Ollama, vLLM, LM Studio) — just set LLM_BASE_URL and LLM_MODEL.

If no API key is configured the pipeline still works: it answers by listing
the matching catalog entries directly (see rag.py, "no-LLM mode").
"""
from __future__ import annotations

import logging

import httpx

from .config import Settings

log = logging.getLogger("llm")


class LLMError(Exception):
    pass


class LLMClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.available = bool(settings.llm_api_key)
        self._client: httpx.AsyncClient | None = None
        if self.available:
            self._client = httpx.AsyncClient(
                base_url=settings.llm_base_url.rstrip("/"),
                headers={"Authorization": f"Bearer {settings.llm_api_key}"},
                timeout=settings.llm_timeout_seconds,
            )
        else:
            log.warning(
                "No LLM API key configured — running in no-LLM mode. "
                "The bot will return matched products without generated answers."
            )

    async def complete(self, system: str, user: str) -> str:
        if not self.available or self._client is None:
            raise LLMError("LLM is not configured.")
        try:
            resp = await self._client.post(
                "/chat/completions",
                json={
                    "model": self.settings.llm_model,
                    "temperature": self.settings.llm_temperature,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                },
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
        except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
            raise LLMError(f"LLM request failed: {exc}") from exc
        content = (content or "").strip()
        if not content:
            raise LLMError("LLM returned an empty response.")
        return content

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
