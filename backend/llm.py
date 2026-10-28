"""Minimal client for any OpenAI-compatible chat-completions endpoint.

Works with OpenAI, Groq, Together, Mistral, OpenRouter, or a local server
(Ollama, vLLM, LM Studio) — just set LLM_BASE_URL and LLM_MODEL.

If no API key is configured the pipeline still works: it answers by listing
the matching catalog entries directly (see rag.py, "no-LLM mode").
"""
from __future__ import annotations

import asyncio
import logging

import httpx

from .config import Settings

log = logging.getLogger("llm")


class LLMError(Exception):
    pass


class LLMClient:
    def __init__(self, settings: Settings,
                 transport: httpx.AsyncBaseTransport | None = None):
        self.settings = settings
        self.available = bool(settings.llm_api_key)
        self._client: httpx.AsyncClient | None = None
        if self.available:
            self._client = httpx.AsyncClient(
                base_url=settings.llm_base_url.rstrip("/"),
                headers={"Authorization": f"Bearer {settings.llm_api_key}"},
                timeout=settings.llm_timeout_seconds,
                transport=transport,
            )
        else:
            log.warning(
                "No LLM API key configured — running in no-LLM mode. "
                "The bot will return matched products without generated answers."
            )

    async def complete(self, system: str, user: str) -> str:
        if not self.available or self._client is None:
            raise LLMError("LLM is not configured.")

        payload = {
            "model": self.settings.llm_model,
            "temperature": self.settings.llm_temperature,
            "max_tokens": self.settings.llm_max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }

        attempts = max(1, self.settings.llm_max_retries + 1)
        last_exc: Exception | None = None

        for attempt in range(attempts):
            try:
                resp = await self._client.post("/chat/completions", json=payload)
                resp.raise_for_status()
                data = resp.json()

                # Some providers (e.g. OpenRouter) return HTTP 200 with an
                # error envelope instead of an HTTP error. Surface it clearly.
                if isinstance(data, dict) and data.get("error"):
                    err = data["error"]
                    message = (
                        err.get("message") if isinstance(err, dict) else str(err)
                    )
                    code = err.get("code") if isinstance(err, dict) else None
                    last_exc = LLMError(f"provider error {code}: {message}")
                    if code in (429, 500, 502, 503, 504) and attempt < attempts - 1:
                        delay = 0.5 * (2 ** attempt)
                        log.warning(
                            "LLM provider error %s, retrying in %.1fs", code, delay
                        )
                        await asyncio.sleep(delay)
                        continue
                    break

                content = data["choices"][0]["message"]["content"]
                content = (content or "").strip()
                if not content:
                    raise LLMError("LLM returned an empty response.")
                return content
            except httpx.HTTPStatusError as exc:
                last_exc = exc
                status = exc.response.status_code
                # Auth / validation errors will not fix themselves: stop.
                if status != 429 and status < 500:
                    break
                if attempt < attempts - 1:
                    delay = 0.5 * (2 ** attempt)
                    log.warning(
                        "LLM HTTP %s, retrying in %.1fs (attempt %d/%d)",
                        status, delay, attempt + 1, attempts,
                    )
                    await asyncio.sleep(delay)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_exc = exc
                if attempt < attempts - 1:
                    delay = 0.5 * (2 ** attempt)
                    log.warning(
                        "LLM %s, retrying in %.1fs (attempt %d/%d)",
                        type(exc).__name__, delay, attempt + 1, attempts,
                    )
                    await asyncio.sleep(delay)
            except (KeyError, IndexError, ValueError) as exc:
                last_exc = exc
                break

        raise LLMError(f"LLM request failed: {last_exc}") from last_exc

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
