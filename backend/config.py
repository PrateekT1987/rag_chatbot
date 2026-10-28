"""Application configuration.

All settings come from environment variables (optionally via a .env file).
Nothing secret is ever exposed to the browser — the widget only talks to
this backend, which holds the API key server-side.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root (next to backend/) if present.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _env_bool(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


@dataclass
class Settings:
    # --- Catalog ---------------------------------------------------------
    catalog_path: Path = field(
        default_factory=lambda: Path(
            os.getenv("CATALOG_PATH", str(PROJECT_ROOT / "data" / "products.csv"))
        )
    )

    # --- LLM (any OpenAI-compatible endpoint) ----------------------------
    llm_api_key: str = os.getenv("LLM_API_KEY", os.getenv("OPENAI_API_KEY", ""))
    llm_base_url: str = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
    llm_model: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    llm_temperature: float = _env_float("LLM_TEMPERATURE", 0.1)
    llm_timeout_seconds: float = _env_float("LLM_TIMEOUT_SECONDS", 30.0)
    # Cap on generated answer length — protects latency and cost.
    llm_max_tokens: int = _env_int("LLM_MAX_TOKENS", 400)
    # Extra attempts for transient LLM failures (429 / 5xx / network).
    llm_max_retries: int = _env_int("LLM_MAX_RETRIES", 2)

    # --- Retrieval -------------------------------------------------------
    # "tfidf" (default, zero infrastructure) or "openai" (embeddings API).
    retriever: str = os.getenv("RETRIEVER", "tfidf")
    top_k: int = _env_int("TOP_K", 4)
    # Minimum cosine similarity required to answer at all. Questions below
    # this score fall back instead of letting the LLM guess. 0.10 catches
    # clearly off-topic questions while accepting every in-domain phrasing
    # we tested; the LLM system prompt is the second guard for the rest.
    relevance_threshold: float = _env_float("RELEVANCE_THRESHOLD", 0.10)

    # Extra sentence appended to fallback answers, e.g.
    # "email sales@example.com or call +1 555 0100".
    fallback_contact: str = os.getenv("FALLBACK_CONTACT", "")

    # --- Server ----------------------------------------------------------
    cors_origins: list[str] = field(
        default_factory=lambda: [
            o.strip()
            for o in os.getenv("CORS_ORIGINS", "").split(",")
            if o.strip()
        ]
    )
    rate_limit_per_minute: int = _env_int("RATE_LIMIT_PER_MINUTE", 30)
    # Optional token required for POST /api/reindex (leave empty to disable).
    admin_token: str = os.getenv("ADMIN_TOKEN", "")
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = _env_int("PORT", 8000)

    def __post_init__(self) -> None:
        self.catalog_path = Path(self.catalog_path)
