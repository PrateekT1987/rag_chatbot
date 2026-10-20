"""Shared fixtures: a fake LLM so tests never hit a real API."""
from __future__ import annotations

from pathlib import Path

import pytest

from backend.catalog import Catalog
from backend.config import Settings
from backend.retriever import TfidfIndex

CATALOG_CSV = "data/products.csv"


class FakeLLM:
    """Duck-typed stand-in for LLMClient."""

    def __init__(self, reply: str = "The AUR-H700 costs 129.99 USD.", fail: bool = False):
        self.available = True
        self.reply = reply
        self.fail = fail
        self.calls: list[dict] = []

    async def complete(self, system: str, user: str) -> str:
        self.calls.append({"system": system, "user": user})
        if self.fail:
            from backend.llm import LLMError

            raise LLMError("simulated outage")
        return self.reply

    async def close(self) -> None:  # pragma: no cover
        pass


class NoLLM:
    available = False

    async def complete(self, system: str, user: str) -> str:  # pragma: no cover
        raise AssertionError("should not be called")

    async def close(self) -> None:  # pragma: no cover
        pass


@pytest.fixture
def settings() -> Settings:
    s = Settings()
    s.catalog_path = __import__("pathlib").Path(CATALOG_CSV)
    return s


@pytest.fixture
def catalog() -> Catalog:
    return Catalog(CATALOG_CSV)


@pytest.fixture
def index(catalog) -> TfidfIndex:
    return TfidfIndex(catalog)
