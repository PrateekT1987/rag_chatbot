"""Pydantic schemas for the HTTP API."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=2000)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=1000)
    # Last few turns of conversation, for follow-up questions.
    history: list[ChatTurn] = Field(default_factory=list, max_length=10)


class ProductCard(BaseModel):
    """A matched product the widget can render as a card."""

    sku: str
    name: str
    category: str
    price: float
    currency: str
    availability: str
    url: str | None = None


class ChatResponse(BaseModel):
    answer: str
    # True when we could not find relevant catalog data and refused to guess.
    fallback: bool = False
    products: list[ProductCard] = Field(default_factory=list)


class Product(BaseModel):
    """One row of the catalog CSV."""

    sku: str
    name: str
    category: str = ""
    price: float = 0.0
    currency: str = "USD"
    availability: str = ""
    description: str = ""
    # Raw specs string, e.g. "Battery life: 30h; Bluetooth: 5.3".
    specs: str = ""
    tags: str = ""
    url: str = ""
