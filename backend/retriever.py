"""Lightweight vector retrieval over the product catalog.

Default backend is TF-IDF (word 1-2 grams + character 3-5 grams) with cosine
similarity, computed with scikit-learn. With ~50 products this is fast
(indexing takes milliseconds), needs no model downloads, no vector database
and no extra services — and it handles product names, SKUs, misspellings and
spec keywords very well.

An OpenAI-embeddings backend is available for anyone who wants semantic
embeddings: set RETRIEVER=openai (uses the same API key as the LLM).

To swap in another engine, implement `BaseIndex.search()` and return
`list[ScoredProduct]` — nothing else in the pipeline changes.
"""
from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import (
    ENGLISH_STOP_WORDS,
    TfidfVectorizer,
)
from sklearn.pipeline import FeatureUnion

from .catalog import Catalog
from .models import Product

log = logging.getLogger("retriever")

_TOKEN_RE = re.compile(r"(?u)\b\w\w+\b")


def _stem(token: str) -> str:
    """Very light plural stemming ("laptops" -> "laptop", "earbuds" ->
    "earbud"). Applied to both documents and queries, so even aggressive
    strips only affect matching, never display."""
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def word_analyzer(text: str) -> list[str]:
    """Tokenizer + stopword removal + light stemming + word 1-2 grams.

    Passed to TfidfVectorizer as a callable analyzer (which bypasses
    ngram_range/stop_words parameters)."""
    tokens = [
        _stem(t)
        for t in _TOKEN_RE.findall(text.lower())
        if t not in ENGLISH_STOP_WORDS
    ]
    bigrams = [f"{a} {b}" for a, b in zip(tokens, tokens[1:])]
    return tokens + bigrams


@dataclass
class ScoredProduct:
    product: Product
    score: float


class BaseIndex:
    def search(self, query: str, k: int) -> list[ScoredProduct]:
        raise NotImplementedError

    def rebuild_if_needed(self) -> bool:
        """Re-index if the underlying catalog changed. Returns True when rebuilt."""
        raise NotImplementedError

    @property
    def size(self) -> int:
        raise NotImplementedError


class TfidfIndex(BaseIndex):
    """Sparse TF-IDF vector search. No model download, no services."""

    def __init__(self, catalog: Catalog):
        self._catalog = catalog
        self._lock = threading.Lock()
        self._matrix = None
        self._vectorizer = None
        self._products: list[Product] = []
        self.rebuild()

    def rebuild(self) -> None:
        with self._lock:
            catalog_products = self._catalog.products
            docs = self._catalog.documents
            if not docs:
                raise ValueError("Cannot build index: catalog is empty.")

            vectorizer = FeatureUnion(
                [
                    (
                        "word",
                        TfidfVectorizer(
                            analyzer=word_analyzer,
                            sublinear_tf=True,
                            min_df=1,
                        ),
                    ),
                    (
                        "char",
                        TfidfVectorizer(
                            analyzer="char_wb",
                            ngram_range=(3, 5),
                            sublinear_tf=True,
                            min_df=1,
                        ),
                    ),
                ],
                # Char n-grams help with typos and SKUs but also create
                # spurious matches on short off-topic queries, so they get a
                # lower weight than the word features.
                transformer_weights={"word": 1.0, "char": 0.3},
            )
            matrix = vectorizer.fit_transform(docs)

            # Atomic swap so concurrent searches never see a partial index.
            self._vectorizer = vectorizer
            self._matrix = matrix
            self._products = list(catalog_products)
            log.info("TF-IDF index built: %d products, %d features",
                     matrix.shape[0], matrix.shape[1])

    def rebuild_if_needed(self) -> bool:
        if self._catalog.load():  # hot-reload: returns True only if CSV changed
            self.rebuild()
            return True
        return False

    def search(self, query: str, k: int = 4) -> list[ScoredProduct]:
        self.rebuild_if_needed()
        with self._lock:
            if self._matrix is None or not self._products:
                return []
            q = self._vectorizer.transform([query])
            # Rows are L2-normalized by TfidfVectorizer, so the dot product
            # of query and document vectors is the cosine similarity.
            scores = (self._matrix @ q.T).toarray().ravel()
            order = np.argsort(scores)[::-1][:k]
            return [
                ScoredProduct(product=self._products[i], score=float(scores[i]))
                for i in order
                if scores[i] > 0
            ]

    @property
    def size(self) -> int:
        return len(self._products)


class OpenAIIndex(BaseIndex):
    """Semantic retrieval using an OpenAI-compatible embeddings endpoint.

    Enable with RETRIEVER=openai. Embeddings are cached in memory and only
    recomputed when the catalog changes.
    """

    def __init__(self, catalog: Catalog, api_key: str, base_url: str,
                 model: str = "text-embedding-3-small"):
        import httpx

        self._catalog = catalog
        self._model = model
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=30.0,
        )
        self._lock = threading.Lock()
        self._products: list[Product] = []
        self._matrix: np.ndarray | None = None
        self.rebuild()

    def rebuild(self) -> None:
        with self._lock:
            products = list(self._catalog.products)
            docs = self._catalog.documents
            embeddings = self._embed_batch(docs)
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            self._matrix = embeddings / norms
            self._products = products
            log.info("Embedding index built: %d products", len(products))

    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        import httpx

        # The API accepts ~2048 inputs per request; batches of 128 are plenty.
        vectors: list[list[float]] = []
        for start in range(0, len(texts), 128):
            batch = texts[start:start + 128]
            try:
                resp = self._client.post(
                    "/embeddings",
                    json={"model": self._model, "input": batch},
                )
                resp.raise_for_status()
                data = sorted(resp.json()["data"], key=lambda d: d["index"])
                vectors.extend(item["embedding"] for item in data)
            except (httpx.HTTPError, KeyError, ValueError) as exc:
                raise RuntimeError(f"Embedding request failed: {exc}") from exc
        return np.asarray(vectors, dtype=np.float32)

    def rebuild_if_needed(self) -> bool:
        if self._catalog.load():
            self.rebuild()
            return True
        return False

    def search(self, query: str, k: int = 4) -> list[ScoredProduct]:
        self.rebuild_if_needed()
        with self._lock:
            if self._matrix is None or not self._products:
                return []
            q = self._embed_batch([query])[0]
            q = q / (np.linalg.norm(q) or 1.0)
            scores = self._matrix @ q
            order = np.argsort(scores)[::-1][:k]
            return [
                ScoredProduct(product=self._products[i], score=float(scores[i]))
                for i in order
                if scores[i] > 0
            ]

    @property
    def size(self) -> int:
        return len(self._products)


def build_index(settings, catalog: Catalog) -> BaseIndex:
    if settings.retriever.lower() == "openai":
        if not settings.llm_api_key:
            raise ValueError(
                "RETRIEVER=openai requires LLM_API_KEY (or OPENAI_API_KEY)."
            )
        return OpenAIIndex(catalog, settings.llm_api_key, settings.llm_base_url)
    return TfidfIndex(catalog)
